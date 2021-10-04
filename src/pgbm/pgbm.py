"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   https://github.com/elephaint/pgbm/blob/main/LICENSE

   Olivier Sprangers, Sebastian Schelter, Maarten de Rijke. 
   Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression.
   https://arxiv.org/abs/2106.01682 
   Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and
   Data Mining (KDD ’21), August 14–18, 2021, Virtual Event, Singapore.
   https://doi.org/10.1145/3447548.3467278

"""
#%% Import packages
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.autograd import grad
from torch.distributions import Normal, NegativeBinomial, Poisson, StudentT, LogNormal, Laplace, Uniform, TransformedDistribution, SigmoidTransform, AffineTransform, Gamma, Gumbel, Weibull
from torch.utils.cpp_extension import load
from pathlib import Path
import pickle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import r2_score
#%% Probabilistic Gradient Boosting Machines
class PGBM(nn.Module):
    def __init__(self, size=1, rank=0):
        super(PGBM, self).__init__()
        self.cwd = Path().cwd()
        self.world_size = size
        self.rank = rank
        if size > 1:
            self.distributed = True
        else:
            self.distributed = False
    
    def _init_params(self, params):       
        self.params = {}
        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'max_leaves',
                       'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds', 'feature_fraction', 'bagging_fraction', 
                       'seed', 'device', 'gpu_device_id', 'derivatives', 'distribution','checkpoint', 'tree_correlation', 
                       'monotone_constraints', 'monotone_iterations']
        param_defaults = [0.0, 2, 0.1, 1.0, 32, 
                          256, 100, 2, 100, 1, 1, 
                          2147483647, 'cpu', 0, 'exact', 'normal', False, np.log10(self.n_samples) / 100, 
                          np.zeros(self.n_features), 1]
        
        # Choose device
        if 'device' in params:
            if params['device'] == 'gpu':
                current_path = Path(__file__).parent.absolute()
                self.parallelsum_kernel = load(name='parallelsum', sources=[f'{current_path}/parallelsum.cpp', f'{current_path}/parallelsum_kernel.cu'])
                if 'gpu_device_id' in params:
                    self.device = torch.device(params['gpu_device_id'])
                else:
                    self.device = torch.device(0)
                    params['gpu_device_id'] = 0
                
            # This is experimental and has been tested only on Google Colab
            # See this: https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb
            elif params['device'] == 'tpu':
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            else:
                self.device = torch.device('cpu')
        else: 
            self.device = torch.device('cpu')
            params['device'] = 'cpu'
                   
        for i, param in enumerate(param_names):
            self.params[param] = params[param] if param in params else param_defaults[i]
            if param in ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'tree_correlation', 'monotone_constraints']:
                self.params[param] = torch.tensor(self.params[param], device=self.device, dtype=torch.float32)
                
        # Make sure we bound certain parameters
        self.params['min_data_in_leaf'] = torch.clamp(self.params['min_data_in_leaf'], 2)
        self.params['min_split_gain'] = torch.clamp(self.params['min_split_gain'], 0.0)
        self.params['monotone_iterations'] = np.maximum(self.params['monotone_iterations'], 1)
        self.feature_fraction = torch.tensor(self.params['feature_fraction'] * self.n_features, device = self.device, dtype=torch.int64).clamp(1, self.n_features)
        self.bagging_fraction = torch.tensor(self.params['bagging_fraction'] * self.n_samples, device = self.device, dtype=torch.int64).clamp(1, self.n_samples)
            
        # Set some additional params
        assert len(self.params['monotone_constraints']) == self.n_features, "The number of items in the monotonicity constraint list should be equal to the number of features in your dataset."
        self.params['max_nodes'] = self.params['max_leaves'] - 1
        self.any_monotone = torch.any(self.params['monotone_constraints'] != 0)
        if self.params['feature_fraction'] < 1:
            self.rng = torch.Generator(self.device)
            self.rng.manual_seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])  # cpu
        torch.cuda.manual_seed_all(self.params['seed'])  
        self.epsilon = 1.0e-4

    def _create_feature_bins(self, X):
        # Create array that contains the bins
        max_bin = self.params['max_bin']
        bins = torch.zeros((X.shape[1], max_bin), device=X.device)
        quantiles = torch.linspace(0, 1, max_bin, device=X.device)
        # For each feature, create max_bins based on frequency bins. 
        for i in range(X.shape[1]):
            xs = X[:, i]
            if self.distributed:
                current_bin = torch.quantile(xs, quantiles)
                # Synchronize across processes
                all_bins = torch.zeros((self.world_size, max_bin), dtype=torch.float32, device=X.device)
                all_bins[self.rank] = current_bin
                dist.all_reduce(all_bins, op=dist.ReduceOp.SUM)
                current_bin = torch.unique(torch.quantile(all_bins, quantiles))     
            else:
                current_bin = torch.unique(torch.quantile(xs, quantiles))
            # A bit inefficiency created here... some features usually have less than max_bin values (e.g. 1/0 indicator features). 
            bins[i, :len(current_bin)] = current_bin
            bins[i, len(current_bin):] = current_bin.max()
            
        return bins
    
    def _objective_approx(self, yhat_train, y, levels=None):
        yhat = yhat_train.detach()
        yhat.requires_grad = True
        yhat_upper = yhat + self.epsilon
        yhat_lower = yhat - self.epsilon
        loss = self.loss(yhat, y, levels)
        loss_upper = self.loss(yhat_upper, y, levels)
        loss_lower = self.loss(yhat_lower, y, levels)           
        gradient = grad(loss, yhat)[0]
        gradient_upper = grad(loss_upper, yhat_upper)[0]
        gradient_lower = grad(loss_lower, yhat_lower)[0]
        hessian = (gradient_upper - gradient_lower) / (2 * self.epsilon)
        
        return gradient, hessian

    def _leaf_prediction(self, gradient, hessian, node, estimator, return_mu=False):
        # Empirical mean
        gradient_sum = gradient.sum()
        hessian_sum = hessian.sum()
        N = torch.tensor(gradient.shape[0], device=gradient.device, dtype=torch.float32)       
        if self.distributed:
            dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(hessian_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(N, op=dist.ReduceOp.SUM)
        gradient_mean = gradient_sum / N
        hessian_mean = hessian_sum / N   
        # Empirical variance and covariance
        factor = 1 / (N - 1)
        gradient_variance = factor * ((gradient - gradient_mean)**2).sum()
        hessian_variance = factor * ((hessian - hessian_mean)**2).sum()
        covariance = factor * ((gradient - gradient_mean)*(hessian - hessian_mean)).sum()
        if self.distributed:
            # [Synchronize gradients and hessian variances and covariance]
            dist.all_reduce(gradient_variance, op=dist.ReduceOp.SUM)
            dist.all_reduce(hessian_variance, op=dist.ReduceOp.SUM)
            dist.all_reduce(covariance, op=dist.ReduceOp.SUM)       
        # Mean and variance of the leaf prediction
        lambda_scaled = self.params['lambda'] / N
        mu = gradient_mean / ( hessian_mean + lambda_scaled) - covariance / (hessian_mean + lambda_scaled)**2 + (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3
        if return_mu:
            return mu
        else:
            epsilon = 1e-6
            var = mu**2 * (gradient_variance / (gradient_mean + epsilon)**2 + hessian_variance / (hessian_mean + lambda_scaled)**2
                            - 2 * covariance / (gradient_mean * (hessian_mean + lambda_scaled) + epsilon) )
            # Save optimal prediction and node information
            self.leaves_idx[estimator, self.leaf_idx] = node
            self.leaves_mu[estimator, self.leaf_idx] = mu             
            self.leaves_var[estimator, self.leaf_idx] = var
            # Increase leaf idx       
            self.leaf_idx += 1   
            
    def _create_split(self, estimator, node, sample_features, split_feature_sample, split_bin, X_node, split_gain, n_samples, split_left, split_right, gradient, hessian):
        # Save split information
        self.nodes_idx[estimator, self.node_idx] = node
        self.nodes_split_feature[estimator, self.node_idx] = sample_features[split_feature_sample] 
        self.nodes_split_bin[estimator, self.node_idx] = split_bin
        self.node_idx += 1
        X_node_samples = torch.tensor(X_node.shape[1], device=X_node.device)
        # Synchronize
        if self.distributed:
            dist.all_reduce(X_node_samples, op=dist.ReduceOp.SUM)
        self.feature_importance[sample_features[split_feature_sample]] += split_gain * X_node_samples / n_samples 
        # Assign next node to samples if next node does not exceed max n_leaves
        criterion = 2 * (self.params['max_nodes'] - self.node_idx + 1)
        n_leaves_old = self.leaf_idx
        if (criterion >  self.params['max_leaves'] - n_leaves_old):
            self.train_nodes += split_left * node
        else:
            self._leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator)
        if (criterion >  self.params['max_leaves'] - n_leaves_old + 1):
            self.train_nodes += split_right * (node + 1)
        else:
            self._leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator)
    
    def _create_tree(self, X, gradient, hessian, estimator):
        # Set start node and start leaf index
        self.leaf_idx = torch.tensor(0, dtype=torch.int64, device=X.device)
        self.node_idx = 0
        node_constraint_idx = 0
        node = 1
        node_constraints = torch.zeros((self.params['max_nodes'] * 2 + 1, 3), dtype=torch.float32, device=X.device)
        node_constraints[0, 0] = node
        node_constraints[:, 1] = -np.inf
        node_constraints[:, 2] = np.inf
        n_features = X.shape[0]
        n_samples = torch.tensor(X.shape[1], device=X.device)
        # Synchronize total samples in node for feature importance
        if self.distributed:
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
        # Choose random subset of features
        if self.params['feature_fraction'] < 1:
            sample_features = torch.randperm(n_features, device=X.device, dtype=torch.int64, generator=self.rng)[:self.feature_fraction]
            Xe = X[sample_features]
        else:
            sample_features = torch.arange(n_features, device=X.device, dtype=torch.int64)
            Xe = X
        # Create tree
        drange = torch.arange(self.params['max_bin'], device=X.device);
        while (self.leaf_idx < self.params['max_leaves']):
            split_node = self.train_nodes == node
            # Only create node if there are samples in it
            split_any = torch.any(split_node).float()
            if self.distributed:
                dist.all_reduce(split_any, op=dist.ReduceOp.SUM)
            if split_any > 0:
                # Choose feature subset
                X_node = Xe[:, split_node]
                gradient_node = gradient[split_node]
                hessian_node = hessian[split_node]
                # Compute split decision
                if self.params['device'] == 'gpu':
                    Gl, Hl, Glc = self.parallelsum_kernel.split_decision(X_node, gradient_node, hessian_node, self.params['max_bin'])
                else:
                    left_idx = torch.gt(X_node.unsqueeze(-1), drange).float();
                    # Compute counts
                    Glc = left_idx.sum(1);
                    # Compute sum
                    Gl = torch.einsum("i, jik -> jk", gradient_node, left_idx);
                    Hl = torch.einsum("i, jik -> jk", hessian_node, left_idx);
                    
                # Compute counts of right leafs
                Grc = len(gradient_node) - Glc;
                Glc = Glc >= self.params['min_data_in_leaf']
                Grc = Grc >= self.params['min_data_in_leaf']
                # Sum gradients and hessian
                G = gradient_node.sum()
                H = hessian_node.sum()
                # Synchronize
                if self.distributed:
                    dist.all_reduce(Gl, op=dist.ReduceOp.SUM)
                    dist.all_reduce(Hl, op=dist.ReduceOp.SUM)
                    dist.all_reduce(G, op=dist.ReduceOp.SUM)
                    dist.all_reduce(H, op=dist.ReduceOp.SUM)
                    dist.all_reduce(Glc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(Grc, op=dist.ReduceOp.SUM)
                # Compute split_gain, only consider split gain when enough samples in leaves.
                split_gain_tot = (Gl * Gl) / (Hl + self.params['lambda']) + (G - Gl)*(G - Gl) / (H - Hl + self.params['lambda']) - (G * G) / (H + self.params['lambda'])
                split_gain_tot = split_gain_tot * Glc * Grc
                split_gain = split_gain_tot.max()
                # Split if split_gain exceeds minimum
                if split_gain > self.params['min_split_gain']:
                    argmaxsg = split_gain_tot.argmax()
                    split_feature_sample = argmaxsg // self.params['max_bin']
                    split_bin = argmaxsg - split_feature_sample * self.params['max_bin']
                    split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                    split_right = ~split_left * split_node
                    split_left *= split_node
                    # Check for monotone constraints if applicable
                    if self.any_monotone:
                        split_gain_tot_flat = split_gain_tot.flatten()
                        # Find min and max for leaf (mu) weights of current node
                        node_min = node_constraints[node_constraints[:, 0] == node, 1].squeeze()
                        node_max = node_constraints[node_constraints[:, 0] == node, 2].squeeze()
                        # Check if current split proposal has a monotonicity constraint
                        split_constraint = self.params['monotone_constraints'][sample_features[split_feature_sample]]
                        # Perform check only if parent node has a constraint or if the current proposal is constrained. Todo: this might be a CPU check due to np.inf. Replace np.inf by: torch.tensor(float("Inf"), dtype=torch.float32, device=X.device)
                        if (node_min > -np.inf) or (node_max < np.inf) or (split_constraint != 0):
                            # We precompute the child left- and right weights and evaluate whether they satisfy the constraints. If not, we seek another split and repeat.
                            mu_left = self._leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, return_mu=True)
                            mu_right = self._leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, return_mu=True)
                            split = True
                            split_iters = 1
                            condition = split * (((mu_left < node_min) + (mu_left > node_max) + (mu_right < node_min) + (mu_right > node_max)) + ((split_constraint != 0) * (torch.sign(mu_right - mu_left) != split_constraint)))
                            while condition > 0:
                                # Set gain of current split to -1, as this split is not allowed
                                split_gain_tot_flat[argmaxsg] = -1
                                # Get new split. Check if split_gain is still sufficient, because we might end up with having only constraint invalid splits (i.e. all split_gain <= 0).
                                split_gain = split_gain_tot_flat.max()
                                # Check if new proposed split is allowed, otherwise break loop
                                split = (split_gain > self.params['min_split_gain']) * (split_iters < self.params['monotone_iterations'])
                                if not split: break
                                # Find new split
                                argmaxsg = split_gain_tot_flat.argmax()
                                split_feature_sample = argmaxsg // self.params['max_bin']
                                split_bin = argmaxsg - split_feature_sample * self.params['max_bin']
                                split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                                split_right = ~split_left * split_node
                                split_left *= split_node
                                # Compute new leaf weights
                                mu_left = self._leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, return_mu=True)
                                mu_right = self._leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, return_mu=True)
                                # Check if new proposed split has a monotonicity constraint
                                split_constraint = self.params['monotone_constraints'][sample_features[split_feature_sample]]
                                condition = split * (((mu_left < node_min) + (mu_left > node_max) + (mu_right < node_min) + (mu_right > node_max)) + ((split_constraint != 0) * (torch.sign(mu_right - mu_left) != split_constraint)))
                                split_iters += 1
                            # Only create a split if there still is a split to make...
                            if split:
                                # Compute min and max values for children nodes
                                if split_constraint == 1:
                                    left_node_min = node_min
                                    left_node_max = mu_right
                                    right_node_min = mu_left
                                    right_node_max = node_max                                    
                                elif split_constraint == -1:
                                    left_node_min = mu_right
                                    left_node_max = node_max
                                    right_node_min = node_min
                                    right_node_max = mu_left
                                else:
                                    left_node_min = node_min
                                    left_node_max = node_max
                                    right_node_min = node_min
                                    right_node_max = node_max
                                # Set left children constraints
                                node_constraints[node_constraint_idx, 0] = 2 * node
                                node_constraints[node_constraint_idx, 1] = left_node_min
                                node_constraints[node_constraint_idx, 2] = left_node_max
                                node_constraint_idx += 1
                                # Set right children constraints
                                node_constraints[node_constraint_idx, 0] = 2 * node + 1
                                node_constraints[node_constraint_idx, 1] = right_node_min
                                node_constraints[node_constraint_idx, 2] = right_node_max
                                node_constraint_idx += 1
                                # Create split
                                self._create_split(estimator, node, sample_features, split_feature_sample, split_bin, X_node, split_gain, n_samples, split_left, split_right, gradient, hessian)
                            else:
                                self._leaf_prediction(gradient_node, hessian_node, node, estimator)
                        else:
                            # Set left children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node
                            node_constraint_idx += 1
                            # Set right children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node + 1
                            node_constraint_idx += 1                                                        
                            # Create split
                            self._create_split(estimator, node, sample_features, split_feature_sample, split_bin, X_node, split_gain, n_samples, split_left, split_right, gradient, hessian)
                    else:
                        self._create_split(estimator, node, sample_features, split_feature_sample, split_bin, X_node, split_gain, n_samples, split_left, split_right, gradient, hessian)
                else:
                    self._leaf_prediction(gradient_node, hessian_node, node, estimator)
                    
            # Choose next node (based on breadth-first)
            next_nodes = self.train_nodes[self.train_nodes > node]
            n_elements_next_nodes = torch.tensor(next_nodes.nelement(), device=X.device)
            if self.distributed:
                dist.barrier()
                dist.all_reduce(n_elements_next_nodes, op=dist.ReduceOp.SUM)
            if n_elements_next_nodes == 0:
                break
            else:
                if next_nodes.nelement() == 0:
                    node = torch.tensor(10000000000, device = X.device, dtype=torch.int64)
                else: 
                    node = next_nodes.min()
                if self.distributed:
                    # Synchronize next node choice
                    dist.all_reduce(node, op=dist.ReduceOp.MIN)                    
                                     
    def _predict_tree(self, X, mu, variance, estimator):
        # Get prediction for a single tree
        predictions = torch.zeros(X.shape[1], device=X.device, dtype=torch.int)
        nodes_predict = torch.ones(X.shape[1], device=X.device, dtype=torch.int)
        lr = self.params['learning_rate']
        corr = self.params['tree_correlation']
        leaves_idx = self.leaves_idx[estimator]
        leaves_mu = self.leaves_mu[estimator]
        leaves_var = self.leaves_var[estimator]
        nodes_idx = self.nodes_idx[estimator]
        nodes_split_feature = self.nodes_split_feature[estimator]
        nodes_split_bin = self.nodes_split_bin[estimator]
        node = torch.ones(1, device = X.device, dtype=torch.int64)
        # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
        leaf_idx = torch.eq(node, leaves_idx)
        if torch.any(leaf_idx):
            var_y = leaves_var[leaf_idx]
            mu += -lr * leaves_mu[leaf_idx]
            variance += lr * lr * var_y - 2 * lr * corr * variance.sqrt() * var_y.sqrt()
            predictions += 1
        else: 
            # Loop until every sample has a prediction (this allows earlier stopping than looping over all possible tree paths)
            while predictions.sum() < len(predictions):
                # Choose next node (based on breadth-first)
                condition = (nodes_predict >= node) * (predictions == 0)
                node = nodes_predict[condition].min()
                # Select current node information
                split_node = nodes_predict == node
                node_idx = (nodes_idx == node)
                current_feature = (nodes_split_feature * node_idx).sum()
                current_bin = (nodes_split_bin * node_idx).sum()
                # Split node
                split_left = (X[current_feature] > current_bin).squeeze()
                split_right = ~split_left * split_node
                split_left *= split_node
                # Check if children are leaves
                leaf_idx_left = torch.eq(2 * node, leaves_idx)
                leaf_idx_right = torch.eq(2 * node + 1, leaves_idx)
                # Update mu and variance with left leaf prediction
                if torch.any(leaf_idx_left):
                    mu += -lr * split_left * leaves_mu[leaf_idx_left]
                    var_left = split_left * leaves_var[leaf_idx_left]
                    variance += lr**2 * var_left - 2 * lr * corr * variance.sqrt() * var_left.sqrt()
                    predictions += split_left
                else:
                    nodes_predict += split_left * node
                # Update mu and variance with right leaf prediction
                if torch.any(leaf_idx_right):
                    mu += -lr * split_right * leaves_mu[leaf_idx_right]
                    var_right = split_right * leaves_var[leaf_idx_right]
                    variance += lr**2 * var_right - 2 * lr * corr * variance.sqrt() * var_right.sqrt()
                    predictions += split_right
                else:
                    nodes_predict += split_right * (node + 1)
                           
        return mu, variance      

    def _predict_forest(self, X):
        # Parallel prediction of a tree ensemble
        predictions = torch.zeros((X.shape[1], self.best_iteration), device=X.device, dtype=torch.int64)
        nodes_predict = torch.ones_like(predictions)
        mu = torch.zeros((X.shape[1], self.best_iteration), device=X.device, dtype=torch.float32)
        variance = torch.zeros_like(mu)
        node = nodes_predict.min()
        # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
        leaf_idx = torch.eq(node, self.leaves_idx)
        index = torch.any(leaf_idx, dim=1)
        mu[:, index] = self.leaves_mu[leaf_idx]
        variance[:, index] = self.leaves_var[leaf_idx]
        predictions[:, index] = 1
        # Loop until every sample has a prediction from every tree (this allows earlier stopping than looping over all possible tree paths)
        while predictions.sum() < predictions.nelement():
            # Choose next node (based on breadth-first)
            condition = (nodes_predict >= node) * (predictions == 0)
            node = nodes_predict[condition].min()
            # Select current node information
            split_node = nodes_predict == node
            node_idx = (self.nodes_idx == node)
            current_features = (self.nodes_split_feature * node_idx).sum(1)
            current_bins = (self.nodes_split_bin * node_idx).sum(1, keepdim=True)
            # Split node
            split_left = (X[current_features] > current_bins).T
            split_right = ~split_left * split_node
            split_left *= split_node
            # Check if children are leaves
            leaf_idx_left = torch.eq(2 * node, self.leaves_idx)
            leaf_idx_right = torch.eq(2 * node + 1, self.leaves_idx)
            # Update mu and variance with left leaf prediction
            mu += split_left * (self.leaves_mu * leaf_idx_left).sum(1)
            variance += split_left * (self.leaves_var * leaf_idx_left).sum(1)
            sum_left = leaf_idx_left.sum(1)
            predictions += split_left * sum_left
            nodes_predict += (1 - sum_left) * split_left * node
            # Update mu and variance with right leaf prediction
            mu += split_right * (self.leaves_mu * leaf_idx_right).sum(1)
            variance += split_right * (self.leaves_var * leaf_idx_right).sum(1)
            sum_right = leaf_idx_right.sum(1)
            predictions += split_right * sum_right
            nodes_predict += (1  - sum_right) * split_right * (node + 1)

        # Each prediction only for the amount of learning rate in the ensemble
        lr = self.params['learning_rate']
        mu = (-lr * mu).sum(1)
        if self.dist:
            corr = self.params['tree_correlation']
            variance = variance.T
            variance_total = torch.zeros(X.shape[1], dtype=torch.float32, device=X.device)
            variance_total += lr**2 * variance[0]
            # I have not figured out how to parallelize the variance estimate calculation in the ensemble, so we iterate over 
            # the variance per estimator
            for estimator in range(1, self.best_iteration):
                variance_total += lr**2 * variance[estimator] - 2 * lr * corr * variance_total.sqrt() * variance[estimator].sqrt()       
            variance = variance_total
        else:
            variance = variance[:, 0]
    
        return mu, variance  
    
    def _create_X_splits(self, X):
        # Pre-compute split decisions for Xtrain
        if self.params['max_bin'] <= 256:
            X_splits = torch.zeros((X.shape[1], X.shape[0]), device=self.device, dtype=torch.uint8)
        else:
            X_splits = torch.zeros((X.shape[1], X.shape[0]), device=self.device, dtype=torch.int16)
        
        for i in range(self.params['max_bin']):
            X_splits += (X > self.bins[:, i]).T
        
        return X_splits
    
    def _convert_array(self, array):
        if (type(array) == np.ndarray) or (type(array) == np.memmap):
            array = torch.from_numpy(array).float()
        elif type(array) == torch.Tensor:
            array = array.float()
        
        return array.to(self.device)
    
    def train(self, train_set, objective, metric, params=None, valid_set=None, sample_weight=None, eval_sample_weight=None):
        """
        Train a PGBM model.
        
        Example::
            >> train_set = (X_train, y_train)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
        
        Args:
            train_set (tuple of ([n_trainig_samples x n_features], [n_training_samples])): sample set (X, y) on which to train the PGBM model, 
                where X contains the features of the samples and y is the ground truth.
            objective (function): The objective function is the loss function that will be optimized during the gradient boosting process.
                The function should consume a torch vector of predictions yhat and ground truth values y and output the gradient and 
                hessian with respect to yhat of the loss function. For more complicated loss functions, it is possible to add a 
                levels variable, but this can be set to None in case it is not required.
            metric (function): The metric function is the function that generates the error metric. The evaluation 
                metric should consume a torch vector of predictions yhat and ground truth values y, and output a scalar loss. For 
                more complicated evaluation metrics, it is possible to add a levels variable, but this can be set to None in case 
                it is not required.
            params (dictionary): Dictionary containing the learning parameters of a PGBM model. 
            valid_set (tuple of ([n_validation_samples x n_features], [n_validation_samples])): sample set (X, y) on which to validate 
                the PGBM model, where X contains the features of the samples and y is the ground truth.
            sample_weight (vector of shape [n_training_samples] or dictionary of arrays): sample weights for the training data.
            eval_sample_weight (vector of shape [n_training_samples] or dictionary of arrays): sample weights for the validation data.
        """
        # Create parameters
        if params is None:
            params = {}
        self.n_samples = train_set[0].shape[0]
        self.n_features = train_set[0].shape[1]
        self._init_params(params)
        # Create train data
        X_train, y_train = self._convert_array(train_set[0]), self._convert_array(train_set[1]).squeeze()
        # Set objective & metric
        if self.params['derivatives'] == 'exact':
            self.objective = objective
        else:
            self.loss = objective
            self.objective = self._objective_approx
        self.metric = metric
        # Initialize predictions
        n_samples = torch.tensor(X_train.shape[0], device=X_train.device)
        y_train_sum = y_train.sum()
        # Synchronize
        if self.distributed:
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(y_train_sum, op=dist.ReduceOp.SUM) 
        # Pre-allocate arrays
        nodes_idx = torch.zeros((self.params['n_estimators'], self.params['max_nodes']), dtype=torch.int64, device = self.device)
        nodes_split_feature = torch.zeros_like(nodes_idx)
        nodes_split_bin = torch.zeros_like(nodes_idx)
        leaves_idx = torch.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=torch.int64, device = self.device)
        leaves_mu = torch.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=torch.float32, device = self.device)
        leaves_var = torch.zeros_like(leaves_mu)
        # Continue training from existing model or train new model, depending on whether a model was loaded.
        if not hasattr(self, 'yhat_0'):
            self.yhat_0 = y_train_sum / n_samples                           
            self.best_iteration = 0
            yhat_train = self.yhat_0.repeat(self.n_samples).to(self.device)
            self.bins = self._create_feature_bins(X_train)
            self.feature_importance = torch.zeros(self.n_features, dtype=torch.float32, device=self.device)
            self.nodes_idx = nodes_idx
            self.nodes_split_feature = nodes_split_feature
            self.nodes_split_bin = nodes_split_bin
            self.leaves_idx = leaves_idx
            self.leaves_mu = leaves_mu
            self.leaves_var = leaves_var
            start_iteration = 0
            new_model = True
        else:
            yhat_train = self.predict(X_train, parallel=False)
            self.nodes_idx = torch.cat((self.nodes_idx, nodes_idx))
            self.nodes_split_feature = torch.cat((self.nodes_split_feature, nodes_split_feature))
            self.nodes_split_bin = torch.cat((self.nodes_split_bin, nodes_split_bin))
            self.leaves_idx = torch.cat((self.leaves_idx, leaves_idx))
            self.leaves_mu = torch.cat((self.leaves_mu, leaves_mu))
            self.leaves_var = torch.cat((self.leaves_var, leaves_var))
            start_iteration = self.best_iteration
            new_model = False
        # Initialize
        self.train_nodes = torch.ones(self.bagging_fraction, dtype=torch.int64, device = self.device)
        self.dist = False
        self.n_forecasts_dist = 1
        # Pre-compute split decisions for X_train
        X_train_splits = self._create_X_splits(X_train)
        # Initialize validation
        validate = False
        if valid_set is not None:
            validate = True
            early_stopping = 0
            X_validate, y_validate = self._convert_array(valid_set[0]), self._convert_array(valid_set[1]).squeeze()
            if new_model:
                yhat_validate = self.yhat_0.repeat(y_validate.shape[0])
                self.best_score = torch.tensor(float('inf'), device = self.device, dtype=torch.float32)
            else:
                yhat_validate = self.predict(X_validate)
                validation_metric = metric(yhat_validate, y_validate, eval_sample_weight)
                if self.distributed:
                    dist.all_reduce(validation_metric, op=dist.ReduceOp.SUM)
                    validation_metric /= self.world_size
                self.best_score = validation_metric
            # Pre-compute split decisions for X_validate
            X_validate_splits = self._create_X_splits(X_validate)

        # Retrieve initial loss and gradient
        gradient, hessian = self.objective(yhat_train, y_train, sample_weight)      
        # Loop over estimators
        for estimator in range(start_iteration, self.params['n_estimators'] + start_iteration):
            # Retrieve bagging batch
            if self.params['bagging_fraction'] < 1:
                samples = torch.randperm(self.n_samples, device=self.device)[:self.bagging_fraction]
                # Create tree
                self._create_tree(X_train_splits[:, samples], gradient[samples], hessian[samples], estimator)
            else:
                self._create_tree(X_train_splits, gradient, hessian, estimator)                
            # Get predictions for all samples
            yhat_train, _ = self._predict_tree(X_train_splits, yhat_train, yhat_train*0., estimator)
            # Compute new gradient and hessian
            gradient, hessian = self.objective(yhat_train, y_train, sample_weight)
            # Compute metric
            train_metric = metric(yhat_train, y_train, sample_weight)
            # Synchronize across processes: we just add all process metrics and take the mean, this is a simplification
            if self.distributed:
                dist.all_reduce(train_metric, op=dist.ReduceOp.SUM)
                train_metric /= self.world_size
            # Reset train nodes
            self.train_nodes.fill_(1)
            # Validation statistics
            if validate:
                yhat_validate, _ =  self._predict_tree(X_validate_splits, yhat_validate, yhat_validate*0., estimator) 
                validation_metric = metric(yhat_validate, y_validate, eval_sample_weight)
                if self.distributed:
                    dist.all_reduce(validation_metric, op=dist.ReduceOp.SUM)
                    validation_metric /= self.world_size
                if (self.params['verbose'] > 1) & (self.rank == 0):
                    print(f"Estimator {estimator}/{self.params['n_estimators'] + start_iteration}, Train metric: {train_metric:.4f}, Validation metric: {validation_metric:.4f}")
                if validation_metric < self.best_score:
                    self.best_score = validation_metric
                    self.best_iteration = estimator + 1
                    early_stopping = 1
                else:
                    early_stopping += 1
                    if early_stopping == self.params['early_stopping_rounds']:
                        break
            else:
                if (self.params['verbose'] > 1) & (self.rank == 0):
                    print(f"Estimator {estimator}/{self.params['n_estimators'] + start_iteration}, Train metric: {train_metric:.4f}")
                self.best_iteration = estimator + 1
                self.best_score = 0
            
            # Save current model checkpoint to current working directory
            if self.params['checkpoint']:
                self.save(f'{self.cwd}\checkpoint')
            
        # Truncate tree arrays
        self.leaves_idx             = self.leaves_idx[:self.best_iteration]
        self.leaves_mu              = self.leaves_mu[:self.best_iteration]
        self.leaves_var             = self.leaves_var[:self.best_iteration]
        self.nodes_idx              = self.nodes_idx[:self.best_iteration]
        self.nodes_split_bin        = self.nodes_split_bin[:self.best_iteration]
        self.nodes_split_feature    = self.nodes_split_feature[:self.best_iteration]
                       
    def predict(self, X, parallel=True):
        """
        Generate point estimates/forecasts for a given sample set X
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> yhat_test = model.predict(X_test)
        
        Args:
            X (array of [n_samples x n_features]): sample set for which to create the estimates/forecasts.
            parallel (boolean): whether to generate the estimates in parallel or using a serial computation.
                Use serial if you experience RAM or GPU out-of-memory errors when using this function. Otherwise,
                parallel is recommended as it is much faster.
        """
        X = self._convert_array(X)
        self.dist = False
        self.n_forecasts_dist = 1
        yhat0 = self.yhat_0.repeat(X.shape[0])
        mu = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        variance = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        # Construct split decision tensor
        X_test_splits = self._create_X_splits(X)
        
        # Predict samples
        if parallel:
            mu, _ = self._predict_forest(X_test_splits)            
        else:
            for estimator in range(self.best_iteration):
                mu, _ = self._predict_tree(X_test_splits, mu, variance, estimator)

        return yhat0 + mu
       
    def predict_dist(self, X, n_forecasts=100, parallel=True, output_sample_statistics=False):
        """
        Generate probabilistic estimates/forecasts for a given sample set X
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> yhat_test_dist = model.predict_dist(X_test)
        
        Args:
            X (array of [n_samples x n_features]): sample set for which to create 
            the estimates/forecasts.
            n_forecasts (integer): number of estimates/forecasts to create.
            parallel (boolean): whether to generate the estimates in parallel or using a serial computation.
                Use serial if you experience RAM or GPU out-of-memory errors when using this function. Otherwise,
                parallel is recommended as it is much faster.
            output_sample_statistics (boolean): whether to also output the learned sample mean and variance. If True,
                the function will return a tuple (forecasts, mu, variance) with the latter arrays containing the learned
                mean and variance per sample that can be used to parameterize a distribution.
        """
        X = self._convert_array(X)
        self.dist = True
        self.n_forecasts_dist = n_forecasts
        yhat0 = self.yhat_0.repeat(X.shape[0])
        mu = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        variance = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        # Construct split decision tensor
        X_test_splits = self._create_X_splits(X)
        
        # Compute aggregate mean and variance
        if parallel:
            mu, variance = self._predict_forest(X_test_splits)
        else:
            for estimator in range(self.best_iteration):
                mu, variance = self._predict_tree(X_test_splits, mu, variance, estimator)
        
        # Sample from distribution
        mu += yhat0
        if self.params['distribution'] == 'normal':
            loc = mu
            scale = torch.nan_to_num(variance.sqrt(), 1e-9)
            yhat = Normal(loc, scale).rsample([n_forecasts])
        elif self.params['distribution'] == 'studentt':
            v = 3
            loc = mu
            factor = v / (v - 2)
            scale = torch.nan_to_num( (variance / factor).sqrt(), 1e-9)
            yhat = StudentT(v, loc, scale).rsample([n_forecasts])
        elif self.params['distribution'] == 'laplace':
            loc = mu
            scale = torch.nan_to_num( (0.5 * variance).sqrt(), 1e-9)
            yhat = Laplace(loc, scale).rsample([n_forecasts])
        elif self.params['distribution'] == 'logistic':
            loc = mu
            scale = torch.nan_to_num( ((3 * variance) / np.pi**2).sqrt(), 1e-9)
            base_dist = Uniform(torch.zeros(X.shape[0], device=X.device), torch.ones(X.shape[0], device=X.device))
            yhat = TransformedDistribution(base_dist, [SigmoidTransform().inv, AffineTransform(loc, scale)]).rsample([n_forecasts])
        elif self.params['distribution'] == 'lognormal':
            mu_adj = mu.clamp(1e-9)
            variance = torch.nan_to_num(variance, 1e-9).clamp(1e-9)
            scale = ((variance + mu_adj**2) / mu_adj**2).log().clamp(1e-9)
            loc = (mu_adj.log() - 0.5 * scale).clamp(1e-9)
            yhat = LogNormal(loc, scale.sqrt()).rsample([n_forecasts])
        elif self.params['distribution'] == 'gamma':
            variance = torch.nan_to_num(variance, 1e-9)
            mu_adj = torch.nan_to_num(mu, 1e-9)
            rate = (mu_adj.clamp(1e-9) / variance.clamp(1e-9))
            shape = mu_adj.clamp(1e-9) * rate
            yhat = Gamma(shape, rate).rsample([n_forecasts])
        elif self.params['distribution'] == 'gumbel':
            variance = torch.nan_to_num(variance, 1e-9)
            scale = (6 * variance / np.pi**2).sqrt().clamp(1e-9)
            loc = mu - scale * np.euler_gamma
            yhat = Gumbel(loc, scale).rsample([n_forecasts]) 
        elif self.params['distribution'] == 'weibull':
            variance = torch.nan_to_num(variance, 1e-9)
            # This is a little bit hacky..
            def weibull_params(k):
                gamma = lambda x: torch.exp(torch.lgamma(1 + x))   
                f = -gamma(2 / k) / gamma(1 / k)**2 + 1   
            
                return f
            
            def weibull_params_grad(k):
                gamma = lambda x: torch.exp(torch.lgamma(1 + x))
                psi0 = lambda x: torch.polygamma(0, 1 + x)
                psi1 = lambda x: torch.polygamma(1, 1 + x)
                grad = -(2 * gamma(2 / k) * ( psi0(1/k) - psi0(2/k) ) ) / (k**2 * gamma(1 / k)**2)
                hess = -(2 * gamma(2 / k) * (2 * psi0(2/k)**2 - 2*k*psi0(1/k) - psi1(1/k) + 2 * k * psi0(2 / k) - 4 * psi0(1 / k) * psi0( 2 / k) + 2 * psi1(2 / k) + 2 * psi0(1 / k)**2) ) / ( k**4 * gamma(1 / k)**2)
            
                return grad, hess

            # Initialize k - Todo: initialization should be somehow proportional to ratio of mu/var
            k = torch.ones(mu.shape[0], device=mu.device, dtype=torch.float) * 1e6
            f = weibull_params(k) 
            threshold = -variance / mu**2
            # Gradient descent to fit Weibull parameters to empirical statistics
            i = 0
            alpha = 0.1
            while (torch.any(f > threshold)) and torch.all(k > 0):
                grad, hess = weibull_params_grad(k)
                k = k + alpha * grad / hess * (f > threshold)
                f = weibull_params(k)
                i += 1
                if i > 10000:
                    print('Weibull not converged')
                    break
                
            # Clip values
            scale = (mu / torch.exp(torch.lgamma(1 + 1 / k))).clamp(self.epsilon)
            k = k.clamp(self.epsilon)

            # Sample from fitted Weibull
            yhat = Weibull(scale, k).rsample([n_forecasts]) 

        elif self.params['distribution'] == 'negativebinomial':
            loc = mu.clamp(1e-9)
            eps = 1e-9
            variance = torch.nan_to_num(variance, 1e-9)
            scale = torch.maximum(loc + eps, variance).clamp(1e-9)
            probs = (1 - (loc / scale)).clamp(0, 0.99999)
            counts = (-loc**2 / (loc - scale)).clamp(eps)
            yhat = NegativeBinomial(counts, probs).sample([n_forecasts])
        elif self.params['distribution'] == 'poisson':
            yhat = Poisson(mu.clamp(1e-9)).sample([n_forecasts])
        else:
            print('Distribution not (yet) supported')
          
        if output_sample_statistics:
            return (yhat, mu, variance)
        else:
            return yhat
    
    def crps_ensemble(self, yhat_dist, y):
        """
        Calculate the empirical Continuously Ranked Probability Score for a set 
        of forecasts for a number of samples.
        
        Based on `crps_ensemble` from `properscoring`
        https://pypi.org/project/properscoring/
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> yhat_test_dist = model.predict_dist(X_test)
            >> crps = model.crps_ensemble(yhat_test_dist, y_test)
        
        Args:
            yhat_dist (array of [n_forecasts x n_samples]): array containing forecasts 
            for each sample.
            y (array of [n_samples]): ground truth value of each sample.
        """
        y = self._convert_array(y)
        yhat_dist = self._convert_array(yhat_dist)
        n_forecasts = yhat_dist.shape[0]
        # Sort the forecasts in ascending order
        yhat_dist_sorted, _ = torch.sort(yhat_dist, 0)
        # Create temporary tensors
        y_cdf = torch.zeros_like(y)
        yhat_cdf = torch.zeros_like(y)
        yhat_prev = torch.zeros_like(y)
        crps = torch.zeros_like(y)
        # Loop over the samples generated per observation
        for yhat in yhat_dist_sorted:
            flag = (y_cdf == 0) * (y < yhat)
            crps += flag * ((y - yhat_prev) * yhat_cdf ** 2)
            crps += flag * ((yhat - y) * (yhat_cdf - 1) ** 2)
            y_cdf += flag
            crps += ~flag * ((yhat - yhat_prev) * (yhat_cdf - y_cdf) ** 2)
            yhat_cdf += 1 / n_forecasts
            yhat_prev = yhat
        
        # In case y_cdf == 0 after the loop
        flag = (y_cdf == 0)
        crps += flag * (y - yhat)
        
        return crps       
    
    def save(self, filename):
        """
        Save a PGBM model to a file. The model parameters are saved as numpy arrays and dictionaries. 
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> model.save('model.pt')
        
        Args:
            filename (string): name and location to save the model to
        """
        params = self.params.copy()
        params['learning_rate'] = params['learning_rate'].cpu().numpy()
        params['tree_correlation'] = params['tree_correlation'].cpu().numpy()
        params['lambda'] = params['lambda'].cpu().numpy()
        params['min_split_gain'] = params['min_split_gain'].cpu().numpy()
        params['min_data_in_leaf'] = params['min_data_in_leaf'].cpu().numpy()
        params['monotone_constraints'] = params['monotone_constraints'].cpu().numpy()

        state_dict = {'nodes_idx': self.nodes_idx[:self.best_iteration].cpu().numpy(),
                      'nodes_split_feature':self.nodes_split_feature[:self.best_iteration].cpu().numpy(),
                      'nodes_split_bin':self.nodes_split_bin[:self.best_iteration].cpu().numpy(),
                      'leaves_idx':self.leaves_idx[:self.best_iteration].cpu().numpy(),
                      'leaves_mu':self.leaves_mu[:self.best_iteration].cpu().numpy(),
                      'leaves_var':self.leaves_var[:self.best_iteration].cpu().numpy(),
                      'feature_importance':self.feature_importance.cpu().numpy(),
                      'best_iteration':self.best_iteration,
                      'params':params,
                      'yhat0':self.yhat_0.cpu().numpy(),
                      'bins':self.bins.cpu().numpy()}
        
        with open(filename, 'wb') as handle:
            pickle.dump(state_dict, handle)   
    
    def load(self, filename, device=None):
        """
        Load a PGBM model from a file to a device. 
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.load('model.pt') # Load to default device (cpu)
            >> model.load('model.pt', device=torch.device(0)) # Load to default GPU at index 0
        
        Args:
            filename (string): location of model file.
            device (torch.device): device to which to load the model. Default = 'cpu'.
        """
        if device is None:
            device = torch.device('cpu')
        with open(filename, 'rb') as handle:
            state_dict = pickle.load(handle)
        
        torch_float = lambda x: torch.from_numpy(x).float().to(device)
        torch_long = lambda x: torch.from_numpy(x).long().to(device)
        
        self.nodes_idx = torch_long(state_dict['nodes_idx'])
        self.nodes_split_feature  = torch_long(state_dict['nodes_split_feature'])
        self.nodes_split_bin  = torch_long(state_dict['nodes_split_bin'])
        self.leaves_idx  = torch_long(state_dict['leaves_idx'])
        self.leaves_mu  = torch_float(state_dict['leaves_mu'])
        self.leaves_var  = torch_float(state_dict['leaves_var'])
        self.feature_importance  = torch_float(state_dict['feature_importance'])
        self.best_iteration  = state_dict['best_iteration']
        self.params  = state_dict['params']
        self.params['learning_rate'] = torch_float(np.array(self.params['learning_rate']))
        self.params['tree_correlation'] = torch_float(np.array(self.params['tree_correlation']))
        self.params['lambda'] = torch_float(np.array(self.params['lambda']))
        self.params['min_split_gain'] = torch_float(np.array(self.params['min_split_gain']))
        self.params['min_data_in_leaf'] = torch_float(np.array(self.params['min_data_in_leaf']))
        self.params['monotone_constraints'] = torch_float(np.array(self.params['monotone_constraints']))        
        self.yhat_0  = torch_float(np.array(state_dict['yhat0']))
        self.bins = torch_float(state_dict['bins'])
        self.device = device
        
    def permutation_importance(self, X, y=None, n_permutations=10, levels=None):
        """
        Calculate feature importance of a PGBM model for a sample set X by randomly 
        permuting each feature. 
        
        This function can be executed in a supervised and unsupervised manner, depending
        on whether y is given. If y is given, the output of this function is the change
        in error metric when randomly permuting a feature. If y is not given, the output
        is the weighted average change in prediction when randomly permuting a feature. 
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> perm_importance_supervised = model.permutation_importance(X_test, y_test)  # Supervised
            >> perm_importance_unsupervised = model.permutation_importance(X_test)  # Unsupervised
        
        Args:
            X (array of [n_samples x n_features]): sample set for which to determine the feature importance.
            y (array of [n_samples]): ground truth for sample set X.
            n_permutations (integer): number of random permutations to perform for each feature.
            levels (dict of arrays):only applicable when using a levels argument in the error metric.
        """
        X = self._convert_array(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        permutation_importance_metric = torch.zeros((n_features, n_permutations), device=X.device, dtype=X.dtype)
        # Calculate base score
        yhat_base = self.predict(X)
        if y is not None:
            y = self._convert_array(y)
            base_metric = self.metric(yhat_base, y, levels)
        # Loop over permuted features
        for feature in range(n_features):
            X_permuted = torch.zeros((n_permutations, n_samples, n_features), device=X.device, dtype=X.dtype)
            for permutation in range(n_permutations):
                indices = torch.randperm(n_samples, device=X.device, dtype=torch.int64)
                X_current = X.clone()
                X_current[:, feature] = X_current[indices, feature]
                X_permuted[permutation] = X_current
            
            X_permuted = X_permuted.reshape(n_permutations * n_samples, n_features)
            yhat = self.predict(X_permuted)
            yhat = yhat.reshape(n_permutations, n_samples)
            if y is not None:
                for permutation in range(n_permutations):
                    permuted_metric = self.metric(yhat[permutation], y, levels)
                    permutation_importance_metric[feature, permutation] = ((permuted_metric / base_metric) - 1) * 100
            else:
                permutation_importance_metric[feature] = (yhat_base.unsqueeze(0) - yhat).abs().sum(1) / yhat_base.sum() * 100                
        
        return permutation_importance_metric

    def optimize_distribution(self, X, y, distributions=None, tree_correlations=None):
        """
        Find the distribution and tree correlation that best fits the data according to lowest CRPS score. The parameters
        'distribution' and 'tree_correlation' of a PGBM model will be adjusted to the best values after running this script.
        
        This function returns the best found distribution and tree correlation.
        
        Example::
            >> train_set = (X_train, y_train)
            >> validation_set = (X_validation, y_validation)
            >> model = PGBM()
            >> model.train(train_set, objective, metric)
            >> yhat_dist = model.optimize_distribution(X_validation, y_validation)
        
        Args:
            X (array of [n_samples x n_features]): sample set for which to create 
            the estimates/forecasts.
            y (array of [n_samples]): ground truth for sample set X.
            distributions (list of strings): optional, list containing distributions to choose from. Options are:
                normal, studentt, laplace, logistic, lognormal, gamma, gumbel, weibull, negativebinomial, poisson.
            tree_correlations (vector): optional, vector containing tree correlations to use in optimization procedure.
        """               
        # Convert input data if not right type
        X = self._convert_array(X)
        y = self._convert_array(y)
 
        # List of distributions and tree correlations
        if distributions == None:
            distributions = ['normal', 'studentt', 'laplace', 'logistic', 
                             'lognormal', 'gamma', 'gumbel', 'weibull', 'negativebinomial', 'poisson']
        if tree_correlations == None:
            tree_correlations = torch.arange(start=0, end=0.2, step=0.01, dtype=torch.float32, device=X.device)
        else:
            tree_correlations = tree_correlations.float().to(X.device)
               
        # Loop over choices
        crps_best = torch.tensor(float('inf'), device = X.device, dtype=torch.float32)
        distribution_best = self.params['distribution']
        tree_correlation_best = self.params['tree_correlation']
        for distribution in distributions:
            for tree_correlation in tree_correlations:
                self.params['distribution'] = distribution
                self.params['tree_correlation'] = tree_correlation
                yhat_dist = self.predict_dist(X)
                crps = self.crps_ensemble(yhat_dist, y).mean()
                if (self.params['verbose'] > 1) & (self.rank == 0):
                    print(f'CRPS: {crps:.2f} (Distribution: {distribution}, Tree correlation: {tree_correlation:.3f})')     
                if crps < crps_best:
                    crps_best = crps
                    distribution_best = distribution
                    tree_correlation_best = tree_correlation
        
        # Set to best values
        if (self.params['verbose'] > 1) & (self.rank == 0):
            print(f'Lowest CRPS: {crps_best:.4f} (Distribution: {distribution_best}, Tree correlation: {tree_correlation_best:.3f})')  
        self.params['distribution'] = distribution_best
        self.params['tree_correlation'] = tree_correlation_best
        
        return (distribution_best, tree_correlation_best)   

class PGBMRegressor(BaseEstimator):
    """
    Probabilistic Gradient Boosting Machines (PGBM) Regressor.
    
    PGBMRegressor fits a Gradient Boosting Machine regression model and returns
    point and probabilistic predictions.
    
    This class uses PyTorch as backend.
    
    Ref:
       Olivier Sprangers, Sebastian Schelter, Maarten de Rijke. 
       Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression.
       https://arxiv.org/abs/2106.0168 
       Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and
       Data Mining (KDD ’21), August 14–18, 2021, Virtual Event, Singapore.
       https://doi.org/10.1145/3447548.3467278
            
    ----------
    objective : str or function, default='mse'
        Objective function to minimize. If not 'mse', a user defined objective 
        function should be supplied in the form of:
            
        (A): in case parameter "derivatives='exact'"
            
            def objective(y, yhat, sample_weight=None):
                [......]
                
                return gradient, hessian
            
            in which gradient and hessian are of the same shape as y.
        
        (B): in case parameter "derivatives='approx'"

            def objective(y, yhat, sample_weight=None):
                [......]
                
                return loss
            
            in which loss is a scalar, differentiable loss.

    metric : str or function, default='rmse'
        Metric to evaluate predictions during training and evaluation. If not 
        'rmse', a user defined metric should be supplied in the form of:
            
            def metric(y, yhat, sample_weight=None):
                [......]
                
                return metric
        
            in which metric is a scalar.

    max_leaves : int, default=32 
        The maximum number of leaves per tree. Increase this value to create 
        more complicated trees, and reduce the value to create simpler trees 
        (reduce overfitting).
        
    learning_rate : float, default=0.1, constraint>0
        The learning rate of the algorithm; the amount of each new tree 
        prediction that should be added to the ensemble.

    n_estimators : int, default=100, constraint>0
        The number of trees to create. Typically setting this value higher may 
        improve performance, at the expense of training speed and potential for 
        overfit. Use in conjunction with learning rate and max_leaves; more 
        trees generally requires a lower learning_rate and/or a lower max_leaves.
        
    min_split_gain : float, default = 0.0, constraint >= 0.0 
        The minimum gain for a node to split when building a tree.
        
    min_data_in_leaf : int, default= 3, constraint>= 3. 
        The minimum number of samples in a leaf of a tree. Increase this value 
        to reduce overfit.
        
    bagging_fraction : float, default=1, constraint>0, constraint<=1. 
        Fraction of samples to use when building a tree. Set to a value between
        0 and 1 to randomly select a portion of samples to construct each new 
        tree. A lower fraction speeds up training (and can be used to deal with
        out-of-memory issues when training on GPU) and may reduce overfit.
        
    feature_fraction : float, default=1, constraint>0, constraint<=1.
        Fraction of features to use when building a tree. Set to a value between
        0 and 1 to randomly select a portion of features to construct each new 
        tree. A lower fraction speeds up training (and can be used to deal with
        out-of-memory issues when training on GPU) and may reduce overfit. 

    max_bin : int, default=256, constraint<32,767
        The maximum number of bins used to bin continuous features. Increasing 
        this value can improve prediction performance, at the cost of training 
        speed and potential overfit.

    reg_lambda : float, default=1.0, constraint>0
        Regularization parameter.
        
    random_state : int, default=2147483647
        Random seed to use for feature_fraction and bagging_fraction (latter 
        only for Numba backend - for speed considerations the Torch backend 
        bagging_fraction determination is not yet deterministic).
        
    device : str, default=cpu
        Traininig device. Choices are 'cpu' or 'gpu'.
        
    gpu_device_id : int, default=0 
        Integer with the index of the GPU used for training. Change this when
        you'd like to train on a different GPU within your node. For multi-gpu
        and multinode training, use the Python API (not this sklearn wrapper)
        
    derivatives : str, default=exact. 
        If a loss function with an analytical gradient and hessian is provided,
        use 'exact'. If a loss function with a scalar, differentiable loss is
        provided, use approx to let PyTorch use auto-differentiation to 
        calculate the gradient and (approximate) hessian.
     
    distribution : str, default='normal'
        Choice of output distribution for probabilistic predictions. Choices 
        are normal, studentt, laplace, logistic, gamma, gumbel, poisson. Note 
        that the studentt distribution has a constant degree-of-freedom of 3.
    
    checkpoint : bool, default=False
        Boolean to save a model checkpoint after each iteration to the current 
        working directory.
    
    tree_correlation : float, default=np.log_10(n_samples) / 100
        Tree correlation hyperparameter. This controls the amount of 
        correlation we assume to exist between each subsequent tree in the 
        ensemble.
    
    monotone_contraints : array, default=None. Array detailing
        monotone constraints for each feature in the dataset, where 0 represents
        no constraint, 1 a positive monotone constraint, and -1 a negative
        monotone constraint. For example, for a dataset with 3 features, this 
        parameter could be [1, 0, -1] for respectively a positive, none and 
        negative monotone contraint on feature 1, 2 and 3.
            
    monotone_iterations : int, default=1. The number of alternative splits that
        will be considered if a monotone constraint is violated by the current
        split proposal. Increase this to improve accuracy at the expense of 
        training speed. Note: the monotone_constraints need to be set in the
        .fit() method. 
        
    verbose : int, default=2.
        Flag to output metric results for each iteration. Set to 1 to supress 
        output.

    
    Attributes
    ----------
    learner_ : Fitted PGBM learner.         
    
    Examples
    --------
    >>> from pgbm import PGBMRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_boston
    >>> X, y = load_boston(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    >>> model = PGBMRegressor().fit(X_train, y_train)  
    >>> yhat_point = model.predict(X_test)
    >>> yhat_dist = model.predict_dist(X_test)
    """
    def __init__(self, objective='mse', metric='rmse', max_leaves=32, learning_rate=0.1, n_estimators=100,
                 min_split_gain=0.0, min_data_in_leaf=3, bagging_fraction=1, feature_fraction=1, max_bin=256,
                 reg_lambda=1.0, random_state=2147483647, device='cpu', gpu_device_id=0, derivatives='exact',
                 distribution='normal', checkpoint=False, tree_correlation=None, monotone_constraints=None, 
                 monotone_iterations=1, verbose=2):
        # Set parameters
        self.objective = objective
        self.metric = metric
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_split_gain = min_split_gain
        self.min_data_in_leaf = min_data_in_leaf
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.device = device
        self.gpu_device_id = gpu_device_id
        self.derivatives = derivatives
        self.distribution = distribution
        self.checkpoint = checkpoint
        self.tree_correlation = tree_correlation
        self.monotone_constraints = monotone_constraints
        self.monotone_iterations = monotone_iterations
        self.verbose = verbose
        
    def fit(self, X, y, eval_set=None, sample_weight=None, eval_sample_weight=None,
            early_stopping_rounds=None):
        # Set estimator type
        self._estimator_type = "regressor"
        # Check that X and y have correct shape and convert to float32
        X, y = check_X_y(X, y)
        X, y = X.astype(np.float32), y.astype(np.float32)
        self.n_features_in_ = X.shape[1]
        # Check eval set
        if eval_set is not None:
            X_valid, y_valid = eval_set[0], eval_set[1]
            X_valid, y_valid = check_X_y(X_valid, y_valid)
            X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)

        # Check parameter values and create parameter dict
        params = {'min_split_gain':self.min_split_gain,
                  'min_data_in_leaf':self.min_data_in_leaf,
                  'max_leaves':self.max_leaves,
                  'max_bin':self.max_bin,
                  'learning_rate':self.learning_rate,
                  'n_estimators':self.n_estimators,
                  'verbose':self.verbose,
                  'early_stopping_rounds': early_stopping_rounds,
                  'feature_fraction':self.feature_fraction,
                  'bagging_fraction':self.bagging_fraction,
                  'seed':self.random_state,
                  'lambda':self.reg_lambda,
                  'device':self.device,
                  'gpu_device_id':self.gpu_device_id,
                  'derivatives':self.derivatives,
                  'distribution':self.distribution,
                  'monotone_iterations': self.monotone_iterations,
                  'checkpoint': self.checkpoint}
        if self.tree_correlation is not None: 
            params['tree_correlation'] = self.tree_correlation
        if self.monotone_constraints is not None:
            params['monotone_constraints'] = self.monotone_constraints

        # Set objective and metric. If default, override derivatives argument
        if (self.objective == 'mse'):
            self._objective = self._mseloss_objective
            params['derivatives'] = 'exact'
        else:
            self._objective = self.objective
        if (self.metric == 'rmse'):
            self._metric = self._rmseloss_metric
        else:
            self._metric = self.metric    
        # Check sample weight shape
        torch_array = lambda array: torch.from_numpy(np.array(array).astype(np.float32)).float()
        device = torch.device(self.gpu_device_id) if self.device=='gpu' else torch.device('cpu')
        if sample_weight is not None: 
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if len(sample_weight) != len(y):
                raise ValueError('Length of sample_weight does not equal length of X and y')
            sample_weight = torch_array(sample_weight).to(device)
        if eval_sample_weight is not None: 
            eval_sample_weight = check_array(eval_sample_weight, ensure_2d=False)
            if len(eval_sample_weight) != len(y_valid):
                raise ValueError("Length of eval_sample_weight does not equal length of X_valid and y_valid")
            eval_sample_weight = torch_array(eval_sample_weight).to(device)

        # Train model
        self.learner_ = PGBM()
        self.learner_.train(train_set=(X, y), valid_set=eval_set, params=params, objective=self._objective, 
                         metric=self._metric, sample_weight=sample_weight, 
                         eval_sample_weight=eval_sample_weight)
        
        return self

    def predict(self, X, parallel=True):
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(np.float32)
        
        return self.learner_.predict(X, parallel).cpu().numpy()
    
    def score(self, X, y, sample_weight=None, parallel=True):
        # Checks
        X, y = check_X_y(X, y)
        X, y = X.astype(np.float32), y.astype(np.float32)
        # Make prediction
        yhat = self.predict(X, parallel)
        
        # Check sample weight shape
        torch_array = lambda array: torch.from_numpy(np.array(array).astype(np.float32)).float()
        device = torch.device(self.gpu_device_id) if self.device=='gpu' else torch.device('cpu')
        if sample_weight is not None: 
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if len(sample_weight) != len(y):
                raise ValueError("Length of sample_weight does not equal length of X and y")
            sample_weight = torch_array(sample_weight).to(device)
                
        # Score prediction with r2
        score = r2_score(y, yhat, sample_weight)
        
        return score
    
    def predict_dist(self, X, n_forecasts=100, parallel=True, output_sample_statistics=False):
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(np.float32)
        
        return self.learner_.predict_dist(X, n_forecasts, parallel, output_sample_statistics).cpu().numpy()
        
    def _mseloss_objective(self, yhat, y, sample_weight=None):
        gradient = (yhat - y)
        hessian = torch.ones_like(yhat)
        
        if sample_weight is not None:
            gradient *= sample_weight
            hessian *= sample_weight   
    
        return gradient, hessian
    
    def _rmseloss_metric(self, yhat, y, sample_weight=None):
        error = (yhat - y)
        if sample_weight is not None:
            error *= sample_weight
        
        loss = torch.sqrt(torch.mean(torch.square(error)))
    
        return loss
