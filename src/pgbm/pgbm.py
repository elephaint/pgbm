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
import torch.nn.functional as F
import torch.nn.parallel as parallel
import numpy as np
from torch.autograd import grad
from torch.distributions import Normal, NegativeBinomial, Poisson, StudentT, LogNormal, Laplace, Uniform, TransformedDistribution, SigmoidTransform, AffineTransform, Gamma, Gumbel, Weibull
from torch.utils.cpp_extension import load
from pathlib import Path
import pickle
#%% Probabilistic Gradient Boosting Machines
class PGBM(nn.Module):
    def __init__(self):
        super(PGBM, self).__init__()
    
    def _init_params(self, params):       
        self.params = {}
        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'tree_correlation', 'max_leaves',
                       'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds', 'feature_fraction', 'bagging_fraction', 
                       'seed', 'device', 'output_device', 'gpu_device_ids', 'derivatives', 'distribution']
        param_defaults = [0.0, 2, 0.1, 1.0, 0.03, 32, 256, 100, 2, 100, 1, 1, 0, 'cpu', 'cpu', (0,), 'exact', 'normal']
        
        # Create gpu functions for split decision
        if 'device' in params:
            if params['device'] == 'gpu':
                current_path = Path(__file__).parent.absolute()
                parallelsum_kernel = load(name='parallelsum', sources=[f'{current_path}/parallelsum.cpp', f'{current_path}/parallelsum_kernel.cu'])
                self.device_ids = params['gpu_device_ids'] if 'gpu_device_ids' in params else (0,)
                params['gpu_device_ids'] = self.device_ids
                max_bin = params['max_bin'] if 'max_bin' in params else 256
                self.parallel_split_decision = parallel.replicate(_split_decision(max_bin, parallelsum_kernel), self.device_ids)
                if 'output_device' in params:
                    if params['output_device'] == 'gpu':
                        output_id = params['gpu_device_ids'][0]
                        self.output_device = torch.device(output_id)
                    else:
                        self.output_device = torch.device('cpu')
                else:
                    self.output_device = torch.device(0)
            else:
                self.output_device = torch.device('cpu')
        else: 
            self.output_device = torch.device('cpu')
                   
        for i, param in enumerate(param_names):
            self.params[param] = params[param] if param in params else param_defaults[i]
            if param in ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'tree_correlation']:
                self.params[param] = torch.tensor(self.params[param], device=self.output_device, dtype=torch.float32)
                
        # Set some additional params
        self.params['max_nodes'] = self.params['max_leaves'] - 1
        torch.manual_seed(self.params['seed'])
        self.epsilon = 1.0e-4

    def _create_feature_bins(self, X):
        # Create array that contains the bins
        max_bin = self.params['max_bin']
        bins = torch.zeros((X.shape[1], max_bin), device=X.device)
        # For each feature, create max_bins based on frequency bins. 
        for i in range(X.shape[1]):
            xs = X[:, i].sort()[0]
            current_bin = torch.unique(F.interpolate(xs[None, None, :], max_bin, mode='linear', align_corners=False).squeeze())
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

    def _leaf_prediction(self, gradient, hessian, node, estimator):
        # Empirical mean
        gradient_mean = gradient.mean()
        hessian_mean = hessian.mean()
        # Empirical variance
        N = len(gradient)
        factor = 1 / (N - 1)
        gradient_variance = factor * ((gradient - gradient_mean)**2).sum()
        hessian_variance = factor * ((hessian - hessian_mean)**2).sum()
        # Empirical covariance
        covariance = factor * ((gradient - gradient_mean)*(hessian - hessian_mean)).sum()
        # Mean and variance of the leaf prediction
        lambda_scaled = self.params['lambda'] / N
        mu = gradient_mean / ( hessian_mean + lambda_scaled) - covariance / (hessian_mean + lambda_scaled)**2 + (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3
        var = mu**2 * (gradient_variance / gradient_mean**2 + hessian_variance / (hessian_mean + lambda_scaled)**2
                        - 2 * covariance / (gradient_mean * (hessian_mean + lambda_scaled) ) )
        # Save optimal prediction and node information
        self.leaves_idx[estimator, self.leaf_idx] = node
        self.leaves_mu[estimator, self.leaf_idx] = mu             
        self.leaves_var[estimator, self.leaf_idx] = var
        # Increase leaf idx       
        self.leaf_idx += 1           
    
    def _create_tree(self, X, gradient, hessian, estimator):
        # Set start node and start leaf index
        self.leaf_idx = 0
        self.node_idx = 0
        node = 1
        n_features = X.shape[0]
        n_samples = X.shape[1]
        sample_features = torch.randperm(n_features, device=X.device, dtype=torch.int64)[:self.feature_fraction]
        Xe = X[sample_features]
        drange = torch.arange(self.params['max_bin'], device=X.device);
        # Create tree
        while (self.leaf_idx < self.params['max_leaves']):
            split_node = self.train_nodes == node
            # Only create node if there are samples in it
            if split_node.any():
                # Choose feature subset
                X_node = Xe[:, split_node]
                gradient_node = gradient[split_node]
                hessian_node = hessian[split_node]
                # Compute split decision, parallelize across GPUs if required
                if self.params['device'] == 'gpu':
                    inputs = parallel.scatter((X_node.T, gradient_node, hessian_node), self.device_ids)
                    outputs = parallel.parallel_apply(self.parallel_split_decision, inputs)
                    Gl, Hl = parallel.gather(outputs, self.device_ids[0])
                    Gl, Hl = Gl.sum(0).to(self.output_device), Hl.sum(0).to(self.output_device)
                else:
                    left_idx = torch.gt(X_node.unsqueeze(-1), drange).float();
                    Gl = torch.einsum("i, jik -> jk", gradient_node, left_idx);
                    Hl = torch.einsum("i, jik -> jk", hessian_node, left_idx);
                # Comput split_gain
                G = gradient_node.sum()
                H = hessian_node.sum()
                split_gain_tot = (Gl * Gl) / (Hl + self.params['lambda']) + (G - Gl)*(G - Gl) / (H - Hl + self.params['lambda']) - (G * G) / (H + self.params['lambda'])
                split_gain = split_gain_tot.max()      
                # Split if split_gain exceeds minimum
                if split_gain > self.params['min_split_gain']:
                    argmaxsg = split_gain_tot.argmax()
                    split_feature_sample = argmaxsg // self.params['max_bin']
                    split_bin = argmaxsg - split_feature_sample * self.params['max_bin']
                    split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                    split_right = ~split_left * split_node
                    split_left = split_left * split_node
                    # Split when enough data in leafs                        
                    if (split_left.sum() > self.params['min_data_in_leaf']) & (split_right.sum() > self.params['min_data_in_leaf']):
                        # Save split information
                        self.nodes_idx[estimator, self.node_idx] = node
                        self.nodes_split_feature[estimator, self.node_idx] = sample_features[split_feature_sample] 
                        self.nodes_split_bin[estimator, self.node_idx] = split_bin
                        self.node_idx += 1
                        self.feature_importance[sample_features[split_feature_sample]] += split_gain * X_node.shape[1] / n_samples 
                        # Assign next node to samples if next node does not exceed max n_leaves
                        criterion = 2 * (self.params['max_nodes'] - self.node_idx + 1)
                        n_leaves_old = self.leaf_idx
                        if (criterion >  self.params['max_leaves'] - n_leaves_old):
                            self.train_nodes[split_left] = 2 * node
                        else:
                            self._leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator)
                        if (criterion >  self.params['max_leaves'] - n_leaves_old + 1):
                            self.train_nodes[split_right] = 2 * node + 1
                        else:
                            self._leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator)
                    else:
                        self._leaf_prediction(gradient_node, hessian_node, node, estimator)
                else:
                    self._leaf_prediction(gradient_node, hessian_node, node, estimator)
                    
            # Choose next node (based on breadth-first)
            next_nodes = self.train_nodes[self.train_nodes > node]
            if next_nodes.nelement() == 0:
                break
            else:
                node = next_nodes.min()
                                     
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
                split_left = split_left * split_node
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
            split_left = split_left * split_node
            # Check if children are leaves
            leaf_idx_left = torch.eq(2 * node, self.leaves_idx)
            leaf_idx_right = torch.eq(2 * node + 1, self.leaves_idx)
            # Update mu and variance with left leaf prediction
            mu += split_left * (self.leaves_mu * leaf_idx_left).sum(1)
            variance += split_left * (self.leaves_var * leaf_idx_left).sum(1)
            predictions += split_left * leaf_idx_left.sum(1)
            nodes_predict += (1 - leaf_idx_left.sum(1)) * split_left * node
            # Update mu and variance with right leaf prediction
            mu += split_right * (self.leaves_mu * leaf_idx_right).sum(1)
            variance += split_right * (self.leaves_var * leaf_idx_right).sum(1)
            predictions += split_right * leaf_idx_right.sum(1)
            nodes_predict += ( (1  - leaf_idx_right.sum(1)) * split_right * (node + 1))

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
        if (self.params['max_bin'] <= 32) or (self.params['max_bin'] == 64) or (self.params['max_bin'] == 128) or (self.params['max_bin'] == 256):
            X_splits = torch.zeros((X.shape[1], X.shape[0]), device=self.output_device, dtype=torch.uint8)
        else:
            X_splits = torch.zeros((X.shape[1], X.shape[0]), device=self.output_device, dtype=torch.int32)
        
        for i in range(self.params['max_bin']):
            X_splits += (X > self.bins[:, i]).T
        
        return X_splits
    
    def _convert_array(self, array):
        if type(array) == np.ndarray:
            array = torch.from_numpy(array).float()
        elif type(array) == torch.Tensor:
            array = array.float()
        
        return array.to(self.output_device)
    
    def train(self, train_set, objective, metric, params=None, valid_set=None, levels_train=None, levels_valid=None):
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
            levels_train (dictionary of arrays): if the objective requires a levels variable, it can supplied here.
            levels_valid (dictionary of arrays): if the metric requires a levels variable, it can supplied here.
        """
        # Create parameters
        if params is None:
            params = {}
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
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        self.yhat_0 = y_train.mean()
        yhat_train = self.yhat_0.repeat(self.n_samples).to(self.output_device)
        # Fractions of features and samples
        self.feature_fraction = torch.tensor(self.params['feature_fraction'] * self.n_features, device = self.output_device, dtype=torch.int64).clamp(1, self.n_features)
        self.bagging_fraction = torch.tensor(self.params['bagging_fraction'] * self.n_samples, device = self.output_device, dtype=torch.int64).clamp(1, self.n_samples)
        # Create feature bins
        self.bins = self._create_feature_bins(X_train)
        # Pre-allocate arrays
        self.train_nodes = torch.ones(self.bagging_fraction, dtype=torch.int64, device = self.output_device)
        self.nodes_idx = torch.zeros((self.params['n_estimators'], self.params['max_nodes']), dtype=torch.int64, device = self.output_device)
        self.nodes_split_feature = torch.zeros((self.params['n_estimators'], self.params['max_nodes']), dtype=torch.int64, device = self.output_device)
        self.nodes_split_bin = torch.zeros((self.params['n_estimators'], self.params['max_nodes']), dtype=torch.int64, device = self.output_device)
        self.leaves_idx = torch.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=torch.int64, device = self.output_device)
        self.leaves_mu = torch.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=torch.float32, device = self.output_device)
        self.leaves_var = torch.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=torch.float32, device = self.output_device)
        self.feature_importance = torch.zeros(self.n_features, dtype=torch.float32, device=self.output_device)
        self.dist = False
        self.n_forecasts_dist = 1
        self.best_iteration = 0
        # Pre-compute split decisions for X_train
        X_train_splits = self._create_X_splits(X_train)
        # Initialize validation
        validate = False
        if valid_set is not None:
            validate = True
            early_stopping = 0
            X_validate, y_validate = self._convert_array(valid_set[0]), self._convert_array(valid_set[1]).squeeze()
            yhat_validate = self.yhat_0.repeat(y_validate.shape[0])
            self.best_score = torch.tensor(float('inf'), device = self.output_device, dtype=torch.float32)
            # Pre-compute split decisions for X_validate
            X_validate_splits = self._create_X_splits(X_validate)

        # Retrieve initial loss and gradient
        gradient, hessian = self.objective(yhat_train, y_train, levels_train)      
        # Loop over estimators
        for estimator in range(self.params['n_estimators']):
            # Retrieve bagging batch
            samples = torch.randperm(self.n_samples, device=self.output_device)[:self.bagging_fraction]
            # Create tree
            self._create_tree(X_train_splits[:, samples], gradient[samples], hessian[samples], estimator)
            # Get predictions for all samples
            yhat_train, _ = self._predict_tree(X_train_splits, yhat_train, yhat_train*0., estimator)
            # Compute new gradient and hessian
            gradient, hessian = self.objective(yhat_train, y_train, levels_train)
            # Compute metric
            train_metric = metric(yhat_train, y_train, levels_train)
            # Reset train nodes
            self.train_nodes.fill_(1)
            # Validation statistics
            if validate:
                yhat_validate, _ =  self._predict_tree(X_validate_splits, yhat_validate, yhat_validate*0., estimator) 
                validation_metric = metric(yhat_validate, y_validate, levels_valid)
                if (self.params['verbose'] > 1):
                    print(f"Estimator {estimator}/{self.params['n_estimators']}, Train metric: {train_metric:.4f}, Validation metric: {validation_metric:.4f}")
                if validation_metric < self.best_score:
                    self.best_score = validation_metric
                    self.best_iteration = estimator
                    early_stopping = 1
                else:
                    early_stopping += 1
                    if early_stopping == self.params['early_stopping_rounds']:
                        break
            else:
                if (self.params['verbose'] > 1):
                    print(f"Estimator {estimator}/{self.params['n_estimators']}, Train metric: {train_metric:.4f}")
                self.best_iteration = estimator
                self.best_score = 0
            
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
       
    def predict_dist(self, X, n_forecasts=100, parallel=True):
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
            mu = mu.clamp(1e-9)
            variance = torch.nan_to_num(variance, 1e-9).clamp(1e-9)
            scale = ((variance + mu**2) / mu**2).log().clamp(1e-9)
            loc = (mu.log() - 0.5 * scale).clamp(1e-9)
            yhat = LogNormal(loc, scale.sqrt()).rsample([n_forecasts])
        elif self.params['distribution'] == 'gamma':
            variance = torch.nan_to_num(variance, 1e-9)
            mu = torch.nan_to_num(mu, 1e-9)
            rate = (mu.clamp(1e-9) / variance.clamp(1e-9))
            shape = mu.clamp(1e-9) * rate
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

        state_dict = {'nodes_idx': self.nodes_idx.cpu().numpy(),
                      'nodes_split_feature':self.nodes_split_feature.cpu().numpy(),
                      'nodes_split_bin':self.nodes_split_bin.cpu().numpy(),
                      'leaves_idx':self.leaves_idx.cpu().numpy(),
                      'leaves_mu':self.leaves_mu.cpu().numpy(),
                      'leaves_var':self.leaves_var.cpu().numpy(),
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
        self.params['learning_rate'] = torch_float(self.params['learning_rate'])
        self.params['tree_correlation'] = torch_float(self.params['tree_correlation'])
        self.params['lambda'] = torch_float(self.params['lambda'])
        self.params['min_split_gain'] = torch_float(self.params['min_split_gain'])
        self.params['min_data_in_leaf'] = torch_float(self.params['min_data_in_leaf'])
        self.yhat_0  = torch_float(state_dict['yhat0'])
        self.bins = torch_float(state_dict['bins'])
        self.output_device = device
        
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
        
# Create split decision in module for parallelization                
class _split_decision(nn.Module):
    def __init__(self, max_bin, parallelsum_kernel):
        super(_split_decision, self).__init__()
        self.max_bin = max_bin
        self.parallelsum_kernel = parallelsum_kernel
    
    def forward(self, X, gradient, hessian):
        Gl, Hl = self.parallelsum_kernel.split_decision(X.T, gradient, hessian, self.max_bin)
        
        return Gl.unsqueeze(0), Hl.unsqueeze(0)       