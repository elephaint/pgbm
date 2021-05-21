"""
   Copyright (c) 2021 Olivier Sprangers 

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

   Olivier Sprangers, Sebastian Schelter, Maarten de Rijke. Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression (https://linktopaper). Accepted for publication at SIGKDD '21.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as parallel
import numpy as np
from torch.autograd import grad
from torch.distributions import Normal, NegativeBinomial, Poisson, StudentT, LogNormal, Laplace, Uniform, TransformedDistribution, SigmoidTransform, AffineTransform, Gamma, Gumbel, Weibull
from torch.utils.cpp_extension import load
parallelsum_kernel = load(name='parallelsum', sources=['pgbm/cuda/parallelsum.cpp', 'pgbm/cuda/parallelsum_kernel.cu'])
#%% Probabilistic Gradient Boosting Machines
class PGBM(nn.Module):
    def __init__(self, params=None):
        super(PGBM, self).__init__()
        # Load params
        if params is None:
            params = {}
        self.params = self._init_params(params)                      
        
        # Set some additional params
        self.params['max_nodes'] = self.params['max_leaves'] - 1
        torch.manual_seed(self.params['seed'])
        self.epsilon = 1e-6
    
    def _init_params(self, params):
        # self.params = {}
        params_new = {}
        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'tree_correlation', 'max_leaves',
                       'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds', 'feature_fraction', 'bagging_fraction', 
                       'seed', 'device', 'output_device', 'gpu_device_ids', 'derivatives', 'distribution']
        param_defaults = [0.0, 2, 0.1, 1.0, 0.03, 32, 256, 100, 2, 100, 1, 1, 0, 'cpu', 'cpu', (0,), 'exact', 'normal']
        
        # Create gpu functions for split decision
        if 'device' in params:
            if params['device'] == 'gpu':
                self.device_ids = params['gpu_device_ids'] if 'gpu_device_ids' in params else (0,)
                max_bin = params['max_bin'] if 'max_bin' in params else 256
                self.parallel_split_decision = parallel.replicate(_split_decision(max_bin), self.device_ids)
                if params['output_device'] == 'gpu':
                    output_id = params['gpu_device_ids'][0]
                    self.output_device = params['gpu_device_ids'][output_id]
                else:
                    self.output_device = torch.device('cpu')
            else:
                self.output_device = torch.device('cpu')
        else: 
            self.output_device = torch.device('cpu')
                   
        for i, param in enumerate(param_names):
            params_new[param] = params[param] if param in params else param_defaults[i]
            if i < 5:
                params_new[param] = torch.tensor(params_new[param], device=self.output_device, dtype=torch.float32)
        
        return params_new
        
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
        if levels is not None:
            loss = self.loss(yhat, y, levels)
            loss_upper = self.loss(yhat_upper, y, levels)
            loss_lower = self.loss(yhat_lower, y, levels)
        else: 
            loss = self.loss(yhat, y)
            loss_upper = self.loss(yhat_upper, y)
            loss_lower = self.loss(yhat_lower, y)            
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
        sample_features = torch.randperm(n_features, device=X.device, dtype=torch.int64)[:self.feature_fraction]
        Xe = X[sample_features]
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
                    Gl, Hl = parallelsum_kernel.split_decision(X_node, gradient_node, hessian_node, self.params['max_bin'])
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
                              
    def _predict_leaf(self, mu_x, var_x, leaves_idx, leaves_mu, leaves_var, node):
        leaf_idx = leaves_idx == node
        lr = self.params['learning_rate']
        corr = self.params['tree_correlation']
        mu_y = leaves_mu[leaf_idx]
        mu_y = lr * mu_y
        mu = mu_x - mu_y
        if self.dist == True:
            var_y = leaves_var[leaf_idx]
            variance = var_x + lr * lr * var_y - 2 * lr * corr * var_x.sqrt() * var_y.sqrt()
        else:
            variance = var_x
        return mu, variance        
    
    def _predict_tree(self, X, mu, variance, estimator):
        predictions = torch.zeros(X.shape[1], device=X.device, dtype=torch.int)
        nodes_predict = torch.ones(X.shape[1], device=X.device, dtype=torch.int)
        mu_total = torch.zeros_like(mu)
        variance_total = torch.zeros_like(variance)
        leaves_idx = self.leaves_idx[estimator]
        leaves_mu = self.leaves_mu[estimator]
        leaves_var = self.leaves_var[estimator]
        nodes_idx = self.nodes_idx[estimator]
        nodes_split_feature = self.nodes_split_feature[estimator]
        nodes_split_bin = self.nodes_split_bin[estimator]
        node = torch.ones(1, device = X.device, dtype=torch.int64)
        # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
        if torch.any(torch.eq(node, leaves_idx)):
            mu_current, variance_current = self._predict_leaf(mu, variance, leaves_idx, leaves_mu, leaves_var, node)
            predictions += 1
            mu_total += mu_current
            variance_total += variance_current
        else: 
            # Loop until every sample has a prediction (this allows earlier stopping than looping over all possible tree paths)
            while predictions.sum() < len(predictions):
                # Choose next node (based on breadth-first)
                node = nodes_predict[(nodes_predict >= node) & (predictions == 0)].min()
                # Select current node information
                split_node = nodes_predict == node
                current_feature = nodes_split_feature[nodes_idx == node]
                current_bin = nodes_split_bin[nodes_idx == node]
                # Split node
                split_left = (X[current_feature] > current_bin).squeeze()
                split_right = ~split_left * split_node
                split_left = split_left * split_node
                # Assign information
                # Get prediction left if it exists
                if torch.any(torch.eq(2 * node, leaves_idx)):
                    mu_current, variance_current = self._predict_leaf(mu[split_left], variance[split_left], leaves_idx, leaves_mu, leaves_var, 2 * node)
                    predictions[split_left] = 1
                    mu_total[split_left] = mu_current
                    variance_total[split_left] = variance_current
                else:
                    nodes_predict[split_left] = 2 * node
                # Get prediction right if it exists
                if torch.any(torch.eq(2 * node + 1, leaves_idx)):
                    mu_current, variance_current = self._predict_leaf(mu[split_right], variance[split_right], leaves_idx, leaves_mu, leaves_var, 2 * node + 1)
                    predictions[split_right] = 1
                    mu_total[split_right] = mu_current
                    variance_total[split_right] = variance_current
                else:
                    nodes_predict[split_right] = 2 * node + 1
                   
        return mu_total, variance_total      
    
    def _create_X_splits(self, X):
        # Pre-compute split decisions for Xtrain
        if (self.params['max_bin'] <= 32) or (self.params['max_bin'] == 64) or (self.params['max_bin'] == 128) or (self.params['max_bin'] == 256):
            X_splits = torch.zeros((X.shape[1], X.shape[0]), device=self.output_device, dtype=torch.uint8)
        else:
            X_splits = torch.zeros((X.shape[1], X.shape[0]), device=self.output_device, dtype=torch.int32)
        
        for i in range(self.params['max_bin']):
            X_splits += (X > self.bins[:, i]).T
        
        return X_splits
    
    def train(self, train_set, objective, metric, valid_set=None, levels=None):
        # Create train data, send to device
        X_train, y_train = train_set
        X_train, y_train = X_train.to(self.output_device), y_train.to(self.output_device)
        if self.params['derivatives'] == 'exact':
            self.objective = objective
        else:
            self.loss = objective
            self.objective = self._objective_approx
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
        self.dist = False
        self.n_samples_dist = 1
        self.best_iteration = 0
        # Pre-compute split decisions for X_train
        X_train_splits = self._create_X_splits(X_train)
        del X_train
        # Initialize validation
        validate = False
        if valid_set is not None:
            validate = True
            early_stopping = 0
            X_validate, y_validate = valid_set
            X_validate, y_validate = X_validate.to(self.output_device), y_validate.to(self.output_device)
            yhat_validate = self.yhat_0.repeat(y_validate.shape[0])
            self.best_score = torch.tensor(float('inf'), device = self.output_device, dtype=torch.float32)
            # Pre-compute split decisions for X_validate
            X_validate_splits = self._create_X_splits(X_validate)
            del X_validate

        # Retrieve initial loss and gradient
        if levels is not None:
            gradient, hessian = self.objective(yhat_train, y_train, levels)      
        else:
            gradient, hessian = self.objective(yhat_train, y_train)
        # Loop over estimators
        for estimator in range(self.params['n_estimators']):
            # Retrieve bagging batch
            samples = torch.randperm(self.n_samples, device=self.output_device)[:self.bagging_fraction]
            # Create tree
            self._create_tree(X_train_splits[:, samples], gradient[samples], hessian[samples], estimator)
            # Get predictions for all samples
            yhat_train, _ = self._predict_tree(X_train_splits, yhat_train, yhat_train*0., estimator)
            # Compute new gradient and hessian
            if levels is not None:
                gradient, hessian = self.objective(yhat_train, y_train, levels)
            else:
                gradient, hessian = self.objective(yhat_train, y_train)
            # Compute metric
            train_metric = metric(yhat_train, y_train)
            # Reset train nodes
            self.train_nodes.fill_(1)
            # Validation statistics
            if validate:
                yhat_validate, _ =  self._predict_tree(X_validate_splits, yhat_validate, yhat_validate*0., estimator) 
                validation_metric = metric(yhat_validate, y_validate)
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
                       
    def predict(self, X):
        X = X.to(self.output_device)
        self.dist = False
        self.n_samples_dist = 1
        yhat0 = self.yhat_0.repeat(X.shape[0])
        mu = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        variance = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        # Construct split decision tensor
        X_test_splits = self._create_X_splits(X)
        
        # Predict samples
        for estimator in range(self.best_iteration):
            self.estimator = estimator
            mu, variance = self._predict_tree(X_test_splits, mu, variance, estimator)

        return yhat0 + mu
    
    def predict_dist(self, X, n_samples):
        X = X.to(self.output_device)
        self.dist = True
        self.n_samples_dist = n_samples
        yhat0 = self.yhat_0.repeat(X.shape[0])
        mu = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        variance = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        # Construct split decision tensor
        X_test_splits = self._create_X_splits(X)
        
        # Compute aggregate mean and variance
        for estimator in range(self.best_iteration):
            self.estimator = estimator
            mu, variance = self._predict_tree(X_test_splits, mu, variance, estimator)
        
        # Sample from distribution
        mu += yhat0
        if self.params['distribution'] == 'normal':
            loc = mu
            scale = torch.nan_to_num(variance.sqrt(), 1e-9)
            yhat = Normal(loc, scale).rsample([n_samples])
        elif self.params['distribution'] == 'studentt':
            v = 3
            loc = mu
            factor = v / (v - 2)
            scale = torch.nan_to_num( (variance / factor).sqrt(), 1e-9)
            yhat = StudentT(v, loc, scale).rsample([n_samples])
        elif self.params['distribution'] == 'laplace':
            loc = mu
            scale = torch.nan_to_num( (0.5 * variance).sqrt(), 1e-9)
            yhat = Laplace(loc, scale).rsample([n_samples])
        elif self.params['distribution'] == 'logistic':
            loc = mu
            scale = torch.nan_to_num( ((3 * variance) / np.pi**2).sqrt(), 1e-9)
            base_dist = Uniform(torch.zeros(X.shape[0], device=X.device), torch.ones(X.shape[0], device=X.device))
            yhat = TransformedDistribution(base_dist, [SigmoidTransform().inv, AffineTransform(loc, scale)]).rsample([n_samples])
        elif self.params['distribution'] == 'lognormal':
            mu = mu.clamp(1e-9)
            variance = torch.nan_to_num(variance, 1e-9).clamp(1e-9)
            scale = ((variance + mu**2) / mu**2).log().clamp(1e-9)
            loc = (mu.log() - 0.5 * scale).clamp(1e-9)
            yhat = LogNormal(loc, scale.sqrt()).rsample([n_samples])
        elif self.params['distribution'] == 'gamma':
            variance = torch.nan_to_num(variance, 1e-9)
            rate = (mu.clamp(1e-9) / variance.clamp(1e-9))
            shape = mu.clamp(1e-9) * rate
            yhat = Gamma(shape, rate).rsample([n_samples])
        elif self.params['distribution'] == 'gumbel':
            variance = torch.nan_to_num(variance, 1e-9)
            scale = (6 * variance / np.pi**2).sqrt().clamp(1e-9)
            loc = mu - scale * np.euler_gamma
            yhat = Gumbel(loc, scale).rsample([n_samples]) 
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
            yhat = Weibull(scale, k).rsample([n_samples]) 

        elif self.params['distribution'] == 'negativebinomial':
            loc = mu.clamp(1e-9)
            eps = 1e-9
            variance = torch.nan_to_num(variance, 1e-9)
            scale = torch.maximum(loc + eps, variance).clamp(1e-9)
            probs = (1 - (loc / scale)).clamp(0, 0.99999)
            counts = (-loc**2 / (loc - scale)).clamp(eps)
            yhat = NegativeBinomial(counts, probs).sample([n_samples])
        elif self.params['distribution'] == 'poisson':
            yhat = Poisson(mu.clamp(1e-9)).sample([n_samples])
          
        
        return yhat
    
class _split_decision(nn.Module):
    def __init__(self, max_bin):
        super(_split_decision, self).__init__()
        self.max_bin = max_bin
    
    def forward(self, X, gradient, hessian):
        Gl, Hl = parallelsum_kernel.split_decision(X.T, gradient, hessian, self.max_bin)
        
        return Gl.unsqueeze(0), Hl.unsqueeze(0)

# Calculates the empirical CRPS for a set of forecasts for a number of samples
def crps_ensemble(y, yhat_dist):
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
    
    
    