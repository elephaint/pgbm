""""""
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
import numpy as np
from torch.autograd import grad
from torch.distributions import Normal, NegativeBinomial, Poisson, StudentT, Laplace, Uniform, TransformedDistribution, SigmoidTransform, AffineTransform, Gamma, Gumbel, Weibull
from torch.utils.cpp_extension import load
from pathlib import Path
import pickle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import r2_score
#%% Load custom kernel
current_path = Path(__file__).parent
if torch.cuda.is_available():
    load(name="split_decision",
        sources=[current_path.joinpath("splitgain_cuda.cpp"),
                 current_path.joinpath("splitgain_kernel.cu")],
        is_python_module=False,
        verbose=True)  
else:
    load(name="split_decision",
        sources=[current_path.joinpath("splitgain_cpu.cpp")],
        is_python_module=False,
        verbose=True)      
#%% Probabilistic Gradient Boosting Machines
class PGBM(object):
    """ Probabilistic Gradient Boosting Machines (PGBM) (Python class)
	
	PGBM fits a Probabilistic Gradient Boosting Machine regression model and returns point and probabilistic predictions. 
	
	This class uses Torch as backend.

	Example: 
		
        .. code-block:: python
        
            from pgbm import PGBM
            model = PGBM()	

    """
    def __init__(self):
        super(PGBM, self).__init__()
        self.cwd = Path().cwd()
        self.new_model = True
 
    def _init_single_param(self, param_name, default, dtype, params=None):
        # Lambda function to convert parameter to correct dtype
        if dtype == 'int' or dtype == 'str' or dtype == 'bool' or dtype == 'other':
            convert = lambda x: x
        elif dtype == 'torch_float':
            convert = lambda x: torch.tensor(x, dtype=torch.float32, device=self.torch_device)
        elif dtype == 'torch_long':
            convert = lambda x: torch.tensor(x, dtype=torch.int64, device=self.torch_device)
        # Check if the parameter is in the supplied dict, else use existing or default
        if param_name in params:
            setattr(self, param_name, convert(params[param_name]))   
        else:
            if not hasattr(self, param_name):
                setattr(self, param_name, convert(default)) 
 
    def _init_params(self, params=None):               
        # Set device
        if 'device' in params:
            if (params['device'] == 'gpu') and torch.cuda.is_available():
                print('Training on GPU')
                if 'gpu_device_id' in params:
                    self.torch_device = torch.device(params['gpu_device_id'])
                    self.device = 'gpu'
                    self.gpu_device_id = params['gpu_device_id']
                else:
                    self.torch_device = torch.device(0)
                    self.device = 'gpu'
                    self.gpu_device_id = 0
            else:
                print('Training on CPU')
                self.torch_device = torch.device('cpu')
                self.device = 'cpu'
        else: 
            print('Training on CPU')
            self.device = 'cpu'
            self.torch_device = torch.device('cpu')  
        # Arrays of parameters
        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate',  'reg_lambda', 
                       'max_leaves', 'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds', 
                       'feature_fraction', 'bagging_fraction', 'seed', 'derivatives', 'distribution', 
                       'checkpoint', 'tree_correlation', 'monotone_constraints', 'monotone_iterations']
        param_dtypes = ['torch_float', 'torch_float', 'torch_float', 'torch_float',
                        'int', 'int', 'int', 'int', 'int', 
                        'torch_float', 'torch_float', 'int', 'str', 'str', 
                        'bool', 'torch_float', 'torch_long', 'int']
        param_defaults = [0.0, 2, 0.1, 1.0, 
                          32, 256, 100, 2, 100, 
                          1, 1, 2147483647, 'exact', 'normal', 
                          False, np.log10(self.n_samples) / 100, 
                          np.zeros(self.n_features), 1]
        # Initialize all parameters
        for i, param in enumerate(param_names):
            self._init_single_param(param, param_defaults[i], param_dtypes[i], params)
        # Check monotone constraints
        assert self.monotone_constraints.shape[0] == self.n_features, "The number of items in the monotonicity constraint list should be equal to the number of features in your dataset."
        self.any_monotone = torch.any(self.monotone_constraints != 0)
        # Make sure we bound certain parameters
        self.min_data_in_leaf = torch.clamp(self.min_data_in_leaf, 2)
        self.min_split_gain = torch.clamp(self.min_split_gain, 0.0)
        self.feature_samples = (self.feature_fraction * self.n_features).clamp(1, self.n_features).type(torch.int64)
        self.bagging_samples = (self.bagging_fraction * self.n_samples).clamp(1, self.n_samples).type(torch.int64)
        self.monotone_iterations = np.maximum(self.monotone_iterations, 1)
        # Set some additional params
        self.max_nodes = self.max_leaves - 1
        torch.manual_seed(self.seed)  # cpu
        torch.cuda.manual_seed_all(self.seed)  
        self.epsilon = 1.0e-4   
    
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
                                      
    def _convert_array(self, array):
        if (type(array) == np.ndarray) or (type(array) == np.memmap):
            array = torch.from_numpy(array).float()
        elif type(array) == torch.Tensor:
            array = array.float()
        
        return array.to(self.torch_device)
    
    def train(self, train_set, objective, metric, params=None, valid_set=None, sample_weight=None, eval_sample_weight=None):
        """Train a PGBM model.
		        
		:param train_set: sample set (X, y) of size ([n_training_samples x n_features], [n_training_samples])) on which to train the PGBM model, where X contains the features of the samples and y is the ground truth.
		:type train_set: tuple
		:param objective: The objective function is the loss function that will be optimized during the gradient boosting process. The function should consume a numpy vector of predictions yhat and ground truth values y and output the gradient and hessian with respect to yhat of the loss function. 
		:type objective: function
		:param metric: The metric function is the function that generates the error metric. The evaluation metric should consume a numpy vector of predictions yhat and ground truth values y, and output a scalar loss. 
		:type metric: function
		:param params: Dictionary containing the learning parameters of a PGBM model, defaults to None.
		:type params: dictionary, optional
		:param valid_set: sample set (X, y) of size ([n_validation_samples x n_features], [n_validation_samples])) on which to validate the PGBM model, where X contains the features of the samples and y is the ground truth, defaults to None.
		:type valid_set: tuple, optional
		:param sample_weight: sample weights for the training data, defaults to None.
		:type sample_weight: torch.Tensor, optional
		:param eval_sample_weight: sample weights for the validation data, defaults to None.
		:type eval_sample_weight: torch.Tensor, optional
		
		:return: `self`
		:rtype: PGBM object
		
		Example:
		
		.. code-block:: python
		
			# Load packages
			from pgbm import PGBM
			import numpy as np
			from sklearn.model_selection import train_test_split
			from sklearn.datasets import fetch_california_housing
			#%% Objective for pgbm
			def mseloss_objective(yhat, y, sample_weight=None):
				gradient = (yhat - y)
				hessian = np.ones_like(yhat)

				return gradient, hessian

			def rmseloss_metric(yhat, y, sample_weight=None):
				loss = torch.sqrt(torch.mean(torch.square(yhat - y)))

				return loss
			#%% Load data
			X, y = fetch_california_housing(return_X_y=True)
			#%% Train pgbm
			# Split data
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
			train_data = (X_train, y_train)
			# Train on set 
			model = PGBM()  
			model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)	
			
        """
        # Create parameters
        self.n_samples = train_set[0].shape[0]
        self.n_features = train_set[0].shape[1]
        if params == None:
            params = {}
        self._init_params(params)
        # Create train data
        X_train, y_train = self._convert_array(train_set[0]), self._convert_array(train_set[1]).squeeze()
        # Set objective & metric
        if self.derivatives == 'exact':
            self.objective = objective
        else:
            self.loss = objective
            self.objective = self._objective_approx
        self.metric = metric
        # Initialize predictions
        n_samples = torch.tensor(X_train.shape[0], device=X_train.device)
        y_train_sum = y_train.sum()
        # Pre-allocate arrays
        nodes_idx = torch.zeros((self.n_estimators, self.max_nodes), dtype=torch.int64, device = self.torch_device)
        nodes_split_feature = torch.zeros_like(nodes_idx)
        nodes_split_bin = torch.zeros_like(nodes_idx)
        leaves_idx = torch.zeros((self.n_estimators, self.max_leaves), dtype=torch.int64, device = self.torch_device)
        leaves_mu = torch.zeros((self.n_estimators, self.max_leaves), dtype=torch.float32, device = self.torch_device)
        leaves_var = torch.zeros_like(leaves_mu)
        # Continue training from existing model or train new model, depending on whether a model was loaded.
        if self.new_model:
            self.initial_estimate = y_train_sum / n_samples                           
            self.best_iteration = 0
            yhat_train = self.initial_estimate.repeat(n_samples)
            self.bins = _create_feature_bins(X_train, self.max_bin)
            self.feature_importance = torch.zeros(self.n_features, dtype=torch.float32, device=self.torch_device)
            self.nodes_idx = nodes_idx
            self.nodes_split_feature = nodes_split_feature
            self.nodes_split_bin = nodes_split_bin
            self.leaves_idx = leaves_idx
            self.leaves_mu = leaves_mu 
            self.leaves_var = leaves_var 
            start_iteration = 0
        else:
            yhat_train = self.predict(X_train, parallel=False)
            self.nodes_idx = torch.cat((self.nodes_idx, nodes_idx))
            self.nodes_split_feature = torch.cat((self.nodes_split_feature, nodes_split_feature))
            self.nodes_split_bin = torch.cat((self.nodes_split_bin, nodes_split_bin))
            self.leaves_idx = torch.cat((self.leaves_idx, leaves_idx))
            self.leaves_mu = torch.cat((self.leaves_mu, leaves_mu))
            self.leaves_var = torch.cat((self.leaves_var, leaves_var))
            start_iteration = self.best_iteration
        # Initialize
        train_nodes = torch.ones(self.n_samples, dtype=torch.int64, device = self.torch_device)
        # Pre-compute split decisions for X_train
        X_train_splits = _create_X_splits(X_train, self.bins)
        # Prepare logs for train metrics
        self.train_metrics = torch.zeros(self.n_estimators + start_iteration, dtype=torch.float32, device=self.torch_device)
        # Initialize validation
        validate = False
        self.best_score = torch.tensor(0., device = self.torch_device, dtype=torch.float32)
        if valid_set is not None:
            validate = True
            early_stopping = 0
            X_validate, y_validate = self._convert_array(valid_set[0]), self._convert_array(valid_set[1]).squeeze()
            # Prepare logs for validation metrics
            self.validation_metrics = torch.zeros(self.n_estimators + start_iteration, dtype=torch.float32, device=self.torch_device)
            if self.new_model:
                yhat_validate = self.initial_estimate.repeat(y_validate.shape[0])
                self.best_score += float('inf')
            else:
                yhat_validate = self.predict(X_validate)
                validation_metric = metric(yhat_validate, y_validate, eval_sample_weight)
                self.best_score += validation_metric
            # Pre-compute split decisions for X_validate
            X_validate_splits = _create_X_splits(X_validate, self.bins)

        # Retrieve initial loss and gradient
        gradient, hessian = self.objective(yhat_train, y_train, sample_weight)      
        # Loop over estimators
        for estimator in range(start_iteration, self.n_estimators + start_iteration):
            # Retrieve bagging batch
            samples = ~torch.round(torch.rand(self.n_samples, device=self.torch_device) * (1 / (2 * self.bagging_fraction))).bool()
            sample_features = torch.arange(self.n_features, device=self.torch_device) if self.feature_fraction == 1.0 else torch.randperm(self.n_features, device=self.torch_device)[:self.feature_samples]
            # Create tree
            self.nodes_idx, self.nodes_split_bin, self.nodes_split_feature, self.leaves_idx,\
            self.leaves_mu, self.leaves_var, self.feature_importance, yhat_train =\
                _create_tree(X_train_splits, gradient,
                            hessian, estimator, train_nodes, 
                            self.nodes_idx, self.nodes_split_bin, self.nodes_split_feature, 
                            self.leaves_idx, self.leaves_mu, self.leaves_var, 
                            self.feature_importance, yhat_train, self.learning_rate,
                            self.max_nodes, samples, sample_features, self.max_bin, 
                            self.min_data_in_leaf, self.reg_lambda, 
                            self.min_split_gain, self.any_monotone,
                            self.monotone_constraints, self.monotone_iterations)                       
            # Compute new gradient and hessian
            gradient, hessian = self.objective(yhat_train, y_train, sample_weight)
            # Compute metric
            train_metric = self.metric(yhat_train, y_train, sample_weight)
            self.train_metrics[estimator] = train_metric
            # Reset train nodes
            train_nodes.fill_(1)
            # Validation statistics
            if validate:
                yhat_validate += _predict_tree_mu(X_validate_splits, self.nodes_idx[estimator],
                                                  self.nodes_split_bin[estimator], self.nodes_split_feature[estimator],
                                                  self.leaves_idx[estimator], self.leaves_mu[estimator], 
                                                  self.learning_rate)
                validation_metric = self.metric(yhat_validate, y_validate, eval_sample_weight)
                self.validation_metrics[estimator] = validation_metric
                if self.verbose > 1:
                    print(f"Estimator {estimator}/{self.n_estimators + start_iteration}, Train metric: {train_metric:.4f}, Validation metric: {validation_metric:.4f}")
                if validation_metric < self.best_score:
                    self.best_score = validation_metric
                    self.best_iteration = estimator + 1
                    early_stopping = 1
                else:
                    early_stopping += 1
                    if early_stopping == self.early_stopping_rounds:
                        break
            else:
                if self.verbose > 1:
                    print(f"Estimator {estimator}/{self.n_estimators + start_iteration}, Train metric: {train_metric:.4f}")
                self.best_iteration = estimator + 1
            
            # Save current model checkpoint to current working directory
            if self.checkpoint:
                self.save(f'{self.cwd}/checkpoint')
            
        # Truncate tree arrays
        self.nodes_idx              = self.nodes_idx[:self.best_iteration]
        self.nodes_split_bin        = self.nodes_split_bin[:self.best_iteration]
        self.nodes_split_feature    = self.nodes_split_feature[:self.best_iteration]
        self.leaves_idx             = self.leaves_idx[:self.best_iteration]
        self.leaves_mu              = self.leaves_mu[:self.best_iteration]
        self.leaves_var             = self.leaves_var[:self.best_iteration]
        self.train_metrics          = self.train_metrics[:self.best_iteration]
        if validate:
            self.validation_metrics = self.validation_metrics[:self.best_iteration]

    def predict(self, X, parallel=True):
        """Generate point estimates/forecasts for a given sample set X.
        
		:param X: sample set of size [n_samples x n_features] for which to create the estimates/forecasts.
		:type X: torch.Tensor
		:param parallel: compute predictions for all trees in parallel (`True`) or serial (`False`). Use `False` when experiencing out-of-memory errors.
		:type parallel: boolean, optional
		
		:return: predictions of size [n_samples]
		:rtype: torch.Tensor
		
		Example:
		
		.. code-block:: python
			
			yhat_test = model.predict(X_test)
        
        """
        X = self._convert_array(X)
        initial_estimate = self.initial_estimate.repeat(X.shape[0])
        mu = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        # Construct split decision tensor
        X_test_splits = _create_X_splits(X, self.bins)
        
        # Predict samples
        if parallel:
            mu = _predict_forest_mu(X_test_splits, self.nodes_idx,
                                    self.nodes_split_bin, self.nodes_split_feature,
                                    self.leaves_idx, self.leaves_mu, 
                                    self.learning_rate, self.best_iteration)          
        else:
            for estimator in range(self.best_iteration):
                mu += _predict_tree_mu(X_test_splits, self.nodes_idx[estimator],
                                      self.nodes_split_bin[estimator], self.nodes_split_feature[estimator],
                                      self.leaves_idx[estimator], self.leaves_mu[estimator], 
                                      self.learning_rate)

        return initial_estimate + mu
       
    def predict_dist(self, X, n_forecasts=100, parallel=True, output_sample_statistics=False):
        """Generate probabilistic estimates/forecasts for a given sample set X
        
		:param X: sample set of size [n_samples x n_features] for which to create the estimates/forecasts.
		:type X: torch.Tensor
		:param n_forecasts: number of estimates/forecasts to create, defaults to 100
		:type n_forecasts: int, optional
		:param parallel: compute predictions for all trees in parallel (`True`) or serial (`False`). Use `False` when experiencing out-of-memory errors.
		:type parallel: boolean, optional
		:param output_sample_statistics: whether to also output the learned sample mean and variance. If True, the function will return a tuple (forecasts, mu, variance) with the latter arrays containing the learned mean and variance per sample that can be used to parameterize a distribution, defaults to False
		:type output_sample_statistics: boolean, optional
		
		:return: predictions of size [n_forecasts x n_samples]
		:rtype: torch.Tensor
		
		Example:
		
		.. code-block:: python
			
			yhat_test = model.predict_dist(X_test)
        
        """
        X = self._convert_array(X)
        initial_estimate = self.initial_estimate.repeat(X.shape[0])
        # Construct split decision tensor
        X_test_splits = _create_X_splits(X, self.bins)
        
        # Compute aggregate mean and variance
        if parallel:
            mu, variance = _predict_forest_muvar(X_test_splits, self.nodes_idx,
                                         self.nodes_split_bin, self.nodes_split_feature,
                                         self.leaves_idx, self.leaves_mu, 
                                         self.leaves_var, self.learning_rate,
                                         self.tree_correlation, self.best_iteration)  
        else:
            mu = torch.zeros(X_test_splits.shape[1], dtype=torch.float32, device=X.device)
            variance = torch.zeros_like(mu)
            for estimator in range(self.best_iteration):
                mu, variance = _predict_tree_muvar(X_test_splits, mu, variance, self.nodes_idx[estimator],
                                      self.nodes_split_bin[estimator], self.nodes_split_feature[estimator],
                                      self.leaves_idx[estimator], self.leaves_mu[estimator], 
                                      self.leaves_var[estimator], self.learning_rate,
                                      self.tree_correlation)        

        # Sample from distribution
        mu += initial_estimate
        if self.distribution == 'normal':
            loc = mu
            scale = torch.nan_to_num(variance.sqrt().clamp(1e-9), 1e-9)
            yhat = Normal(loc, scale).rsample([n_forecasts])
        elif self.distribution == 'studentt':
            v = 3
            loc = mu
            factor = v / (v - 2)
            scale = torch.nan_to_num( (variance / factor).sqrt(), 1e-9)
            yhat = StudentT(v, loc, scale).rsample([n_forecasts])
        elif self.distribution == 'laplace':
            loc = mu
            scale = torch.nan_to_num( (0.5 * variance).sqrt(), 1e-9)
            yhat = Laplace(loc, scale).rsample([n_forecasts])
        elif self.distribution == 'logistic':
            loc = mu
            scale = torch.nan_to_num( ((3 * variance) / np.pi**2).sqrt(), 1e-9)
            base_dist = Uniform(torch.zeros(X.shape[0], device=X.device), torch.ones(X.shape[0], device=X.device))
            yhat = TransformedDistribution(base_dist, [SigmoidTransform().inv, AffineTransform(loc, scale)]).rsample([n_forecasts])
        elif self.distribution == 'lognormal':
            mu_adj = mu.clamp(1e-9)
            variance = torch.nan_to_num(variance, 1e-9).clamp(1e-9)
            loc = torch.log(mu_adj**2 / torch.sqrt(variance + mu_adj**2))
            scale = torch.log(1 + variance / mu_adj**2).clamp(1e-9)
            yhat = torch.exp(Normal(loc, torch.sqrt(scale)).rsample([n_forecasts]))
        elif self.distribution == 'gamma':
            variance = torch.nan_to_num(variance, 1e-9)
            mu_adj = torch.nan_to_num(mu, 1e-9)
            rate = (mu_adj.clamp(1e-9) / variance.clamp(1e-9))
            shape = mu_adj.clamp(1e-9) * rate
            yhat = Gamma(shape, rate).rsample([n_forecasts])
        elif self.distribution == 'gumbel':
            variance = torch.nan_to_num(variance, 1e-9)
            scale = (6 * variance / np.pi**2).sqrt().clamp(1e-9)
            loc = mu - scale * np.euler_gamma
            yhat = Gumbel(loc, scale).rsample([n_forecasts]) 
        elif self.distribution == 'weibull':
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

        elif self.distribution == 'negativebinomial':
            loc = mu.clamp(1e-9)
            eps = 1e-9
            variance = torch.nan_to_num(variance, 1e-9)
            scale = torch.maximum(loc + eps, variance).clamp(1e-9)
            probs = (1 - (loc / scale)).clamp(0, 0.99999)
            counts = (-loc**2 / (loc - scale)).clamp(eps)
            yhat = NegativeBinomial(counts, probs).sample([n_forecasts])
        elif self.distribution == 'poisson':
            yhat = Poisson(mu.clamp(1e-9)).sample([n_forecasts])
        else:
            print('Distribution not (yet) supported')
          
        if output_sample_statistics:
            return (yhat, mu, variance)
        else:
            return yhat
    
    def crps_ensemble(self, yhat_dist, y):
        """Calculate the empirical Continuously Ranked Probability Score (CRPS) for a set of forecasts for a number of samples (lower is better). 
		
		Based on `crps_ensemble` from `properscoring` https://pypi.org/project/properscoring/
        
		:param yhat_dist: forecasts for each sample of size [n_forecasts x n_samples].
		:type yhat_dist: torch.Tensor
		:param y: ground truth value of each sample of size [n_samples].
		:type y: torch.Tensor
		
		:return: CRPS score for each sample
		:rtype: torch.Tensor
		
		Example:
		
		.. code-block:: python
			
			train_set = (X_train, y_train)
			test_set = (X_test, y_test)
			model = PGBM()
			model.train(train_set, objective, metric)
			yhat_test_dist = model.predict_dist(X_test)
			crps = model.crps_ensemble(yhat_test_dist, y_test)
			
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
        """Save a PGBM model to a file. The model parameters are saved as numpy arrays and dictionaries.
		
		:param filename: location of model file
		:type filename: str
		
		:return: dictionary saved in filename
		:rtype: dictionary
		
		Example:
		
		.. code-block:: python
			
			model = PGBM()
			model.train(train_set, objective, metric)
			model.save('model.pt')
        """
        state_dict = {'nodes_idx': self.nodes_idx[:self.best_iteration].cpu().numpy(),
                      'nodes_split_feature':self.nodes_split_feature[:self.best_iteration].cpu().numpy(),
                      'nodes_split_bin':self.nodes_split_bin[:self.best_iteration].cpu().numpy(),
                      'leaves_idx':self.leaves_idx[:self.best_iteration].cpu().numpy(),
                      'leaves_mu':self.leaves_mu[:self.best_iteration].cpu().numpy(),
                      'leaves_var':self.leaves_var[:self.best_iteration].cpu().numpy(),
                      'feature_importance':self.feature_importance.cpu().numpy(),
                      'best_iteration':self.best_iteration,
                      'initial_estimate':self.initial_estimate.cpu().numpy(),
                      'bins':self.bins.cpu().numpy(),
                      'min_split_gain': self.min_split_gain.cpu().numpy(),
                      'min_data_in_leaf': self.min_data_in_leaf.cpu().numpy(),
                      'learning_rate':self.learning_rate.cpu().numpy(),
                      'reg_lambda':self.reg_lambda.cpu().numpy(),
                      'max_leaves': self.max_leaves,
                      'max_bin': self.max_bin,                     
                      'n_estimators': self.n_estimators,
                      'verbose': self.verbose,
                      'early_stopping_rounds': self.early_stopping_rounds,
                      'feature_fraction': self.feature_fraction.cpu().numpy(),
                      'bagging_fraction': self.bagging_fraction.cpu().numpy(),
                      'seed': self.seed,
                      'derivatives': self.derivatives,
                      'distribution': self.distribution,
                      'checkpoint': self.checkpoint,
                      'tree_correlation': self.tree_correlation.cpu().numpy(), 
                      'monotone_constraints': self.monotone_constraints.cpu().numpy(), 
                      'monotone_iterations': self.monotone_iterations
                      }
        
        with open(filename, 'wb') as handle:
            pickle.dump(state_dict, handle)   
    
    def load(self, filename, device=None):
        """Load a PGBM model from a file.
		
		:param filename: location of model file
		:type filename: str
		:param device: torch device, defaults to torch.device('cpu')
		:type device: torch.device, optional
		
		:return: `self`
		:rtype: PGBM object
		
		Example:
		
		.. code-block:: python
			
			model = PGBM()
			model.load('model.pt')
        """
        if device is None:
            self.torch_device = torch.device('cpu')
            self.device = 'cpu'
        else:
            self.torch_device = device
            self.device = 'gpu' if device.type == 'cuda' else 'cpu'

        with open(filename, 'rb') as handle:
            state_dict = pickle.load(handle)
        
        # Helper functions
        torch_float = lambda x: torch.from_numpy(x).float().to(self.torch_device)
        torch_long = lambda x: torch.from_numpy(x).long().to(self.torch_device)
        # Load dict
        self.nodes_idx = torch_long(state_dict['nodes_idx'])
        self.nodes_split_feature  = torch_long(state_dict['nodes_split_feature'])
        self.nodes_split_bin  = torch_long(state_dict['nodes_split_bin'])
        self.leaves_idx  = torch_long(state_dict['leaves_idx'])
        self.leaves_mu  = torch_float(state_dict['leaves_mu'])
        self.leaves_var  = torch_float(state_dict['leaves_var'])
        self.feature_importance  = torch_float(state_dict['feature_importance'])
        self.best_iteration  = state_dict['best_iteration']
        self.initial_estimate  = torch_float(state_dict['initial_estimate'])
        self.bins = torch_float(state_dict['bins'])
        self.min_split_gain = torch_float(state_dict['min_split_gain'])
        self.min_data_in_leaf = torch_float(state_dict['min_data_in_leaf'])
        self.learning_rate = torch_float(state_dict['learning_rate'])
        self.reg_lambda = torch_float(state_dict['reg_lambda'])
        self.max_leaves = state_dict['max_leaves']
        self.max_bin = state_dict['max_bin']
        self.n_estimators = state_dict['n_estimators']
        self.verbose = state_dict['verbose']
        self.early_stopping_rounds = state_dict['early_stopping_rounds']
        self.feature_fraction = torch_float(state_dict['feature_fraction'])
        self.bagging_fraction = torch_float(state_dict['bagging_fraction'])
        self.seed = state_dict['seed']
        self.derivatives = state_dict['derivatives']
        self.distribution = state_dict['distribution']
        self.checkpoint = state_dict['checkpoint']
        self.tree_correlation = torch_float(state_dict['tree_correlation'])
        self.monotone_constraints = torch_long(state_dict['monotone_constraints'])
        self.monotone_iterations = state_dict['monotone_iterations']
        # Set flag to indicate this is not a new model
        self.new_model = False
        
    def permutation_importance(self, X, y=None, n_permutations=10, levels=None):
        """Calculate feature importance of a PGBM model for a sample set X by randomly permuting each feature. 
		
		This function can be executed in a supervised and unsupervised manner, depending on whether y is given. 
		
		If y is provided, the output of this function is the change in error metric when randomly permuting a feature. 
		
		If y is not provided, the output is the weighted average change in prediction when randomly permuting a feature. 
		
		:param X: sample set of size [n_samples x n_features] for which to determine the feature importance.
		:type X: torch.Tensor
		:param y: ground truth of size [n_samples] for sample set X, defaults to None
		:type y: torch.Tensor, optional
		:param n_permutations: number of random permutations to perform for each feature, defaults to 10
		:type n_permutations: int, optional
		
		:return: permutation importance score per feature
		:rtype: torch.Tensor
		
		Example:
		
		.. code-block:: python
		
			train_set = (X_train, y_train)
			test_set = (X_test, y_test)
			model = PGBM()
			model.train(train_set, objective, metric)
			perm_importance_supervised = model.permutation_importance(X_test, y_test)  # Supervised
			perm_importance_unsupervised = model.permutation_importance(X_test)  # Unsupervised
        
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
        """Find the distribution and tree correlation that best fits the data according to lowest CRPS score. 
		
		The parameters 'distribution' and 'tree_correlation' of a PGBM model will be adjusted to the best values after running this script. 
		
		This function returns the best found distribution and tree correlation.
		
		:param X: sample set of size [n_samples x n_features] for which to optimize the distribution.
		:type X: torch.Tensor
		:param y: ground truth of size [n_samples] for sample set X
		:type y: torch.Tensor
		:param distributions: list containing distributions to choose from. Options are: `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gamma`, `gumbel`, `weibull`, `negativebinomial`, `poisson`. Defaults to None (corresponds to iterating over all distributions)
		:type distributions: list, optional
		:param tree_correlations: vector containing tree correlations to use in optimization procedure, defaults to None (corresponds to iterating over a default range).
		:type tree_correlations: torch.Tensor, optional
		
		:return: distribution and tree correlation that yields lowest CRPS
		:rtype: tuple

		Example:
		
		.. code-block:: python
		
			train_set = (X_train, y_train)
			validation_set = (X_validation, y_validation)
			model = PGBM()
			(best_dist, best_tree_corr) = model.optimize_distribution(X_validation, y_validation)

        """               
        # Convert input data if not right type
        X = self._convert_array(X)
        y = self._convert_array(y)
 
        # List of distributions and tree correlations
        if distributions == None:
            distributions = ['normal', 'studentt', 'laplace', 'logistic', 
                             'lognormal', 'gamma', 'gumbel', 'weibull', 'negativebinomial', 'poisson']
        if tree_correlations == None:
            tree_correlations = torch.arange(start=-0.2, end=0.2, step=0.02, dtype=torch.float32, device=X.device)
        else:
            tree_correlations = tree_correlations.float().to(X.device)
               
        # Loop over choices
        crps_best = torch.tensor(float('inf'), device = X.device, dtype=torch.float32)
        distribution_best = self.distribution
        tree_correlation_best = self.tree_correlation
        for distribution in distributions:
            for tree_correlation in tree_correlations:
                self.distribution = distribution
                self.tree_correlation = tree_correlation
                yhat_dist = self.predict_dist(X)
                crps = self.crps_ensemble(yhat_dist, y).mean()
                if self.verbose > 1:
                    print(f'CRPS: {crps:.2f} (Distribution: {distribution}, Tree correlation: {tree_correlation:.3f})')     
                if crps < crps_best:
                    crps_best = crps
                    distribution_best = distribution
                    tree_correlation_best = tree_correlation
        
        # Set to best values
        if self.verbose > 1:
            print(f'Lowest CRPS: {crps_best:.4f} (Distribution: {distribution_best}, Tree correlation: {tree_correlation_best:.3f})')  
        self.distribution = distribution_best
        self.tree_correlation = tree_correlation_best
        
        return (distribution_best, tree_correlation_best)   

@torch.jit.script
def _create_X_splits(X: torch.Tensor, bins: torch.Tensor):
    # Pre-compute split decisions for X
    max_bin = bins.shape[1]
    dtype_split = torch.uint8 if max_bin <= 256 else torch.int16
    X_splits = torch.zeros((X.shape[1], X.shape[0]), device=X.device, dtype=dtype_split)
    for i in range(max_bin):
        X_splits += (X > bins[:, i]).T
    
    return X_splits

@torch.jit.script
def _predict_tree_muvar(X: torch.Tensor, mu: torch.Tensor, variance: torch.Tensor,
                  nodes_idx: torch.Tensor, nodes_split_bin: torch.Tensor, 
                  nodes_split_feature: torch.Tensor, leaves_idx: torch.Tensor, 
                  leaves_mu: torch.Tensor, leaves_var: torch.Tensor, lr: torch.Tensor,
                  corr: torch.Tensor):
    # Get prediction for a single tree
    nodes_predict = torch.ones(X.shape[1], device=X.device, dtype=torch.int)
    node = torch.ones(1, device = X.device, dtype=torch.int64)
    # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
    leaf_idx = torch.eq(node, leaves_idx)
    if torch.any(leaf_idx):
        var_y = leaves_var[leaf_idx]
        mu += -lr * leaves_mu[leaf_idx]
        variance += lr * lr * var_y - 2 * lr * corr * variance.sqrt() * var_y.sqrt()
    else: 
        # Loop until every sample has a prediction (this allows earlier stopping than looping over all possible tree paths)
        for node in nodes_idx:
            if node == 0: break
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
            else:
                nodes_predict += split_left * node
            # Update mu and variance with right leaf prediction
            if torch.any(leaf_idx_right):
                mu += -lr * split_right * leaves_mu[leaf_idx_right]
                var_right = split_right * leaves_var[leaf_idx_right]
                variance += lr**2 * var_right - 2 * lr * corr * variance.sqrt() * var_right.sqrt()
            else:
                nodes_predict += split_right * (node + 1)
                       
    return mu, variance    

@torch.jit.script
def _predict_tree_mu(X: torch.Tensor, nodes_idx: torch.Tensor, 
                        nodes_split_bin: torch.Tensor, nodes_split_feature: torch.Tensor, 
                        leaves_idx: torch.Tensor, leaves_mu: torch.Tensor, 
                        lr: torch.Tensor):
    # Get prediction for a single tree
    nodes_predict = torch.ones(X.shape[1], device=X.device, dtype=torch.int)
    mu = torch.zeros(X.shape[1], device=X.device, dtype=torch.float32)
    node = torch.ones(1, device = X.device, dtype=torch.int64)
    # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
    leaf_idx = torch.eq(node, leaves_idx)
    mu += (leaves_mu * leaf_idx).sum()
    # Loop over nodes
    for node in nodes_idx:
        if node == 0: break
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
        # Left leaf prediction
        mu += split_left * (leaves_mu * leaf_idx_left).sum()
        sum_left = leaf_idx_left.sum()
        nodes_predict += (1 - sum_left) * split_left * node
        # Right leaf prediction
        mu += split_right * (leaves_mu * leaf_idx_right).sum()
        sum_right = leaf_idx_right.sum()
        nodes_predict += (1 - sum_right) * split_right * (node + 1)
                       
    return -lr * mu

@torch.jit.script
def _predict_forest_muvar(X: torch.Tensor, nodes_idx: torch.Tensor, nodes_split_bin: torch.Tensor, 
                    nodes_split_feature: torch.Tensor, leaves_idx: torch.Tensor, 
                    leaves_mu: torch.Tensor, leaves_var: torch.Tensor, lr: torch.Tensor,
                    corr: torch.Tensor, best_iteration: int):
    # Parallel prediction of a tree ensemble
    nodes_predict = torch.ones((X.shape[1], best_iteration), device=X.device, dtype=torch.int64)
    unique_nodes = torch.unique(nodes_idx, sorted=True)
    unique_nodes = unique_nodes[unique_nodes != 0]
    node = torch.ones(1, device = X.device, dtype=torch.int64)
    mu = torch.zeros((X.shape[1], best_iteration), device=X.device, dtype=torch.float32)
    variance = torch.zeros_like(mu)
    # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
    leaf_idx = torch.eq(node, leaves_idx)
    index = torch.any(leaf_idx, dim=1)
    mu[:, index] = leaves_mu[leaf_idx]
    variance[:, index] = leaves_var[leaf_idx]
    # Loop over nodes
    for node in unique_nodes:
        # Select current node information
        split_node = nodes_predict == node
        node_idx = (nodes_idx == node)
        current_features = (nodes_split_feature * node_idx).sum(1)
        current_bins = (nodes_split_bin * node_idx).sum(1, keepdim=True)
        # Split node
        split_left = (X[current_features] > current_bins).T
        split_right = ~split_left * split_node
        split_left *= split_node
        # Check if children are leaves
        leaf_idx_left = torch.eq(2 * node, leaves_idx)
        leaf_idx_right = torch.eq(2 * node + 1, leaves_idx)
        # Update mu and variance with left leaf prediction
        mu += split_left * (leaves_mu * leaf_idx_left).sum(1)
        variance += split_left * (leaves_var * leaf_idx_left).sum(1)
        sum_left = leaf_idx_left.sum(1)
        nodes_predict += (1 - sum_left) * split_left * node
        # Update mu and variance with right leaf prediction
        mu += split_right * (leaves_mu * leaf_idx_right).sum(1)
        variance += split_right * (leaves_var * leaf_idx_right).sum(1)
        sum_right = leaf_idx_right.sum(1)
        nodes_predict += (1  - sum_right) * split_right * (node + 1)

    # Each prediction only for the amount of learning rate in the ensemble
    mu = -lr * mu.sum(1)
    # Variance
    variance = variance.T
    variance_total = torch.zeros(X.shape[1], dtype=torch.float32, device=X.device)
    variance_total += lr**2 * variance[0]
    # I have not figured out how to parallelize the variance estimate calculation in the ensemble, so we iterate over 
    # the variance per estimator
    for estimator in range(1, best_iteration):
        variance_total += lr**2 * variance[estimator] - 2 * lr * corr * variance_total.sqrt() * variance[estimator].sqrt()       

    return mu, variance_total 

@torch.jit.script
def _predict_forest_mu(X: torch.Tensor, nodes_idx: torch.Tensor, nodes_split_bin: torch.Tensor, 
                    nodes_split_feature: torch.Tensor, leaves_idx: torch.Tensor, 
                    leaves_mu: torch.Tensor, lr: torch.Tensor,
                    best_iteration: int):
    # Parallel prediction of a tree ensemble - mean only
    nodes_predict = torch.ones((X.shape[1], best_iteration), device=X.device, dtype=torch.int64)
    unique_nodes = torch.unique(nodes_idx, sorted=True)
    unique_nodes = unique_nodes[unique_nodes != 0]
    node = torch.ones(1, device = X.device, dtype=torch.int64)
    mu = torch.zeros((X.shape[1], best_iteration), device=X.device, dtype=torch.float32)
    # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
    leaf_idx = torch.eq(node, leaves_idx)
    index = torch.any(leaf_idx, dim=1)
    mu[:, index] = leaves_mu[leaf_idx]
    # Loop over nodes
    for node in unique_nodes:
        # Select current node information
        split_node = nodes_predict == node
        node_idx = (nodes_idx == node)
        current_features = (nodes_split_feature * node_idx).sum(1)
        current_bins = (nodes_split_bin * node_idx).sum(1, keepdim=True)
        # Split node
        split_left = (X[current_features] > current_bins).T
        split_right = ~split_left * split_node
        split_left *= split_node
        # Check if children are leaves
        leaf_idx_left = torch.eq(2 * node, leaves_idx)
        leaf_idx_right = torch.eq(2 * node + 1, leaves_idx)
        # Update mu and variance with left leaf prediction
        mu += split_left * (leaves_mu * leaf_idx_left).sum(1)
        sum_left = leaf_idx_left.sum(1)
        nodes_predict += (1 - sum_left) * split_left * node
        # Update mu and variance with right leaf prediction
        mu += split_right * (leaves_mu * leaf_idx_right).sum(1)
        sum_right = leaf_idx_right.sum(1)
        nodes_predict += (1  - sum_right) * split_right * (node + 1)

    # Each prediction only for the amount of learning rate in the ensemble
    mu = -lr * mu.sum(1)

    return mu 

@torch.jit.script
def _create_feature_bins(X: torch.Tensor, max_bin: int = 256):
    # Create array that contains the bins
    bins = torch.zeros((X.shape[1], max_bin), device=X.device)
    quantiles = torch.linspace(0, 1, max_bin, device=X.device)
    # For each feature, create max_bins based on frequency bins. 
    for i in range(X.shape[1]):
        xs = X[:, i]   
        current_bin = torch.unique(torch.quantile(xs, quantiles))
        # A bit inefficiency created here... some features usually have less than max_bin values (e.g. 1/0 indicator features). 
        bins[i, :current_bin.shape[0]] = current_bin
        bins[i, current_bin.shape[0]:] = current_bin.max()
        
    return bins

@torch.jit.script
def _leaf_prediction(gradient: torch.Tensor, hessian: torch.Tensor, node: torch.Tensor,
                     estimator: int, reg_lambda: torch.Tensor, leaves_idx: torch.Tensor,
                     leaves_mu: torch.Tensor, leaves_var: torch.Tensor, 
                     leaf_idx: int, split_node: torch.Tensor, yhat_train: torch.Tensor,
                     lr: torch.Tensor):
    # Empirical mean
    N = gradient.shape[0]       
    gradient_mean = gradient.mean()
    hessian_mean = hessian.mean()   
    # Empirical variance and covariance
    factor = 1 / (N - 1)
    gradient_variance = factor * ((gradient - gradient_mean)**2).sum()
    hessian_variance = factor * ((hessian - hessian_mean)**2).sum()
    covariance = factor * ((gradient - gradient_mean)*(hessian - hessian_mean)).sum()      
    # Mean and variance of the leaf prediction
    lambda_scaled = reg_lambda / N
    mu = gradient_mean / ( hessian_mean + lambda_scaled) -\
        covariance / (hessian_mean + lambda_scaled)**2 +\
        (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3
    
    epsilon = 1e-6
    variance = mu**2 * (gradient_variance / (gradient_mean + epsilon)**2 + \
                   hessian_variance / (hessian_mean + lambda_scaled)**2 \
                    - 2 * covariance / (gradient_mean * (hessian_mean + lambda_scaled) + epsilon) )
    # Save optimal prediction and node information
    leaves_idx[estimator, leaf_idx] = node
    leaves_mu[estimator, leaf_idx] = mu             
    leaves_var[estimator, leaf_idx] = variance
    yhat_train[split_node] -= lr * mu
    # Increase leaf idx       
    leaf_idx += 1  
    
    return leaves_idx, leaves_mu, leaves_var, leaf_idx, yhat_train

@torch.jit.script
def _leaf_prediction_mu(gradient: torch.Tensor, hessian: torch.Tensor, 
                        reg_lambda: torch.Tensor):
    # Empirical mean
    N = gradient.shape[0]       
    gradient_mean = gradient.mean()
    hessian_mean = hessian.mean()   
    # Empirical variance and covariance
    factor = 1 / (N - 1)
    hessian_variance = factor * ((hessian - hessian_mean)**2).sum()
    covariance = factor * ((gradient - gradient_mean)*(hessian - hessian_mean)).sum()      
    # Mean and variance of the leaf prediction
    lambda_scaled = reg_lambda / N
    mu = gradient_mean / ( hessian_mean + lambda_scaled) -\
        covariance / (hessian_mean + lambda_scaled)**2 +\
            (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3
    
    return mu

@torch.jit.script
def _create_tree(X: torch.Tensor, gradient: torch.Tensor, hessian: torch.Tensor, 
                 estimator: int, train_nodes: torch.Tensor, nodes_idx: torch.Tensor, 
                 nodes_split_bin: torch.Tensor, nodes_split_feature: torch.Tensor, 
                 leaves_idx: torch.Tensor, leaves_mu: torch.Tensor, 
                 leaves_var: torch.Tensor, feature_importance: torch.Tensor, 
                 yhat_train: torch.Tensor, learning_rate: torch.Tensor,
                 max_nodes: int, samples: torch.Tensor, 
                 sample_features: torch.Tensor, max_bin: int, 
                 min_data_in_leaf: torch.Tensor, reg_lambda: torch.Tensor, 
                 min_split_gain: torch.Tensor, any_monotone: torch.Tensor,
                 monotone_constraints: torch.Tensor, monotone_iterations: int):
    # Set start node and start leaf index
    leaf_idx = 0
    node_idx = 0
    node = torch.tensor(1, dtype=torch.int64, device=X.device)
    next_nodes = torch.zeros((max_nodes * 2 + 1), dtype=torch.int64, device=X.device)
    next_node_idx = 0
    # Set constraint matrices for monotone constraints
    node_constraint_idx = 0
    node_constraints = torch.zeros((max_nodes * 2 + 1, 3), dtype=torch.float32, device=X.device)
    node_constraints[0, 0] = node
    node_constraints[:, 1] = -np.inf
    node_constraints[:, 2] = np.inf
    # Choose random subset of features
    Xe = X[sample_features]
    # Set other initial variables
    n_samples = samples.sum()
    grad_hess = torch.cat((gradient.unsqueeze(1), hessian.unsqueeze(1)), dim=1)
    # Create tree
    while (leaf_idx < max_nodes + 1) and (node != 0):
        # Retrieve samples in current node, and for train only the samples in the bagging batch
        split_node = train_nodes == node
        split_node_train = split_node * samples
        # Continue making splits until we exceed max_nodes, after that create leaves only
        if node_idx < max_nodes:
            # Select samples in current node
            X_node = Xe[:, split_node_train]
            grad_hess_node = grad_hess[split_node_train]
            # Compute split gain histogram
            Gl, Hl, Glc = torch.ops.pgbm.split_gain(X_node, grad_hess_node, max_bin)                   
            # Compute counts of right leaves
            Grc = grad_hess_node.shape[0] - Glc;
            # Sum gradients and hessian of the node
            G, H = grad_hess_node.sum(0).chunk(2, dim=0)
            # Compute total split_gain
            split_gain_tot = (Gl * Gl) / (Hl + reg_lambda) +\
                            (G - Gl)*(G - Gl) / (H - Hl + reg_lambda) -\
                                (G * G) / (H + reg_lambda)
            # Only consider split gain when enough samples in leaves.
            split_gain_tot *= (Glc >= min_data_in_leaf) * (Grc >= min_data_in_leaf)
            split_gain = split_gain_tot.max()
            # Split if split_gain exceeds minimum
            if split_gain > min_split_gain:
                argmaxsg = split_gain_tot.argmax()
                split_feature_sample = torch.div(argmaxsg, max_bin, rounding_mode='floor')
                split_bin = argmaxsg - split_feature_sample * max_bin
                split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                split_right = ~split_left * split_node
                split_left *= split_node
                # Check for monotone constraints if applicable
                if any_monotone:
                    split_gain_tot_flat = split_gain_tot.flatten()
                    # Find min and max for leaf (mu) weights of current node
                    node_min = node_constraints[node_constraints[:, 0] == node, 1].squeeze()
                    node_max = node_constraints[node_constraints[:, 0] == node, 2].squeeze()
                    # Check if current split proposal has a monotonicity constraint
                    split_constraint = monotone_constraints[sample_features[split_feature_sample]]
                    # Perform check only if parent node has a constraint or if the current proposal is constrained. Todo: this might be a CPU check due to np.inf. Replace np.inf by: torch.tensor(float("Inf"), dtype=torch.float32, device=X.device)
                    if (node_min > -np.inf) or (node_max < np.inf) or (split_constraint != 0):
                        # We precompute the child left- and right weights and evaluate whether they satisfy the constraints. If not, we seek another split and repeat.
                        mu_left = _leaf_prediction_mu(gradient[split_left], hessian[split_left], reg_lambda)
                        mu_right = _leaf_prediction_mu(gradient[split_right], hessian[split_right], reg_lambda)
                        split = 1
                        split_iters = 1
                        condition = split * (((mu_left < node_min) + (mu_left > node_max) +\
                                              (mu_right < node_min) + (mu_right > node_max)) +\
                                             ((split_constraint != 0) * (torch.sign(mu_right - mu_left) != split_constraint)))
                        while condition > 0:
                            # Set gain of current split to -1, as this split is not allowed
                            split_gain_tot_flat[argmaxsg] = -1
                            # Get new split. Check if split_gain is still sufficient, because we might end up with having only constraint invalid splits (i.e. all split_gain <= 0).
                            split_gain = split_gain_tot_flat.max()
                            # Check if new proposed split is allowed, otherwise break loop
                            split = (split_gain > min_split_gain) * int(split_iters < monotone_iterations)
                            if not split: break
                            # Find new split
                            argmaxsg = split_gain_tot_flat.argmax()
                            split_feature_sample = torch.div(argmaxsg, max_bin, rounding_mode='floor')
                            split_bin = argmaxsg - split_feature_sample * max_bin
                            split_left = (Xe[split_feature_sample] > split_bin).squeeze()
                            split_right = ~split_left * split_node
                            split_left *= split_node
                            # Compute new leaf weights
                            mu_left = _leaf_prediction_mu(gradient[split_left], hessian[split_left], reg_lambda)
                            mu_right = _leaf_prediction_mu(gradient[split_right], hessian[split_right], reg_lambda)
                            # Check if new proposed split has a monotonicity constraint
                            split_constraint = monotone_constraints[sample_features[split_feature_sample]]
                            condition = split * (((mu_left < node_min) + (mu_left > node_max) +\
                                                  (mu_right < node_min) + (mu_right > node_max)) +\
                                                 ((split_constraint != 0) * (torch.sign(mu_right - mu_left) != split_constraint)))
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
                            node_constraints[node_constraint_idx, 1] = torch.maximum(left_node_min, node_constraints[node_constraint_idx, 1])
                            node_constraints[node_constraint_idx, 2] = torch.minimum(left_node_max, node_constraints[node_constraint_idx, 2])
                            node_constraint_idx += 1
                            # Set right children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node + 1
                            node_constraints[node_constraint_idx, 1] = torch.maximum(right_node_min, node_constraints[node_constraint_idx, 1])
                            node_constraints[node_constraint_idx, 2] = torch.minimum(right_node_max, node_constraints[node_constraint_idx, 2])
                            node_constraint_idx += 1
                            # Create split
                            feature = sample_features[split_feature_sample]
                            nodes_idx[estimator, node_idx] = node
                            nodes_split_feature[estimator, node_idx] = feature 
                            nodes_split_bin[estimator, node_idx] = split_bin
                            # Feature importance
                            feature_importance[feature] += split_gain * X_node.shape[1] / n_samples 
                            # Assign samples to next node
                            train_nodes += split_left * node + split_right * (node + 1)
                            next_nodes[2 * node_idx] = 2 * node
                            next_nodes[2 * node_idx + 1] = 2 * node + 1
                            node_idx += 1
                        else:
                            leaves_idx, leaves_mu, leaves_var, leaf_idx, yhat_train =\
                                _leaf_prediction(gradient[split_node_train], hessian[split_node_train], node, 
                                                 estimator, reg_lambda, leaves_idx, leaves_mu,
                                                 leaves_var, leaf_idx, split_node, yhat_train,
                                                 learning_rate)
                    else:
                        # Set left children constraints
                        node_constraints[node_constraint_idx, 0] = 2 * node
                        node_constraint_idx += 1
                        # Set right children constraints
                        node_constraints[node_constraint_idx, 0] = 2 * node + 1
                        node_constraint_idx += 1                                                        
                        # Save split information
                        feature = sample_features[split_feature_sample]
                        nodes_idx[estimator, node_idx] = node
                        nodes_split_feature[estimator, node_idx] = feature 
                        nodes_split_bin[estimator, node_idx] = split_bin
                        # Feature importance
                        feature_importance[feature] += split_gain * X_node.shape[1] / n_samples 
                        # Assign samples to next node
                        train_nodes += split_left * node + split_right * (node + 1)
                        next_nodes[2 * node_idx] = 2 * node
                        next_nodes[2 * node_idx + 1] = 2 * node + 1
                        node_idx += 1
                else:
                    # Save split information
                    feature = sample_features[split_feature_sample]
                    nodes_idx[estimator, node_idx] = node
                    nodes_split_feature[estimator, node_idx] = feature 
                    nodes_split_bin[estimator, node_idx] = split_bin
                    # Feature importance
                    feature_importance[feature] += split_gain * X_node.shape[1] / n_samples 
                    # Assign samples to next node
                    train_nodes += split_left * node + split_right * (node + 1)
                    next_nodes[2 * node_idx] = 2 * node
                    next_nodes[2 * node_idx + 1] = 2 * node + 1
                    node_idx += 1
            else:
                leaves_idx, leaves_mu, leaves_var, leaf_idx, yhat_train =\
                    _leaf_prediction(gradient[split_node_train], hessian[split_node_train], node, 
                                     estimator, reg_lambda, leaves_idx, leaves_mu,
                                     leaves_var, leaf_idx, split_node, yhat_train,
                                     learning_rate)
        else:
            leaves_idx, leaves_mu, leaves_var, leaf_idx, yhat_train =\
                _leaf_prediction(gradient[split_node_train], hessian[split_node_train], node, 
                                 estimator, reg_lambda, leaves_idx, leaves_mu,
                                 leaves_var, leaf_idx, split_node, yhat_train,
                                 learning_rate)
                                                                       
        node = next_nodes[next_node_idx]
        next_node_idx += 1
    
    return nodes_idx, nodes_split_bin, nodes_split_feature, leaves_idx,\
        leaves_mu, leaves_var, feature_importance, yhat_train

class PGBMRegressor(BaseEstimator):
    """ Probabilistic Gradient Boosting Machines (PGBM) Regressor (Scikit-learn wrapper)
	
	PGBMRegressor fits a Probabilistic Gradient Boosting Machine regression model and returns point and probabilistic predictions. 
	
	This class uses Torch as backend.
	
	:param objective: Objective function to minimize. If not `mse`, a user defined objective function should be supplied in the form of:
    
		.. code-block:: python	

			def objective(y, yhat, sample_weight=None):
				[......]
				
				return gradient, hessian
			
    
		in which gradient and hessian are of the same shape as y.
	
	:type objective: str or function, default='mse'
	
	:param metric: Metric to evaluate predictions during training and evaluation. If not `rmse`, a user defined metric function should be supplied in the form of:
    
		.. code-block:: python	

			def metric(y, yhat, sample_weight=None):
				[......]
				
				return metric
			
    
		in which metric is a scalar.
	
	:type metric: str or function, default='rmse'
	
	:param max_leaves: The maximum number of leaves per tree. Increase this value to create more complicated trees, and reduce the value to create simpler trees (reduce overfitting).
	:type max_leaves: int, default=32
	
	:param learning_rate: The learning rate of the algorithm; the amount of each new tree prediction that should be added to the ensemble.
	:type learning_rate: float, default=0.1

	:param n_estimators:  The number of trees to create. Typically setting this value higher may  improve performance, at the expense of training speed and potential for overfit. Use in conjunction with learning rate and max_leaves; more trees generally requires a lower learning_rate and/or a lower max_leaves.
	:type n_estimators: int, default=100, constraint>0
	
	:param min_split_gain: The minimum gain for a node to split when building a tree.
	:type min_split_gain: float, default = 0.0, constraint >= 0.0
	
	:param min_data_in_leaf: The minimum number of samples in a leaf of a tree. Increase this value to reduce overfit.
	:type min_data_in_leaf: int, default=3, constraint>=2
	
	:param bagging_fraction: Fraction of samples to use when building a tree. Set to a value between 0 and 1 to randomly select a portion of samples to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
	:type bagging_fraction: float, default=1, constraint>0, constraint<=1
	
	:param feature_fraction: Fraction of features to use when building a tree. Set to a value between 0 and 1 to randomly select a portion of features to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
	:type feature_fraction: float, default=1, constraint>0, constraint<=1
	
	:param max_bin: The maximum number of bins used to bin continuous features. Increasing this value can improve prediction performance, at the cost of training speed and potential overfit.
	:type max_bin: int, default=256, constraint<=32,767
	
	:param reg_lambda: Regularization parameter.
	:type reg_lambda: float, default=1.0, constraint>0
	
	:param random_state: Random seed to use for feature_fraction and bagging_fraction.
	:type random_state: int, default=2147483647
	
	:param device: Choose from `cpu` or `gpu`. Set Torch training device.
	:type device: str, default=`cpu`

	:param gpu_device_id: id of gpu device in case multiple gpus are present in the system, defaults to 0.
	:type gpu_device_id: int, default=0

	:param derivatives: Choose from `exact` or `approx`. Determines whether to compute the derivatives exactly or approximately. If `exact`, PGBMRegressor expects a loss function that outputs a gradient and hessian vector of size [n_training_samples]. If `approx`, PGBMRegressor expects a loss function with a scalar output.
	:type derivatives: str, default=`exact`

	:param distribution: Choice of output distribution for probabilistic predictions. Options are: `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gamma`, `gumbel`, `weibull`, `negativebinomial`, `poisson`. 
	:type distribution: str, default=`normal`
	
	:param checkpoint: Set to `True` to save a model checkpoint after each iteration to the current working directory.
	:type checkpoint: bool, default=`False`
	
	:param tree_correlation: Tree correlation hyperparameter. This controls the amount of correlation we assume to exist between each subsequent tree in the ensemble.
	:type tree_correlation: float, default=np.log_10(n_samples_train)/100
	
	:param monotone_constraints: List detailing monotone constraints for each feature in the dataset, where 0 represents no constraint, 1 a positive monotone constraint, and -1 a negative monotone constraint. For example, for a dataset with 3 features, this parameter could be [1, 0, -1] for respectively a positive, none and negative monotone contraint on feature 1, 2 and 3.
	:type monotone_constraints: List or torch.Tensor
	
	:param monotone_iterations: The number of alternative splits that will be considered if a monotone constraint is violated by the current split proposal. Increase this to improve accuracy at the expense of training speed.  
	:type monotone_iterations: int, default=1
	
	:param verbose: Flag to output metric results for each iteration. Set to 1 to supress output.
	:type verbose: int, default=2
	
	:param init_model: Path to an initial model for which continual training is desired. The model will use the parameters from the initial model.
	:type init_model: str, default=None
	
	:return: `self`
	:rtype: PGBM object
	
	Example: 
		
        .. code-block:: python
        
            from pgbm import PGBMRegressor
            model = PGBMRegressor()	
    
    """
    def __init__(self, objective='mse', metric='rmse', max_leaves=32, learning_rate=0.1, n_estimators=100,
                 min_split_gain=0.0, min_data_in_leaf=3, bagging_fraction=1, feature_fraction=1, max_bin=256,
                 reg_lambda=1.0, random_state=2147483647, device='cpu', gpu_device_id=0, derivatives='exact',
                 distribution='normal', checkpoint=False, tree_correlation=None, monotone_constraints=None, 
                 monotone_iterations=1, verbose=2, init_model=None):
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
        self.init_model = init_model
        
        if self.init_model is not None:
            self._torch_device = torch.device(self.gpu_device_id) if self.device=='gpu' else torch.device('cpu')
            self.learner_ = PGBM()
            self.learner_.load(self.init_model, device=self._torch_device)
    
    def _torch_float_array(self, array):
        return torch.from_numpy(np.array(array).astype(np.float32)).float().to(self._torch_device)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": "This test gives error due to incorrect rtol setting in sklearn's estimator testing package",
            },
        }
    
    def fit(self, X, y, eval_set=None, sample_weight=None, eval_sample_weight=None,
            early_stopping_rounds=None):
        """Fit a PGBMRegressor model.

        :param X: sample set of size [n_training_samples x n_features]
        :type X: torch.Tensor
        :param y: ground truth of size [n_training_samples] for sample set X 
        :type y: torch.Tensor
        :param eval_set: validation set of size ([n_validation_samples x n_features], [n_validation_samples]), defaults to None
        :type eval_set: tuple, optional
        :param sample_weight: sample weights for the training data, defaults to None
        :type sample_weight: torch.Tensor, optional
        :param eval_sample_weight: sample weights for the eval_set, defaults to None
        :type eval_sample_weight: torch.Tensor, optional
        :param early_stopping_rounds: stop training if metric on the eval_set has not improved for `early_stopping_rounds`, defaults to None
        :type early_stopping_rounds: int, optional
        
        :return: `self`
        :rtype: fitted PGBM object

        Example: 
            
        .. code-block:: python
        
            from pgbm import PGBMRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import fetch_california_housing
            X, y = fetch_california_housing(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            model = PGBMRegressor()
            model.fit(X_train, y_train)

        """            
        # Set estimator type
        self._estimator_type = "regressor"
        # Check that X and y have correct shape and convert to float32
        X, y = check_X_y(X, y)
        X, y = X.astype(np.float32), y.astype(np.float32)
        self.n_features_in_ = X.shape[1]
        if X.shape[0] == 1:
            raise ValueError("Data contains only 1 sample")
        # Check eval set
        if eval_set is not None:
            X_valid, y_valid = eval_set[0], eval_set[1]
            X_valid, y_valid = check_X_y(X_valid, y_valid)
            X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
            eval_set = (X_valid, y_valid)

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
                  'reg_lambda':self.reg_lambda,
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
            self._metric = self.rmseloss_metric
        else:
            self._metric = self.metric    
        # Check sample weight shape
        self._torch_device = torch.device(self.gpu_device_id) if self.device=='gpu' else torch.device('cpu')
        if sample_weight is not None: 
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if len(sample_weight) != len(y):
                raise ValueError('Length of sample_weight does not equal length of X and y')
            sample_weight = self._torch_float_array(sample_weight)
        if eval_sample_weight is not None: 
            eval_sample_weight = check_array(eval_sample_weight, ensure_2d=False)
            if len(eval_sample_weight) != len(y_valid):
                raise ValueError("Length of eval_sample_weight does not equal length of X_valid and y_valid")
            eval_sample_weight = self._torch_float_array(eval_sample_weight)

        # Train model
        if self.init_model is None:
            self.learner_ = PGBM()
            self.learner_.train(train_set=(X, y), valid_set=eval_set, params=params, objective=self._objective, 
                         metric=self._metric, sample_weight=sample_weight, 
                         eval_sample_weight=eval_sample_weight)
        else:
            self.learner_.train(train_set=(X, y), valid_set=eval_set, objective=self._objective, 
                     metric=self._metric, sample_weight=sample_weight, 
                     eval_sample_weight=eval_sample_weight)
        return self

    def predict(self, X, parallel=True):
        """Generate point estimates/forecasts for a given sample set X.

		:param X: sample set of size [n_samples x n_features] for which to create the estimates/forecasts.
		:type X: torch.Tensor
		:param parallel: compute predictions for all trees in parallel (`True`) or serial (`False`). Use `False` when experiencing out-of-memory errors.
		:type parallel: boolean, optional
		
		:return: predictions of size [n_samples]
		:rtype: np.array

		Example:
		
		.. code-block:: python
			
			yhat_test = model.predict(X_test)
        
        """         
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(np.float32)
        
        return self.learner_.predict(X, parallel).cpu().numpy()
    
    def score(self, X, y, sample_weight=None, parallel=True):
        """Compute R2 score of fitted PGBMRegressor

        :param X: sample set of size [n_samples x n_features] for which to create the estimates/forecasts.
        :type X: torch.Tensor
        :param y: ground truth of size [n_samples]
        :type y: torch.Tensor
        :param sample_weight: sample weights, defaults to None
        :type sample_weight: torch.Tensor, optional
		:param parallel: compute predictions for all trees in parallel (`True`) or serial (`False`). Use `False` when experiencing out-of-memory errors.
		:type parallel: boolean, optional
        :return: R2 score
        :rtype: float
		
        Example:
		
		.. code-block:: python
			
			model = PGBMRegressor()
			model.fit(X_train, y_train)
			r2_score = model.score(X_test, y_test)       
        """        
        # Checks
        X, y = check_X_y(X, y)
        X, y = X.astype(np.float32), y.astype(np.float32)
        # Make prediction
        yhat = self.predict(X, parallel)
        
        # Check sample weight shape
        if sample_weight is not None: 
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if len(sample_weight) != len(y):
                raise ValueError("Length of sample_weight does not equal length of X and y")
            sample_weight = self._torch_float_array(sample_weight)
                
        # Score prediction with r2
        score = r2_score(y_true=y, y_pred=yhat, sample_weight=sample_weight)
        
        return score
    
    def predict_dist(self, X, n_forecasts=100, parallel=True, output_sample_statistics=False):
        """Generate probabilistic estimates/forecasts for a given sample set X
        
		:param X: sample set of size [n_samples x n_features] for which to create the estimates/forecasts.
		:type X: torch.Tensor
		:param n_forecasts: number of estimates/forecasts to create, defaults to 100
		:type n_forecasts: int, optional
		:param parallel: compute predictions for all trees in parallel (`True`) or serial (`False`). Use `False` when experiencing out-of-memory errors.
		:type parallel: boolean, optional
		:param output_sample_statistics: whether to also output the learned sample mean and variance. If True, the function will return a tuple (forecasts, mu, variance) with the latter arrays containing the learned mean and variance per sample that can be used to parameterize a distribution, defaults to False
		:type output_sample_statistics: boolean, optional
		
		:return: predictions of size [n_forecasts x n_samples]
		:rtype: np.array

		Example:
		
		.. code-block:: python
			
			yhat_test = model.predict_dist(X_test)
        
        """            
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(np.float32)
        
        if output_sample_statistics:
            yhat, mu, variance =  self.learner_.predict_dist(X, n_forecasts, parallel, output_sample_statistics)
            return (yhat.cpu().numpy(), mu.cpu().numpy(), variance.cpu().numpy()) 
        else:
            return self.learner_.predict_dist(X, n_forecasts, parallel, output_sample_statistics).cpu().numpy()
        
    def save(self, filename):
        """Save a fitted PGBM model to a file. The model parameters are saved as numpy arrays and dictionaries.
		
		:param filename: location of model file
		:type filename: str
		
		:return: dictionary saved in filename
		:rtype: dictionary
		
		Example:
		
		.. code-block:: python
			
			model = PGBMRegressor()
			model.fit(X, y)
			model.save('model.pt')
        """        
        self.learner_.save(filename)
    
    def _mseloss_objective(self, yhat, y, sample_weight=None):
        gradient = (yhat - y)
        hessian = torch.ones_like(yhat)
        
        if sample_weight is not None:
            if sample_weight.shape != y.shape:
                raise ValueError("Sample weight should have same shape as y_true")
            gradient *= sample_weight
            hessian *= sample_weight   
    
        return gradient, hessian
    
    def rmseloss_metric(self, yhat, y, sample_weight=None):
        """Root Mean Squared Error Loss 
		
		:param yhat: forecasts for each sample of size [n_samples].
		:type yhat: np.array
		:param y: ground truth value of each sample of size [n_samples].
		:type y: np.array
		:param sample_weight: sample weights of size [n_samples].
		:type sample_weight: np.array
		
		:return: RMSE
		:rtype: float
		
		Example:
		
		.. code-block:: python
			
			model = PGBMRegressor()
			model.fit(X_train, y_train)
			yhat_test = model.predict(X_test)
			rmse = model.rmseloss_metric(yhat_test, y_test)
			
        """     
        error = (yhat - y)
        if sample_weight is not None:
            if sample_weight.shape != y.shape:
                raise ValueError("Sample weight should have same shape as y_true")
            error *= sample_weight
                
        loss = ((error**2).mean())**(0.5)
    
        return loss

    def crps_ensemble(self, yhat_dist, y):
        """Calculate the empirical Continuously Ranked Probability Score (CRPS) for a set of forecasts for a number of samples (lower is better). 
		
		Based on `crps_ensemble` from `properscoring` https://pypi.org/project/properscoring/
        
		:param yhat_dist: forecasts for each sample of size [n_forecasts x n_samples].
		:type yhat_dist: np.array
		:param y: ground truth value of each sample of size [n_samples].
		:type y: np.array
		
		:return: CRPS score for each sample
		:rtype: np.array
		
		Example:
		
		.. code-block:: python
			
			model = PGBMRegressor()
			model.fit(X_train, y_train)
			yhat_test_dist = model.predict_dist(X_test)
			crps = model.crps_ensemble(yhat_test_dist, y_test)
			
        """
        
        return self.learner_.crps_ensemble(yhat_dist, y)
    
    
        
