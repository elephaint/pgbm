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
   https://arxiv.org/abs/2106.0168 
   Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and
   Data Mining (KDD ’21), August 14–18, 2021, Virtual Event, Singapore.
   https://doi.org/10.1145/3447548.3467278

"""
#%% Import packages
import numpy as np
from numba import njit, prange, config
import pickle
#%% Probabilistic Gradient Boosting Machines
class PGBM(object):
    def __init__(self):
        super(PGBM, self).__init__()
    
    def _init_params(self, params):       
        self.params = {}
        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'tree_correlation', 'max_leaves',
                       'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds', 'feature_fraction', 'bagging_fraction', 
                       'seed', 'split_parallel', 'distribution']
        param_defaults = [0.0, 2, 0.1, 1.0, 0.03, 32, 256, 100, 2, 100, 1, 1, 0, 'feature', 'normal']
                          
        for i, param in enumerate(param_names):
            self.params[param] = params[param] if param in params else param_defaults[i]
                
        # Set some additional params
        self.params['max_nodes'] = self.params['max_leaves'] - 1
        self.epsilon = 1.0e-4
        np.random.seed(self.params['seed'])            

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _create_feature_bins(X, max_bin):
        # Create array that contains the bins
        bins = np.zeros((X.shape[1], max_bin))
        # For each feature, create max_bins based on frequency bins. 
        # Note: this function will fail if a feature has only a single unique value.
        for i in prange(X.shape[1]):
            xc = X[:, i]
            nunique = len(np.unique(xc))
            if (nunique <= max_bin + 1):
                current_bin = np.unique(xc)
                current_bin = current_bin[1:]
            elif nunique > 1:
                current_bin = np.percentile(xc, np.linspace(0, 100, max_bin + 2))[1:-1]
                
            # A bit inefficiency created here... some features usually have less than max_bin values (e.g. 1/0 indicator features). 
            # However, my current implementation just does one massive matrix multiplication over all the bins, requiring equal bin size for all features.
            bins[i, :len(current_bin)] = current_bin
            bins[i, len(current_bin):] = current_bin.max()
            
        return bins
    
    @staticmethod
    @njit(fastmath=True)
    def _create_tree(X, gradient, hessian, estimator, train_nodes, bins, nodes_idx, nodes_split_feature, nodes_split_bin, leaves_idx, leaves_mu, leaves_var, feature_importance, lampda, max_nodes, max_leaves, max_bin, feature_fraction, min_split_gain, min_data_in_leaf, split_parallel):
        # Set start node and start leaf index
        leaf_idx = 0
        node_idx = 0
        node = 1
        n_features = X.shape[0]
        n_samples = X.shape[1]
        sample_features = np.random.choice(n_features, feature_fraction, replace=False)
        Xe = X[sample_features]
        # Create tree
        while leaf_idx < max_leaves:
            split_node = train_nodes == node
            # Only create node if there are samples in it
            if split_node.any():
                # Choose feature subset
                X_node = Xe[:, split_node]
                gradient_node = gradient[split_node]
                hessian_node = hessian[split_node]
                # Comput split_gain
                G = np.sum(gradient_node)
                H = np.sum(hessian_node)
                if split_parallel == 'sample':
                    Gl, Hl = _split_decision_sample_parallel(X_node, gradient_node, hessian_node, max_bin)
                else:
                    Gl, Hl = _split_decision_feature_parallel(X_node, gradient_node, hessian_node, max_bin)
                    
                split_gain_tot = (Gl * Gl) / (Hl + lampda) + (G - Gl)*(G - Gl) / (H - Hl + lampda) - (G * G) / (H + lampda)
                split_gain = split_gain_tot.max()              
                # Split if split_gain exceeds minimum
                if split_gain > min_split_gain:
                    argmaxsg = split_gain_tot.argmax()
                    split_feature_sample = argmaxsg // max_bin
                    split_bin = argmaxsg - split_feature_sample * max_bin
                    split_left = (Xe[split_feature_sample] > split_bin)
                    # print(split_left.shape)
                    split_right = ~split_left * split_node
                    split_left = split_left * split_node
                    # Split when enough data in leafs                        
                    if (np.sum(split_left) > min_data_in_leaf) & (np.sum(split_right) > min_data_in_leaf):
                        # Save split information
                        nodes_idx[estimator, node_idx] = node
                        nodes_split_feature[estimator, node_idx] = sample_features[split_feature_sample] 
                        nodes_split_bin[estimator, node_idx] = split_bin
                        node_idx += 1
                        feature_importance[sample_features[split_feature_sample]] += split_gain * X_node.shape[1] / n_samples 
                        # Assign next node to samples if next node does not exceed max n_leaves
                        criterion = 2 * (max_nodes - node_idx + 1)
                        n_leaves_old = leaf_idx
                        if (criterion >  max_leaves - n_leaves_old):
                            train_nodes[split_left] = 2 * node
                        else:
                            leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                        if (criterion >  max_leaves - n_leaves_old + 1):
                            train_nodes[split_right] = 2 * node + 1
                        else:
                            leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                    else:
                        leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient_node, hessian_node, node, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                else:
                    leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient_node, hessian_node, node, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                    
            # Choose next node (based on breadth-first)
            next_nodes = train_nodes[train_nodes > node]
            if next_nodes.shape[0] == 0:
                break
            else:
                node = next_nodes.min()
            
        return nodes_idx, nodes_split_feature, nodes_split_bin, leaves_idx, leaves_mu, leaves_var, feature_importance
        
    @staticmethod
    @njit(fastmath=True)
    def _predict_tree(X, mu, variance, estimator, nodes_idx, nodes_split_feature, nodes_split_bin, leaves_idx, leaves_mu, leaves_var, learning_rate, tree_correlation, dist):
        predictions = np.zeros(X.shape[1], dtype=np.int64)
        nodes_predict = np.ones(X.shape[1], dtype=np.int64)
        lr = learning_rate
        corr = tree_correlation
        node = 1
        # Capture edge case where first node is a leaf (i.e. there is no decision tree, only a single leaf)
        leaf_idx = np.equal(node, leaves_idx)
        if np.any(leaf_idx):
            var_y = leaves_var[leaf_idx]
            mu += -lr * leaves_mu[leaf_idx]
            variance += lr * lr * var_y - 2 * lr * corr * np.sqrt(variance) * np.sqrt(var_y)
            predictions += 1
        else: 
            # Loop until every sample has a prediction (this allows earlier stopping than looping over all possible tree paths)
            while np.sum(predictions) < len(predictions):
                # Choose next node (based on breadth-first)
                condition = (nodes_predict >= node) * (predictions == 0)
                node = nodes_predict[condition].min()
                # Select current node information
                split_node = nodes_predict == node
                node_idx = (nodes_idx == node)
                current_feature = np.sum(nodes_split_feature * node_idx)
                current_bin = np.sum(nodes_split_bin * node_idx)
                # Split node
                split_left = (X[current_feature] > current_bin)
                split_right = ~split_left * split_node
                split_left = split_left * split_node
                # Check if children are leaves
                leaf_idx_left = np.equal(2 * node, leaves_idx)
                leaf_idx_right = np.equal(2 * node + 1, leaves_idx)
                # Update mu and variance with left leaf prediction
                if np.any(leaf_idx_left):
                    mu += -lr * split_left * leaves_mu[leaf_idx_left]
                    var_left = split_left * leaves_var[leaf_idx_left]
                    variance += lr**2 * var_left - 2 * lr * corr * np.sqrt(variance) * np.sqrt(var_left)
                    predictions += split_left
                else:
                    nodes_predict += split_left * node
                # Update mu and variance with right leaf prediction
                if np.any(leaf_idx_right):
                    mu += -lr * split_right * leaves_mu[leaf_idx_right]
                    var_right = split_right * leaves_var[leaf_idx_right]
                    variance += lr**2 * var_right - 2 * lr * corr * np.sqrt(variance) * np.sqrt(var_right)
                    predictions += split_right
                else:
                    nodes_predict += split_right * (node + 1)
                   
        return mu, variance
    
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _create_X_splits(X, bins, max_bin):
        # Pre-compute split decisions for Xtrain
        X_splits = np.zeros((X.shape[1], X.shape[0]), dtype=np.int32)
        
        for i in prange(max_bin):
            X_splits += (X > bins[:, i]).T
    
        return X_splits
    
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
                The function should consume a numpy vector of predictions yhat and ground truth values y and output the gradient and 
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
        X_train, y_train = train_set[0], train_set[1].squeeze()
        # Set objective & metric
        self.objective = objective
        self.metric = metric
        # Initialize predictions
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        self.yhat_0 = y_train.mean()
        yhat_train = self.yhat_0.repeat(self.n_samples)
        # Fractions of features and samples
        self.feature_fraction = np.clip(int(self.params['feature_fraction'] * self.n_features), 1, self.n_features, dtype=np.int64)
        self.bagging_fraction = np.clip(int(self.params['bagging_fraction'] * self.n_samples), 1, self.n_samples, dtype=np.int64)
        # Create feature bins
        self.bins = self._create_feature_bins(X_train, self.params['max_bin'])
        # Pre-allocate arrays
        train_nodes = np.ones(self.bagging_fraction, dtype=np.int64)
        self.nodes_idx = np.zeros((self.params['n_estimators'], self.params['max_nodes']), dtype=np.int64)
        self.nodes_split_feature = np.zeros_like(self.nodes_idx)
        self.nodes_split_bin = np.zeros_like(self.nodes_idx)
        self.leaves_idx = np.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=np.int64)
        self.leaves_mu = np.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=np.float64)
        self.leaves_var = np.zeros_like(self.leaves_mu)
        self.feature_importance = np.zeros(self.n_features, dtype=np.float64)
        dist = False
        # Pre-compute split decisions for X_train
        X_train_splits = self._create_X_splits(X_train, self.bins, self.params['max_bin'])
        # Initialize validation
        validate = False
        if valid_set is not None:
            validate = True
            early_stopping = 0
            X_validate, y_validate = valid_set[0], valid_set[1].squeeze()
            yhat_validate = self.yhat_0.repeat(y_validate.shape[0])
            self.best_score = np.inf
            # Pre-compute split decisions for X_validate
            X_validate_splits = self._create_X_splits(X_validate, self.bins, self.params['max_bin'])

        # Retrieve initial loss and gradient
        gradient, hessian = self.objective(yhat_train, y_train, levels_train)      
        # Loop over estimators
        for estimator in range(self.params['n_estimators']):
            # Retrieve bagging batch
            samples = np.random.choice(self.n_samples, self.bagging_fraction)
            # Create tree
            self.nodes_idx, self.nodes_split_feature, self.nodes_split_bin, self.leaves_idx, self.leaves_mu, self.leaves_var, self.feature_importance = self._create_tree(X_train_splits[:, samples], gradient[samples], hessian[samples], estimator, train_nodes, self.bins, self.nodes_idx, self.nodes_split_feature, self.nodes_split_bin, self.leaves_idx, self.leaves_mu, self.leaves_var, self.feature_importance, self.params['lambda'], self.params['max_nodes'], self.params['max_leaves'], self.params['max_bin'], self.feature_fraction, self.params['min_split_gain'], self.params['min_data_in_leaf'], self.params['split_parallel'])
            # Get predictions for all samples
            yhat_train, _ = self._predict_tree(X_train_splits, yhat_train, yhat_train*0., estimator, self.nodes_idx[estimator], self.nodes_split_feature[estimator], self.nodes_split_bin[estimator], self.leaves_idx[estimator], self.leaves_mu[estimator], self.leaves_var[estimator], self.params['learning_rate'], self.params['tree_correlation'], dist)
            # Compute new gradient and hessian
            gradient, hessian = self.objective(yhat_train, y_train, levels_train)
            # Compute metric
            train_metric = metric(yhat_train, y_train, levels_train)
            # Reset train nodes
            train_nodes.fill(1)
            # Validation statistics
            if validate:
                yhat_validate, _ =  self._predict_tree(X_validate_splits, yhat_validate, yhat_validate*0., estimator, self.nodes_idx[estimator],  self.nodes_split_feature[estimator], self.nodes_split_bin[estimator], self.leaves_idx[estimator], self.leaves_mu[estimator], self.leaves_var[estimator], self.params['learning_rate'], self.params['tree_correlation'], dist)
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
            parallel (boolean): not applicable. Only in place to support easy switching between 
                Torch and Numba backend.
        """
        dist = False
        yhat0 = self.yhat_0.repeat(X.shape[0])
        mu = np.zeros(X.shape[0], dtype=np.float64)
        variance = np.zeros(X.shape[0], dtype=np.float64)
        # Construct split decision tensor
        X_test_splits = self._create_X_splits(X, self.bins, self.params['max_bin'])
        
        # Predict samples
        for estimator in range(self.best_iteration):
            mu, variance =  self._predict_tree(X_test_splits, mu, variance, estimator, self.nodes_idx[estimator], self.nodes_split_feature[estimator], self.nodes_split_bin[estimator], self.leaves_idx[estimator], self.leaves_mu[estimator], self.leaves_var[estimator], self.params['learning_rate'], self.params['tree_correlation'], dist)

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
            parallel (boolean): not applicable. Only in place to support easy switching between 
                Torch and Numba backend.
        """
        dist = True
        yhat0 = self.yhat_0.repeat(X.shape[0])
        mu = np.zeros(X.shape[0], dtype=np.float64)
        variance = np.zeros(X.shape[0], dtype=np.float64)
        # Construct split decision tensor
        X_test_splits = self._create_X_splits(X, self.bins, self.params['max_bin'])
        
        # Compute aggregate mean and variance
        for estimator in range(self.best_iteration):
            mu, variance =  self._predict_tree(X_test_splits, mu, variance, estimator, self.nodes_idx[estimator], self.nodes_split_feature[estimator], self.nodes_split_bin[estimator], self.leaves_idx[estimator], self.leaves_mu[estimator], self.leaves_var[estimator], self.params['learning_rate'], self.params['tree_correlation'], dist)
        
        # Sample from distribution
        mu += yhat0
        if self.params['distribution'] == 'normal':
            loc = mu
            scale = np.sqrt(variance)
            yhat = np.random.normal(loc, scale, (n_forecasts, loc.shape[0]))
        elif self.params['distribution'] == 'laplace':
            loc = mu
            scale = np.clip(np.sqrt(0.5 * variance), 1e-9)
            yhat =  np.random.laplace(loc, scale, (n_forecasts, loc.shape[0]))
        elif self.params['distribution'] == 'logistic':
            loc = mu
            scale = np.clip(np.sqrt((3 * variance) / np.pi**2), 1e-9)
            yhat =  np.random.logistic(loc, scale, (n_forecasts, loc.shape[0]))
        elif self.params['distribution'] == 'gamma':
            variance = np.clip(variance, 1e-9)
            mu = np.clip(mu, 1e-9)
            rate = (mu / variance)
            shape = mu * rate
            yhat =  np.random.gamma(shape, rate, (n_forecasts, loc.shape[0]))
        elif self.params['distribution'] == 'gumbel':
            variance = np.clip(variance, 1e-9)
            scale = np.clip(np.sqrt(6 * variance / np.pi**2), 1e-9)
            loc = mu - scale * np.euler_gamma
            yhat =  np.random.gumbel(loc, scale, (n_forecasts, loc.shape[0]))
        elif self.params['distribution'] == 'poisson':
            yhat = np.random.poisson(loc, scale, (n_forecasts, loc.shape[0]))
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
        n_forecasts = yhat_dist.shape[0]
        # Sort the forecasts in ascending order
        yhat_dist_sorted = np.sort(yhat_dist, 0)
        # Create temporary tensors
        y_cdf = np.zeros_like(y)
        yhat_cdf = np.zeros_like(y)
        yhat_prev = np.zeros_like(y)
        crps = np.zeros_like(y)
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
        state_dict = {'nodes_idx': self.nodes_idx,
                      'nodes_split_feature':self.nodes_split_feature,
                      'nodes_split_bin':self.nodes_split_bin,
                      'leaves_idx':self.leaves_idx,
                      'leaves_mu':self.leaves_mu,
                      'leaves_var':self.leaves_var,
                      'feature_importance':self.feature_importance,
                      'best_iteration':self.best_iteration,
                      'params':self.params,
                      'yhat0':self.yhat_0,
                      'bins':self.bins}
        
        with open(filename, 'wb') as handle:
            pickle.dump(state_dict, handle)   
    
    def load(self, filename, device=None):
        """
        Load a PGBM model from a file 
        
        Example::
            >> train_set = (X_train, y_train)
            >> test_set = (X_test, y_test)
            >> model = PGBM()
            >> model.load('model.pt')
        
        Args:
            filename (string): location of model file.
            device (): not applicable, only included to support convenient 
                switching between the Torch and Numba backend packages
        """
        with open(filename, 'rb') as handle:
            state_dict = pickle.load(handle)
        
        self.nodes_idx = state_dict['nodes_idx']
        self.nodes_split_feature  = state_dict['nodes_split_feature']
        self.nodes_split_bin  = state_dict['nodes_split_bin']
        self.leaves_idx  = state_dict['leaves_idx']
        self.leaves_mu  = state_dict['leaves_mu']
        self.leaves_var  = state_dict['leaves_var']
        self.feature_importance  = state_dict['feature_importance']
        self.best_iteration  = state_dict['best_iteration']
        self.params  = state_dict['params']
        self.yhat_0  = state_dict['yhat0']
        self.bins = state_dict['bins']
        
    # Calculate permutation importance of a PGBM model
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
        n_samples = X.shape[0]
        n_features = X.shape[1]
        permutation_importance_metric = np.zeros((n_features, n_permutations), dtype=np.float64)
        # Calculate base score
        yhat_base = self.predict(X)
        if y is not None:
            y = self._convert_array(y)
            base_metric = self.metric(yhat_base, y, levels)
        # Loop over permuted features
        for feature in range(n_features):
            X_permuted = np.zeros((n_permutations, n_samples, n_features), device=X.device, dtype=X.dtype)
            for permutation in range(n_permutations):
                indices = np.random.choice(n_samples, n_samples)
                X_current = X.copy()
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
                permutation_importance_metric[feature] = (yhat_base[None, :] - yhat).abs().sum(1) / yhat_base.sum() * 100                
        
        return permutation_importance_metric
    
@njit(fastmath=True, parallel=True)
def _leaf_prediction(gradient, hessian, node, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var):
    # Empirical mean
    gradient_mean = np.mean(gradient)
    hessian_mean = np.mean(hessian)
    # Empirical variance
    N = len(gradient)
    factor = 1 / (N - 1)
    gradient_variance = factor * np.sum((gradient - gradient_mean)**2)
    hessian_variance = factor * np.sum((hessian - hessian_mean)**2)
    # Empirical covariance
    covariance = factor * np.sum((gradient - gradient_mean)*(hessian - hessian_mean))
    # Mean and variance of the leaf prediction
    lambda_scaled = lampda / N
    mu = gradient_mean / ( hessian_mean + lambda_scaled) - covariance / (hessian_mean + lambda_scaled)**2 + (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3
    var = mu**2 * (gradient_variance / gradient_mean**2 + hessian_variance / (hessian_mean + lambda_scaled)**2
                    - 2 * covariance / (gradient_mean * (hessian_mean + lambda_scaled) ) )
    # Save optimal prediction and node information
    leaves_idx[estimator, leaf_idx] = node
    leaves_mu[estimator, leaf_idx] = mu             
    leaves_var[estimator, leaf_idx] = var
    # Increase leaf idx       
    leaf_idx += 1           
    
    return leaf_idx, leaves_idx, leaves_mu, leaves_var

@njit(fastmath=True)
def wrapper_split_decision(thread_idx, start_idx, stop_idx, X, gradient, hessian, n_bins, Gl_t, Hl_t):
    for i in range(X.shape[0]):
        for j in range(start_idx[thread_idx], stop_idx[thread_idx]):
            current_grad = gradient[j]
            current_hess = hessian[j]
            Xc = X[i, j]
            for k in range(n_bins):
                idx = k <  Xc
                Gl_t[thread_idx, i, k] += idx * current_grad
                Hl_t[thread_idx, i, k] += idx * current_hess

@njit(parallel=True, fastmath=True)
def _split_decision_sample_parallel(X, gradient, hessian, n_bins):
    n_threads = config.NUMBA_NUM_THREADS
    n_samples = X.shape[1]
    n_features = X.shape[0]
    Gl_t = np.zeros((n_threads, n_features, n_bins), dtype=np.float64)
    Hl_t = np.zeros((n_threads, n_features, n_bins), dtype=np.float64)
    idx = np.linspace(0, n_samples, n_threads + 1)
    start_idx = idx[:-1]
    stop_idx = idx[1:]
    
    for thread_idx in prange(n_threads):
        wrapper_split_decision(thread_idx, start_idx, stop_idx, X, gradient, hessian, n_bins, Gl_t, Hl_t)
        
    return Gl_t.sum(0), Hl_t.sum(0)
       
@njit(parallel=True, fastmath=True)
def _split_decision_feature_parallel(X, gradient, hessian, n_bins):
    n_samples = X.shape[1]
    n_features = X.shape[0]
    Gl = np.zeros((n_features, n_bins), dtype=np.float64)
    Hl = np.zeros((n_features, n_bins), dtype=np.float64)    
    
    for i in prange(n_features):
        for j in range(n_samples):
            Xc = X[i, j]
            current_grad = gradient[j]
            current_hess = hessian[j]
            for k in range(n_bins):
                flag = (k < Xc)
                Gl[i, k] += flag * current_grad
                Hl[i, k] += flag * current_hess
    
    return Gl, Hl