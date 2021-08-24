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
from pathlib import Path
import pickle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import r2_score
#%% Probabilistic Gradient Boosting Machines
class PGBM(object):
    def __init__(self):
        super(PGBM, self).__init__()
        self.cwd = Path().cwd()
    
    def _init_params(self, params):       
        self.params = {}
        param_names = ['min_split_gain', 'min_data_in_leaf', 'learning_rate', 'lambda', 'max_leaves',
                       'max_bin', 'n_estimators', 'verbose', 'early_stopping_rounds', 'feature_fraction', 'bagging_fraction', 
                       'seed', 'split_parallel', 'distribution', 'checkpoint', 'tree_correlation', 
                       'monotone_constraints', 'monotone_iterations']
        param_defaults = [0.0, 2, 0.1, 1.0, 32, 
                          256, 100, 2, 100, 1, 1, 
                          2147483647, 'feature', 'normal', False, np.log10(self.n_samples) / 100, 
                          np.zeros(self.n_features), 1]
                          
        for i, param in enumerate(param_names):
            self.params[param] = params[param] if param in params else param_defaults[i]
        
        # Make sure we bound certain parameters
        self.params['min_data_in_leaf'] = np.maximum(self.params['min_data_in_leaf'], 2)
        self.params['min_split_gain'] = np.maximum(self.params['min_split_gain'], 0.0)
        self.params['monotone_iterations'] = np.maximum(self.params['monotone_iterations'], 1)
        self.feature_fraction = np.clip(int(self.params['feature_fraction'] * self.n_features), 1, self.n_features, dtype=np.int64)
        self.bagging_fraction = np.clip(int(self.params['bagging_fraction'] * self.n_samples), 1, self.n_samples, dtype=np.int64)
        
        # Set some additional params
        assert len(self.params['monotone_constraints']) == self.n_features, "The number of items in the monotonicity constraint list should be equal to the number of features in your dataset."
        self.params['max_nodes'] = self.params['max_leaves'] - 1
        self.params['monotone_constraints'] = np.array(self.params['monotone_constraints'])
        self.any_monotone = np.any(self.params['monotone_constraints'] != 0)
        np.random.seed(self.params['seed'])            

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _create_feature_bins(X, max_bin):
        # Create array that contains the bins
        bins = np.zeros((X.shape[1], max_bin), dtype=np.float64)
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
            bins[i, len(current_bin):] = np.max(current_bin)
            
        return bins
    
    @staticmethod
    @njit(fastmath=True)
    def _create_tree(X, gradient, hessian, estimator, train_nodes, bins, nodes_idx, nodes_split_feature, nodes_split_bin, leaves_idx, leaves_mu, leaves_var, feature_importance, lampda, max_nodes, max_leaves, max_bin, feature_fraction, min_split_gain, min_data_in_leaf, split_parallel, monotone_constraints, any_monotone, monotone_iterations):
        # Set start node and start leaf index
        leaf_idx = 0
        node_idx = 0
        node_constraint_idx = 0
        node = 1
        node_constraints = np.zeros((max_nodes * 2 + 1, 3), dtype=np.float64)
        node_constraints[0, 0] = node
        node_constraints[:, 1] = -np.inf
        node_constraints[:, 2] = np.inf
        n_features = X.shape[0]
        n_samples = X.shape[1]
        if feature_fraction < 1:
            sample_features = np.random.choice(n_features, feature_fraction, replace=False)
            Xe = X[sample_features]
        else:
            sample_features = np.arange(n_features)
            Xe = X
        # Create tree
        while leaf_idx < max_leaves:
            split_node = train_nodes == node
            # Only create node if there are samples in it
            if split_node.any():
                # Choose feature subset
                X_node = Xe[:, split_node]
                gradient_node = gradient[split_node]
                hessian_node = hessian[split_node]
                # Compute split_gain
                if split_parallel == 'sample':
                    Gl, Hl, Glc = _split_decision_sample_parallel(X_node, gradient_node, hessian_node, max_bin)
                else:
                    Gl, Hl, Glc = _split_decision_feature_parallel(X_node, gradient_node, hessian_node, max_bin)
                # Compute counts of right leafs
                Grc = len(gradient_node) - Glc
                Glc = Glc >= min_data_in_leaf
                Grc = Grc >= min_data_in_leaf
                # Sum gradients and hessian
                G = np.sum(gradient_node)
                H = np.sum(hessian_node)
                # Compute split_gain, only consider split gain when enough samples in leaves.
                split_gain_tot = (Gl * Gl) / (Hl + lampda) + (G - Gl)*(G - Gl) / (H - Hl + lampda) - (G * G) / (H + lampda)
                split_gain_tot = split_gain_tot * Glc * Grc
                split_gain = np.max(split_gain_tot)              
                # Split if split_gain exceeds minimum
                if split_gain > min_split_gain:
                    argmaxsg = np.argmax(split_gain_tot)
                    split_feature_sample = argmaxsg // max_bin
                    split_bin = argmaxsg - split_feature_sample * max_bin
                    split_left = (Xe[split_feature_sample] > split_bin)
                    split_right = ~split_left * split_node
                    split_left *= split_node
                    # Check for monotone constraints if applicable
                    if any_monotone:
                        split_gain_tot_flat = split_gain_tot.flatten()
                        # Find min and max for leaf (mu) weights of current node
                        node_min = node_constraints[node_constraints[:, 0] == node, 1][0]
                        node_max = node_constraints[node_constraints[:, 0] == node, 2][0]
                        # Check if current split proposal has a monotonicity constraint
                        split_constraint = monotone_constraints[sample_features[split_feature_sample]]
                        # Perform check only if parent node has a constraint or if the current proposal is constrained. Todo: this might be a CPU check due to np.inf. Replace np.inf by: torch.tensor(float("Inf"), dtype=torch.float32, device=X.device)
                        if (node_min > -np.inf) or (node_max < np.inf) or (split_constraint != 0):
                            # We precompute the child left- and right weights and evaluate whether they satisfy the constraints. If not, we seek another split and repeat.
                            mu_left = _mu_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                            mu_right = _mu_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                            split = True
                            split_iters = 1
                            condition = split * (((mu_left < node_min) + (mu_left > node_max) + (mu_right < node_min) + (mu_right > node_max)) + ((split_constraint != 0) * (np.sign(mu_right - mu_left) != split_constraint)))
                            while condition > 0:
                                # Set gain of current split to -1, as this split is not allowed
                                split_gain_tot_flat[argmaxsg] = -1
                                # Get new split. Check if split_gain is still sufficient, because we might end up with having only constraint invalid splits (i.e. all split_gain <= 0).
                                split_gain = split_gain_tot_flat.max()
                                # Check if new proposed split is allowed, otherwise break loop
                                split = (split_gain > min_split_gain) * (split_iters < monotone_iterations)
                                if not split: break
                                # Find new split 
                                argmaxsg = split_gain_tot_flat.argmax()
                                split_feature_sample = argmaxsg // max_bin
                                split_bin = argmaxsg - split_feature_sample * max_bin
                                split_left = (Xe[split_feature_sample] > split_bin)
                                split_right = ~split_left * split_node
                                split_left *= split_node
                                # Compute new leaf weights
                                mu_left = _mu_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                                mu_right = _mu_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                                # Compute new condition
                                split_constraint = monotone_constraints[sample_features[split_feature_sample]]
                                condition = split * (((mu_left < node_min) + (mu_left > node_max) + (mu_right < node_min) + (mu_right > node_max)) + ((split_constraint != 0) * (np.sign(mu_right - mu_left) != split_constraint)))
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
                                nodes_idx[estimator, node_idx] = node
                                nodes_split_feature[estimator, node_idx] = sample_features[split_feature_sample] 
                                nodes_split_bin[estimator, node_idx] = split_bin
                                node_idx += 1
                                feature_importance[sample_features[split_feature_sample]] += split_gain * X_node.shape[1] / n_samples 
                                # Assign next node to samples if next node does not exceed max n_leaves
                                criterion = 2 * (max_nodes - node_idx + 1)
                                n_leaves_old = leaf_idx
                                if (criterion >  max_leaves - n_leaves_old):
                                    train_nodes += split_left * node
                                else:
                                    leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                                if (criterion >  max_leaves - n_leaves_old + 1):
                                    train_nodes += split_right * (node + 1)
                                else:
                                    leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                            else:
                                leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient_node, hessian_node, node, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                        else:
                            # Set left children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node
                            node_constraint_idx += 1
                            # Set right children constraints
                            node_constraints[node_constraint_idx, 0] = 2 * node + 1
                            node_constraint_idx += 1    
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
                                train_nodes += split_left * node
                            else:
                                leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                            if (criterion >  max_leaves - n_leaves_old + 1):
                                train_nodes += split_right * (node + 1)
                            else:
                                leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                    else:
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
                            train_nodes += split_left * node
                        else:
                            leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_left], hessian[split_left], node * 2, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
                        if (criterion >  max_leaves - n_leaves_old + 1):
                            train_nodes += split_right * (node + 1)
                        else:
                            leaf_idx, leaves_idx, leaves_mu, leaves_var = _leaf_prediction(gradient[split_right], hessian[split_right], node * 2 + 1, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var)
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
        # Pre-compute split decisions for X
        X_splits = np.zeros((X.shape[1], X.shape[0]), dtype=np.uint16)
        
        for j in prange(X.shape[1]):
            for i in range(X.shape[0]):
                for k in range(max_bin):
                    X_splits[j, i] += (X[i, j] > bins[j, k])
    
        return X_splits
    
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
                The function should consume a numpy vector of predictions yhat and ground truth values y and output the gradient and 
                hessian with respect to yhat of the loss function. 
            metric (function): The metric function is the function that generates the error metric. The evaluation 
                metric should consume a torch vector of predictions yhat and ground truth values y, and output a scalar loss. 
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
        X_train, y_train = train_set[0].astype(np.float64), train_set[1].squeeze().astype(np.float64)
        # Set objective & metric
        self.objective = objective
        self.metric = metric
        # Initialize predictions
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        y_train_sum = y_train.sum()
        # Pre-allocate arrays
        nodes_idx = np.zeros((self.params['n_estimators'], self.params['max_nodes']), dtype=np.int64)
        nodes_split_feature = np.zeros_like(nodes_idx)
        nodes_split_bin = np.zeros_like(nodes_idx)
        leaves_idx = np.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=np.int64)
        leaves_mu = np.zeros((self.params['n_estimators'], self.params['max_leaves']), dtype=np.float64)
        leaves_var = np.zeros_like(leaves_mu)
        # Continue training from existing model or train new model, depending on whether a model was loaded.
        if not hasattr(self, 'yhat_0'):
            self.yhat_0 = y_train_sum / self.n_samples                           
            self.best_iteration = 0
            yhat_train = self.yhat_0.repeat(self.n_samples)
            self.bins = self._create_feature_bins(X_train, self.params['max_bin'])
            self.feature_importance = np.zeros(self.n_features, dtype=np.float64)
            self.nodes_idx = nodes_idx
            self.nodes_split_feature = nodes_split_feature
            self.nodes_split_bin = nodes_split_bin
            self.leaves_idx = leaves_idx
            self.leaves_mu = leaves_mu
            self.leaves_var = leaves_var
            start_iteration = 0
            new_model = True
        else:
            yhat_train = self.predict(X_train)
            self.nodes_idx = np.concatenate((self.nodes_idx, nodes_idx))
            self.nodes_split_feature = np.concatenate((self.nodes_split_feature, nodes_split_feature))
            self.nodes_split_bin = np.concatenate((self.nodes_split_bin, nodes_split_bin))
            self.leaves_idx = np.concatenate((self.leaves_idx, leaves_idx))
            self.leaves_mu = np.concatenate((self.leaves_mu, leaves_mu))
            self.leaves_var = np.concatenate((self.leaves_var, leaves_var))
            start_iteration = self.best_iteration
            new_model = False
        # Initialize
        train_nodes = np.ones(self.bagging_fraction, dtype=np.int64)
        dist = False
        # Pre-compute split decisions for X_train
        X_train_splits = self._create_X_splits(X_train, self.bins, self.params['max_bin'])
        # Initialize validation
        validate = False
        if valid_set is not None:
            validate = True
            early_stopping = 0
            X_validate, y_validate = valid_set[0].astype(np.float64), valid_set[1].squeeze().astype(np.float64)
            if new_model:
                yhat_validate = self.yhat_0.repeat(y_validate.shape[0])
                self.best_score = np.inf
            else:
                yhat_validate = self.predict(X_validate)
                validation_metric = metric(yhat_validate, y_validate, eval_sample_weight)
                self.best_score = validation_metric
            # Pre-compute split decisions for X_validate
            X_validate_splits = self._create_X_splits(X_validate, self.bins, self.params['max_bin'])

        # Retrieve initial loss and gradient
        rng = np.random.default_rng(self.params['seed'])
        gradient, hessian = self.objective(yhat_train, y_train, sample_weight)      
        # Loop over estimators
        for estimator in range(start_iteration, self.params['n_estimators'] + start_iteration):
            if self.params['bagging_fraction'] < 1:
                # Retrieve bagging batch
                samples = rng.choice(self.n_samples, self.bagging_fraction)
                # Create tree
                self.nodes_idx, self.nodes_split_feature, self.nodes_split_bin, self.leaves_idx, self.leaves_mu, self.leaves_var, self.feature_importance = self._create_tree(X_train_splits[:, samples], gradient[samples], hessian[samples], estimator, train_nodes, self.bins, self.nodes_idx, self.nodes_split_feature, self.nodes_split_bin, self.leaves_idx, self.leaves_mu, self.leaves_var, self.feature_importance, self.params['lambda'], self.params['max_nodes'], self.params['max_leaves'], self.params['max_bin'], self.feature_fraction, self.params['min_split_gain'], self.params['min_data_in_leaf'], self.params['split_parallel'], self.params['monotone_constraints'], self.any_monotone, self.params['monotone_iterations'])
            else:
                self.nodes_idx, self.nodes_split_feature, self.nodes_split_bin, self.leaves_idx, self.leaves_mu, self.leaves_var, self.feature_importance = self._create_tree(X_train_splits, gradient, hessian, estimator, train_nodes, self.bins, self.nodes_idx, self.nodes_split_feature, self.nodes_split_bin, self.leaves_idx, self.leaves_mu, self.leaves_var, self.feature_importance, self.params['lambda'], self.params['max_nodes'], self.params['max_leaves'], self.params['max_bin'], self.feature_fraction, self.params['min_split_gain'], self.params['min_data_in_leaf'], self.params['split_parallel'], self.params['monotone_constraints'], self.any_monotone, self.params['monotone_iterations'])                
            # Get predictions for all samples
            yhat_train, _ = self._predict_tree(X_train_splits, yhat_train, yhat_train*0., estimator, self.nodes_idx[estimator], self.nodes_split_feature[estimator], self.nodes_split_bin[estimator], self.leaves_idx[estimator], self.leaves_mu[estimator], self.leaves_var[estimator], self.params['learning_rate'], self.params['tree_correlation'], dist)
            # Compute new gradient and hessian
            gradient, hessian = self.objective(yhat_train, y_train, sample_weight)
            # Compute metric
            train_metric = metric(yhat_train, y_train, sample_weight)
            # Reset train nodes
            train_nodes.fill(1)
            # Validation statistics
            if validate:
                yhat_validate, _ =  self._predict_tree(X_validate_splits, yhat_validate, yhat_validate*0., estimator, self.nodes_idx[estimator],  self.nodes_split_feature[estimator], self.nodes_split_bin[estimator], self.leaves_idx[estimator], self.leaves_mu[estimator], self.leaves_var[estimator], self.params['learning_rate'], self.params['tree_correlation'], dist)
                validation_metric = metric(yhat_validate, y_validate, eval_sample_weight)
                if (self.params['verbose'] > 1):
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
                if (self.params['verbose'] > 1):
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
            parallel (boolean): not applicable. Only in place to support easy switching between 
                Torch and Numba backend.
        """
        X = X.astype(np.float64)
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
            parallel (boolean): not applicable. Only in place to support easy switching between 
                Torch and Numba backend.
            output_sample_statistics (boolean): whether to also output the learned sample mean and variance. If True,
                the function will return a tuple (forecasts, mu, variance) with the latter arrays containing the learned
                mean and variance per sample that can be used to parameterize a distribution.
        """
        X = X.astype(np.float64)
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
        rng = np.random.default_rng(self.params['seed'])
        if self.params['distribution'] == 'normal':
            loc = mu
            variance = np.clip(variance, 1e-9, np.max(variance))
            scale = np.sqrt(variance)
            yhat = rng.normal(loc, scale, (n_forecasts, mu.shape[0]))
        elif self.params['distribution'] == 'studentt':
            v = 3
            loc = mu
            variance = np.clip(variance, 1e-9, np.max(variance))
            factor = v / (v - 2)
            yhat = rng.standard_t(v, (n_forecasts, mu.shape[0])) * np.sqrt(variance / factor) + loc
        elif self.params['distribution'] == 'laplace':
            loc = mu
            scale = np.sqrt(0.5 * variance)
            scale = np.clip(scale, 1e-9, np.max(scale))
            yhat =  rng.laplace(loc, scale, (n_forecasts, mu.shape[0]))
        elif self.params['distribution'] == 'logistic':
            loc = mu
            scale = np.sqrt((3 * variance) / np.pi**2)
            scale = np.clip(scale, 1e-9, np.max(scale))
            yhat =  rng.logistic(loc, scale, (n_forecasts, mu.shape[0]))
        elif self.params['distribution'] == 'gamma':
            variance = np.clip(variance, 1e-9, np.max(variance))
            mu_adj = np.clip(mu, 1e-9, np.max(mu))
            shape = mu_adj**2 / variance
            scale = mu_adj / shape
            yhat =  rng.gamma(shape, scale, (n_forecasts, mu_adj.shape[0]))
        elif self.params['distribution'] == 'gumbel':
            variance = np.clip(variance, 1e-9, np.max(variance))
            scale = np.sqrt(6 * variance / np.pi**2)
            scale = np.clip(scale, 1e-9, np.max(scale))
            loc = mu - scale * np.euler_gamma
            yhat = rng.gumbel(loc, scale, (n_forecasts, mu.shape[0]))
        elif self.params['distribution'] == 'poisson':
            yhat = rng.poisson(mu, (n_forecasts, mu.shape[0]))
        else:
            print('Distribution not (yet) supported')
        
        if output_sample_statistics:
            return (yhat, mu, variance)
        else:
            return yhat
    
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def crps_ensemble(yhat_dist, y):
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
        n_samples = yhat_dist.shape[1]
        crps = np.zeros_like(y)
        # Loop over the samples
        for sample in prange(n_samples):
            # Sort the forecasts in ascending order
            yhat_dist_sorted = np.sort(yhat_dist[:, sample])
            y_cdf = 0.0
            yhat_cdf = 0.0
            yhats_prev = 0.0
            ys = y[sample]
            # Loop over the forecasts per sample
            for yhats in yhat_dist_sorted:
                flag = (y_cdf == 0) * (ys < yhats) * 1.
                crps[sample] += flag * ( ((ys - yhats_prev) * yhat_cdf ** 2) + ((yhats - ys) * (yhat_cdf - 1) ** 2) )
                y_cdf += flag
                crps[sample] += (1 - flag) * ((yhats - yhats_prev) * (yhat_cdf - y_cdf) ** 2)
                yhat_cdf += 1 / n_forecasts
                yhats_prev = yhats
            
            # In case y_cdf == 0 after the loop
            flag = (y_cdf == 0) * 1.
            crps[sample] += flag * (ys - yhats)
        
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
        state_dict = {'nodes_idx': self.nodes_idx[:self.best_iteration],
                      'nodes_split_feature':self.nodes_split_feature[:self.best_iteration],
                      'nodes_split_bin':self.nodes_split_bin[:self.best_iteration],
                      'leaves_idx':self.leaves_idx[:self.best_iteration],
                      'leaves_mu':self.leaves_mu[:self.best_iteration],
                      'leaves_var':self.leaves_var[:self.best_iteration],
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
    def permutation_importance(self, X, y=None, n_permutations=10):
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
        """
        X = X.astype(np.float64)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        permutation_importance_metric = np.zeros((n_features, n_permutations), dtype=np.float64)
        # Calculate base score
        yhat_base = self.predict(X)
        if y is not None:
            y = y.astype(np.float64)
            base_metric = self.metric(yhat_base, y)
        # Loop over permuted features
        for feature in range(n_features):
            X_permuted = _permute_X(X, feature, n_permutations)           
            yhat = self.predict(X_permuted)
            yhat = yhat.reshape(n_permutations, n_samples)
            if y is not None:
                for permutation in range(n_permutations):
                    permuted_metric = self.metric(yhat[permutation], y)
                    permutation_importance_metric[feature, permutation] = ((permuted_metric / base_metric) - 1) * 100
            else:
                permutation_importance_metric[feature] = np.sum(np.abs(yhat_base[None, :] - yhat), axis=1) / np.sum(yhat_base) * 100                
        
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
        X, y = X.astype(np.float64), y.astype(np.float64)
        # List of distributions and tree correlations
        if distributions == None:
            distributions = ['normal', 'studentt', 'laplace', 'logistic', 
                             'gamma', 'gumbel', 'poisson']
        if np.all(tree_correlations == None):
            tree_correlations = np.arange(start=0, stop=0.2, step=0.01, dtype=np.float64)
               
        # Loop over choices
        crps_best = np.inf
        distribution_best = self.params['distribution']
        tree_correlation_best = self.params['tree_correlation']
        for distribution in distributions:
            for tree_correlation in tree_correlations:
                self.params['distribution'] = distribution
                self.params['tree_correlation'] = tree_correlation
                yhat_dist = self.predict_dist(X)
                crps = np.mean(self.crps_ensemble(yhat_dist, y))
                if self.params['verbose'] > 1:
                    print(f'CRPS: {crps:.2f} (Distribution: {distribution}, Tree correlation: {tree_correlation:.3f})')     
                if crps < crps_best:
                    crps_best = crps
                    distribution_best = distribution
                    tree_correlation_best = tree_correlation
        
        # Set to best values
        if self.params['verbose'] > 1:
            print(f'Lowest CRPS: {crps_best:.4f} (Distribution: {distribution_best}, Tree correlation: {tree_correlation_best:.3f})')  
        self.params['distribution'] = distribution_best
        self.params['tree_correlation'] = tree_correlation_best
        
        return (distribution_best, tree_correlation_best)
    
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
    epsilon = 1.0e-6
    mu = gradient_mean / ( hessian_mean + lambda_scaled) - covariance / (hessian_mean + lambda_scaled)**2 + (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3
    var = mu**2 * (gradient_variance / (gradient_mean + epsilon)**2 + hessian_variance / (hessian_mean + lambda_scaled)**2
                    - 2 * covariance / (gradient_mean * (hessian_mean + lambda_scaled) + epsilon) )
    # Save optimal prediction and node information
    leaves_idx[estimator, leaf_idx] = node
    leaves_mu[estimator, leaf_idx] = mu             
    leaves_var[estimator, leaf_idx] = var
    # Increase leaf idx       
    leaf_idx += 1           
    
    return leaf_idx, leaves_idx, leaves_mu, leaves_var

@njit(fastmath=True, parallel=True)
def _mu_prediction(gradient, hessian, node, estimator, lampda, leaf_idx, leaves_idx, leaves_mu, leaves_var):
    # Empirical mean
    gradient_mean = np.mean(gradient)
    hessian_mean = np.mean(hessian)
    # Empirical variance
    N = len(gradient)
    factor = 1 / (N - 1)
    hessian_variance = factor * np.sum((hessian - hessian_mean)**2)
    # Empirical covariance
    covariance = factor * np.sum((gradient - gradient_mean)*(hessian - hessian_mean))
    # Mean and variance of the leaf prediction
    lambda_scaled = lampda / N
    mu = gradient_mean / ( hessian_mean + lambda_scaled) - covariance / (hessian_mean + lambda_scaled)**2 + (hessian_variance * gradient_mean) / (hessian_mean + lambda_scaled)**3     
    
    return mu

@njit(fastmath=True)
def wrapper_split_decision(thread_idx, start_idx, stop_idx, X, gradient, hessian, n_bins, Gl_t, Hl_t, Glc_t):
    for i in range(X.shape[0]):
        for j in range(start_idx[thread_idx], stop_idx[thread_idx]):
            current_grad = gradient[j]
            current_hess = hessian[j]
            Xc = X[i, j]
            k = 0
            flag = (k <  Xc)
            while flag:
                Gl_t[thread_idx, i, k] += current_grad
                Hl_t[thread_idx, i, k] += current_hess
                Glc_t[thread_idx, i, k] += 1
                k += 1
                flag = (k <  Xc) 

@njit(parallel=True, fastmath=True)
def _split_decision_sample_parallel(X, gradient, hessian, n_bins):
    n_threads = config.NUMBA_NUM_THREADS
    n_samples = X.shape[1]
    n_features = X.shape[0]
    Gl_t = np.zeros((n_threads, n_features, n_bins), dtype=np.float64)
    Hl_t = np.zeros((n_threads, n_features, n_bins), dtype=np.float64)
    Glc_t = np.zeros((n_threads, n_features, n_bins), dtype=np.float64)
    idx = np.linspace(0, n_samples, n_threads + 1)
    start_idx = idx[:-1]
    stop_idx = idx[1:]
    
    for thread_idx in prange(n_threads):
        wrapper_split_decision(thread_idx, start_idx, stop_idx, X, gradient, hessian, n_bins, Gl_t, Hl_t, Glc_t)
        
    return Gl_t.sum(0), Hl_t.sum(0), Glc_t.sum(0)
       
@njit(parallel=True, fastmath=True)
def _split_decision_feature_parallel(X, gradient, hessian, n_bins):
    n_samples = X.shape[1]
    n_features = X.shape[0]
    Gl = np.zeros((n_features, n_bins), dtype=np.float64)
    Hl = np.zeros((n_features, n_bins), dtype=np.float64)    
    Glc = np.zeros((n_features, n_bins), dtype=np.float64)
    
    for i in prange(n_features):
        for j in range(n_samples):
            Xc = X[i, j]
            current_grad = gradient[j]
            current_hess = hessian[j]
            k = 0
            flag = (k <  Xc)
            while flag:
                Gl[i, k] += current_grad
                Hl[i, k] += current_hess
                Glc[i, k] += 1
                k += 1
                flag = (k <  Xc) 
    
    return Gl, Hl, Glc

@njit(parallel=True, fastmath=True)
def _permute_X(X, feature, n_permutations):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    X_permuted = np.zeros((n_permutations, n_samples, n_features), dtype=X.dtype)
    for permutation in prange(n_permutations):
        indices = np.random.choice(n_samples, n_samples)
        X_current = X.copy()
        X_current[:, feature] = X_current[indices, feature]
        X_permuted[permutation] = X_current
    
    return X_permuted.reshape(n_permutations * n_samples, n_features)

class PGBMRegressor(BaseEstimator):
    """
    Probabilistic Gradient Boosting Machines (PGBM) Regressor.
    
    PGBMRegressor fits a Gradient Boosting Machine regression model and returns
    point and probabilistic predictions.
    
    This class uses Numba as backend.
    
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
            
            def objective(y, yhat, sample_weight=None):
                [......]
                
                return gradient, hessian
            
            in which gradient and hessian are of the same shape as y.

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
        
    split_parallel : str, default='feature'
        Choose from 'feature' or 'sample'. This parameter determines whether to
        parallelize the split decision computation across the sample dimension 
        or across the feature dimension. Typically, for smaller datasets with 
        few features 'feature' is the fastest, whereas for larger datasets and/or 
        datasets with many (e.g. > 50) features, sample will provide better 
        results.
     
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
    >>> from pgbm_nb import PGBMRegressor
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
                 reg_lambda=1.0, random_state=2147483647, split_parallel='feature',
                 distribution='normal', checkpoint=False, tree_correlation=None, monotone_iterations=1, 
                 verbose=2):
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
        self.split_parallel = split_parallel
        self.distribution = distribution
        self.checkpoint = checkpoint
        self.tree_correlation = tree_correlation
        self.monotone_iterations = monotone_iterations
        self.verbose = verbose
        
    def fit(self, X, y, eval_set=None, sample_weight=None, eval_sample_weight=None,
            early_stopping_rounds=None, monotone_constraints=None):
        # Set estimator type
        self._estimator_type = "regressor"
        # Check that X and y have correct shape and convert to float64
        X, y = check_X_y(X, y)
        X, y = X.astype(np.float64), y.astype(np.float64)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        if self.n_samples_ < self.min_data_in_leaf:
            raise ValueError(f"n_samples = {self.n_samples_}, should be at least {self.min_data_in_leaf}.")
        # Check eval set
        if eval_set is not None:
            X_valid, y_valid = eval_set[0], eval_set[1]
            X_valid, y_valid = check_X_y(X_valid, y_valid)
            X_valid, y_valid = X_valid.astype(np.float64), y_valid.astype(np.float64)

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
                  'distribution':self.distribution,
                  'monotone_iterations': self.monotone_iterations,
                  'checkpoint': self.checkpoint,
                  'split_parallel':self.split_parallel}
        if self.tree_correlation is not None: 
            params['tree_correlation'] = self.tree_correlation
        if monotone_constraints is not None:
            params['monotone_constraints'] = monotone_constraints

        # Set objective and metric
        if (self.objective == 'mse'):
            self._objective = self._mseloss_objective
        else:
            self._objective = self.objective
        if (self.metric == 'rmse'):
            self._metric = self._rmseloss_metric
        else:
            self._metric = self.metric    
        # Check sample weight shape
        if sample_weight is not None: 
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if len(sample_weight) != len(y):
                raise ValueError('Length of sample_weight does not equal length of X and y')
            sample_weight = np.array(sample_weight).astype(np.float64)
        if eval_sample_weight is not None: 
            eval_sample_weight = check_array(eval_sample_weight, ensure_2d=False)
            if len(eval_sample_weight) != len(y_valid):
                raise ValueError('Length of eval_sample_weight does not equal length of X_valid and y_valid')
            eval_sample_weight = np.array(eval_sample_weight).astype(np.float64)
        # Train model
        self.learner_ = PGBM()
        self.learner_.train(train_set=(X, y), valid_set=eval_set, params=params, objective=self._objective, 
                         metric=self._metric, sample_weight=sample_weight, 
                         eval_sample_weight=eval_sample_weight)
        
        return self

    def predict(self, X, parallel=True):
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(np.float64)
        
        return self.learner_.predict(X, parallel)
    
    def score(self, X, y, sample_weight=None, parallel=True):
        # Checks
        X, y = check_X_y(X, y)
        X, y = X.astype(np.float64), y.astype(np.float64)
        # Make prediction
        yhat = self.predict(X, parallel)
        
        # Check sample weight shape
        if sample_weight is not None: 
            sample_weight = check_array(sample_weight, ensure_2d=False)
            if len(sample_weight) != len(y):
                raise ValueError('Length of sample_weight does not equal length of X and y')
            sample_weight = np.array(sample_weight).astype(np.float64)
                
        # Score prediction with r2
        score = r2_score(y, yhat, sample_weight)
        
        return score
    
    def predict_dist(self, X, n_forecasts=100, parallel=True, output_sample_statistics=False):
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(np.float64)
        
        return self.learner_.predict_dist(X, n_forecasts, parallel, output_sample_statistics)
        
    def _mseloss_objective(self, yhat, y, sample_weight=None):
        gradient = (yhat - y)
        hessian = np.ones_like(yhat)
        
        if sample_weight is not None:
            gradient *= sample_weight
            hessian *= sample_weight   
    
        return gradient, hessian
    
    def _rmseloss_metric(self, yhat, y, sample_weight=None):
        error = (yhat - y)
        if sample_weight is not None:
            error *= sample_weight
        
        loss = np.sqrt(np.mean(np.square(error)))
    
        return loss