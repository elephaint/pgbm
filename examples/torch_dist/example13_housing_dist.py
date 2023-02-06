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
   
   Shout out to: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html 
   for the distributed training with PyTorch tutorial

"""
#%% Load packages
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pgbm.torch import PGBMDist
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#%% Objective
def mseloss_objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

# Metric
def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Training code
def run(local_rank, args):
    # Set parameters
    params = {'max_bin' :args.max_bin,
              'max_leaves': args.max_leaves,
              'n_estimators': args.n_estimators,
              'device': args.device,
              'gpu_device_id': local_rank,
              'min_split_gain':args.min_split_gain,
              'min_data_in_leaf':args.min_data_in_leaf,
              'learning_rate':args.learning_rate,
              'verbose':args.verbose,
              'early_stopping_rounds':args.early_stopping_rounds,
              'feature_fraction':args.feature_fraction,
              'bagging_fraction':args.bagging_fraction,
              'seed':args.seed,
              'lambda':args.reg_lambda,
              'derivatives':args.derivatives,
              'distribution':args.distribution }
    # Set torch device
    if args.device == 'cpu':
        torch_device = torch.device('cpu')   
    else:
        torch_device = torch.device(local_rank)
    # Set global rank
    global_rank = args.nr * args.processes + local_rank
    # Initialize processes
    dist.init_process_group(
        backend=args.backend,                                         
   		init_method='env://',                                   
    	world_size=args.size,                              
    	rank=global_rank)                                               
    
    # Load data
    X, y = fetch_california_housing(return_X_y=True)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    torchdata = lambda x: torch.from_numpy(x).float()
    X_train, y_train = torchdata(X_train), torchdata(y_train)
    X_test, y_test = torchdata(X_test), torchdata(y_test)
    X_train = X_train.chunk(args.size)[global_rank]
    y_train = y_train.chunk(args.size)[global_rank]
    X_test = X_test.chunk(args.size)[global_rank]
    y_test = y_test.chunk(args.size)[global_rank]
    train_data = (X_train, y_train)    
    # Train on set 
    model = PGBMDist(args.size, global_rank)  
    model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric, params=params)
    #% Point and probabilistic predictions. By default, 100 probabilistic estimates are created
    yhat_point = model.predict(X_test)
    yhat_dist = model.predict_dist(X_test)
    # Scoring
    y_test = y_test.to(torch_device)
    rmse = model.metric(yhat_point, y_test)
    crps = model.crps_ensemble(yhat_dist, y_test).mean()    
    # We simply take the mean of scores across processes - this is a simplification
    dist.all_reduce(rmse, op=dist.ReduceOp.SUM)
    dist.all_reduce(crps, op=dist.ReduceOp.SUM)
    rmse /= args.size
    crps /= args.size
    # Print final scores on rank 0 process.
    if local_rank == 0:
        print(f'RMSE PGBM: {rmse:.2f}')
        print(f'CRPS PGBM: {crps:.2f}')

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', 
                        help='number of nodes')
    parser.add_argument('-p', '--processes', default=1, type=int,
                        help='number of processes per node. For multi-GPU training, this should be equal to the number of GPUs per node.')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-b', '--backend', default='gloo', type=str,
                        help="backend for distributed training. Valid options: 'gloo', 'nccl', 'mpi' ")
    parser.add_argument('-d', '--device', default='cpu', type=str,
                        help="device for training. Valid options: 'cpu', 'gpu' ")
    parser.add_argument('--min_split_gain', default=0.0, type=float,
                        help="minimum gain to split a node")    
    parser.add_argument('--min_data_in_leaf', default=2, type=int,
                        help="minimum datapoints in a leaf")  
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help="learning rate for a PGBM model")      
    parser.add_argument('--reg_lambda', default=1.0, type=float,
                        help="lambda regularization parameter") 
    parser.add_argument('--max_leaves', default=32, type=int,
                        help="maximum number of leaves in a tree") 
    parser.add_argument('--max_bin', default=256, type=int,
                        help="maximum number of bins used to construct histograms for features")
    parser.add_argument('--n_estimators', default=100, type=int,
                        help="number of trees to construct")
    parser.add_argument('-v', '--verbose', default=2, type=int,
                        help="Verbose level, use < 2 to suppress iteration status.")
    parser.add_argument('--early_stopping_rounds', default=100, type=int,
                        help="Number of early stopping rounds in case a validation set is used")
    parser.add_argument('--feature_fraction', default=1, type=float,
                        help="Random subsampled fraction of features to use to construct a tree")
    parser.add_argument('--bagging_fraction', default=1, type=float,
                        help="Random subsampled fraction of samples to use to construct a tree")
    parser.add_argument('--seed', default=2147483647, type=int,
                        help="Seed to use to generate deterministic feature fraction samples")
    parser.add_argument('--derivatives', default='exact', type=str,
                        help="Whether to use exact derivatives or autograd derivatives. Valid options: 'exact', 'approx'")
    parser.add_argument('--distribution', default='normal', type=str,
                        help="Distribution to use to generate probabilistic predictions.")
    parser.add_argument('--checkpoint', default=False, type=bool,
                        help="Whether to save model state checkpoints after each iteration.")
    parser.add_argument('--MASTER_ADDR', default='127.0.0.1', type=str,
                        help="IP address of master process for distributed training.")
    parser.add_argument('--MASTER_PORT', default='29500', type=str,
                        help="Port of node of master process for distributed training.")
    args = parser.parse_args()
    # Set world size
    args.size = args.processes * args.nodes 
    # Set address
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    # Spawn process
    mp.spawn(run, nprocs=args.processes, args=(args,))