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

"""
#%% Import packages
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import requests, zipfile, io
import numpy as np
from pathlib import Path
#%% Load data
def get_dataset(dataset):
    datasets = {'housing': lambda: fetch_california_housing(return_X_y=True),
                'concrete': lambda: pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'),
                'energy': lambda: pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx').iloc[:, :-2],
                'kin8nm': lambda: pd.read_csv('https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff'),
                'protein': lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv')[["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]],
                'wine': lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=";"),
                'yacht': lambda: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data', header=None, delim_whitespace=True)}
    if dataset == 'higgs':
        # Pre-download Higgs from https://archive.ics.uci.edu/ml/datasets/HIGGS, extract HIGGS.csv to pgbm/datasets/, load in Python and save as higgs.feather
        data = pd.read_feather('higgs.feather') 
    elif dataset == 'msd':
        if Path('msd.feather').is_file():
            data = pd.read_feather('msd.feather')
        else:
            print('Downloading MSD dataset...')
            data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip', header=None).iloc[:, ::-1]
            data.columns = data.columns.astype('str')
            data.to_feather('msd.feather')
    elif dataset == 'housing':
        X, y = datasets[dataset]()
        data = pd.DataFrame(np.concatenate((X, y[:, None]), axis=1))
    elif dataset == 'naval':
        file = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip', stream=True)
        z = zipfile.ZipFile(io.BytesIO(file.content))
        z.infolist()[5].filename = 'naval.txt'
        z.extract(z.infolist()[5])
        data = pd.read_csv('naval.txt', delim_whitespace=True, header=None).iloc[:, :-1].drop(columns = [8, 11])
    elif dataset == 'power':
        file = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip', stream=True)
        z = zipfile.ZipFile(io.BytesIO(file.content))
        z.infolist()[2].filename = 'power.xlsx'
        z.extract(z.infolist()[2])
        data = pd.read_excel('power.xlsx')
    else:
        data = datasets[dataset]()
    
    return data

def get_fold(dataset, data, random_state):
    if dataset == 'higgs':
        n_training = 500000
        X_train, y_train = data.iloc[:-n_training, 1:].values, data.iloc[:-n_training, 0].values
        X_test, y_test = data.iloc[-n_training:, 1:].values, data.iloc[-n_training:, 0].values    
    elif dataset == 'msd':
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        n_training = 463715
        X_train, X_test, y_train, y_test = X[:n_training], X[n_training:], y[:n_training], y[n_training:]
    else:
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state) 
    
    return X_train, X_test, y_train, y_test
       