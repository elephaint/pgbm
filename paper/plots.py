# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 08:53:20 2021

@author: ospra
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'text.usetex' : True}, style='white')
# sns.set_style("white")
#%% Load values
algorithms = ['pgbm_gpu','lightgbm_cpu','ngboost_cpu']
filename = 'experiments/01_uci_benchmark/exp1_'
df = pd.DataFrame()
for algorithm in algorithms:
    dfc = pd.read_csv(filename + algorithm + '.csv', index_col=0)
    df = pd.concat((df, dfc), axis=0)
#%% CRPS
data = df[['method', 'dataset', 'crps_test']].copy()
data = data[data.method != 'lightgbm']
data = data[data.dataset != 'higgs']
#data['dataset'] = pd.Categorical(data.dataset, categories = ['yacht', 'boston', 'energy', 'concrete', 'wine', 'kin8nm', 'power', 'naval', 'protein','msd'])
data.columns = ['Method','Dataset','CRPS']
pgbm_median = data[data.Method == 'pgbm'].groupby(['Dataset'])['CRPS'].median()
pgbm_median.name = 'crps_test_pgbm_median'
data = pd.merge(data, pgbm_median, how='left', on='Dataset')
data['NCRPS'] = data['CRPS'] / data['crps_test_pgbm_median']
colors = ['#ff7f0e','#1f77b4','#2ca02c']
sns.set_palette(sns.color_palette(colors))
datasets = ['yacht', 'boston', 'energy', 'concrete', 'wine', 'kin8nm', 'power', 'naval', 'protein','msd']
fig, axes = plt.subplots(2, 5)

for i, ax in enumerate(fig.axes):
    dataset = datasets[i]
    subset = data[data.Dataset == dataset]
    sns.boxplot(ax = ax, x = 'Dataset', y='NCRPS', data = subset, hue='Method', width=1, showfliers=False)
    ax.set_xticks([])
    ax.set_xlabel(xlabel='')
    ax.set_title(label=dataset, fontsize=24)
    if i == 0 or i == 5:
        ax.set_ylabel(ylabel = 'CRPS', fontsize=24)
    else:
        ax.set_ylabel(ylabel = '', fontsize=0)
    ax.tick_params(labelsize=24)
    # ax.set_yticklabels(ax.get_yticks(), size = 15)
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.legend_.remove()

handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc = 'lower center', ncol=2, fontsize=24)
leg.get_frame().set_linewidth(0.0)
fig.tight_layout()
#%% RMSE
data = df[['method', 'dataset', 'rmse_test']].copy()
#data = data[data.method != 'lightgbm']
data = data[data.dataset != 'higgs']
#data['dataset'] = pd.Categorical(data.dataset, categories = ['yacht', 'boston', 'energy', 'concrete', 'wine', 'kin8nm', 'power', 'naval', 'protein','msd'])
data['method'] = pd.Categorical(data.method, categories = ['pgbm', 'ngboost', 'lightgbm'])
data.columns = ['Method','Dataset','RMSE']
pgbm_median = data[data.Method == 'pgbm'].groupby(['Dataset'])['RMSE'].median()
pgbm_median.name = 'rmse_test_pgbm_median'
data = pd.merge(data, pgbm_median, how='left', on='Dataset')
data['NRMSE'] = data['RMSE'] / data['rmse_test_pgbm_median']
colors = ['#ff7f0e','#1f77b4','#2ca02c']
sns.set_palette(sns.color_palette(colors))
datasets = ['yacht', 'boston', 'energy', 'concrete', 'wine', 'kin8nm', 'power', 'naval', 'protein','msd']
fig, axes = plt.subplots(2, 5)

for i, ax in enumerate(fig.axes):
    dataset = datasets[i]
    subset = data[data.Dataset == dataset]
    sns.boxplot(ax = ax, x = 'Dataset', y='NRMSE', data = subset, hue='Method', width=1, showfliers=False)
    ax.set_xticks([])
    ax.set_xlabel(xlabel='')
    ax.set_title(label=dataset, fontsize=24)
    if i == 0 or i == 5:
        ax.set_ylabel(ylabel = 'RMSE', fontsize=24)
    else:
        ax.set_ylabel(ylabel = '', fontsize=0)
    ax.tick_params(labelsize=24)
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.legend_.remove()

handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc = 'lower center', ncol=3, fontsize=24)
leg.get_frame().set_linewidth(0.0)
fig.tight_layout()