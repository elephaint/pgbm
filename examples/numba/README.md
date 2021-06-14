# Examples #

This folder contains examples of PGBM. The examples illustrate the following:
* Example 1: How to train PGBM on CPU.
* Example 4: How to train PGBM using a validation loop.
* Examples 5-6: How PGBM compares to other methods such as NGBoost and LightGBM.
* Example 7: How the choice of output distribution can be optimized after training.
* Example 9: How to plot the feature importance of a learner after training.
* Example 10: How we employed PGBM to forecast Covid-19 daily hospital admissions in the Netherlands.
* Example 11: How to save and load a PGBM model. 

Note: to use the `higgs` dataset in any of the examples, download [here](https://archive.ics.uci.edu/ml/datasets/HIGGS), unpack and save `HIGGS.csv` to your local working directory.

Below is an example of a probabilistic regression task: predict housing prices for the [Boston Housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/). The code for this example can be found [here](https://github.com/elephaint/pgbm/blob/main/examples/example1_bostonhousing.py).

First, we import the necessary packages. In this simple example we will train on the CPU.
```
from pgbm_nb import PGBM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
```
Second, we define our loss function and evaluation metric. 
* The loss function should consume a torch vector of predictions `yhat` and ground truth values `y` and output the gradient and hessian with respect to `yhat` of the loss function. For more complicated loss functions, it is possible to add a `levels` variable, but this can be set to `None` in case it is not required.
* The evaluation metric should consume a torch vector of predictions `yhat` and ground truth values `y`, and output a scalar loss. For more complicated evaluation metrics, it is possible to add a `levels` variable, but this can be set to `None` in case it is not required.
```
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = np.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y):
    loss = np.sqrt(np.mean(np.square(yhat - y)))

    return loss
```
Third, we load our data:
```
X, y = load_boston(return_X_y=True)
``` 
Finally, we train our model:
```
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
train_data = (X_train, y_train)
# Train on set 
model = PGBM()  
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
#% Point and probabilistic predictions. By default, 100 probabilistic estimates are created
yhat_point = model.predict(X_test)
yhat_dist = model.predict_dist(X_test)
# Scoring
rmse = model.metric(yhat_point, y_test)
crps = model.crps_ensemble(yhat_dist, y_test).mean()    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
```
We can now plot the point and probabilistic predictions (indicated by max and min bound on the predictions):
```
plt.rcParams.update({'font.size': 22})
plt.plot(y_test, 'o', label='Actual')
plt.plot(yhat_point, 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist.max(axis=0), 'k--', label='Max bound PGBM')
plt.plot(yhat_dist.min(axis=0), 'k--', label='Min bound PGBM')
plt.legend()
```
which will give us the point forecast and probabilistic forecast:
![Boston Housing probabilistic forecast](/examples/pytorch/example01_figure.png)