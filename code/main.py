import sys
import numpy as np
import scipy as sc

from CreateData import *
from QPProblem import *
from Estimator import *
from Quality import *
# from criteria import *
# from alg import *
# from mi import *
# Parameters for data sets generation
np.random.seed(0)

# Number of samples
objects = 200
# Number of random features
features = dict()
param = dict()
features['rand_features'] = 50
# Number of orthogonal features
features['ortfeat_features'] = 0
# Number of features collinearing with target vector
features['coltarget_features'] = 0
# Number of features corellating with other orthogonal features
features['colfeat_features'] = 0
# Number of features orthogonal to target vector and collinearing to each other
features['ortcol_features'] = 0
# A parameter of multicollinearity
param['multpar'] = 0.8
# A set of the considered feature selection methods
alg = ['Lasso', 'LARS', 'Stepwise', 'ElasticNet', 'Ridge', 'Genetic']
# A set of the considered criteria
crit = ['complexity', 'Cp', 'RSS', 'CondNumber', 'Vif', 'bic']
# Number of the iteration in AlgCrit function
param['iter'] = 1
param['crit'] = crit
# A limit error
param['s_0'] = 0.5
param['threshold'] = 10e-10  # to shrink the small coefficients in w*
param['data'] = 'real'  # or 'real' 'artificial'
# Parameters of the real data set
param['real_data_filename'] = 'BP50GATEST.mat'
param['real_data_X'] = 'bp50_s1d_ll_a'
param['real_data_y'] = 'bp50_y1_ll_a'
# Parameters for genetic algorithm
param['Genetic'] = dict()
param['Genetic']['nGenerations'] = 10
param['Genetic']['nIndividuals'] = 20
# Generate the target vector
param['target'] = np.random.randint(1.5*objects, size=(objects, 1))
X = create_data(objects, features, param)
y = param['target']
# Optional normalization
X = X / np.linalg.norm(X, axis=0)
y = y / np.linalg.norm(y)
# Split test and train set
test_set_ratio = 0.7
X_train = X[:int(test_set_ratio*X.shape[0]), :]
y_train = y[:int(test_set_ratio*X.shape[0])]
X_test = X[int(test_set_ratio*X.shape[0]):, :]
y_test = y[int(test_set_ratio*X.shape[0]):]

# Create and solve quadratic optimization problem
# String indications for way to compute similarities and relevances
sim = 'correl'
rel = 'correl'
[Q, b] = create_opt_problem(X_train, y_train, sim, rel)

print np.sum(np.isnan(Q))
z, val = solve_opt_problem(Q, b)

threshold = sorted(z)
rss = np.zeros(len(threshold))
rss_test = np.zeros(len(threshold))
stability = np.zeros(len(threshold))
vif = np.zeros(len(threshold))
complexity = np.zeros(len(threshold))
bic = np.zeros(len(threshold))
cp = np.zeros(len(threshold))
A = np.zeros([len(threshold), X.shape[1]])
for i in range(len(threshold)):
    active_idx = z >= threshold[i]
    A[i, :] = active_idx
    if sum(active_idx) > 0:
        w, train_pred, test_pred = estimator(X_train, y_train, X_test,
                                             active_idx)
        rss[i] = RSS(train_pred, y_train)
        rss_test[i] = RSS(test_pred, y_test)
        stability[i] = np.linalg.cond(
                X_train[:, active_idx].T.dot(X_train[:, active_idx]))
#       vif(i) = Vif(X_train(:, active_idx))
        complexity[i] = Complexity(active_idx)
        bic[i] = Bic(rss[i], complexity[i], X_train.shape[0])
        cp[i] = Cp(rss[i], rss[0], complexity[i], X_train.shape[0])

idx_min = np.argmin(rss_test)
active_idx_opt = A[idx_min, :]
print active_idx_opt
