import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, log_loss, accuracy_score

from qpfs import QPFS

plt.rcParams['text.usetex'] = True
cmap = sns.light_palette((210, 90, 60), 10, input="husl")
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20


class RegGD:
    def __init__(self, n_hidden=2, lr=1e-3, add_bias=True, max_iter=100, tol=1e-6, verbose=True):
        self.n_hidden = n_hidden
        self.lr = lr
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._w_history = []
        self._loss_history = []
        self.msg = ''
        
    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W1)).dot(self.W2)
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.n_samples, self.n_features = X.shape
        self.W1 = 1. * np.random.randn(self.n_features, self.n_hidden)
        self.W2 = 1. * np.random.randn(self.n_hidden)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append((self.W1, self.W2))
            self._loss_history.append(mean_squared_error(y, f))
            W1_old = self.W1
            W2_old = self.W2
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W1, self.W2 = self._update_weights(X, y, f)
            if self.verbose:
                print(self.msg)
                
            W1_diff = (np.sum((W1_old - self.W1) ** 2) / np.sum(self.W1 ** 2))
            W2_diff = (np.sum((W2_old - self.W2) ** 2) / np.sum(self.W2 ** 2))
            if (W1_diff < self.tol) and (W2_diff < self.tol):
                break
        self._w_history.append((self.W1, self.W2))
        self._loss_history.append(mean_squared_error(y, f))
        
    def _get_J1(self, X):
        J1 = np.zeros((self.n_samples, self.n_features * self.n_hidden))
        dfl = self.dsigmoid(X.dot(self.W1)) * self.W2.T
        for i in range(self.n_features):
            J1[:, i * self.n_hidden: (i + 1) * self.n_hidden] = X[:, [i]] * dfl
        return J1
        
    def _get_J2(self, X):
        return self.sigmoid(X.dot(self.W1))

    def _update_weights(self, X, y, f):
        J1 = self._get_J1(X)
        J2 = self._get_J2(X)

        jac1 = J1.T.dot(f - y).reshape(self.W1.shape)
        jac2 = J2.T.dot(f - y)
        
        W1_update = -self.lr * jac1
        W2_update = -self.lr * jac2
        
        return self.W1 + W1_update, self.W2 + W2_update
    

class RegNM:
    def __init__(self, n_hidden=2, lr=1e-3, momentum = 0.9, add_bias=True, max_iter=100, tol=1e-6, verbose=True):
        self.n_hidden = n_hidden
        self.lr = lr
        self.momentum = momentum
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._w_history = []
        self._loss_history = []
        self.msg = ''
        
    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W1)).dot(self.W2)
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.n_samples, self.n_features = X.shape
        self.W1 = 1. * np.random.randn(self.n_features, self.n_hidden)
        self.W2 = 1. * np.random.randn(self.n_hidden)
        self.V1 = np.zeros_like(self.W1)
        self.V2 = np.zeros_like(self.W2)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append((self.W1, self.W2))
            self._loss_history.append(mean_squared_error(y, f))
            W1_old = self.W1
            W2_old = self.W2
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W1, self.W2 = self._update_weights(X, y, f)
            if self.verbose:
                print(self.msg)
                
            W1_diff = (np.sum((W1_old - self.W1) ** 2) / np.sum(self.W1 ** 2))
            W2_diff = (np.sum((W2_old - self.W2) ** 2) / np.sum(self.W2 ** 2))
            if (W1_diff < self.tol) and (W2_diff < self.tol):
                break
        self._w_history.append((self.W1, self.W2))
        self._loss_history.append(mean_squared_error(y, f))
        
    def _get_J1(self, X):
        J1 = np.zeros((self.n_samples, self.n_features * self.n_hidden))
        dfl = self.dsigmoid(X.dot(self.W1)) * self.W2.T
        for i in range(self.n_features):
            J1[:, i * self.n_hidden: (i + 1) * self.n_hidden] = X[:, [i]] * dfl
        return J1
        
    def _get_J2(self, X):
        return self.sigmoid(X.dot(self.W1))

    def _update_weights(self, X, y, f):
        
        J1 = self._get_J1(X)
        J2 = self._get_J2(X)

        jac1 = J1.T.dot(f - y).reshape(self.W1.shape)
        jac2 = J2.T.dot(f - y)
        
        V1_prev = self.V1.copy()
        V2_prev = self.V2.copy()
        
        self.V1 = self.momentum * self.V1 - self.lr * jac1
        self.V2 = self.momentum * self.V2 - self.lr * jac2
        
        W1_update = -self.momentum * V1_prev + (1. + self.momentum) * self.V1
        W2_update = -self.momentum * V2_prev + (1. + self.momentum) * self.V2
        
        return self.W1 + W1_update, self.W2 + W2_update
    
    
class RegADAM:
    def __init__(self, n_hidden=2, eps=1e-8, beta1=0.9, beta2=0.999, 
                 lr=1e-2, add_bias=True, max_iter=100, tol=1e-6, verbose=True):
        self.n_hidden = n_hidden
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._w_history = []
        self._loss_history = []
        self.msg = ''
        
    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W1)).dot(self.W2)
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.n_samples, self.n_features = X.shape
        self.W1 = 1. * np.random.randn(self.n_features, self.n_hidden)
        self.W2 = 1. * np.random.randn(self.n_hidden)
        self.M1 = np.zeros_like(self.W1)
        self.V1 = np.zeros_like(self.W1)
        self.M2 = np.zeros_like(self.W2)
        self.V2 = np.zeros_like(self.W2)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append((self.W1, self.W2))
            self._loss_history.append(mean_squared_error(y, f))
            W1_old = self.W1
            W2_old = self.W2
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W1, self.W2 = self._update_weights(X, y, f, i)
            if self.verbose:
                print(self.msg)
                
            W1_diff = (np.sum((W1_old - self.W1) ** 2) / np.sum(self.W1 ** 2))
            W2_diff = (np.sum((W2_old - self.W2) ** 2) / np.sum(self.W2 ** 2))
            if (W1_diff < self.tol) and (W2_diff < self.tol):
                break
        self._w_history.append((self.W1, self.W2))
        self._loss_history.append(mean_squared_error(y, f))
        
    def _get_J1(self, X):
        J1 = np.zeros((self.n_samples, self.n_features * self.n_hidden))
        dfl = self.dsigmoid(X.dot(self.W1)) * self.W2.T
        for i in range(self.n_features):
            J1[:, i * self.n_hidden: (i + 1) * self.n_hidden] = X[:, [i]] * dfl
        return J1
        
    def _get_J2(self, X):
        return self.sigmoid(X.dot(self.W1))

    def _update_weights(self, X, y, f, i):
        J1 = self._get_J1(X)
        J2 = self._get_J2(X)

        jac1 = J1.T.dot(f - y).reshape(self.W1.shape)
        jac2 = J2.T.dot(f - y)
        
        self.M1 = self.beta1 * self.M1 + (1. - self.beta1) * jac1
        mt = self.M1 / (1 - self.beta1 ** (i + 1))
        self.V1 = self.beta2 * self.V1 + (1 - self.beta2) * (jac1 ** 2)
        vt = self.V1 / (1 - self.beta2 ** (i + 1))
        W1_update = - self.lr * mt / (np.sqrt(vt) + self.eps)
        
        self.M2 = self.beta1 * self.M2 + (1. - self.beta1) * jac2
        mt = self.M2 / (1 - self.beta1 ** (i + 1))
        self.V2 = self.beta2 * self.V2 + (1 - self.beta2) * (jac2 ** 2)
        vt = self.V2 / (1 - self.beta2 ** (i + 1))
        W2_update = - self.lr * mt / (np.sqrt(vt) + self.eps)

        return self.W1 + W1_update, self.W2 + W2_update


class RegGN:
    def __init__(self, n_hidden=2, add_bias=True, qpfs=False, max_iter=100, tol=1e-6, verbose=True):
        self.n_hidden = n_hidden
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.qpfs = (QPFS(sim='corr'), QPFS(sim='corr')) if qpfs else None
        self._w_history = []
        self._loss_history = []
        self.msg = ''
        
    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W1)).dot(self.W2)
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.n_samples, self.n_features = X.shape
        self.W1 = 1. * np.random.randn(self.n_features, self.n_hidden)
        self.W2 = 1. * np.random.randn(self.n_hidden)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append((self.W1, self.W2))
            self._loss_history.append(mean_squared_error(y, f))
            W1_old = self.W1
            W2_old = self.W2
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W1, self.W2 = self._update_weights(X, y, f, i)
            if self.verbose:
                print(self.msg)
                
            W1_diff = (np.sum((W1_old - self.W1) ** 2) / np.sum(self.W1 ** 2))
            W2_diff = (np.sum((W2_old - self.W2) ** 2) / np.sum(self.W2 ** 2))
            if (W1_diff < self.tol) and (W2_diff < self.tol):
                break
        self._w_history.append((self.W1, self.W2))
        self._loss_history.append(mean_squared_error(y, f))
        
    def _get_J1(self, X):
        J1 = np.zeros((self.n_samples, self.n_features * self.n_hidden))
        dfl = self.dsigmoid(X.dot(self.W1)) * self.W2.T
        for i in range(self.n_features):
            J1[:, i * self.n_hidden: (i + 1) * self.n_hidden] = X[:, [i]] * dfl
        return J1
        
    def _get_J2(self, X):
        return self.sigmoid(X.dot(self.W1))

    def _update_weights(self, X, y, f, it=1):
        J1 = self._get_J1(X)
        J2 = self._get_J2(X)
        
        if (self.qpfs is None) or (it % 50 == 0):
            active_idxs1 = np.ones(self.n_features * self.n_hidden).astype(bool)
            active_idxs2 = np.ones(self.n_hidden).astype(bool)
        else:
            self.qpfs[0].fit(J1, f - y)
            self.qpfs[1].fit(J2, f - y)
            self.msg += '\t ' + ' '.join(['{:.2f}'.format(s) for s in self.qpfs[0].score]) + '\n'
#             self.msg += '\t b:' + '\t ' + ' '.join(['{:.3f}'.format(b) for b in self.qpfs[0].b.squeeze()]) + '\n'
            self.msg += '\t ' + ' '.join(['{:.2f}'.format(s) for s in self.qpfs[1].score]) + '\n'
#             self.msg += '\t b:' + '\t ' + ' '.join(['{:.3f}'.format(b) for b in self.qpfs[1].b.squeeze()]) + '\n'
            
            active_idxs1 = self.qpfs[0].score > np.min([1e-2, (1. / 10 * self.n_features * self.n_hidden)])
            active_idxs2 = self.qpfs[1].score > np.min([1e-2, (1. / 10 * self.n_hidden)])
            active_idxs2 = np.ones(self.n_hidden).astype(bool)
            J1 = J1[:, active_idxs1]
            J2 = J2[:, active_idxs2]

        jac1 = J1.T.dot(f - y)
        hes1 = J1.T.dot(J1)

        jac2 = J2.T.dot(f - y)
        hes2 = J2.T.dot(J2)

        cond_hes1_1 = np.linalg.cond(hes1)
        hes1 += .01 * np.eye(hes1.shape[0])
        cond_hes1_2 = np.linalg.cond(hes1)
        
        cond_hes2_1 = np.linalg.cond(hes2)
        hes2 += .01 * np.eye(hes2.shape[0])
        cond_hes2_2 = np.linalg.cond(hes2)
        
        self.msg += '\t hes1 cond: {:.3f} / {:.3f}'.format(cond_hes1_1, cond_hes1_2) + '\n'
        self.msg += '\t hes2 cond: {:.3f} / {:.3f}'.format(cond_hes2_1, cond_hes2_2) + '\n'
        
        hes1_pinv = np.linalg.pinv(hes1)
        hes2_pinv = np.linalg.pinv(hes2)
        
        W1_update = np.zeros_like(self.W1)
        W2_update = np.zeros_like(self.W2)
        
        W1_update[active_idxs1.reshape((self.n_features, self.n_hidden))] = -hes1_pinv.dot(jac1)
        W2_update[active_idxs2] = -hes2_pinv.dot(jac2)
        
        t1, t2 = self._backtracking_linesearch(X, y, 
                                               (jac1, jac2), 
                                               (W1_update, W2_update), 
                                               (active_idxs1, active_idxs2)
                                              )
        return self.W1 + t1 * W1_update, self.W2 + t2 * W2_update
    
    def _backtracking_linesearch(self, X, y, jacs, W_updates, active_idxs, alpha=0.002, beta=0.8):
        t1 = 1.
        t2 = 1.
        loss_old = mean_squared_error(y, self.predict(X, add_bias=False))
        for i in range(20):
            f = self.sigmoid(X.dot(self.W1 + t1 * W_updates[0])).dot(self.W2)
            if mean_squared_error(y, f) < loss_old + alpha * t1 * W_updates[0].ravel()[active_idxs[0]].dot(jacs[0]):
                break
            t1 = beta * t1
        for i in range(20):
            f = self.sigmoid(X.dot(self.W1)).dot(self.W2 + t2 * W_updates[1])
            if mean_squared_error(y, f) < loss_old + alpha * t2 * W_updates[1][active_idxs[1]].dot(jacs[1]):
                break
            t2 = beta * t2
        self.msg += f'\t t1: {t1:.4f}'
        self.msg += f'\t t2: {t2:.4f}'
        return t1, t2
    
    def plot_w_updates(self, figsize):
        df = np.zeros((np.prod(self.W1.shape), len(self._w_history) - 1))
        for i in range(len(self._w_history) - 1):
            w_prev = self._w_history[i][0].ravel()
            w_next = self._w_history[i + 1][0].ravel()
            df[:, i] = np.abs(w_next - w_prev)

        df = pd.DataFrame(df)

        plt.figure(figsize=figsize)
        sns.heatmap(df, cmap=cmap, vmin=0., vmax=1e-9, annot=True, cbar=False, fmt='.2f')

        plt.tight_layout()
        plt.show()


class ClassGD:
    def __init__(self, lr=1e-3, add_bias=True, max_iter=100, tol=1e-6, verbose=True):
        self.lr = lr
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._w_history = []
        self._loss_history = []
        self.msg = ''
        
    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W))
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        y = y[:, np.newaxis] if len(y.shape) == 1 else y
        self.n_samples, self.n_features = X.shape
        self.W = 1. * np.random.randn(self.n_features, 1)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append(self.W)
            self._loss_history.append(log_loss(y, f))
            W_old = self.W
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W = self._update_weights(X, y, f)
            if self.verbose:
                print(self.msg)
                
            W_diff = (np.sum((W_old - self.W) ** 2) / np.sum(self.W ** 2))
            if W_diff < self.tol:
                break
        self._w_history.append(self.W)
        self._loss_history.append(log_loss(y, f))

    def _update_weights(self, X, y, f):
        jac = X.T.dot(f - y)
        
        W_update = -self.lr * jac
        
        return self.W + W_update


class ClassNM:
    def __init__(self, lr=1e-3, momentum = 0.9, add_bias=True, max_iter=100, tol=1e-6, verbose=True):
        self.lr = lr
        self.momentum = momentum
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._w_history = []
        self._loss_history = []
        self.msg = ''

    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W))
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        y = y[:, np.newaxis] if len(y.shape) == 1 else y
        self.n_samples, self.n_features = X.shape
        self.W = 1. * np.random.randn(self.n_features, 1)
        self.V = np.zeros_like(self.W)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append(self.W)
            self._loss_history.append(log_loss(y, f))
            W_old = self.W
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W = self._update_weights(X, y, f)
            if self.verbose:
                print(self.msg)
                
            W_diff = (np.sum((W_old - self.W) ** 2) / np.sum(self.W ** 2))
            if W_diff < self.tol:
                break
        self._w_history.append(self.W)
        self._loss_history.append(log_loss(y, f))

    def _update_weights(self, X, y, f):

        jac = X.T.dot(f - y)
        V_prev = self.V.copy()
        self.V = self.momentum * self.V - self.lr * jac
        W_update = -self.momentum * V_prev + (1. + self.momentum) * self.V
        
        return self.W + W_update

    
class ClassADAM:
    def __init__(self, eps=1e-8, beta1=0.9, beta2=0.999, 
                 lr=1e-2, add_bias=True, max_iter=100, tol=1e-6, verbose=True):
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._w_history = []
        self._loss_history = []
        self.msg = ''
        
    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.W))
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s
    
    @staticmethod
    def dsigmoid(x):
        return np.exp(-x * np.sign(x)) / (1. + 2 * np.exp(-x * np.sign(x)) + np.exp(-2 * x * np.sign(x)))
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        y = y[:, np.newaxis] if len(y.shape) == 1 else y
        self.n_samples, self.n_features = X.shape
        self.W = 1. * np.random.randn(self.n_features, 1)
        self.M = np.zeros_like(self.W)
        self.V = np.zeros_like(self.W)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append(self.W)
            self._loss_history.append(log_loss(y, f))
            W_old = self.W
            self.msg = 'iter: {}, mse: {:.4f}\n'.format(i, self._loss_history[-1])
            self.W = self._update_weights(X, y, f, i)
            if self.verbose:
                print(self.msg)
                
            W_diff = (np.sum((W_old - self.W) ** 2) / np.sum(self.W ** 2))
            if W_diff < self.tol:
                break
        self._w_history.append(self.W)
        self._loss_history.append(log_loss(y, f))

    def _update_weights(self, X, y, f, i):
        jac = X.T.dot(f - y)
        
        self.M = self.beta1 * self.M + (1. - self.beta1) * jac
        mt = self.M / (1 - self.beta1 ** (i + 1))
        self.V = self.beta2 * self.V + (1 - self.beta2) * (jac ** 2)
        vt = self.V / (1 - self.beta2 ** (i + 1))
        W_update = - self.lr * mt / (np.sqrt(vt) + self.eps)

        return self.W + W_update


class ClassIRLS:
    def __init__(self, add_bias=True, qpfs=False, max_iter=100, tol=1e-6, verbose=True):
        self.add_bias = add_bias
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.qpfs = QPFS() if qpfs else None
        self._w_history = []
        self._loss_history = []
        self._acc_history = []
        self.msg = ''
    
    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        n_objects, n_features = X.shape
        self.w = 1. * np.random.randn(n_features)
        for i in range(self.max_iter):
            f = self.predict(X, add_bias=False)
            self._w_history.append(self.w)
            self._acc_history.append(accuracy_score(y, f > 0.5))
            self._loss_history.append(log_loss(y, f))
            w_old = self.w
            self.msg = 'iter: {}, acc: {:.2f}, loss: {:.4f}\n'.format(i, self._acc_history[-1], 
                                                                      self._loss_history[-1])
            self.w = self._update_weights(X, y)
            if self.verbose:
                print(self.msg)
            if (np.sum((w_old - self.w) ** 2) / np.sum(self.w ** 2)) < self.tol:
                break
        self._w_history.append(self.w)
        self._acc_history.append(accuracy_score(y, f > 0.5))
        self._loss_history.append(log_loss(y, f))

    
    def _update_weights(self, X, y):
        f = self.predict(X, add_bias=False)
        n_features = X.shape[1]
        R = np.diag(np.maximum(1e-12, f * (1. - f)))
        if self.qpfs is None:
            hes = X.T.dot(R).dot(X)
            jac = X.T.dot(f - y)
            active_idxs = np.ones(n_features).astype(bool)
        else:
            F = np.sqrt(R).dot(X)
            z = (f - y) * np.sqrt(1. / np.maximum(1e-12, f * (1. - f)))
            self.qpfs.fit(F, z)
            self.msg += '\t ' + ' '.join(['{:.3f}'.format(s) for s in self.qpfs.score]) + '\n'
            self.msg += '\t b:' + ' '.join(['{:.3f}'.format(b) for b in self.qpfs.b.squeeze()]) + '\n'
            active_idxs = self.qpfs.score > np.min([0.01, (1. / 5 * n_features)])
            F = F[:, active_idxs]
            hes = F.T.dot(F)
            jac = F.T.dot(z)
        cond_1 = np.linalg.cond(hes)
        hes += .01 * np.eye(hes.shape[0])
        jac += .01 * self.w[active_idxs]
        cond_2 = np.linalg.cond(hes)
        self.msg += '\t cond before: {:.3f}, cond after: {:.3f}'.format(cond_1, cond_2) + '\n'
        hes_pinv = np.linalg.pinv(hes)
        w_update = np.zeros_like(self.w)
        w_update[active_idxs] = -hes_pinv.dot(jac)
        t = self._backtracking_linesearch(X, y, jac, w_update, active_idxs)
        return self.w + t * w_update
    
    def _backtracking_linesearch(self, X, y, jac, w_update, active_idxs, alpha=0.002, beta=0.8):
        t = 1
        loss_old = log_loss(y, self.predict(X, add_bias=False))
        for i in range(20):
            f = self.sigmoid(X.dot(self.w + t * w_update))
            if log_loss(y, f) < loss_old + alpha * t * w_update[active_idxs].dot(jac):
                break
            t = beta * t
        self.msg += f'\t t: {t:.4f}\n'
        return t
    
    @staticmethod
    def sigmoid(x):
        s = np.zeros_like(x)
        s[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
        s[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1.)
        return s

    def predict(self, X, add_bias=True):
        if self.add_bias and add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X.dot(self.w))
    
    def plot_w_updates(self, figsize):
        df = np.zeros((len(self.w), len(self._w_history) - 1))
        for i in range(len(self._w_history) - 1):
            w_prev = self._w_history[i]
            w_next = self._w_history[i + 1]
            df[:, i] = np.abs(w_next - w_prev)

        df = pd.DataFrame(df)

        plt.figure(figsize=figsize)
        sns.heatmap(df, cmap=cmap, vmin=0., vmax=1e-9, annot=True, cbar=False, fmt='.2f')

        plt.tight_layout()
        plt.show()