import numpy as np
from sklearn.metrics import mean_squared_error


class GradientDescent:
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
    

class NesterovMomentum:
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
    
    
class ADAM:
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