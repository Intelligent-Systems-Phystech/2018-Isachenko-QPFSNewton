import numpy as np
import sklearn.feature_selection as sklfs
import scipy as sc
import cvxpy as cvx


class QPFS:
    def __init__(self, sim='corr'):
        if sim not in ['corr', 'info']:
            raise ValueError('Similarity measure should be "corr" or "info"')
        self.sim = sim
    
    def get_Qb(self, X, y):

        n_features = X.shape[1]

        if self.sim == 'corr':
            y_mat = y[:, np.newaxis] if len(y.shape) == 1 else y[:]
            corr = np.corrcoef(np.hstack([X, y_mat]).T)
            self.Q = np.abs(corr[:-1, :-1])
            self.b = np.abs(corr[:-1, [-1]])
        elif self.sim == 'info':
            self.Q = np.zeros([n_features, n_features])
            for j in range(n_features):
                self.Q[:, j] = sklfs.mutual_info_regression(X, X[:, j])
            self.b = sklfs.mutual_info_regression(X, y)
        
        self.lamb_min = sc.linalg.eigh(self.Q)[0][0]
        if self.lamb_min < 0:
            self.Q = self.Q - (self.lamb_min - eps) * np.eye(*self.Q.shape)
    
    def get_alpha(self, kind=0, **kwargs):
        if kind == 0:
            return self.Q.mean() / (self.Q.mean() + self.b.mean())
        elif kind == 1:
            return self.Q.mean() * len(self.b) * kwargs['k'] / self.b.mean()
        else:
            return 2 * kwargs['k'] * self.Q.sum() / self.b.sum()

    def fit(self, X, y, alpha=None):
        self.get_Qb(X, y)
        self.alpha = alpha if alpha else self.get_alpha(kind=0)
        self.solve_problem()
    
    def solve_problem(self):
        n = self.Q.shape[0]
        x = cvx.Variable(n)
        objective = cvx.Minimize(cvx.quad_form(x, self.Q) - self.alpha * self.b.T * x)
        constraints = [x >= 0, x <= 1]
        prob = cvx.Problem(objective, constraints)

        prob.solve()

        self.status = prob.status
        self.score = np.array(x.value).flatten()
        
    def __repr__(self):
        return f'QPFS(sim="{self.sim}")'


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    qpfs = QPFS()
    qpfs.fit(X, y)
    print(qpfs.score)
    
    