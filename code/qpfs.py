import numpy as np
import sklearn.feature_selection as sklfs
import scipy as sc
import cvxpy as cvx

np.random.seed(42)


class QPFS:
    def __init__(self, sim='corr'):
        if sim not in ['corr', 'info', 'symuncert']:
            raise ValueError('Similarity measure should be "corr" or "info"')
        self.sim = sim
    
    def get_Qb(self, X, y, eps=1e-12):

        n_features = X.shape[1]
        self.Q = np.ones([n_features, n_features])
        self.b = np.zeros([n_features, 1])
        
        if self.sim == 'corr':
            y_mat = y[:, np.newaxis] if len(y.shape) == 1 else y[:]
            idxs_nz = np.where(np.sum(X ** 2, axis = 0) != 0)[0]
            if len(idxs_nz) > 0:
                corr = np.corrcoef(np.hstack([X[:, idxs_nz], y_mat]).T)
                for i, idx in enumerate(idxs_nz):
                    self.Q[idx, idxs_nz] = np.abs(corr[i, :-1])
                self.b[idxs_nz] = np.abs(corr[:-1, [-1]])
        elif self.sim == 'info':
            for j in range(n_features):
                self.Q[:, j] = sklfs.mutual_info_regression(X, X[:, j])
            self.b = sklfs.mutual_info_regression(X, y)[:, np.newaxis]
        elif self.sim == 'symuncert':
            X_ent = np.zeros([n_features + 1])
            for j in range(n_features):
                X_ent[j] = sklfs.mutual_info_regression(X[:, [j]], X[:, j])
            X_ent[j + 1] = sklfs.mutual_info_regression(y[:, np.newaxis], y)
            for j in range(n_features):
                self.Q[:, j] = sklfs.mutual_info_regression(X, X[:, j], n_neighbors=5) * 2 / (X_ent[:-1] + X_ent[j] + eps)
            self.b = sklfs.mutual_info_regression(X, y, n_neighbors=5) * 2 / (X_ent[-1] + X_ent[:-1] + eps)
            self.b = self.b[:, np.newaxis]
        self.Q = np.nan_to_num(self.Q)
        self.b = np.nan_to_num(self.b)
        
        self.lamb_min = sc.linalg.eigh(self.Q)[0][0]
        if self.lamb_min < 0:
            self.Q = self.Q - (self.lamb_min - eps) * np.eye(*self.Q.shape)
    
    def get_alpha(self):
        return self.Q.mean() / (self.Q.mean() + self.b.mean())

    def fit(self, X, y, alpha=None):
        self.get_Qb(X, y)
        self.alpha = alpha if alpha else self.get_alpha()
        self.solve_problem()
    
    def solve_problem(self):
        n = self.Q.shape[0]
        x = cvx.Variable(n)
        c = np.ones((n, 1))
        objective = cvx.Minimize(0.5 * (1 - self.alpha) * cvx.quad_form(x, self.Q) - 
                                 self.alpha * self.b.T * x)
        constraints = [x >= 0, c.T * x <= 1]
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
    
    