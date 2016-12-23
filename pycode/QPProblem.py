import numpy as np
from cvxpy import *
from scipy.stats.stats import pearsonr


def create_opt_problem(X, y, sim, rel):
    """
    % Function generates matrix Q and vector b
    % which represent feature similarities and feature relevances
    %
    % Input:
    % X - [m, n] - design matrix
    % y - [m, 1] - target vector
    % sim - string - indicator of the way to compute feature similarities,
    % support values are 'correl' and 'mi'
    % rel - string - indicator of the way to compute feature significance,
    % support values are 'correl', 'mi' and 'signif'
    %
    % Output:
    % Q - [n ,n ] - matrix of features similarities
    % b - [n, 1] - vector of feature relevances
    %
    % Author: Alexandr Katrutsa, 2016
    % E-mail: aleksandr.katrutsa@phystech.edu
    """
    if sim == 'correl':
        Q = np.zeros([X.shape[1], X.shape[1]])
        for i in range(Q.shape[1]):
            for j in range(i, Q.shape[1]):
                xi = X[:, i] - X[:, i].mean()
                xj = X[:, j] - X[:, j].mean()
                sdi = np.sqrt(np.sum(xi**2))
                sdj = np.sqrt(np.sum(xj**2))
                Q[i, j] = xi.dot(xj) / (sdi*sdj)
                Q[j, i] = Q[i, j]
    else:
        if sim == 'mi':
            Q = np.zeros([X.shape[1], X.shape[1]])
            for i in range(Q.shape[1]):
                for j in range(i, Q.shape[1]):
                    Q[i, j] = information(X[:, i], X[:, j])
            Q = Q + Q.T - np.diag(np.diag(Q))
        lambdas = np.linalg.eig(Q)
        min_lambda = min(lambdas)
        if min_lambda < 0:
            Q = Q - min_lambda * np.eye(Q.shape[0])
    if rel == 'correl':
        b = np.zeros([X.shape[1], 1])
        for i in range(X.shape[1]):
            b[i] = np.abs(pearsonr(X[:, i], y.flatten())[0])
    if rel == 'mi':
        b = np.zeros([X.shape[1], 1])
        for i in range(X.shape[1]):
            b[i] = information(y.T, X[:, i].T)
    # if rel == 'signif':
    #    lm = fitlm(X, y)
    #    p_val = lm.Coefficients.pValue(2:end);
    #    idx_zero_coeff = find(abs(lm.Coefficients.Estimate(2:end)) < 1e-7);
    #    nan_idx = isnan(p_val);
    #    p_val(nan_idx) = ones(sum(nan_idx), 1);
    #    b = 1 - p_val ./ sum(p_val);
    #    b(idx_zero_coeff) = zeros(length(idx_zero_coeff), 1);
    #    return
    # end

    # end

    return Q, b


def solve_opt_problem(Q, b):
    """
     Function solves the quadratic optimization problem stated to select
     significance and noncollinear features

     Input:
     Q - [n, n] - matrix of features similarities
     b - [n, 1] - vector of feature relevances

     Output:
     x - [n, 1] - solution of the quadratic optimization problem

     Author: Alexandr Katrutsa, 2016
     E-mail: aleksandr.katrutsa@phystech.edu
    """

    n = Q.shape[0]
    x = Variable(n)

    objective = Minimize(quad_form(x, Q) - b.T*x)
    constraints = [x >= 0, norm(x, 1) <= 1]
    prob = Problem(objective, constraints)

    prob.solve()

    if prob.status == 'optimal':
        return np.array(x.value).flatten(), prob.value
    else:
        return None
