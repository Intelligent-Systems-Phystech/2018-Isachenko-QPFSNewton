import numpy as np
from sklearn.linear_model import LinearRegression


def estimator(X_train, y_train, X_test, active_idx):
    est = LinearRegression()
    est.fit(X_train[:, active_idx], y_train)
    train_pred = est.predict(X_train[:, active_idx])
    test_pred = est.predict(X_test[:, active_idx])
    return est.coef_, train_pred, test_pred
