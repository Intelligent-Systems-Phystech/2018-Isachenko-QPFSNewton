import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_breast():
    """
    scikit-learn
    """
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def load_cardiotocography():
    """
    https://archive.ics.uci.edu/ml/datasets/Cardiotocography
    """
    data = pd.read_excel('./../data/CTG.xls', sheetname='Data', header=1)
    columns = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS',
                'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
                'Mode', 'Mean', 'Median', 'Variance',
                'Tendency', 'NSP']
    data = data[columns]
    X, y = data.iloc[:, :-1].as_matrix(), (data['NSP'] > 1.5).astype(float).as_matrix()
    X = (X - X.mean()) / (X.std(axis=0))
    return X, y


def load_climate():
    """
    https://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes
    """
    with open('./../data/pop_failures.dat') as f:
        columns = f.readline().split()
        data = []
        while True:
            line = f.readline()
            if line == '':
                break
            data.append(list(map(float, line.split())))
    data = np.array(data)
    X, y = data[:, 2:-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y