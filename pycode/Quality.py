import numpy as np


def RSS(pred, y):
    return np.sum((pred - y)**2)


def Complexity(act_idx):
    return np.sum(act_idx)


def Cp(RSS_p, RSS, p, m):
    return RSS_p / RSS + 2*p - m


def Bic(RSS, p, m):
    return RSS + p*np.log(m)
