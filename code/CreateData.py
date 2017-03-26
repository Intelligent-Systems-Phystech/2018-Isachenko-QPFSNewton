import numpy as np
import scipy as sp
import scipy.linalg

def create_data(m, features, par):
    """
    Function creates design matrix according to values of the
    features structure fields
    
    Input:
    m - [1, 1] - number objects, number rows in created matrix
    features - structure with fields:
          rand_features - [1, 1] - number of random features
          ortfeat_features - [1, 1] - number of orthogonal features
          coltarget_features - [1, 1] - number of features collinearing 
                                      with target vector
          colfeat_features - [1, 1] - number of features corellating
                                      with other orthogonal features
          ortcol_features - [1, 1] - number of features orthogonal 
                                     target vector and collinearing to each other
    par - structure with fields:
          multpar - [1, 1] - parameter of multicollinearity, if multpar = 1, 
                             then all selected features are collinearing
          target - [m, 1] - target vector
    
    Output:
    X - [m, total_features] - matrix with test data set
    
    Author: Alexandr Katrutsa, 2014-2015 
    E-mail: aleksandr.katrutsa@phystech.edu
    """


    random = features['rand_features']
    ortfeat = features['ortfeat_features']
    coltarget = features['coltarget_features']
    colfeat = features['colfeat_features']
    ortcol = features['ortcol_features']
    k = par['multpar']
    target = par['target']

    total_features = random + ortfeat + coltarget + colfeat + ortcol
    assert m >= total_features, 'Not enough objects, objects must be more than features'

    # Create for every kind of features a matrix and concatenate it

    # Create random features
    mat_random = []
    if random > 0:
        mat_random = np.hstack([np.random.rand(m, random - 1), target + 0.01*np.random.randn(m, 1)])
    
    # Create orthogonal features, a linear combination of them equals target vector
    if ortfeat > 0:
        vec_1 = np.zeros([target.shape[0], 1])
        vec_1[0:len(target):2] = target[0:len(target):2]
        vec_2 = np.zeros([target.shape[0], 1])
        vec_2[1:len(target):2] = target[1:len(target):2]
        if ortfeat < 3:
            mat_ortfeat = np.hstack([vec_1, vec_2])
        else:
            mat_ort_ortfeat = null_space(vec_1.T)
            mat_ortfeat = np.hstack([vec_1, vec_2, 
                    mat_ort_ortfeat[:, np.random.choice(mat_ort_ortfeat.shape[1], ortfeat-2)]])
    else:
        mat_ortfeat = np.zeros([m, ortfeat]);

    # Create features, which are collinearing to a target vector according to k
    mat_coltarget = np.zeros([m, coltarget])
    if coltarget > 0:
        mat_ort_coltarget = null_space(target.T)
        for i in range(coltarget):
            mat_coltarget[:, i] = k*target.flatten() + (1-k)*mat_ort_coltarget[:, i]

    # Create features, which are correlated to the orthogonal features from mat_ortfeat 
    mat_colfeat = np.zeros([m, colfeat])
    if ortfeat > 1 and colfeat > 0:
        idx_first = 0
        idx_last = 0
        colfeat_per_ortfeat = (colfeat / ortfeat) * np.ones(ortfeat)
        for i in range(colfeat % ortfeat):
            colfeat_per_ortfeat[i] += 1
        colfeat_per_ortfeat = colfeat_per_ortfeat.astype('int')
        for i in range(ortfeat):
            mat_ort_ortfeat = null_space(mat_ortfeat[:, [i]].T)
            mat_col_perfeat = np.zeros([m, colfeat_per_ortfeat[i]])
            idx_last += colfeat_per_ortfeat[i]
            for j in range(colfeat_per_ortfeat[i]):
                mat_col_perfeat[:, j] = k*mat_ortfeat[:, j] + (1-k)*mat_ort_ortfeat[:, j]
            mat_colfeat[:, idx_first:idx_last] = mat_col_perfeat
            idx_first = idx_last

    # Create features, which are orthogonal to the target vector and collinearing each other
    mat_ortcol = np.zeros([m, ortcol])
    if ortcol > 0:
        mat_ort_coltarget = null_space(target.T)
        mid = mat_ort_coltarget.shape[1] / 2
        for i in range(ortcol):
           mat_ortcol[:, i]= k*mat_ort_coltarget[:, mid]  + (1-k)*mat_ort_coltarget[:, i]
    X = np.hstack([mat_ortcol, mat_ortfeat, mat_colfeat, mat_coltarget, mat_random])
    return X

def null_space(A, eps=1e-9):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = np.ones(A.shape[1])
    null_mask[:len(s)] = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)