import tensorflow as tf
import numpy as np
import scipy.linalg as la

def var_single(x):
    '''find variance matrix '''
    v = np.cov(x)
    r = np.matrix('v,1;1,v')
    return r

def CCA(X,Y):
    '''
    Canonical Correlation Analysis
    Input: observation matrix (X,Y), one data point perl column
    Output: basis in (X,Y) space, correlation
    Example: X = np.vstack() Y=np.vstack() rx, ry = CCA(X, Y)
    '''
    # find variance and covariance matrix
    if len(X) == 1:
        cov_xx = var_single(X)
    else:
        cov_xx = np.cov(X)
    if len(Y) == 1:
        cov_yy = var_single(Y)
    else:
        cov_yy = np.cov(Y)
    
    n = len(X)

    cov_xy = np.cov(X, Y)[:n, :n]
    cov_yx = np.cov(Y, X)[:n, :n] # cov_yx = np.transpose(cov_xy)

    # eigenvector & eigenvalue
    cov_xx_evalue, cov_xx_evector = la.eig(cov_xx) #
    cov_xx_isqrt = dot(dot(cov_xx_evector, np.diag(1/np.sqrt(cov_xx_evalue))), np.transpose(cov_xx_evector))

    cov_yy_evalue, cov_yy_evector = la.eig(cov_yy)
    cov_yy_isqrt = dot(dot(cov_yy_evector, np.diag(1/np.sqrt(cov_yy_evalue))), np.transpose(cov_yy_evector))

    #Xmat Ymat
    Xmat = dot(dot(dot(dot(cov_xx_isqrt, cov_xy), la.inv(cov_yy)), cov_yx), cov_xx_isqrt)
    Ymat = dot(dot(dot(dot(cov_yy_isqrt, cov_yx), la.inv(cov_xx)), cov_xy), cov_yy_isqrt)

    rx = la.eig(Xmat)
    ry = la.eig(Ymat)

    return rx, ry

def LinearHSIC(XX,YY):
    '''
    Linear Hilbert-Schmidt Independence Criterion
    '''
    # trace matrix tr()
    LHSIC = np.trace(np.matmul(XX, YY))
    return LHSIC

def CKA(X,Y):
    '''
    Centered Kernel Analysis ||Y^T X||^2_F/ (||X^T X||_F ||Y^T Y||_F)
    Input: Observation matrix X, Observation matrix Y 
    Output: orthonormal basis in X space, orthonormal basis in Y space, correlation
    Example: 
    '''
    
    # find variance and covariance matrix 
    if len(X) == 1:
        cov_xx = var_single(X)
    else:
        cov_xx = np.cov(X)
    if len(Y) == 1:
        cov_yy = var_single(Y)
    else:
        cov_yy = np.cov(Y)

    p = len(X) # p<=q
    q = len(Y)
    
    cov_xy = np.cov(X, Y)[:p, :q]
    cov_yx = np.transpose(cov_xy)

    # left-singular eigenvector & squared singular eigevalue
    cov_xx_evalue, cov_xx_evector = la.eig(cov_xx)
    cov_xx_isqrt = dot(dot(cov_xx_evector, np.diag(1/np.sqrt(cov_xx_evalue))), np.transpose(cov_xx_evector))

    cov_yy_evalue, cov_yy_evector = la.eig(cov_yy)
    cov_yy_isqrt = dot(dot(cov_yy_evector, np.diag(1/np.sqrt(cov_yy_evalue))), np.transpose(cov_yy_evector))
    
    # linear CKA
    LHSIC = LinearHSIC(cov_xx, cov_yy)
    LHSIC_X = np.sqrt(LinearHSIC(cov_xx, cov_xx))
    LHSIC_Y = np.sqrt(LinearHSIC(cov_yy, cov_yy))
    CKA = LHSIC/np.matmul(LHSIC_X, LHSIC_Y)
    
    # orthonormal basis
    Xmat = dot(dot(dot(dot(cov_xx_isqrt, cov_xy), la.inv(cov_yy)), cov_yx), cov_xx_isqrt)
    Ymat = dot(dot(dot(dot(cov_yy_isqrt, cov_yx), la.inv(cov_xx)), cov_xy), cov_yy_isqrt)    
    
    rx = la.eig(Xmat)
    ry = la.eig(Ymat)
    return rx, ry
