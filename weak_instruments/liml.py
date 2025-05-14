import numpy as np
from scipy.stats import norm
#from scipy.stats import t
import logging
from numpy.typing import NDArray
from scipy.linalg import eigvals


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)



class LIMLResult:
    def __init__(self, betas, se_list, tstat_list, pval_list, ci_list):
        self.betas = betas
        self.se_list = se_list
        self.tstat_list = tstat_list
        self.pval_list = pval_list
        self.ci_list = ci_list

    def __getitem__(self, key: str):
        if key == 'betas':
            return self.betas
        elif key == 'se_list':
            return self.se_list
        elif key == 'tstat_list':
            return self.tstat_list
        elif key == 'pval_list':
            return self.pval_list
        elif key == 'ci_list':
            return self.ci_list
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'betas', 'se_list', 'tstat_list', 'pval_list', or 'ci_list'.")

    def __repr__(self):
        return f"LIMLResult(betas={self.betas}, se_list={self.se_list}, tstat_list={self.tstat_list}, pval_list={self.pval_list}, ci_list={self.ci_list})"
    

def LIML(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, G: NDArray[np.float64] | None = None, talk: bool = False, colnames=None) -> LIMLResult:
    N = Y.shape[0]

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if G is not None:
        if G.ndim == 1:
            G = G.reshape(-1, 1)
        X = np.hstack((X, G))
        Z = np.hstack((Z, G))

    ones = np.ones((N,1))
    X = np.hstack((ones, X))
    Z = np.hstack((ones, Z))

    if Y.ndim == 1:
        Y = Y.reshape(-1,1)

    YX = np.hstack([Y, X])

    # Adjust logging level based on the `talk` parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    #Set up the Projection Matrix
    U = Z @ np.linalg.inv(Z.T @ Z)
    P = Z.T

    #Find the LIML eigenvalue:
    A = YX.T @ YX
    B = YX.T @ (np.eye(N) - U@P) @ YX 

    eigs = eigvals(A,B)
    eigs = np.real(eigs)
    k_liml = np.min(eigs)

    #We have everything needed to compute the point estimates
    bhat_liml = np.linalg.inv(X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ X) @ (X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ Y)
    
    #Now, lets work on variance:
    eps = Y - X @ bhat_liml
    omega = np.diag(np.diag(eps))
    om_2 = omega @ omega
    bread = (X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ X)
    meat = (X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ om_2 @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ X)
    robust_var = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread)

    return LIMLResult(betas=bhat_liml)
