import numpy as np
from numpy.typing import NDArray
from scipy.stats import t
import warnings
import logging
from repo import *

# Set up the logger This helps with error outputs and stuff. We can use this instead of printing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class CJIVEResult: 
    def __init__(self, 
                 beta: NDArray[np.float64], 
                 standard_errors: NDArray[np.float64],
                 r_squared: NDArray[np.float64], 
                 f_stat: NDArray[np.float64], 
                 cis: NDArray[np.float64]):
        self.beta = beta
        self.standard_errors = standard_errors
        self.r_squared = r_squared
        self.f_stat = f_stat
        self.cis = cis


    def __getitem__(self, key: str):
        if key == 'beta':
            return self.beta
        elif key == 'standard_errors':
            return self.standard_errors
        elif key == 'r_squared':
            return self.r_squared
        elif key == 'f_stat':
            return self.f_stat
        elif key == 'cis':
            return self.cis
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'beta', 'standard_errors, 'r_squared', 'f_stat', or 'cis'.")

    def __repr__(self):
        return f"CJIVEResult(beta={self.beta}, standard_errors={self.standard_errors}, r_squared={self.r_squared}, f_stat={self.f_stat}, cis={self.cis})"


def CJIVE(Y: NDArray[np.float64], W: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], cluster_ids: NDArray[np.int32], talk: bool = False) -> CJIVEResult:
    """
    Implements CJIVE estimator from Frandsen, ....
    Parameters
    ----------
    Y : NDArray[np.float64]
        The dependent variable.
    W : NDArray[np.float64]
        The matrix of control variables.
    X : NDArray[np.float64]
        The matrix of exogenous regressors.
    Z : NDArray[np.float64]
        The matrix of instruments.
    cluster_ids : NDArray[np.int32]
        The cluster ids for the observations.
    talk : bool, optional
        If True, prints additional information. The default is False.
    Returns
    -------
    beta : NDArray[np.float64]
        The estimated coefficients for the CJIVE model.
    se : NDArray[np.float64]
        The standard errors of the estimated coefficients.
    r2 : NDArray[np.float64]
        The R-squared value of the model.
    F : NDArray[np.float64]
        The F-statistic of the model.
    cis : NDArray[np.float64]
        The confidence intervals for the estimated coefficients.
    """

    # Set logging level based on the talk parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    N = Z.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if W.ndim == 1:
        W = W.reshape(-1, 1)

    # Add intercepts
    ones = np.ones((N, 1))
    X = np.hstack((ones, X))   
    Z = np.hstack((ones, Z)) 

    # Partial out W from Z: Z_tilde = (I - P_W)Z
    P_W = W @ np.linalg.inv(W.T @ W) @ W.T
    Z_tilde = (np.eye(N) - P_W) @ Z

    # Compute full projection matrix
    P_Zt = Z_tilde @ np.linalg.inv(Z_tilde.T @ Z_tilde) @ Z_tilde.T

    # Build D(P_Zt, n_G): block-diagonal version by clusters
    D_P = np.zeros_like(P_Zt)
    unique_clusters = np.unique(cluster_ids)
    for g in unique_clusters:
        idx = np.where(cluster_ids == g)[0]
        D_P[np.ix_(idx, idx)] = P_Zt[np.ix_(idx, idx)]

    # CJIVE projection matrix
    I = np.eye(N)
    C_CJIVE = np.linalg.inv(I - D_P) @ (P_Zt - D_P)

    # Estimate CJIVE beta
    bhat_CJIVE = np.linalg.inv(X.T @ C_CJIVE.T @ X) @ (X.T @ C_CJIVE.T @ Y)

    #Now, lets get some standard errors. We use Greene (2008)
    Xg_sum_1 = np.zeros((X.shape[1], X.shape[1]))
    Xg_sum_2 = np.zeros((X.shape[1], X.shape[1]))    
    S_sum = np.zeros((X.shape[1], X.shape[1]))

    w_hat = Y - X @ bhat_CJIVE

    for g in unique_clusters:
        idx = np.where(cluster_ids == g)[0]
        Xg = X[idx, :]
        Cg = C_CJIVE[idx,:]
        w_hat_g = w_hat[idx]
        
        Xg_sum_1 += (Xg @ Cg).T @ Xg
        Xg_sum_2 += Xg.T @ (Xg @ Cg)
        S_sum += (Xg @ Cg).T @ np.outer(w_hat_g, w_hat_g) @ (Xg @ Cg)

    G = np.unique(cluster_ids).size

    cluster_var = (G/(G-1)) * np.linalg.inv(Xg_sum_1) @ S_sum @ np.linalg.inv(Xg_sum_2)

    se = np.sqrt(np.diag(cluster_var))

    #Now lets just do a traditional t-test

    #Lets do a hypothesis test that B1=0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = (bhat_CJIVE[i])/((cluster_var[i,i])**.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = bhat_CJIVE[i] - t_crit_i * (cluster_var[i,i])**.5
        ci_upper = bhat_CJIVE[i] + t_crit_i * (cluster_var[i,i])**.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(t_stat_i)
        pvals.append(pval_i)
        cis.append(ci_i)  

    #Grab the R^2 for the model:
    yfit = X @ bhat_CJIVE
    ybar = np.mean(Y)
    r2 = 1 - np.sum((Y-yfit)**2) / np.sum((Y-ybar)**2)
    
    #Overall F-stat for the model:
    q = X.shape[1]
    e = Y-yfit
    F = ((np.sum((yfit-ybar)**2)) / (q-1)) / ((e.T @ e)/(N-q))

    #Root mean-squared error:
    root_mse = ((1/(N-q)) * (np.sum((Y - yfit)**2)))**.5

    #Adjusted R2
    ar2 = 1 - (((1-r2)*(N-1))/(N-q))

    #Now, we can add some first stage statistics if the number of endogenous regressors is 1
    if X.ndim == 2:
        X_fs = X[:,1]
        fs_fit = Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ X_fs
        xbar = np.mean(X_fs)

        #First Stage R2
        fs_r2 = 1 - np.sum((X_fs - fs_fit) ** 2) / np.sum((X_fs - xbar) ** 2)

        #First stage F-stat
        q_fs = Z.shape[1]
        e_fs = X_fs - fs_fit
        fs_F = ((np.sum((fs_fit - xbar) ** 2))/(q_fs-1))/((e_fs.T @ e_fs)/(N-q_fs))    

    return bhat_CJIVE, se, r2, F, cis


