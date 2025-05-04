import numpy as np
from numpy.typing import NDArray

def CJIVE(Y: NDArray[np.float64], W: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], cluster_ids: NDArray[np.int32]):
    """
    Implements CJIVE estimator from Frandsen, ....
    """
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

    return bhat_CJIVE
