# JIVE (and JIVE-related estimators---there are a lot of these in the literature)
import numpy as np

def jive1_estimator(Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Calculates the JIVE1 estimator using a two-pass approach.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.

    Returns:
        np.ndarray: A 1-D numpy array of the JIVE1 estimates (L x 1).
    """
    # First Pass
    ZT_Z = Z.T @ Z
    ZT_Z_inv = np.linalg.inv(ZT_Z)
    pi_hat = ZT_Z_inv @ Z.T @ X
    X_hat = Z @ pi_hat
    H = Z @ ZT_Z_inv @ Z.T
    h = np.diag(H)

    # Second Pass: Construct the X_jive1 matrix
    N = Y.shape
    L = X.shape[1]
    X_jive1 = np.zeros_like(X, dtype=float)

    for i in range(N):
        numerator = X_hat[i, :] - h[i] * X[i, :]
        denominator = 1 - h[i]
        X_jive1[i, :] = numerator / denominator

    # Second Stage: IV estimation using X_jive1 as the instrument for X
    X_jive1_T_X = X_jive1.T @ X
    X_jive1_T_Y = X_jive1.T @ Y

    try:
        beta_jive1 = np.linalg.inv(X_jive1_T_X) @ X_jive1_T_Y
    except np.linalg.LinAlgError:
        print("Singular matrix encountered during the second stage of JIVE1 estimation.")
        return np.full(L, np.nan)

    return beta_jive1



def JIVE2(X, y):
    """Implement the JIVE2 estimator.
    
    Example:
    JIVE2(X, y)"""
    
    return 1






import numpy as np 
 
def JIVE1(X:np.ndarray, Y:np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    JIVE1 algorithm.
    Parameters
    ----------
    X : np.ndarray
        The control data matrix.
    Y : np.ndarray
        The outcome vector.
    n_components : int
        The number of components to extract.
    Returns
    -------
    np.ndarray
        The estimated coefficients.
    """
    
    # First pass, run n regressions of X in Z leaving out the ith row

    # Second pass, 

    # Construct the beta estimates from the formula ::

    beta_est = 1

    return beta_est

def JIVE2(X:np.ndarray, Y:np.ndarray, Z: np.ndarray, const:bool  = False) -> np.ndarray:
    """
    JIVE1 algorithm.
    Parameters
    ----------
    X : np.ndarray
        The control data matrix. X is a one-dimensional array. Single endogenous variable.
    Y : np.ndarray
        The outcome vector.
    Z : np.ndarray
        The instrument matrix. N x K dimensional matrix.
    Returns
    -------
    np.ndarray
        The estimated coefficients.
    """

    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    # Cleaning up we need to add the column of ones to the data matrix
    # Maybe not
    top =0
    bottom = 0

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i != j:
                top = P[i,j]*X[i]*Y[j] # ith row, jth column
                bottom = P[i,j]*X[i]*X[j]

    # First pass, run n regressions of X in Z leaving out the ith row

    # Second pass, 

    # Construct the beta estimates from the formula ::
    
    beta_est = 1

    return beta_est


if __name__=="__main__":
    pass