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
    ZT_Z_inv = np.linalg.inv(Z.T @ Z)
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

def JIVE2(X:np.ndarray, Y:np.ndarray, Z: np.ndarray) -> np.ndarray:
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
    # Check if X is a one-dimensional array
    if X.ndim != 1:
        raise ValueError("X must be a one-dimensional array.")
    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError("Y must be a one-dimensional array.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError("Z must be at least a one-dimensional array.")
    # Check if the number of rows in X, Y, and Z are the same
    if X.shape[0] != Y.shape[0] or X.shape[0] != Z.shape[0]:
        raise ValueError("The number of rows in X, Y, and Z must be the same.")


    # Create the projection matrix P
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    # Initialize the top and bottom of the fraction from the formula
    top =0
    bottom = 0
    # Initialize the outcome vector of beta estimates
    beta_vec = np.zeros(X.shape[1])
    # Loop through all rows
    for i in range(X.shape[0]):
        # Loop through all rows not including the ith row
        for j in range(X.shape[0]):
            # Check to see if the ith row is not equal to the jth row
            if i != j:
                # Calculate the top and bottom of the fraction from the formula
                top = P[i,j]*X[i]*Y[j] # ith row, jth column
                bottom = P[i,j]*X[i]*X[j] # ith row, jth column
                # Calculate the beta estimate for the ith row
                beta_vec[i] = top/bottom
    # Return the vector of beta estimates
    return beta_vec


if __name__=="__main__":
    pass