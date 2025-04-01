# JIVE (and JIVE-related estimators---there are a lot of these in the literature)

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