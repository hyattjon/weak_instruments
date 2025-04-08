# JIVE (and JIVE-related estimators---there are a lot of these in the literature)
import numpy as np

def JIVE1(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk:bool = False) -> np.ndarray:
    """
    Calculates the JIVE1 estimator using a two-pass approach reccommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.

    Returns:
        np.ndarray: A 1-D numpy array of the JIVE1 estimates (L x 1).
    """

    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError("Y must be a one-dimensional array.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError("Z must be at least a one-dimensional array.")
    # Check that Y and X and Z have N columns. Base off of Y.shape[0]
    N = Y.shape[0]
    if X.shape[0] != N:
        raise ValueError("X and Y must have the same number of rows.")
    if Z.shape[0] != N:
        raise ValueError("Z and Y must have the same number of rows.")
    if Z.shape[1] <= X.shape[1]:
        print(f"Normally this estimator is used when Z has more columns than X. In this case Z has {Z.shape[1]} columns and X has {X.shape[1]} columns.")

    if talk:
        print(f"Y has {Y.shape[0]} rows and {Y.shape[1]} columns.\n")
        print(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
        print(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")


    ### I need this double checked to ensure accuracy. I'm not confident that the matrix multiplication will give us a scalar. i.e. the sizes of the matrices with one row subtracted might be messed up.
    # First Pass
    # pi[i] = np.linalg.inv(z[i].T @ z[i]) @ (z[i].T @ X[i])
    Z_j1 = np.zeros((Z.shape[0], X.shape[1]))
    for i in range(Z.shape[0]):
        Z_j1[i] = (Z[i] @ np.linalg.inv(Z.T @ Z) @ (Z.T @ X - Z[i].T @ X[i])) / (1- Z[i] @ (np.linalg.inv(Z.T @ Z)) @ Z[i].T)

    # Second Pass: Construct the X_jive1 matrix
    # Need some help here 

    # Second Stage: IV estimation using X_jive1 as the instrument for X
    p_jive1 = np.dot(Z_j1, np.dot(np.linalg.inv(Z_j1.T @ Z_j1), Z_j1.T))
    beta_jive1 = np.linalg.inv(X.T @ p_jive1 @ X) @ X.T @ p_jive1 @ Y

    print("JIVE1 Estimates:\n", beta_jive1)
    return beta_jive1


def JIVE2(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk:bool = False) -> np.ndarray:
    """
    Calculates the JIVE2 estimator using a two-pass approach reccommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.

    Returns:
        np.ndarray: A 1-D numpy array of the JIVE2 estimates (L x 1).
    """

    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError("Y must be a one-dimensional array.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError("Z must be at least a one-dimensional array.")
    # Check that Y and X and Z have N columns. Base off of Y.shape[0]
    N = Y.shape[0]
    if X.shape[0] != N:
        raise ValueError("X and Y must have the same number of rows.")
    if Z.shape[0] != N:
        raise ValueError("Z and Y must have the same number of rows.")
    if Z.shape[1] <= X.shape[1]:
        print(f"Normally this estimator is used when Z has more columns than X. In this case Z has {Z.shape[1]} columns and X has {X.shape[1]} columns.")

    if talk:
        print(f"Y has {Y.shape[0]} rows and {Y.shape[1]} columns.\n")
        print(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
        print(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")


    ### I need this double checked to ensure accuracy. I'm not confident that the matrix multiplication will give us a scalar. i.e. the sizes of the matrices with one row subtracted might be messed up.
    # First Pass
    # pi[i] = np.linalg.inv(z[i].T @ z[i]) @ (z[i].T @ X[i])
    Z_j2 = np.zeros((Z.shape[0], X.shape[1]))
    for i in range(Z.shape[0]):
        Z_j2[i] = (Z[i] @ np.linalg.inv(Z.T @ Z) @ (Z.T @ X - Z[i].T @ X[i])) / (1- (1/N))

    # Second Pass: Construct the X_jive1 matrix
    # Need some help here 

    # Second Stage: IV estimation using X_jive1 as the instrument for X
    p_jive2 = np.dot(Z_j2, np.dot(np.linalg.inv(Z_j2.T @ Z_j2), Z_j2.T))
    beta_jive2 = np.linalg.inv(X.T @ p_jive2 @ X) @ X.T @ p_jive2 @ Y

    print("JIVE1 Estimates:\n", beta_jive2)
    return beta_jive2




def JIVE2(X, y):
    """Implement the JIVE2 estimator.
    
    Example:
    JIVE2(X, y)"""
    
    return 1


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