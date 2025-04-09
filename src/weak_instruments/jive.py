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
    fit = np.dot(np.dot(np.dot(Z ,  np.linalg.inv(np.dot(Z.T, Z))) , Z.T) , X)
    leverage = (np.diag(np.dot(np.dot(Z , np.linalg.inv(np.dot(Z.T , Z))) , Z.T)))
    leverage = leverage.reshape(-1, 1)

    X_jive = (fit - leverage * X)/(1-leverage) 
    beta_jive1 = np.linalg.inv(X_jive.T @ X) @ X_jive.T @ Y 

    print("JIVE2 Estimates:\n", beta_jive1)

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
    fit = np.dot(np.dot(np.dot(Z ,  np.linalg.inv(np.dot(Z.T, Z))) , Z.T) , X)
    leverage = (np.diag(np.dot(np.dot(Z , np.linalg.inv(np.dot(Z.T , Z))) , Z.T)))
    leverage = leverage.reshape(-1, 1)

    X_jive2 = (fit - leverage * X)/(1-(1/N)) 
    beta_jive2 = np.linalg.inv(X_jive2.T @ X) @ X_jive2.T @ Y 

    print("JIVE2 Estimates:\n", beta_jive2)

    return beta_jive2

