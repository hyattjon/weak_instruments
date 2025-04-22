# JIVE (and JIVE-related estimators---there are a lot of these in the literature)
import numpy as np
import warnings
import logging
from numpy.typing import NDArray
from typing import NamedTuple

# Set up the logger This helps with error outputs and stuff. We can use this instead of printing stuff
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class JIVE1Result(NamedTuple):
    beta: NDArray[np.float64]
    leverage: NDArray[np.float64]
    fitted_values: NDArray[np.float64]

def JIVE1(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], talk: bool = False) -> JIVE1Result:
    """
    Calculates the JIVE1 estimator using a two-pass approach recommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.
        talk (bool): If True, provides detailed output for teaching purposes. Default is False.

    Returns:
        JIVE1Result: A named tuple containing the JIVE1 estimates, leverage values, and fitted values.
    """
    # Adjust logging level based on the `talk` parameter. 
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError(f"Y must be a one-dimensional array, but got shape {Y.shape}.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError(f"Z must be at least a one-dimensional array, but got shape {Z.shape}.")
    
    # Check that Y, X, and Z have consistent dimensions
    N = Y.shape[0]
    if X.shape[0] != N:
        raise ValueError(f"X and Y must have the same number of rows. Got X.shape[0] = {X.shape[0]} and Y.shape[0] = {N}.")
    if Z.shape[0] != N:
        raise ValueError(f"Z and Y must have the same number of rows. Got Z.shape[0] = {Z.shape[0]} and Y.shape[0] = {N}.")
    if Z.shape[1] <= X.shape[1]:
        warnings.warn(f"Normally this estimator is used when Z has more columns than X. In this case Z has {Z.shape[1]} columns and X has {X.shape[1]} columns.", RuntimeWarning)

    logger.debug(f"Y has {Y.shape[0]} rows.\n")
    logger.debug(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
    logger.debug(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")

    # First pass to get fitted values and leverage
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    fit = P @ X
    logger.debug(f"Fitted values obtained.\n")

    # Get the main diagonal from the projection matrix
    leverage = np.diag(P)
    if np.any(leverage >= 1):
        raise ValueError("Leverage values must be strictly less than 1 to avoid division by zero.")
    logger.debug(f"Leverage values obtained.\n")

    # Reshape to get an Nx1 vector
    leverage = leverage.reshape(-1, 1)

    # Second pass to remove the ith row for unbiased estimates
    X_jive = (fit - leverage * X) / (1 - leverage)
    logger.debug(f"Second pass complete.\n")

    # Calculate the optimal estimate
    beta_jive1 = np.linalg.inv(X_jive.T @ X) @ X_jive.T @ Y
    logger.debug(f"JIVE1 Estimates:\n{beta_jive1}\n")

    return JIVE1Result(beta=beta_jive1, leverage=leverage, fitted_values=fit)


