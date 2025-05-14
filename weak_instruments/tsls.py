# Here we implement the 2SLS estimator for the weak instruments case.
import numpy as np
import logging
from numpy.typing import NDArray
from repo import *


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s') 
handler.setFormatter(formatter)
logger.addHandler(handler)


class TSLSResult:
    def __init__(self, 
                 beta: NDArray[np.float64]):
        self.beta = beta

    def __getitem__(self, key: str):
        if key == 'beta':
            return self.beta
        else:
            raise KeyError(f"Invalid key '{key}'. The valid key is 'beta'.")

    def __repr__(self):
        return f"JIVE1Result(beta={self.beta})"


def TSLS(Y: NDArray[np.float64],
         X: NDArray[np.float64],
         Z: NDArray[np.float64]) -> TSLSResult:
    """
    Two-Stage Least Squares (2SLS) estimator for weak instruments.

    Parameters
    ----------
    Y : NDArray[np.float64]
        The dependent variable.
    X : NDArray[np.float64]
        The independent variable.
    Z : NDArray[np.float64]
        The instrument variable.

    Returns
    -------
    TSLSResult
        An object containing the estimated coefficients.
    """

    # Check dimensions
    if Y.shape[0] != X.shape[0] or Y.shape[0] != Z.shape[0]:
        raise ValueError("All input arrays must have the same number of rows.")
    
    # Get the pi hats
    X_hat = X @ np.linalg.inv(Z.T @ Z) @ (Z.T @ X)

    # Get the beta hats
    beta_hat = np.linalg.inv(X_hat.T @ X_hat) @ (X_hat.T @ Y)

    # Return the result
    return TSLSResult(beta=beta_hat)