import numpy as np
import logging
from numpy.typing import NDArray
from scipy.stats import t

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class JIVE2Result:
    def __init__(self, beta: NDArray[np.float64], leverage: NDArray[np.float64], fitted_values: NDArray[np.float64]):
        self.beta = beta
        self.leverage = leverage
        self.fitted_values = fitted_values

    def __getitem__(self, key: str):
        if key == 'beta':
            return self.beta
        elif key == 'leverage':
            return self.leverage
        elif key == 'fitted_values':
            return self.fitted_values
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'beta', 'leverage', or 'fitted_values'.")

    def __repr__(self):
        return f"JIVE2Result(beta={self.beta}, leverage={self.leverage}, fitted_values={self.fitted_values})"


def JIVE2(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], talk: bool = False) -> JIVE2Result:
    """
    Calculates the JIVE2 estimator using a two-pass approach recommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (NDArray[np.float64]): A 1-D numpy array of the dependent variable (N x 1).
        X (NDArray[np.float64]): A 2-D numpy array of the endogenous regressors (N x L).
        Z (NDArray[np.float64]): A 2-D numpy array of the instruments (N x K), where K > L.
        talk (bool): If True, provides detailed output for teaching purposes. Default is False.

    Returns:
        JIVE2Result: A custom result object containing the JIVE2 estimates, leverage values, and fitted values.
    """
    # Adjust logging level based on the `talk` parameter
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
        logger.warning(f"Normally this estimator is used when Z has more columns than X. In this case Z has {Z.shape[1]} columns and X has {X.shape[1]} columns.")

    logger.debug(f"Y has {Y.shape[0]} rows.\n")
    logger.debug(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
    logger.debug(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")

    ones = np.ones((N,1))
    X = np.hstack((ones, X))
    Z = np.hstack((ones, Z))

    # First pass to get fitted values and leverage
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    fit = P @ X
    logger.debug(f"Fitted values obtained.\n")

    leverage = np.diag(P)
    if np.any(leverage >= 1):
        raise ValueError("Leverage values must be strictly less than 1 to avoid division by zero.")
    logger.debug(f"Leverage values obtained.\n")

    # Reshape leverage to an Nx1 vector
    leverage = leverage.reshape(-1, 1)
    logger.debug(f"First pass complete.\n")

    # Second pass to remove ith row and reduce bias
    fit = fit[:, 1:]
    X = X[:,1:]
    X_jive2 = (fit - leverage * X) / (1 - (1/N))
    logger.debug(f"Second pass complete.\n")

    ones = np.ones((N,1))
    X_jive2 = np.hstack((ones, X_jive2))
    X = np.hstack((ones, X))

    # Calculate the JIVE2 estimates
    beta_jive2 = np.linalg.inv(X_jive2.T @ X) @ X_jive2.T @ Y
    logger.debug(f"JIVE2 Estimates:\n{beta_jive2}\n")

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ beta_jive2)**2 * np.outer(X_jive2[i], X_jive2[i])
    robust_v = np.linalg.inv(X_jive2.T @ X) @ midsum @ np.linalg.inv(X.T @ X_jive2)


    #Lets do a hypothesis test that B1=0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = (beta_jive2[i])/((robust_v[i,i])**.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = beta_jive2[i] - t_crit_i * (robust_v[i,i])**.5
        ci_upper = beta_jive2[i] + t_crit_i * (robust_v[i,i])**.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(t_stat_i)
        pvals.append(pval_i)
        cis.append(ci_i)  

    return JIVE2Result(beta=beta_jive2, leverage=leverage, fitted_values=fit)
