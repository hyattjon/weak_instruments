# JIVE2 Estimator
import numpy as np
import logging
from numpy.typing import NDArray
from scipy.stats import t
from repo import *

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class JIVE2Result:
    def __init__(self, 
                 beta: NDArray[np.float64], 
                 leverage: NDArray[np.float64], 
                 fitted_values: NDArray[np.float64],
                 r_squared: NDArray[np.float64], 
                 adjusted_r_squared: NDArray[np.float64], 
                 f_stat: NDArray[np.float64],
                 standard_errors: NDArray[np.float64]):
        self.beta = beta
        self.leverage = leverage
        self.fitted_values = fitted_values
        self.r_squared = r_squared
        self.adjusted_r_squared = adjusted_r_squared
        self.f_stat = f_stat
        self.standard_errors = standard_errors

    def __getitem__(self, key: str):
        if key == 'beta':
            return self.beta
        elif key == 'leverage':
            return self.leverage
        elif key == 'fitted_values':
            return self.fitted_values
        elif key == 'r_squared':
            return self.r_squared
        elif key == 'adjusted_r_squared':
            return self.adjusted_r_squared
        elif key == 'f_stat':
            return self.f_stat
        elif key == 'standard_errors':
            return self.standard_errors
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'beta', 'leverage', 'fitted_values', 'r_squared', 'adjusted_r_squared', 'f_stat', or 'standard_errors'.")


    def __repr__(self):
        return f"JIVE1Result(beta={self.beta}, leverage={self.leverage}, fitted_values={self.fitted_values}, r_squared={self.r_squared}, adjusted_r_squared={self.adjusted_r_squared}, f_stat={self.f_stat}, standard_errors={self.standard_errors})"


def JIVE2(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], G: NDArray[np.float64] | None = None, talk: bool = False) -> JIVE2Result:
    """
    Calculates the JIVE2 estimator defined by Blomquist and Dahlberg (1999) in Jackknife IV estimation.

    Args:
        Y (NDArray[np.float64]): A 1-D numpy array of the dependent variable (N x 1).
        X (NDArray[np.float64]): A 2-D numpy array of the endogenous regressors (N x L).
        Z (NDArray[np.float64]): A 2-D numpy array of the instruments (N x K), where K > L.
        W (NDArray[np.float64]): A 2-D numpy array of the exogenous controls (N x G). Do not include the constant. These are not necessary for the function. 
        talk (bool): If True, provides detailed output for teaching purposes. Default is False.

    Returns:
        JIVE2Result: An object containing the following attributes:
            - beta (NDArray[np.float64]): The estimated coefficients for the model.
            - leverage (NDArray[np.float64]): The leverage values for each observation.
            - fitted_values (NDArray[np.float64]): The fitted values from the first pass of the JIVE2 estimator.
            - r_squared (float): The R-squared value for the model.
            - adjusted_r_squared (float): The adjusted R-squared value for the model.
            - f_stat (float): The F-statistic for the model.
            - standard_errors (NDArray[np.float64]): The robust standard errors for the estimated coefficients.

    Raises:
        ValueError: If the dimensions of Y, X, or Z are inconsistent or invalid.
        RuntimeWarning: If the number of instruments (columns in Z) is not greater than the number of regressors (columns in X).

    Notes:
        - The JIVE2 estimator is a jackknife-based instrumental variable estimator designed to reduce bias in the presence of many instruments.
        - The function performs a two-pass estimation:
            1. The first pass calculates fitted values and leverage values using the instruments.
            2. The second pass removes the ith observation to calculate unbiased estimates.
        - Additional statistics such as R-squared, adjusted R-squared, F-statistics, and robust standard errors are calculated for model evaluation.
        - If the number of endogenous regressors is 1, first-stage statistics (R-squared and F-statistic) are also computed.

    Example:
        >>> import numpy as np
        >>> from weak_instruments.jive2 import JIVE2
        >>> Y = np.array([1, 2, 3])
        >>> X = np.array([[1], [2], [3]])
        >>> Z = np.array([[1, 0], [0, 1], [1, 1]])
        >>> result = JIVE2(Y, X, Z)
        >>> print(result.beta)
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
    
    #If X/Z is a single vector:
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Z.ndim == 1:
        Z = Z.reshape(-1,1)
    
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
    #P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    fit = Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ X
    logger.debug(f"Fitted values obtained.\n")

    leverage = np.diag(Z @ np.linalg.inv(Z.T @ Z) @ Z.T)
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

    X_jive2 = np.hstack((ones, X_jive2))
    X = np.hstack((ones, X))

    # Calculate the JIVE2 estimates
    beta_jive2 = np.linalg.inv(X_jive2.T @ X_jive2) @ X_jive2.T @ Y
    logger.debug(f"JIVE2 Estimates:\n{beta_jive2}\n")

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ beta_jive2)**2 * np.outer(X_jive2[i], X_jive2[i])
    robust_v = np.linalg.inv(X_jive2.T @ X_jive2) @ midsum @ np.linalg.inv(X_jive2.T @ X_jive2)


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

    #Grab the R^2 for the model:
    yfit = X @ beta_jive2
    ybar = np.mean(Y)
    r2 = 1 - np.sum((Y-yfit)**2) / np.sum((Y-ybar)**2)
    
    #Overall F-stat for the model:
    q = X.shape[1]
    e = Y-yfit
    F = ((np.sum((yfit-ybar)**2)) / (q-1)) / ((e.T @ e)/(N-q))

    #Mean-square error:
    root_mse = ((1/(N-q)) * (np.sum((Y - yfit)**2)))**.5

    #Adjustred R2
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


    return JIVE2Result(beta=beta_jive2, leverage=leverage, fitted_values=fit, r_squared=r2, adjusted_r_squared=ar2, f_stat=F, standard_errors=robust_v)
