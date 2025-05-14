# JIVE 1
import numpy as np
import warnings
import logging
from numpy.typing import NDArray
from typing import NamedTuple
from scipy.stats import t
from repo import *

# Set up the logger This helps with error outputs and stuff. We can use this instead of printing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class JIVE1Result:
    def __init__(self, 
                 beta: NDArray[np.float64], 
                 leverage: NDArray[np.float64], 
                 fitted_values: NDArray[np.float64], 
                 r_squared: NDArray[np.float64], 
                 adjusted_r_squared: NDArray[np.float64], 
                 f_stat: NDArray[np.float64],
                 standard_errors: NDArray[np.float64],
                 root_mse: NDArray[np.float64],
                 pvals: NDArray[np.float64],
                 tstats: NDArray[np.float64],
                 cis: NDArray[np.float64]):
        self.beta = beta
        self.leverage = leverage
        self.fitted_values = fitted_values
        self.r_squared = r_squared
        self.adjusted_r_squared = adjusted_r_squared
        self.f_stat = f_stat
        self.standard_errors = standard_errors
        self.root_mse=root_mse,
        self.pvals=pvals,
        self.tstats=tstats,
        self.cis=cis

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
        elif key == 'root_mse':
            return self.root_mse
        elif key == 'pvals':
            return self.pvals
        elif key == 'tstats':
            return self.tstats
        elif key == 'cis':
            return self.cis
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'beta', 'leverage', 'fitted_values', 'r_squared', 'adjusted_r_squared', 'f_stat', 'standard_errors', 'root_mse', 'pvals', 'tstats', or 'cis'.")

    def __repr__(self):
        return f"JIVE1Result(beta={self.beta}, leverage={self.leverage}, fitted_values={self.fitted_values}, r_squared={self.r_squared}, adjusted_r_squared={self.adjusted_r_squared}, f_stat={self.f_stat}, standard_errors={self.standard_errors}, root_mse={self.root_mse}, pvals={self.pvals}, tstats={self.tstats}, cis={self.cis})"


    def summary(self, pvals, tstats, cis, root_mse):
        """
        Prints a summary of the JIVE1 results in a tabular format similar to statsmodels OLS.
        """
        import pandas as pd

        # Create a DataFrame for coefficients, standard errors, t-stats, p-values, and confidence intervals
        summary_df = pd.DataFrame({
            "Coefficient": self.beta.flatten(),
            "Std. Error": np.sqrt(np.diag(self.standard_errors)),
            "t-stat": tstats,
            "P>|t|": pvals,
            "Conf. Int. Low": [ci[0] for ci in cis],
            "Conf. Int. High": [ci[1] for ci in cis]
        })

        # Print the summary
        print("\nJIVE1 Regression Results")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("-" * 80)
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.4f}")
        print(f"F-statistic: {self.f_stat:.4f}")
        print(f"Root MSE: {root_mse:.4f}")
        print("=" * 80)

def JIVE1(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], W: NDArray[np.float64] | None = None, talk: bool = False) -> JIVE1Result:
    """
    Calculates the JIVE1 estimator defined by Blomquist and Dahlberg (1999) in Jackknife IV estimation.

    Args:
        Y (NDArray[np.float64]): A 1-D numpy array of the dependent variable (N x 1).
        X (NDArray[np.float64]): A 2-D numpy array of the endogenous regressors (N x L). Do not inlude the constant.
        Z (NDArray[np.float64]): A 2-D numpy array of the instruments (N x K), where K > L. Do not include the constant.
        W (NDArray[np.float64]): A 2-D numpy array of the exogenous controls (N x G). Do not include the constant. These are not necessary for the function. 
        talk (bool): If True, provides detailed output for teaching / debugging purposes. Default is False.

    Returns:
        JIVE1Result: An object containing the following attributes:
            - beta (NDArray[np.float64]): The estimated coefficients for the model.
            - leverage (NDArray[np.float64]): The leverage values for each observation.
            - fitted_values (NDArray[np.float64]): The fitted values from the first pass of the JIVE1 estimator.
            - r_squared (float): The R-squared value for the model.
            - adjusted_r_squared (float): The adjusted R-squared value for the model.
            - f_stat (float): The F-statistic for the model.
            - standard_errors (NDArray[np.float64]): The robust standard errors for the estimated coefficients.

    Raises:
        ValueError: If the dimensions of Y, X, or Z are inconsistent or invalid.
        RuntimeWarning: If the number of instruments (columns in Z) is not greater than the number of regressors (columns in X).

    Notes:
        - The JIVE1 estimator is a jackknife-based instrumental variable estimator designed to reduce bias in the presence of many instruments.
        - The function performs a two-pass estimation:
            1. The first pass calculates fitted values and leverage values using the instruments.
            2. The second pass removes the ith observation to calculate unbiased estimates.
        - Additional statistics such as R-squared, adjusted R-squared, and F-statistics are calculated for model evaluation.
        - If the number of endogenous regressors is 1, first-stage statistics (R-squared and F-statistic) are also computed.

    Example:
        >>> import numpy as np
        >>> from weak_instruments.jive1 import JIVE1
        >>> Y = np.array([1, 2, 3])
        >>> X = np.array([[1], [2], [3]])
        >>> Z = np.array([[1, 0], [0, 1], [1, 1]])
        >>> result = JIVE1(Y, X, Z)
        >>> print(result.beta)
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
    
    #If X/Z is a single vector:
    if X.ndim == 1:
        X = X.reshape(-1,1)
        logger.debug(f"X reshaped to {X.shape}.\n")
    if Z.ndim == 1:
        Z = Z.reshape(-1,1)
        logger.debug(f"Z reshaped to {Z.shape}.\n")
    
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


    # Drop any constant columns from X and Z
    if np.all(np.all(np.isclose(X, X[0, :], atol=1e-8), axis=0)):
        if hasattr(X, 'columns'):  # Check if X has column names (e.g., a DataFrame)
            dropped_columns = X.columns[np.all(np.isclose(X, X[0, :], atol=1e-8), axis=0)]
            logger.debug(f"X has constant columns. Dropping columns: {list(dropped_columns)}")
        else:
            logger.debug("X has constant columns. Dropping constant columns.")
        X = X[:, ~np.all(np.isclose(X, X[0, :], atol=1e-8), axis=0)]

    if np.all(np.all(np.isclose(Z, Z[0, :], atol=1e-8), axis=0)):
        if hasattr(Z, 'columns'):  # Check if Z has column names (e.g., a DataFrame)
            dropped_columns = Z.columns[np.all(np.isclose(Z, Z[0, :], atol=1e-8), axis=0)]
            logger.debug(f"Z has constant columns. Dropping columns: {list(dropped_columns)}")
        else:
            logger.debug("Z has constant columns. Dropping constant columns.")
        Z = Z[:, ~np.all(np.isclose(Z, Z[0, :], atol=1e-8), axis=0)]

        
    #Add the constant
    k = X.shape[1]
    ones = np.ones((N,1))
    X = np.hstack((ones, X))
    Z = np.hstack((ones, Z))

    #Add the controls:
    if W is not None:
        if W.ndim == 1:
            W = W.reshape(-1, 1)
    if W.shape[0] != N:
        raise ValueError(f"G must have the same number of rows as Y. Got G.shape[0] = {W.shape[0]} and Y.shape[0] = {N}.")
    X = np.hstack((X, W))
    Z = np.hstack((Z, W))
    logger.debug("Controls W have been added to both X and Z.\n")

    # First pass to get fitted values and leverage
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    fit = P @ X #  Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ X 
    logger.debug(f"Fitted values obtained.\n")

    # Get the main diagonal from the projection matrix
    leverage = np.diag(Z @ np.linalg.inv(Z.T @ Z) @ Z.T) # np.diag(P)
    if np.any(leverage >= 1): 
        raise ValueError("Leverage values must be strictly less than 1 to avoid division by zero.")
    logger.debug(f"Leverage values obtained.\n")

    # Reshape to get an Nx1 vector
    leverage = leverage.reshape(-1, 1)

    # Second pass to remove the ith row for unbiased estimates
    fit = fit[:, 1:1+k]
    X = X[:,1:1+k]    
    X_jive1 = (fit - leverage * X) / (1 - leverage)
    logger.debug(f"Second pass complete.\n")

    X_jive1 = np.hstack((ones, X_jive1, W))
    X = np.hstack((ones, X, W))

    # Calculate the optimal estimate
    beta_jive1 = np.linalg.inv(X_jive1.T @ X_jive1) @ X_jive1.T @ Y
    logger.debug(f"JIVE1 Estimates:\n{beta_jive1}\n")

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ beta_jive1)**2 * np.outer(X_jive1[i], X_jive1[i])
    robust_v = np.linalg.inv(X_jive1.T @ X_jive1) @ midsum @ np.linalg.inv(X_jive1.T @ X_jive1)


    #Lets do a hypothesis test that B1=0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = (beta_jive1[i])/((robust_v[i,i])**.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = beta_jive1[i] - t_crit_i * (robust_v[i,i])**.5
        ci_upper = beta_jive1[i] + t_crit_i * (robust_v[i,i])**.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(t_stat_i)
        pvals.append(pval_i)
        cis.append(ci_i)

    #Grab the R^2 for the model:
    yfit = X @ beta_jive1
    ybar = np.mean(Y)
    r2 = 1 - np.sum((Y-yfit)**2) / np.sum((Y-ybar)**2)
    
    #Overall F-stat for the model:
    q = X.shape[1]
    e = Y-yfit
    F = ((np.sum((yfit-ybar)**2)) / (q-1)) / ((e.T @ e)/(N-q))

    #Mean-square error:
    root_mse = ((1/(N-q)) * (np.sum((Y - yfit)**2)))**.5

    #Adjusted R2
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

    return JIVE1Result(beta=beta_jive1, 
                       leverage=leverage, 
                       fitted_values=fit, 
                       r_squared=r2, 
                       adjusted_r_squared=ar2, 
                       f_stat=F, 
                       standard_errors=robust_v,
                       root_mse=root_mse,
                       pvals=pvals,
                       tstats=tstats,
                       cis=cis)


## ## Future thoughts ###
"""





"""