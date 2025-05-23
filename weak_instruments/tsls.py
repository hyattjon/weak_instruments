# Here we implement the 2SLS estimator for the weak instruments case.
import numpy as np
import logging
from numpy.typing import NDArray
import warnings
from scipy.stats import t

# Set up the logger This helps with error outputs and stuff. We can use this instead of printing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  
handler.setFormatter(formatter)
logger.addHandler(handler)


class TSLSResult:
    """
    Stores results for the TSLS estimator.

    Attributes
    ----------
    beta : NDArray[np.float64]
        Estimated coefficients from the TSLS regression.
    r_squared : float
        R-squared value.
    adjusted_r_squared : float
        Adjusted R-squared value.
    f_stat : float
        F-statistic for the model.
    standard_errors : NDArray[np.float64]
        Robust standard errors.
    root_mse : float
        Root mean squared error.
    pvals : NDArray[np.float64] or None
        p-values for coefficients.
    tstats : NDArray[np.float64] or None
        t-statistics for coefficients.
    cis : NDArray[np.float64] or None
        Confidence intervals for coefficients.
    """
   
    def __init__(self, 
                 beta: NDArray[np.float64],
                 r_squared: float = None,
                 adjusted_r_squared: float = None,
                 f_stat: float = None,
                 standard_errors: NDArray[np.float64] = None,
                 root_mse: float = None,
                 pvals: NDArray[np.float64] | None = None,
                 tstats: NDArray[np.float64] | None = None,
                 cis: NDArray[np.float64] = None):
        self.beta = beta
        self.r_squared = r_squared
        self.adjusted_r_squared = adjusted_r_squared
        self.f_stat = f_stat
        self.standard_errors = standard_errors
        self.root_mse = root_mse
        self.pvals = pvals
        self.tstats = tstats
        self.cis = cis

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to TSLSResult attributes.

        Parameters
        ----------
        key : str
            The attribute name to retrieve.

        Returns
        -------
        The value of the requested attribute.

        Raises
        ------
        KeyError
            If the key is not a valid attribute name.
        """
        if key == 'beta':
            return self.beta
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
            raise KeyError(f"Invalid key '{key}'. The valid keys are 'beta', ''r_squared', 'adjusted_r_squared', 'f_stat', 'standard_errors', 'root_mse', 'pvals', 'tstat', or 'cis'.")

    def __repr__(self):
        """
        Returns a string representation of the TSLSResult object.
        """
        return f"JIVE1Result(beta={self.beta}, r_squared={self.r_squared}, adjusted_r_squared={self.adjusted_r_squared}, f_stat={self.f_stat}, standard_errors={self.standard_errors}, root_mse={self.root_mse}, pvals={self.pvals}, tstats={self.tstats}, cis={self.cis})"
    
    def summary(self):
        """
        Prints a summary of the TSLS results in a tabular format similar to statsmodels OLS.
        """
        import pandas as pd

        summary_df = pd.DataFrame({
            "Coefficient": self.beta.flatten(),
            "Std. Error": np.sqrt(np.diag(self.standard_errors)) if self.standard_errors is not None else np.nan,
            "t-stat": self.tstats,
            "P>|t|": self.pvals,
            "Conf. Int. Low": [ci[0] for ci in self.cis] if self.cis is not None else np.nan,
            "Conf. Int. High": [ci[1] for ci in self.cis] if self.cis is not None else np.nan
        })

        print("\nTSLS Regression Results")
        print("=" * 80)
        print(summary_df.round(6).to_string(index=False))
        print("-" * 80)
        print(f"R-squared: {self.r_squared:.6f}" if self.r_squared is not None else "R-squared: N/A")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.6f}" if self.adjusted_r_squared is not None else "Adjusted R-squared: N/A")
        print(f"F-statistic: {self.f_stat:.6f}" if self.f_stat is not None else "F-statistic: N/A")
        print(f"Root MSE: {self.root_mse:.6f}" if self.root_mse is not None else "Root MSE: N/A")
        print("=" * 80)


def TSLS(Y: NDArray[np.float64],
         X: NDArray[np.float64],
         Z: NDArray[np.float64],
         G: NDArray[np.float64] | None = None,
         W: NDArray[np.float64] | None = None,
         talk: bool = False) -> TSLSResult:
    """
    Two-Stage Least Squares (2SLS) estimator for weak instruments.

    Parameters
    ----------
    Y : NDArray[np.float64]
        A 1-D numpy array of the dependent variable (N,).
    X : NDArray[np.float64]
        A 2-D numpy array of the endogenous regressors (N, L). Do not include the constant.
    Z : NDArray[np.float64]
        A 2-D numpy array of the instruments (N, K), where K >= L. Do not include the constant.
    W : NDArray[np.float64], optional
        A 2-D numpy array of the exogenous controls (N, G). Do not include the constant. Default is None.
    talk : bool, optional
        If True, provides detailed output for teaching / debugging purposes. Default is False.

    Returns
    -------
    TSLSResult
        An object containing the following attributes:
            - beta (NDArray[np.float64]): The estimated coefficients for the model.
            - r_squared (float): The R-squared value for the model.
            - adjusted_r_squared (float): The adjusted R-squared value for the model.
            - f_stat (float): The F-statistic for the model.
            - standard_errors (NDArray[np.float64]): The robust standard errors for the estimated coefficients.
            - root_mse (float): The root mean squared error.
            - pvals (list of float): p-values for coefficients.
            - tstats (list of float): t-statistics for coefficients.
            - cis (list of tuple): Confidence intervals for coefficients.

    Raises
    ------
    ValueError
        If the dimensions of Y, X, or Z are inconsistent or invalid.
    RuntimeWarning
        If the number of instruments (columns in Z) is not greater than the number of regressors (columns in X).

    Notes
    -----
    - The TSLS estimator is a classic instrumental variable estimator.
    - The function performs two stages:
        1. The first stage projects X onto the space spanned by Z (and W if provided).
        2. The second stage regresses Y on the fitted values from the first stage.
    - Additional statistics such as R-squared, adjusted R-squared, and F-statistics are calculated for model evaluation.
    - If the number of endogenous regressors is 1, first-stage statistics (R-squared and F-statistic) are also computed.

    Example
    -------
    >>> import numpy as np
    >>> from weak_instruments.tsls import TSLS
    >>> Y = np.array([1, 2, 3])
    >>> X = np.array([[1], [2], [3]])
    >>> Z = np.array([[1, 0], [0, 1], [1, 1]])
    >>> result = TSLS(Y, X, Z)
    >>> print(result.summary())
    """

    # Convert pandas DataFrames/Series to numpy arrays
    if hasattr(Y, "values"):
        Y = Y.values
    if hasattr(X, "values"):
        X = X.values
    if hasattr(Z, "values"):
        Z = Z.values
    if G is not None and hasattr(G, "values"):
        G = G.values
    if W is not None and hasattr(W, "values"):
        W = W.values

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
    
    #If X or Z is a single vector:
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

   
    logger.debug(f"Y has {Y.shape[0]} rows.\n")
    logger.debug(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
    logger.debug(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")


    # Drop constant columns from X
    constant_columns_X = np.all(np.isclose(X, X[0, :], atol=1e-8), axis=0)
    if np.any(constant_columns_X):  # Check if there are any constant columns
        logger.debug(f"X has constant columns. Dropping columns: {np.where(constant_columns_X)[0]}")
        X = X[:, ~constant_columns_X]  # Keep only non-constant columns

    # Drop constant columns from Z
    constant_columns_Z = np.all(np.isclose(Z, Z[0, :], atol=1e-8), axis=0)
    if np.any(constant_columns_Z):  # Check if there are any constant columns
        logger.debug(f"Z has constant columns. Dropping columns: {np.where(constant_columns_Z)[0]}")
        Z = Z[:, ~constant_columns_Z]  # Keep only non-constant columns

    logger.debug(f"X shape after dropping constant columns: {X.shape}")
    logger.debug(f"Z shape after dropping constant columns: {Z.shape}")
 

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
            raise ValueError(f"W must have the same number of rows as Y. Got G.shape[0] = {W.shape[0]} and Y.shape[0] = {N}.")
        X = np.hstack((X, W))
        Z = np.hstack((Z, W))
        logger.debug("Controls W have been added to both X and Z.\n")


    # Check dimensions
    if Y.shape[0] != X.shape[0] or Y.shape[0] != Z.shape[0]:
        raise ValueError("All input arrays must have the same number of rows.")
    
    # Get the pi hats
    X_hat = (Z @ np.linalg.inv(Z.T @ Z) @ Z.T) @ X


    # Get the beta hats
    beta_hat = np.linalg.inv(X_hat.T @ X_hat) @ (X_hat.T @ Y)

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ beta_hat)**2 * np.outer(X_hat[i], X_hat[i])
    robust_v = np.linalg.inv(X_hat.T @ X) @ midsum @ np.linalg.inv(X.T @ X_hat)


    #Lets do a hypothesis test that B1=0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = (beta_hat[i])/((robust_v[i,i])**.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = beta_hat[i] - t_crit_i * (robust_v[i,i])**.5
        ci_upper = beta_hat[i] + t_crit_i * (robust_v[i,i])**.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(float(t_stat_i))
        pvals.append(float(pval_i))
        cis.append(ci_i)

    #Grab the R^2 for the model:
    yfit = X @ beta_hat
    ybar = np.mean(Y)
    r2 = 1 - np.sum((Y-yfit)**2) / np.sum((Y-ybar)**2)
    
    #Overall F-stat for the model:
    q = X.shape[1]
    e = Y-yfit
    F = ((np.sum((yfit-ybar)**2)) / (q-1)) / ((e.T @ e)/(N-q))

    #Mean-square error:
    root_mse = float(((1/(N-q)) * (np.sum((Y - yfit)**2)))**.5)

    #Adjusted R2
    ar2 = 1 - (((1-r2)*(N-1))/(N-q))

    #Now, we can add some first stage statistics if the number of endogenous regressors is 1
    if X.shape[1] == 2: 
        X_fs = X[:,1]
        fs_fit = Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ X_fs
        xbar = np.mean(X_fs)

        #First Stage R2
        fs_r2 = 1 - np.sum((X_fs - fs_fit) ** 2) / np.sum((X_fs - xbar) ** 2)

        #First stage F-stat
        q_fs = Z.shape[1]
        e_fs = X_fs - fs_fit
        fs_F = ((np.sum((fs_fit - xbar) ** 2))/(q_fs-1))/((e_fs.T @ e_fs)/(N-q_fs))

    # Return the result
    return TSLSResult(beta=beta_hat, 
                      r_squared=r2, 
                      adjusted_r_squared=ar2, 
                      f_stat=F, 
                      standard_errors=robust_v, 
                      root_mse=root_mse, 
                      pvals=pvals, 
                      tstats=tstats, 
                      cis=cis)