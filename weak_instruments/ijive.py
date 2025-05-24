import numpy as np
from numpy.typing import NDArray
from scipy.stats import t
import logging
import warnings


# Set up the logger This helps with error outputs and stuff. We can use this instead of printing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class IJIVEResult:
    """
    Stores results for the IJIVE estimator.

    Attributes
    ----------
    beta : NDArray[np.float64]
        Estimated coefficients for the IJIVE model.
    f_stat : float
        F-statistic of the model.
    r_squared : float
        R-squared value of the model.
    adjusted_r_squared : float
        Adjusted R-squared value of the model.
    root_mse : float
        Root mean squared error of the model.
    pvals : list of float
        p-values for the estimated coefficients.
    tstats : list of float
        t-statistics for the estimated coefficients.
    cis : list of tuple
        Confidence intervals for the estimated coefficients.
    """
    def __init__(self, 
                 beta: NDArray[np.float64],  
                 f_stat: NDArray[np.float64],
                 r_squared: np.float64, 
                 adjusted_r_squared: np.float64, 
                 root_mse: np.float64,
                 pvals: np.float64,
                 tstats: NDArray[np.float64],
                 cis: NDArray[np.float64]):
        self.beta = beta
        self.f_stat = f_stat
        self.r_squared = r_squared
        self.adjusted_r_squared = adjusted_r_squared
        self.root_mse = root_mse
        self.pvals = pvals
        self.tstats = tstats
        self.cis = cis

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to IJIVEResult attributes.

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
        elif key == 'root_mse':
            return self.root_mse
        elif key == 'pvals':
            return self.pvals
        elif key == 'tstats':
            return self.tstats
        elif key == 'cis':
            return self.cis
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'beta', 'r_squared', 'adjusted_r_squared', 'root_mse', 'pvals', 'tstats', or 'cis'.")

    def __repr__(self):
        return f"IJIVEResult(beta={self.beta}, r_squared={self.r_squared}, adjusted_r_squared={self.adjusted_r_squared}, root_mse={self.root_mse}, pvals={self.pvals}, tstats={self.tstats}, cis={self.cis})"

    def summary(self):
        """
        Prints a summary of the IJIVE results in a tabular format similar to statsmodels OLS and UJIVE1.
        """
        import pandas as pd
        import numpy as np

        summary_df = pd.DataFrame({
            "Coefficient": self.beta.flatten(),
            "Std. Error": [np.sqrt(ci[1] - ci[0]) / (2 * 1.96) if self.cis is not None else np.nan for ci in self.cis],
            "t-stat": self.tstats if self.tstats is not None else np.nan,
            "P>|t|": self.pvals if self.pvals is not None else np.nan,
            "Conf. Int. Low": [ci[0] for ci in self.cis] if self.cis is not None else np.nan,
            "Conf. Int. High": [ci[1] for ci in self.cis] if self.cis is not None else np.nan
        })

        print("\nIJIVE Regression Results")
        print("=" * 80)
        print(summary_df.round(6).to_string(index=False))
        print("-" * 80)
        print(f"R-squared: {self.r_squared:.6f}" if self.r_squared is not None else "R-squared: N/A")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.6f}" if self.adjusted_r_squared is not None else "Adjusted R-squared: N/A")
        print(f"F-statistic: {self.f_stat:.6f}" if self.f_stat is not None else "F-statistic: N/A")
        print(f"Root MSE: {self.root_mse:.6f}" if self.root_mse is not None else "Root MSE: N/A")
        print("=" * 80)


def IJIVE(Y: NDArray[np.float64],  X: NDArray[np.float64], Z: NDArray[np.float64], G: NDArray[np.float64] | None = None, W: NDArray[np.float64] | None = None, talk: bool = False):
    """
    Calculates the Instrumental Variable estimator using the IJIVE method.
    
    Parameters
    ----------
    Y : NDArray[np.float64]
        The dependent variable.
    W : NDArray[np.float64]
        The matrix of controls.
    X : NDArray[np.float64]
        The matrix of endogenous regressors.
    Z : NDArray[np.float64]
        The matrix of instruments.
    talk : bool, optional
        If True, prints additional information. Default is False.   

    Returns
    -------
    beta : NDArray[np.float64]
        The estimated coefficients.
    r2 : NDArray[np.float64]
        The R-squared value of the model.
    F : NDArray[np.float64]
        The F-statistic of the model.
    ar2 : NDArray[np.float64]
        The adjusted R-squared value of the model.
    root_mse : NDArray[np.float64]
        The root mean square error of the model.
    pvals : np.float64
        The p-values for the hypothesis tests.
    tstats : np.float64
        The t-statistics for the hypothesis tests.
    cis : NDArray[np.float64]
        The confidence intervals for the coefficients.  

    Example
    -------
    >>> Y = np.array([1, 2, 3])
    >>> W = np.array([[1, 0], [0, 1], [1, 1]])
    >>> X = np.array([[1], [2], [3]])
    >>> Z = np.array([[1], [2], [3]])
    >>> result = IJIVE(Y, W, X, Z)
    >>> print(result.beta)
    [0.5]
    
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

        
    # Set the logger level based on the talk parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

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
        

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if W is not None and W.ndim == 1:
        W = W.reshape(-1, 1)    

    N = Z.shape[0]
    
    ones = np.ones((N,1))
    X = np.hstack((ones,X))
    Z = np.hstack((ones, Z))   

    Z_tild = (np.eye(N) - W @ np.linalg.inv(W.T @ W) @ W.T)@Z
    #ZZ_t_inv = np.linalg.inv(Z_tild.T @ Z_tild)
    #U = Z @ ZZ_t_inv
    #P  = Z.T
    diags = np.diag(np.diag(Z_tild @ np.linalg.inv(Z_tild.T @ Z_tild) @ Z_tild.T))

    C_IJIVE = np.linalg.inv(np.eye(N) - diags) @ (Z_tild @ np.linalg.inv(Z_tild.T @ Z_tild) @ Z_tild.T - diags)

    bhat_IJIVE = np.linalg.inv(X.T @ C_IJIVE.T @ X) @ (X.T @ C_IJIVE.T @ Y)

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    X_est = C_IJIVE @ X
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ bhat_IJIVE)**2 * np.outer(X_est[i], X_est[i])
    robust_v = np.linalg.inv(X_est.T @ X) @ midsum @ np.linalg.inv(X.T @ X_est)


    #Lets do a hypothesis test that B1=0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = (bhat_IJIVE[i])/((robust_v[i,i])**.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = bhat_IJIVE[i] - t_crit_i * (robust_v[i,i])**.5
        ci_upper = bhat_IJIVE[i] + t_crit_i * (robust_v[i,i])**.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(t_stat_i)
        pvals.append(pval_i)
        cis.append(ci_i)  

    #Grab the R^2 for the model:
    yfit = X @ bhat_IJIVE
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

    return IJIVEResult(beta=bhat_IJIVE,
                       r_squared=r2,
                       f_stat=F,
                       adjusted_r_squared=ar2,
                       root_mse=root_mse,
                       pvals=pvals,
                       tstats=tstats,
                       cis=cis)