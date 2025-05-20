import numpy as np
from scipy.stats import norm
#from scipy.stats import t
import logging
from numpy.typing import NDArray
from scipy.linalg import eigvals


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)



class LIMLResult:
    """
    Stores results for the LIML estimator.

    Attributes
    ----------
    betas : NDArray[np.float64]
        Estimated coefficients for the LIML model.
    se_list : list of float
        Standard errors of the estimated coefficients.
    tstat_list : list of float
        t-statistics for the estimated coefficients.
    pval_list : list of float
        p-values for the estimated coefficients.
    ci_list : list of tuple
        Confidence intervals for the estimated coefficients.
    """
    def __init__(self, betas, se_list, tstat_list, pval_list, ci_list):
        self.betas = betas
        self.se_list = se_list
        self.tstat_list = tstat_list
        self.pval_list = pval_list
        self.ci_list = ci_list

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to LIMLResult attributes.

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
        if key == 'betas':
            return self.betas
        elif key == 'se_list':
            return self.se_list
        elif key == 'tstat_list':
            return self.tstat_list
        elif key == 'pval_list':
            return self.pval_list
        elif key == 'ci_list':
            return self.ci_list
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'betas', 'se_list', 'tstat_list', 'pval_list', or 'ci_list'.")

    def __repr__(self):
        return f"LIMLResult(betas={self.betas}, se_list={self.se_list}, tstat_list={self.tstat_list}, pval_list={self.pval_list}, ci_list={self.ci_list})"
    
    def summary(self):
        """
        Prints a summary of the LIML results in a tabular format similar to statsmodels OLS and UJIVE1.
        """
        import pandas as pd
        import numpy as np

        summary_df = pd.DataFrame({
            "Coefficient": self.betas.flatten(),
            "Std. Error": np.array(self.se_list) if self.se_list is not None else np.nan,
            "t-stat": np.array(self.tstat_list) if self.tstat_list is not None else np.nan,
            "P>|t|": np.array(self.pval_list) if self.pval_list is not None else np.nan,
            "Conf. Int. Low": [ci[0] for ci in self.ci_list] if self.ci_list is not None else np.nan,
            "Conf. Int. High": [ci[1] for ci in self.ci_list] if self.ci_list is not None else np.nan
        })

        print("\nLIML Regression Results")
        print("=" * 80)
        print(summary_df.round(6).to_string(index=False))
        print("=" * 80)


def LIML(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, G: NDArray[np.float64] | None = None, talk: bool = False, colnames=None) -> LIMLResult:
    """
    Calculates the Limited Information Maximum Likelihood (LIML) estimator for weak instrument robust inference.

    Parameters
    ----------
    Y : np.ndarray
        A 1-D or 2-D numpy array of the dependent variable (N,).
    X : np.ndarray
        A 2-D numpy array of the endogenous regressors (N, L).
    Z : np.ndarray
        A 2-D numpy array of the instruments (N, K).
    G : np.ndarray, optional
        A 2-D numpy array of additional controls (N, G). Default is None.
    talk : bool, optional
        If True, provides detailed output for debugging purposes. Default is False.
    colnames : list, optional
        List of column names for the coefficients. Default is None.

    Returns
    -------
    LIMLResult
        An object containing the following attributes:
            - betas (NDArray[np.float64]): The estimated coefficients for the model.
            - se_list (list of float): Standard errors for the estimated coefficients.
            - tstat_list (list of float): t-statistics for the estimated coefficients.
            - pval_list (list of float): p-values for the estimated coefficients.
            - ci_list (list of tuple): Confidence intervals for the estimated coefficients.

    Raises
    ------
    ValueError
        If the dimensions of Y, X, or Z are inconsistent or invalid.

    Notes
    -----
    - The LIML estimator is designed for robust inference in the presence of weak instruments.
    - The function computes coefficient estimates, standard errors, t-statistics, p-values, and confidence intervals.
    - Additional controls can be included via the G argument.

    Example
    -------
    >>> import numpy as np
    >>> from weak_instruments.liml import LIML
    >>> Y = np.random.randn(100)
    >>> X = np.random.randn(100, 1)
    >>> Z = np.random.randn(100, 2)
    >>> result = LIML(Y, X, Z)
    >>> result.summary()
    """
    
    N = Y.shape[0]

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if G is not None:
        if G.ndim == 1:
            G = G.reshape(-1, 1)
        X = np.hstack((X, G))
        Z = np.hstack((Z, G))

    ones = np.ones((N,1))
    X = np.hstack((ones, X))
    Z = np.hstack((ones, Z))

    if Y.ndim == 1:
        Y = Y.reshape(-1,1)

    YX = np.hstack([Y, X])

    # Adjust logging level based on the `talk` parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    #Set up the Projection Matrix
    U = Z @ np.linalg.inv(Z.T @ Z)
    P = Z.T

    #Find the LIML eigenvalue:
    A = YX.T @ YX
    B = YX.T @ (np.eye(N) - U@P) @ YX 

    eigs = eigvals(A,B)
    eigs = np.real(eigs)
    k_liml = np.min(eigs)

    #We have everything needed to compute the point estimates
    bhat_liml = np.linalg.inv(X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ X) @ (X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ Y)
    
    #Now, lets work on variance:
    eps = Y - X @ bhat_liml
    omega = np.diag(np.diag(eps))
    om_2 = omega @ omega
    bread = (X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ X)
    meat = (X.T @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ om_2 @ (np.eye(N) - k_liml*(np.eye(N) - U@P)) @ X)
    robust_var = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread)

    return LIMLResult(betas=bhat_liml)
