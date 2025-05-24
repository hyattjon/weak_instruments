import numpy as np
from scipy.stats import chi2
import logging


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)


class LMTestResult:
    """
    Stores results for the Jackknife Lagrange Multiplier (LM) test.

    Attributes
    ----------
    lm_stat : float
        The LM test statistic.
    p_val : float
        The p-value for the test statistic.

    Methods
    -------
    summary()
        Prints a summary of the LM test results in a tabular format.
    __getitem__(key)
        Allows dictionary-like access to LMTestResult attributes.
    __repr__()
        Returns a string representation of the LMTestResult object.
    """

    def __init__(self, lm_stat: float, p_val: float):
        """
        Initializes the LMTestResult object.

        Parameters
        ----------
        lm_stat : float
            The LM test statistic.
        p_val : float
            The p-value for the test statistic.
        """
        self.lm_stat = lm_stat
        self.p_val = p_val

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to LMTestResult attributes.

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
        if key == 'lm_stat':
            return self.lm_stat
        elif key == 'p_val':
            return self.p_val
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'lm_stat' and 'p_val'.")

    def __repr__(self):
        return f"LMTestResult(lm_stat={self.lm_stat}, p_val={self.p_val})"

    def summary(self):
        """
        Prints a summary of the LM test results in a tabular format.
        """
        import pandas as pd

        summary_df = pd.DataFrame({
            "LM Statistic": [self.lm_stat],
            "P-value": [self.p_val]
        })

        print("\nLagrange Multiplier (LM) Test Results")
        print("=" * 50)
        print(summary_df.round(6).to_string(index=False))
        print("=" * 50)

def lm_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, b: np.ndarray, talk: bool = False) -> LMTestResult:
    """
    Calculates the Jackknife Lagrange Multiplier (LM) test for weak instrument robust inference.

    Parameters
    ----------
    Y : np.ndarray
        A 1-D numpy array of the dependent variable (N,).
    X : np.ndarray
        A 2-D numpy array of the endogenous regressors (N, L).
    Z : np.ndarray
        A 2-D numpy array of the instruments (N, K), where K > L.
    b : np.ndarray
        A 1-D numpy array of the parameter values to test (L,).
    talk : bool, optional
        If True, provides detailed output for debugging purposes. Default is False.

    Returns
    -------
    LMTestResult
        An object containing the following attributes:
            - lm_stat (float): The LM test statistic.
            - p_val (float): The p-value for the test statistic.

    Raises
    ------
    ValueError
        If the dimensions of Y, X, Z, or b are inconsistent or invalid.

    Notes
    -----
    - The LM test is a robust inference method for instrumental variables models, particularly in the presence of many or weak instruments.
    - This implementation uses a jackknife approach for variance estimation.
    - The function computes the LM test statistic and its p-value under the null hypothesis that the parameter vector b is the true value.
    - The test is robust to weak identification and is valid even when the number of instruments is large relative to the sample size.

    Example
    -------
    >>> import numpy as np
    >>> from weak_instruments.lagrange_multiplier import lm_test
    >>> Y = np.random.randn(100)
    >>> X = np.random.randn(100, 1)
    >>> Z = np.random.randn(100, 2)
    >>> b = np.array([0.5])
    >>> result = lm_test(Y, X, Z, b)
    >>> result.summary()
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


    # Adjust logging level based on the `talk` parameter
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

    N, d = X.shape

    # Get the model residuals at b
    u_0 = Y - X @ b

    # Get the projection matrix and residual maker
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    P_star = P - np.diag(np.diag(P))
    M = np.eye(N) - P

    # Sigma_0
    sig_0 = np.diag(u_0**2)

    # Get the first term in Psi
    term1 = X.T @ P_star @ sig_0 @ P_star @ X

    # Now let's get the second term
    term2 = np.zeros((d, d))
    for i in range(N):
        for j in range(N):
            term2 += np.outer(X[i], X[j]) * u_0[i] * u_0[j] * (P_star[i, j] ** 2)

    # Time for the finished product
    psi_hat = term1 + term2

    # Compute the test statistic
    jlm_stat = (u_0.T @ P_star @ X) @ np.linalg.inv(psi_hat) @ (X.T @ P_star @ u_0)

    # Compute p-value
    jlm_pval = 1 - chi2.cdf(jlm_stat, df=d)

    logger.debug(f"LM Statistic: {jlm_stat}")
    logger.debug(f"P-value: {jlm_pval}")

    if talk:
        logger.info(f"LM Statistic: {jlm_stat}")
        logger.info(f"P-value: {jlm_pval}")

    return LMTestResult(lm_stat=jlm_stat, p_val=jlm_pval)