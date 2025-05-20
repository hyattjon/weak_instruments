# Jackknife Anderson-Rubin tests for many weak IV inference
import numpy as np
from scipy.stats import norm
import logging


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)


class ARTestResult:
    """
    Stores results for the Anderson-Rubin (AR) test.

    Attributes
    ----------
    ar_stat : float
        The Anderson-Rubin test statistic.
    p_val : float
        The p-value for the test statistic.

    Methods
    -------
    summary()
        Prints a summary of the Anderson-Rubin test results in a tabular format.
    __getitem__(key)
        Allows dictionary-like access to ARTestResult attributes.
    __repr__()
        Returns a string representation of the ARTestResult object.
    """
    def __init__(self, ar_stat: float, p_val: float):
        self.ar_stat = ar_stat
        self.p_val = p_val

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to ARTestResult attributes.

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
        if key == 'ar_stat':
            return self.ar_stat
        elif key == 'p_val':
            return self.p_val
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'ar_stat' and 'p_val'.")

    def summary(self):
        """
        Prints a summary of the Anderson-Rubin test results in a tabular format.
        """
        import pandas as pd

        summary_df = pd.DataFrame({
            "AR Statistic": [self.ar_stat],
            "P-value": [self.p_val]
        })

        print("\nAnderson-Rubin Test Results")
        print("=" * 50)
        print(summary_df.round(6).to_string(index=False))
        print("=" * 50)

    def __repr__(self):
        """
        Returns a string representation of the ARTestResult object.
        """
        return f"ARTestResult(ar_stat={self.ar_stat}, p_val={self.p_val})"


def ar_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, b: np.ndarray, talk: bool = False) -> ARTestResult:
    """
    Calculates the Jackknife Anderson-Rubin (AR) test with cross-fit variance as described in Mikusheva and Sun (2022).

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
    ARTestResult
        An object containing the following attributes:
            - ar_stat (float): The Anderson-Rubin test statistic.
            - p_val (float): The p-value for the test statistic.

    Raises
    ------
    ValueError
        If the dimensions of Y, X, Z, or b are inconsistent or invalid.

    Notes
    -----
    - The Anderson-Rubin test is a robust inference method for instrumental variables models, particularly in the presence of many or weak instruments.
    - This implementation uses a jackknife approach with cross-fit variance estimation as recommended by Mikusheva and Sun (2022).
    - The function computes the AR test statistic and its p-value under the null hypothesis that the parameter vector b is the true value.
    - The test is robust to weak identification and is valid even when the number of instruments is large relative to the sample size.

    Example
    -------
    >>> import numpy as np
    >>> from weak_instruments.anderson_rubin import ar_test
    >>> Y = np.array([1, 2, 3])
    >>> X = np.array([[1], [2], [3]])
    >>> Z = np.array([[1, 0], [0, 1], [1, 1]])
    >>> b = np.array([0.5])
    >>> result = ar_test(Y, X, Z, b)
    >>> print(result)
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

    N, K = X.shape

    # Get the model residuals at b
    e_0 = Y - X @ b

    # Get the projection matrix (P) and residual maker matrix (M)
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    M = np.eye(N) - P

    # Get the sum part of the AR
    ar_sum = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                ar_sum += np.sum(P[i, j] * e_0[i] * e_0[j])

    logger.debug(f"AR sum: {ar_sum}")

    # Let's get the phi hat
    phi_hat = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                denom = M[i, i] * M[j, j] + M[i, j]**2
                if denom != 0:
                    phi_hat += (2 / K) * (P[i, j] ** 2 / denom) * (e_0[i] * (M @ e_0)[i] * e_0[j] * (M @ e_0)[j])

    logger.debug(f"Phi hat: {phi_hat}")

    # Compute AR statistic
    ar_stat = ar_sum * (np.sqrt(K) * np.sqrt(phi_hat))
    logger.debug(f"AR statistic: {ar_stat}")

    # Compute p-value
    p_val = 2 * (1 - norm.cdf(abs(ar_stat)))
    logger.debug(f"P-value: {p_val}")

    if talk:
        logger.info(f"AR Statistic: {ar_stat}")
        logger.info(f"P-value: {p_val}")

    return ARTestResult(ar_stat=ar_stat, p_val=p_val)