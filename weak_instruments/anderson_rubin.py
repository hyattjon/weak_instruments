# Jackknife Anderson-Rubin tests for many weak IV inference
import numpy as np
from scipy.stats import norm
import logging
from repo import *

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)


class ARTestResult:
    def __init__(self, ar_stat: float, p_val: float):
        self.ar_stat = ar_stat
        self.p_val = p_val

    def __getitem__(self, key: str):
        if key == 'ar_stat':
            return self.ar_stat
        elif key == 'p_val':
            return self.p_val
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'ar_stat' and 'p_val'.")

    def __repr__(self):
        return f"ARTestResult(ar_stat={self.ar_stat}, p_val={self.p_val})"


def ar_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, b: np.ndarray, talk: bool = False) -> ARTestResult:
    """
    Calculates the Jackknife Anderson-Rubin test with cross-fit variance from Mikusheva and Sun (2022).

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.
        b (np.ndarray): A 1-D numpy array of the parameter values to test.
        talk (bool): If True, provides detailed output for debugging purposes. Default is False.

    Returns:
        ARTestResult: A custom result object containing the AR test statistic and p-value.
    """
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