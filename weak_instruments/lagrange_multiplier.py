import numpy as np
from scipy.stats import chi2
import logging
from repo import *

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)


class LMTestResult:
    def __init__(self, lm_stat: float, p_val: float):
        self.lm_stat = lm_stat
        self.p_val = p_val

    def __getitem__(self, key: str):
        if key == 'lm_stat':
            return self.lm_stat
        elif key == 'p_val':
            return self.p_val
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'lm_stat' and 'p_val'.")

    def __repr__(self):
        return f"LMTestResult(lm_stat={self.lm_stat}, p_val={self.p_val})"


def lm_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, b: np.ndarray, talk: bool = False) -> LMTestResult:
    """
    Calculates the Jackknife Lagrange-multiplier test.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.
        b (np.ndarray): A 1-D numpy array of the parameter values to test.
        talk (bool): If True, provides detailed output for debugging purposes. Default is False.

    Returns:
        LMTestResult: A custom result object containing the LM test statistic and p-value.
    """
    # Adjust logging level based on the `talk` parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

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