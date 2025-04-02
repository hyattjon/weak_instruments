# Jackknife Anderson-Rubin tests for many weak IV inference 
import numpy as np
from scipy.stats import norm

def ar_test(Y: np.ndarray , X: np.ndarray, Z: np.ndarray, b) -> np.ndarray:
    """
    Calculates the Jackknife Anderson-Rubin test with cross-fit variance from Mikusheva and Sun (2022).

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.

    Returns:
        np.ndarray: A 1-D numpy array with the test statistic and pvalue.
    """ 

    N, K = X.shape

    # Get the model residuals at b:
    e_0 = Y - X @ b

    # Get the projection matrix (P) and residual maker matrix (M):
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    M = np.eye(N) - P

    # Get the sum part of the AR:
    ar_sum = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                ar_sum += np.sum(P[i, j] * e_0[i] * e_0[j])

    # Let's get the phi hat:
    phi_hat = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                denom = M[i, i] * M[j, j] + M[i, j]**2
                if denom != 0:
                    phi_hat += (2 / K) * (P[i, j] ** 2 / denom) * (e_0[i] * (M @ e_0)[i] * e_0[j] * (M @ e_0)[j])

    # Compute AR statistic:
    ar_stat = ar_sum * (np.sqrt(K) * np.sqrt(phi_hat))

    # Compute p-value:
    p_val = 2 * (1 - norm.cdf(abs(ar_stat)))

    return ar_stat, p_val