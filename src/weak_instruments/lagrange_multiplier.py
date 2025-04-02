import numpy as np
from scipy.stats import chi2

def lm_test(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, b) -> np.ndarray:
    """
    Calculates the Jackknife Lagrange-multiplier test.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.

    Returns:
        np.ndarray: A 1-D numpy array with the test statistic and p-value.
    """ 
    N, d = X.shape

    # Get the model residuals at b:
    u_0 = Y - X @ b

    # Get the projection matrix and residual maker:
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    P_star = P - np.diag(np.diag(P))
    M = np.eye(N) - P

    # Sigma_0
    sig_0 = np.diag(u_0**2)

    # Get the first term in Psi:
    term1 = X.T @ P_star @ sig_0 @ P_star @ X

    # Now let's get the second term:
    term2 = np.zeros((d, d))
    for i in range(N):
        for j in range(N):
            term2 += np.outer(X[i], X[j]) * u_0[i] * u_0[j] * (P_star[i, j] ** 2)    

    # Time for the finished product:
    psi_hat = term1 + term2 

    # Compute the test statistic:
    jlm_stat = (u_0.T @ P_star @ X) @ np.linalg.inv(psi_hat) @ (X.T @ P_star @ u_0)

    # Compute p-value:
    jlm_pval = 1 - chi2.cdf(jlm_stat, df=d)    

    return jlm_stat, jlm_pval