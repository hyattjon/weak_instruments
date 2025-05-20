import numpy as np
from scipy.stats import norm
#from scipy.stats import t
import logging
from numpy.typing import NDArray

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  
handler.setFormatter(formatter)
logger.addHandler(handler)


class HFULResult:
    """
    Stores results for the HFUL estimator.

    Attributes
    ----------
    betas : NDArray[np.float64]
        Estimated coefficients for the HFUL model.
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
        Allows dictionary-like access to HFULResult attributes.

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
        return f"HFULResult(betas={self.betas}, se_list={self.se_list}, tstat_list={self.tstat_list}, pval_list={self.pval_list}, ci_list={self.ci_list})"

    def summary(self):
        """
        Prints a summary of the HFUL results in a tabular format similar to statsmodels OLS.
        """
        import pandas as pd
        import numpy as np

        summary_df = pd.DataFrame({
            "Coefficient": self.betas.flatten(),
            "Std. Error": np.array(self.se_list),
            "t-stat": np.array(self.tstat_list),
            "P>|t|": np.array(self.pval_list),
            "Conf. Int. Low": [ci[0] for ci in self.ci_list],
            "Conf. Int. High": [ci[1] for ci in self.ci_list]
        })

        print("\nHFUL Regression Results")
        print("=" * 80)
        print(summary_df.round(6).to_string(index=False))
        print("=" * 80)

def HFUL(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, G: NDArray[np.float64] | None = None, talk: bool = False, colnames=None) -> HFULResult:
    """
    Calculates the HFUL estimator for weak instrument robust inference.

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
    HFULResult
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
    - The HFUL estimator is designed for robust inference in the presence of weak instruments.
    - The function computes coefficient estimates, standard errors, t-statistics, p-values, and confidence intervals.
    - Additional controls can be included via the G argument.

    Example
    -------
    >>> import numpy as np
    >>> from weak_instruments.HFUL import HFUL
    >>> Y = np.random.randn(100)
    >>> X = np.random.randn(100, 1)
    >>> Z = np.random.randn(100, 2)
    >>> result = HFUL(Y, X, Z)
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

    Xbar = np.hstack([Y, X])

    # Adjust logging level based on the `talk` parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    # Projection matrix and its diagonal
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    diags = np.diag(P)

    # Compute a_hat
    xbarxbar = sum(diags[i] * np.outer(Xbar[i], Xbar[i]) for i in range(N))
    mat = np.linalg.inv(Xbar.T @ Xbar) @ (Xbar.T @ P @ Xbar - xbarxbar)
    eigs = np.linalg.eigvals(mat)
    a_tild = np.min(eigs)
    a_hat = (a_tild - (1 - a_tild) / N) / (1 - (1 - a_tild) / N)

    # Beta estimator
    xy = sum(diags[i] * np.outer(X[i], Y[i]) for i in range(N))
    xx = sum(diags[i] * np.outer(X[i], X[i]) for i in range(N))
    H_hat = X.T @ P @ X - xx - a_hat * (X.T @ X)
    right = X.T @ P @ Y - xy - a_hat * (X.T @ Y)
    betas = np.linalg.inv(H_hat) @ right

    # Residuals and projection-corrected regressors
    eps_hat = Y - X @ betas
    gam_hat = (X.T @ eps_hat) / (eps_hat.T @ eps_hat)
    X_hat = X - np.outer(eps_hat, gam_hat)
    X_dot = P @ X_hat

    # Prepare Z_tild
    Z_tild = Z @ np.linalg.pinv(Z.T @ Z)
    f_sum = 0
    for i in range(N):
        f_sum += (np.outer(X_dot[i], X_dot[i]) - diags[i] * np.outer(X_hat[i], X_dot[i]) - diags[i] * np.outer(X_dot[i], X_hat[i])) * eps_hat[i]**2

    sig_sum1 = 0
    # Precompute xi_ei for all i
    xi_ei = np.array([X_hat[i] * eps_hat[i] for i in range(N)])

    for i in range(N):
        for j in range(N):
            zij = np.dot(Z_tild[i], Z_tild[j])  # scalar
            sig_sum1 += np.outer(xi_ei[i], xi_ei[j]) * zij
    Sig_hat = f_sum + sig_sum1

    V_hat = np.linalg.inv(H_hat) @ Sig_hat @ np.linalg.inv(H_hat)

    #dof = N - X.shape[1]
    #t_crit = t.ppf(0.975, df=dof)
    norm_crit = norm.ppf(0.975)

    # Store results in lists
    se_list = []
    tstat_list = []
    pval_list = []
    ci_list = []

    for i in range(X.shape[1]):
        se_i = np.sqrt(V_hat[i, i])
        tstat_i = betas[i] / se_i
        #pval_i = 2 * (1 - t.cdf(np.abs(tstat_i), df=dof))
        pval_i = 2 * (1 - norm.cdf(np.abs(tstat_i)))
        ci_lower_i = betas[i] - norm_crit * se_i
        ci_upper_i = betas[i] + norm_crit * se_i

        se_list.append(se_i)
        tstat_list.append(tstat_i)
        pval_list.append(pval_i)
        ci_list.append((ci_lower_i, ci_upper_i))

    if talk:
        logger.info("HFUL Betas:\n%s", betas.flatten())
        logger.info("HFUL Var (original):\n%s", V_hat)
        for i in range(X.shape[1]):
            label = colnames[i] if colnames is not None else f"beta_{i}"
            logger.info("\nCoefficient: %s", label)
            logger.info("  Estimate: %f", betas[i][0])
            logger.info("  SE: %f", se_list[i])
            logger.info("  t-stat: %f", tstat_list[i])
            logger.info("  p-value: %f", pval_list[i])
            logger.info("  95%% CI: (%f, %f)", ci_list[i][0], ci_list[i][1])

    return HFULResult(betas=betas,
                      se_list=se_list,
                      tstat_list=tstat_list,
                      pval_list=pval_list,
                      ci_list=ci_list)