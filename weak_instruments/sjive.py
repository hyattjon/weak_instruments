#SJIVE
import numpy as np
import logging
from numpy.typing import NDArray
from scipy.stats import t
from scipy.optimize import minimize
import logging

# Set up the logger This helps with error outputs and stuff. We can use this instead of printing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  
handler.setFormatter(formatter)
logger.addHandler(handler)


class SJIVEResult:
    """
    Stores results for the SJIVE estimator.

    Attributes
    ----------
    beta : NDArray[np.float64]
        Estimated coefficients for the SJIVE model.
    objective_value : float
        Value of the minimized objective function at the solution.
    converged : bool
        Whether the optimization converged.
    n_iter : int
        Number of iterations performed by the optimizer.
    message : str
        Optimizer status message.
    """

    def __init__(
        self,
        beta: NDArray[np.float64],
        objective_value: float,
        converged: bool,
        n_iter: int,
        message: str
    ):
        self.beta = beta
        self.objective_value = objective_value
        self.converged = converged
        self.n_iter = n_iter
        self.message = message

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to SJIVEResult attributes.

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
        elif key == 'objective_value':
            return self.objective_value
        elif key == 'converged':
            return self.converged
        elif key == 'n_iter':
            return self.n_iter
        elif key == 'message':
            return self.message
        else:
            raise KeyError(
                f"Invalid key '{key}'. Valid keys are 'beta', 'objective_value', 'converged', 'n_iter', or 'message'."
            )

    def __repr__(self):
        """
        Returns a string representation of the SJIVEResult object.
        """
        return (f"SJIVEResult(beta={self.beta}, objective_value={self.objective_value}, "
                f"converged={self.converged}, n_iter={self.n_iter}, message='{self.message}')")

    def summary(self):
        """
        Prints a summary of the SJIVE results.
        """
        import pandas as pd
        summary_df = pd.DataFrame({
            "Coefficient": self.beta.flatten()
        })
        print("\nSJIVE Regression Results")
        print("=" * 50)
        print(summary_df.round(6).to_string(index=False))
        print("-" * 50)
        print(f"Objective value: {self.objective_value:.6f}")
        print(f"Converged: {self.converged}")
        print(f"Iterations: {self.n_iter}")
        print(f"Optimizer message: {self.message}")
        print("=" * 50)


def SJIVE(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    Z: NDArray[np.float64],
    W: NDArray[np.float64] | None = None,
    G: NDArray[np.float64] | None = None,
    talk: bool = False
) -> NDArray[np.float64]:
    """
    Calculates the Smoothed Jackknife Instrumental Variables Estimator (SJIVE).

    Parameters
    ----------
    Y : NDArray[np.float64]
        A 1-D numpy array of the dependent variable (N,).
    X : NDArray[np.float64]
        A 2-D numpy array of the endogenous regressors (N, L).
    Z : NDArray[np.float64]
        A 2-D numpy array of the instruments (N, K).
    talk : bool, optional
        If True, provides detailed output for debugging purposes. Default is False.

    Returns
    -------
    NDArray[np.float64]
        The estimated coefficients for the SJIVE model.

    Raises
    ------
    RuntimeError
        If the optimization fails to converge.

    Notes
    -----
    - The SJIVE estimator is a smoothed version of the Jackknife IV estimator, designed for improved finite-sample properties.
    - The estimator solves a nonlinear objective function using numerical optimization.
    - The function returns only the estimated coefficients; standard errors and inference are not provided by default.

    Example
    -------
    >>> import numpy as np
    >>> from weak_instruments.sjive import sjive
    >>> Y = np.random.randn(100)
    >>> X = np.random.randn(100, 1)
    >>> Z = np.random.randn(100, 2)
    >>> bhat = sjive(Y, X, Z)
    >>> print(bhat)
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

    U = Z @ np.linalg.inv(Z.T @ Z)
    P = Z.T
    D = np.diag(np.diag(U@P))
    I = np.eye(Y.shape[0])
    Del = U@P@D@np.linalg.inv(I - D) @ U @ P - .5 * U @ P @ D @ np.linalg.inv(I-D) - .5 * D @ np.linalg.inv(I-D) @ U @ P
    B = (I-U@P) @ D @ np.linalg.inv(I-D) @ (I-U@P)
    A = U@P + Del 
    C = A - B
    
    #Do the optimization
    def objective(beta: NDArray[np.float64]) -> float:
        residual = Y - X @ beta
        num = residual.T @ C @ residual
        denom = residual.T @ B @ residual
        return float(num / denom)
    
    
    # Initial guess: OLS beta
    beta_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    # Minimize the objective function
    result = minimize(objective, beta_ols, method='BFGS')

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    return SJIVEResult(
        beta=result.x,
        objective_value=result.fun,
        converged=result.success,
        n_iter=result.nit,
        message=result.message
    )

"""data = np.loadtxt('new_ijive.csv', delimiter=',', skiprows=1)
z1 = data[:, 0].reshape(-1,1)
z2 = data[:, 1].reshape(-1,1)
x1 = data[:, 2].reshape(-1,1)
W = data[:, 3].reshape(-1,1)
y = data[:, 4]

X = np.hstack((x1))
Z = np.hstack((z1,z2))

bhat = sjive(y,X,Z)

print(bhat)
"""