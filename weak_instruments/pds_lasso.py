# Import necessary libraries
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class LassoResult:
    """
    A class to hold the results of the Lasso regression.
    
    Attributes
    ----------
    coefficients : NDArray[np.float64]
        The estimated coefficients from the Lasso regression.
    intercept : float
        The intercept term from the Lasso regression.
    mse : float
        The mean squared error of the model.
    n_iter : int
        The number of iterations taken by the Lasso algorithm.
    converged : bool
        Whether the algorithm converged successfully.
    message : str
        A message indicating the convergence status.
    """

    def __init__(self, coefficients, intercept, mse, n_iter, converged, message):
        self.coefficients = coefficients
        self.intercept = intercept
        self.mse = mse
        self.n_iter = n_iter
        self.converged = converged
        self.message = message


def pds_lasso(Y, X, Z, d, alpha=0.1, max_iter=1000):
    """
    Post Double Selection (PDS) Lasso regression.
    """
    from sklearn.linear_model import LassoCV

    # Lasso D on X and Z
    lasso_d = LassoCV(alphas=[alpha], max_iter=max_iter, cv=5)
    lasso_d.fit(np.column_stack(Z,X), d)
    d_hat = lasso_d.predict(Z)

    # Lasso Y on X and Z
    lasso_y = LassoCV(alphas=[alpha], max_iter=max_iter, cv=5)
    lasso_y.fit(np.column_stack((X, Z)), Y)
    Y_hat = lasso_y.predict(np.column_stack((X, Z)))


    # Keep only the Z's with non-zero coefficients from lasso_d and lasso_y
    selected_Z = [x for x in np.where(lasso_d.coef_[:Z.shape[1]] != 0)[0] if x in np.where(lasso_y.coef_[:Z.shape[1]] != 0)[0]]

    # 2SLS using the selected Z's


