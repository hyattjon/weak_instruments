#SJIVE
import numpy as np
import logging
from numpy.typing import NDArray
from scipy.stats import t
from scipy.optimize import minimize
from repo import *

def sjive(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], talk: bool = False):
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
    
    return result.x

data = np.loadtxt('new_ijive.csv', delimiter=',', skiprows=1)
z1 = data[:, 0].reshape(-1,1)
z2 = data[:, 1].reshape(-1,1)
x1 = data[:, 2].reshape(-1,1)
W = data[:, 3].reshape(-1,1)
y = data[:, 4]

X = np.hstack((x1))
Z = np.hstack((z1,z2))

bhat = sjive(y,X,Z)

print(bhat)
