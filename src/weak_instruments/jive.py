# JIVE (and JIVE-related estimators---there are a lot of these in the literature)
import numpy as np
from scipy.stats import t

def JIVE1(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk:bool = False) -> np.ndarray:
    """
    Calculates the JIVE1 estimator using a two-pass approach reccommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.
        talk (bool) : If True prints comments along the way for clearer explanation and verification.

    Returns:
        np.ndarray: A 1-D numpy array of the JIVE1 estimates (L x 1).
    """

    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError("Y must be a one-dimensional array.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError("Z must be at least a one-dimensional array.")
    
    # Check that Y and X and Z have N columns. Base off of Y.shape[0]
    N = Y.shape[0]
    if X.shape[0] != N:
        raise ValueError("X and Y must have the same number of rows.")
    if Z.shape[0] != N:
        raise ValueError("Z and Y must have the same number of rows.")
    if Z.shape[1] <= X.shape[1]:
        print(f"Normally this estimator is used when Z has more columns than X. In this case Z has {Z.shape[1]} columns and X has {X.shape[1]} columns.")

    if talk:
        print(f"Y has {Y.shape[0]} rows and {Y.shape[1]} columns.\n")
        print(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
        print(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")

    # First pass to get fitted values and leverage
    fit = np.dot(np.dot(np.dot(Z ,  np.linalg.inv(np.dot(Z.T, Z))) , Z.T) , X)
    if talk:
        print(f'Fitted values obtained... \n')
    # Get the main diagonal from the projection matrix
    leverage = (np.diag(np.dot(np.dot(Z , np.linalg.inv(np.dot(Z.T , Z))) , Z.T)))

    if talk:
        print(f'Leverage obtained... \n')
    # Reshape to get an Nx1 vector
    leverage = leverage.reshape(-1, 1)

    # Second pass to remove the ith row for unbiased estimates
    X_jive = (fit - leverage * X)/(1-leverage) 
    if talk:
        print(f'Second pass complete...')

    # Here is the optimal estimate
    beta_jive1 = np.linalg.inv(X_jive.T @ X) @ X_jive.T @ Y 

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ beta_jive1)**2 * np.outer(X_jive[i], X_jive[i])
    robust_v = np.linalg.inv(X_jive.T @ X) @ midsum @ np.linalg.inv(X.T @ X_jive)

    #Lets do a hypothesis test that B1=0

    #First, get degrees of freedom
    dof = N - X.shape[1]
    #Get the t-stat
    t_stat = (beta_jive1[1])/((robust_v[1,1])**.5)
    #Get the p-val from t-dist w specified degrees of freedom
    pval = 2 * (1 - t.cdf(np.abs(t_stat), df=dof))
    #Get the critical value (95% level)
    t_crit = t.ppf(0.975, df=dof)
    #Confidence interval
    ci_lower = beta_jive1[1] - t_crit * (robust_v[1,1])**.5
    ci_upper = beta_jive1[1] + t_crit * (robust_v[1,1])**.5

    ci = (ci_lower, ci_upper)


    #WITH ONE ENDOGENOUS VARIABLE! Let's also get and report some relevant things from the first stage:
    #First, the first stage R2:
    x_end = X[:,1]
    xfit = Z@np.linalg.inv(Z.T @ Z) @ Z.T @ x_end
    xbar = np.mean(x_end)

    fs_r2 = 1 - np.sum((x_end - xfit) ** 2) / np.sum((x_end - xbar) ** 2)  # R²

    #Now, lets get the first stage F-stat
    q_fs = Z.shape[1]
    e_fs = x_end - xfit
    F_fs = ((np.sum((xfit - xbar) ** 2))/(q_fs-1))/((e_fs.T @ e_fs)/(N-q_fs))



    #Get the r2 for the model:
    yfit = X @ np.linalg.inv(X_jive.T @ X) @ X_jive.T @ Y 
    ybar = np.mean(y)
    r2 = 1 - np.sum((y-yfit)**2) / np.sum((y-ybar)**2)

    #Overall F-stat for the model:
    q = X.shape[1]
    e = y-yfit
    F = ((np.sum((yfit-ybar)**2)) / (q-1)) / ((e.T @ e)/(N-q))
    

    print("JIVE2 Estimates:\n", beta_jive1)
    print("B1 se:\n", (robust_v[1,1])**.5)
    print("B1 95% CI:", ci)
    print("First Stage R2:", fs_r2)
    print(f"First Stage F({q_fs - 1}, {N - q_fs}):", F_fs)
    print("R2:", r2)
    print(f"F({q - 1}, {N - q}):", F)

    return beta_jive1


def JIVE2(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk:bool = False) -> np.ndarray:
    """
    Calculates the JIVE2 estimator using a two-pass approach reccommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.
        talk (bool) : If True prints comments along the way for clearer explanation and verification.

    Returns:
        np.ndarray: A 1-D numpy array of the JIVE2 estimates (L x 1).
    """

    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError("Y must be a one-dimensional array.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError("Z must be at least a one-dimensional array.")
    # Check that Y and X and Z have N columns. Base off of Y.shape[0]
    N = Y.shape[0]
    if X.shape[0] != N:
        raise ValueError("X and Y must have the same number of rows.")
    if Z.shape[0] != N:
        raise ValueError("Z and Y must have the same number of rows.")
    if Z.shape[1] <= X.shape[1]:
        print(f"Normally this estimator is used when Z has more columns than X. In this case Z has {Z.shape[1]} columns and X has {X.shape[1]} columns.")

    if talk:
        print(f"Y has {Y.shape[0]} rows and {Y.shape[1]} columns.\n")
        print(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
        print(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")


    ### First pass to get fitted values and leverage ###
    # Getting fitted values
    fit = np.dot(np.dot(np.dot(Z ,  np.linalg.inv(np.dot(Z.T, Z))) , Z.T) , X)
    if talk:
        print(f'Obtained fitted values... \n')
    # Getting leverage
    leverage = (np.diag(np.dot(np.dot(Z , np.linalg.inv(np.dot(Z.T , Z))) , Z.T)))
    if talk:
        print(f'Obtained Leverage...')
    # Making leverage an Nx1 vector
    leverage = leverage.reshape(-1, 1)

    if talk:
        print(f'First pass complete...')


    ### Second pass to remove ith row and reduce bias ###
    X_jive2 = (fit - leverage * X)/(1-(1/N)) 
    if talk:
        print(f'Second pass complete... /n')
    beta_jive2 = np.linalg.inv(X_jive2.T @ X) @ X_jive2.T @ Y 

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    midsum = 0
    for i in range(N):
        midsum += (Y[i] - X[i] @ beta_jive2)**2 * np.outer(X_jive2[i], X_jive2[i])
    robust_v = np.linalg.inv(X_jive2.T @ X) @ midsum @ np.linalg.inv(X.T @ X_jive2)

    #Lets do a hypothesis test that B1=0

    #First, get degrees of freedom
    dof = N - X.shape[1]
    #Get the t-stat
    t_stat = (beta_jive2[1])/((robust_v[1,1])**.5)
    #Get the p-val from t-dist w specified degrees of freedom
    pval = 2 * (1 - t.cdf(np.abs(t_stat), df=dof))
    #Get the critical value (95% level)
    t_crit = t.ppf(0.975, df=dof)
    #Confidence interval
    ci_lower = beta_jive2[1] - t_crit * (robust_v[1,1])**.5
    ci_upper = beta_jive2[1] + t_crit * (robust_v[1,1])**.5

    ci = (ci_lower, ci_upper)


    print("JIVE2 Estimates:\n", beta_jive2)
    print("B1 se:\n", (robust_v[1,1])**.5)
    print("B1 95% CI:", ci)

    return beta_jive2


data = np.loadtxt('jive_data.csv', delimiter=',', skiprows=1)
y = data[:, 1]
x = data[:, 0]
z = data[:, 2]

ones = np.ones((data.shape[0], 1))

X = np.hstack((ones, x[:, np.newaxis]))
Z = np.hstack((ones, z[:, np.newaxis]))


JIVE1(y,X,Z)