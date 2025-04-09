#HFUL

import numpy as np

def HFUL(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk:bool = False) -> np.ndarray:
    Xbar = np.hstack([Y,X])
    N = Y.shape[0]
    P = Z  @ np.linalg.inv(Z.T @ Z) @ Z.T
    diags = np.diag(P)

    xbarxbar=0
    for i in range(N):
        xbarxbar += diags[i] * np.outer(Xbar[i], Xbar[i])
    
    mat = np.linalg.inv(Xbar.T @ Xbar) @ (Xbar.T @ P @ Xbar - xbarxbar)
    eigs = np.linalg.eigvals(mat)
    a_tild = np.min(eigs)

    a_hat = (a_tild - ((1-a_tild)/N))/(1-((1-a_tild)/N))

    xy = 0
    for i in range(N):
        xy += diags[i] * X[i].reshape(-1, 1) * Y[i]
    xy.reshape(-1, 1)

    xx = 0
    for i in range(N):
        xx += diags[i] * np.outer(X[i], X[i])

    left = np.linalg.inv(X.T @ P @ X - xx - a_hat * X.T @ X)
    right = (X.T @ P @ Y - xy - a_hat * X.T @ Y)

    betas = left @ right

    #Now, lets work on getting standard errors:

    H_hat = X.T @ P @ X - xx - a_hat * X.T @ X

    eps_hat = Y - X@betas 
    gam_hat = (X.T @ eps_hat)/(eps_hat.T @ eps_hat)
    X_hat = X - eps_hat @ gam_hat.T
    X_dot = P@X_hat
    Z_tild = Z @ np.linalg.inv(Z.T @ Z)

    f_sum = 0
    for i in range(N):
        f_sum += (np.outer(X_dot[i], X_dot[i]) - diags[i] * np.outer(X_hat[i],  X_dot[i]) - diags[i] * np.outer(X_dot[i], X_hat[i]))*eps_hat[i]**2

    sig_sum1 = 0
    sig_sum2 = 0
    sig_sum3 = 0
    K = Z.shape[1]
    for k in range(K):
        for l in range(K):
            for i in range(N):
               sig_sum2 += Z_tild[i,k] * Z_tild[i,l] * X_hat[i].T * eps_hat[i]
            for j in range(N):
                sig_sum3 += Z[j,k] * Z[j,l] * X_hat[j].T * eps_hat[j] 
            
            sig_sum1 += sig_sum2 @ sig_sum3.T
    
    Sig_hat = f_sum + sig_sum1

    print(H_hat.shape)
    print(Sig_hat.shape)
    
    V_hat = np.linalg.inv(H_hat) @ Sig_hat @ np.linalg.inv(H_hat)
    se1 = V_hat[0,0]**.5
    se2 = V_hat[1,1]**.5

    print("HFUL Betas:", betas)
    print("HFUL Var:", V_hat)
    print("SE_1:", se1)
    print("SE_2:", se2)

    return betas
