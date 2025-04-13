import numpy as np
from scipy.stats import t


def HFUL(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk: bool = False, colnames=None) -> np.ndarray:
    N = Y.shape[0]
    Xbar = np.hstack([Y, X])

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
        #f_sum += (np.outer(X_dot[i], X_dot[i]) - diags[i] * np.outer(X_hat[i],  X_dot[i]) - diags[i] * np.outer(X_dot[i], X_hat[i]))*eps_hat[i]**2
        f_sum += (np.outer(X_dot[i], X_dot[i]) - diags[i] * np.outer(X_hat[i], X_dot[i]) - diags[i] * np.outer(X_dot[i], X_hat[i]))*eps_hat[i]**2

    sig_sum1 = 0
    # Precompute xi_ei for all i
    xi_ei = np.array([X_hat[i] * eps_hat[i] for i in range(N)])

    for i in range(N):
        for j in range(N):
            zij = np.dot(Z_tild[i], Z_tild[j])  # scalar
            sig_sum1 += np.outer(xi_ei[i], xi_ei[j]) * zij
    Sig_hat = f_sum + sig_sum1 


    V_hat = np.linalg.inv(H_hat) @ Sig_hat @ np.linalg.inv(H_hat)

    dof = N - X.shape[1]
    t_crit = t.ppf(0.975, df=dof)
    # Store results in lists (or dicts, if you prefer named access)
    se_list = []
    tstat_list = []
    pval_list = []
    ci_list = []

    for i in range(X.shape[1]):
        se_i = np.sqrt(V_hat[i, i])
        tstat_i = betas[i] / se_i
        pval_i = 2 * (1 - t.cdf(np.abs(tstat_i), df=dof))
        ci_lower_i = betas[i] - t_crit * se_i
        ci_upper_i = betas[i] + t_crit * se_i

        se_list.append(se_i)
        tstat_list.append(tstat_i)
        pval_list.append(pval_i)
        ci_list.append((ci_lower_i, ci_upper_i))

    if talk:
        print("HFUL Betas:", betas.flatten())
        print("HFUL Var (original):", V_hat)
        for i in range(X.shape[1]):
            label = colnames[i] if colnames is not None else f"beta_{i}"
            print(f"\nCoefficient: {label}")
            print(f"  Estimate: {betas[i][0]}")
            print(f"  SE: {se_list[i]}")
            print(f"  t-stat: {tstat_list[i]}")
            print(f"  p-value: {pval_list[i]}")
            print(f"  95% CI: {ci_list[i]}")

    
    return betas

data = np.loadtxt('jive_data.csv', delimiter=',', skiprows=1)
y = data[:, 1][:, np.newaxis]
x = data[:, 0]
z = data[:, 2]

ones = np.ones((data.shape[0], 1))

X = np.hstack((ones, x[:, np.newaxis]))
Z = np.hstack((ones, z[:, np.newaxis]))

colnames = ["constant", "x"]
HFUL(y,X,Z, talk=True, colnames=colnames)

'''# First variance term (f_sum)
    f_sum = 0
    for i in range(N):
        f_sum += (X_dot[i].T @ X_dot[i] - diags[i] * X_hat[i].T @ X_dot[i] - diags[i] * X_dot[i].T @ X_hat[i]) * eps_hat[i] ** 2
        

    #Like the MATA:
    K = Z.shape[1]
    sigma_4 = 0
    for k in range(K):
        sigma_3 = 0
        for l in range(K):
            sigma_21 = 0
            sigma_22 = 0
            for i in range(N):
                sigma_21 += Z_tild[i, k] * Z_tild[i, l] * (X_hat[i].T * eps_hat[i])
            for j in range(N):
                sigma_22 += Z[j, k] * Z[j, l] * (X_hat[j].T * eps_hat[j]) 
            sigma_2 = sigma_21 @ sigma_22.T
            sigma_3 += sigma_2
        sigma_4 += sigma_3
    
    Sig_hat = f_sum + sigma_4'''