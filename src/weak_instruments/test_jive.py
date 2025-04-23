import numpy as np
from jive1 import *
from jive2 import *
import pandas as pd
"""
def JIVE2(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, talk:bool = False) -> np.ndarray:
    
    Calculates the JIVE2 estimator using a two-pass approach reccommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (np.ndarray): A 1-D numpy array of the dependent variable (N x 1).
        X (np.ndarray): A 2-D numpy array of the endogenous regressors (N x L).
        Z (np.ndarray): A 2-D numpy array of the instruments (N x K), where K > L.

    Returns:
        np.ndarray: A 1-D numpy array of the JIVE2 estimates (L x 1).
    

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


    ### I need this double checked to ensure accuracy. I'm not confident that the matrix multiplication will give us a scalar. i.e. the sizes of the matrices with one row subtracted might be messed up.
    # First Pass
    # pi[i] = np.linalg.inv(z[i].T @ z[i]) @ (z[i].T @ X[i])
    Z_j2 = np.zeros((Z.shape[0], X.shape[1]))
    for i in range(Z.shape[0]):
        Z_j2[i] = (Z[i] @ np.linalg.inv(Z.T @ Z) @ (Z.T @ X - Z[i].T @ X[i])) / (1- (1/N))

    # Second Pass: Construct the X_jive1 matrix
    # Need some help here 

    # Second Stage: IV estimation using X_jive1 as the instrument for X
    p_jive2 = np.dot(Z_j2, np.dot(np.linalg.inv(Z_j2.T @ Z_j2), Z_j2.T))
    beta_jive2 = np.linalg.inv(X.T @ p_jive2 @ X) @ X.T @ p_jive2 @ Y

    print("JIVE1 Estimates:\n", beta_jive2)
    return beta_jive2

    """

#Pick a vector length:
n = 1000

#Getting our Z's and making a Z matrix:
Z = np.random.randn(n, 1)
column_of_ones = np.ones((Z.shape[0], 1))
Z = np.hstack((column_of_ones, Z))

#Parameter vectors:
α = np.array([1, 1])
β = np.array([1,2])

#Error terms:
e1 = np.random.normal(0,5,n)
e2 = np.random.normal(0,5,n)
δ = np.random.normal(0,1)
ε = 5*e1 - 5*e2 + δ

#Making our endogenous variable:
x = np.dot(Z,α) + .2*e1
X = np.column_stack((column_of_ones, x))

#Outcome vector:
Y = np.dot(X,β) + ε

#OLS benchmark:
bhat_ols = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, Y))

#2sls comparison:
Zt_Z = np.dot(Z.T, Z)
Zt_Z_inv = np.linalg.inv(Zt_Z)
pz = np.dot(np.dot(Z, Zt_Z_inv), Z.T)
proj_x = np.dot(pz, X)
first = np.linalg.inv(np.dot(proj_x.T, X))
second = np.dot(proj_x.T, Y)
bhat_2sls = np.dot(first, second)
jive1 = JIVE1(Y,X,Z)
jive2 = JIVE2(Y,X,Z)

# Combine matrices into a single DataFrame
df = pd.DataFrame({
    "Y": Y,  # Outcome vector
    **{f"X{i}": X[:, i] for i in range(X.shape[1])},  # Endogenous variables
    **{f"Z{i}": Z[:, i] for i in range(Z.shape[1])}   # Instrumental variables
})

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)

# Print the DataFrame to verify
#print(df)


#Compare them:
print("OLS:", bhat_ols[1])
print("2SLS:", bhat_2sls[1])
print("Jive 1:", jive1['beta'])
print("Jive 2:",jive2['beta'])