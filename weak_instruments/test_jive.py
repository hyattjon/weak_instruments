import numpy as np
from ujive1 import *
from ujive2 import *
import pandas as pd
from repo import *


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
jive1 = UJIVE1(Y,X[:,1],Z[:,1],talk=True)
#jive2 = UJIVE2(Y,X,Z,talk=True)

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
#print("Jive 2:",jive2['beta'])