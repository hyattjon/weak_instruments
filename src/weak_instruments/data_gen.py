import numpy as np

from weak_instruments.ujive1 import *

#Pick a vector length:
n = 1000

#Getting our Z's and making a Z matrix:
z1 = np.random.randn(n, 1)
z2 = np.random.randn(n, 1)
Z = np.hstack((z1, z2))
column_of_ones = np.ones((Z.shape[0], 1))
Z = np.hstack((column_of_ones, Z))

#Parameter vectors:
α = np.array([1, 1, 1])
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
y = np.dot(X,β) + ε # I changed this to capital X because I was getting shape errors

#OLS benchmark:
bhat_ols = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T, y))

#2sls comparison:
Zt_Z = np.dot(Z.T, Z)
Zt_Z_inv = np.linalg.inv(Zt_Z)
pz = np.dot(np.dot(Z, Zt_Z_inv), Z.T)
proj_x = np.dot(pz, X)
first = np.linalg.inv(np.dot(proj_x.T, X))
second = np.dot(proj_x.T, y)
bhat_2sls = np.dot(first, second)

#Compare them:
print("OLS:", bhat_ols[1])
print("2SLS", bhat_2sls[1])

jive1 = JIVE1(y,X,Z)

print(jive1['beta'])