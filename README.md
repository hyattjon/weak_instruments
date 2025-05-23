# Solutions to Weak Instruments
Two stage least squares with instruments has become a common method for estimation. Issues can arise when there are many instruments and is compounded when there are many weak instruments. Problems also arise when treatment is clustered.  Many methods have been proposed to handle these issues. In this package, we highlight some proposed solutions to many instruments bias and weak instruments bias, and run Monte Carlo simulations on a newly created Python package for handling many instruments bias.


## JIVE1 and JIVE2 estimates based on Blomquist and Dahlberg (1999)


## UJIVE1 and UJIVE2 estimates based on Angrist, Imbens, and Krueger (1999)
We use the following formula for estimation of UJIVE1:

$\[\frac{Z_i \hat{\pi} - h_i X_i}{1-h_i}\]$

where $h_i$ is the leverage for observation $i$.

We use the following formula for estimation of UJIVE2:

$\[\frac{Z_i \hat{\pi} - h_i X_i}{1-(\frac{1}{N})}\]$


## Cluster Jive (CJIVE)
Cluster Jive accomplishes a simliar task but is used when treatment is correlated for clusters instead of individuals.

## IJIVE


## SJIVE


## LIML


## Two Stage Least Squares
We create two stage least squares for comparison purposes


## Weak identification with many instruments (Mikushueva and Sun)


## Lim et al.




## Jacknife Anderson-Rubin tests for many weak IV inference



## Lagrange Multiplier



## HFUL



Each file should check for:
- Multicollinearity
- Perfect collinearity
- Dimensions of variables (Single column of controls etc)
    - Check to see if dimensions are the same for all variables
- Constant columns
- Y and Z must be one dimensional vectors