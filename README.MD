# A Python3 library for basic unmixing functions

## Functions list

The three functions implemented in this library are:
- **[Vertex Component Analysis (VCA)](#vca-function)**. Related article is J. Nascimento and J. Dias, "Vertex Component Analysis: A fast algorithm to unmix hyperspectral data", IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898-910, 2005. 
- **[Simplex Identification via Split Augmented Lagrangian (SISAL)](#sisal-function)**. Related article is J. Bioucas-Dias, "A variable splitting augmented Lagrangian approach to linear spectral unmixing", in First IEEE GRSS Workshop on Hyperspectral Image and Signal Processing-WHISPERS'2009, Grenoble, France, 2009. 
- **[Sparse Unmixing via variable Splitting and Augmented Lagrangian methods(SUNSAL)](#sunsal-function)**. Related article is Bioucas-Dias, J. M., & Figueiredo, M. A., "Alternating direction algorithms for constrained sparse regression: Application to hyperspectral unmixing." in Workshop in Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), 2010  (pp. 1-4).

Matlab versions of these codes are available in the [Jose Bioucas Dias website](http://www.lx.it.pt/~bioucas/code.htm).

## VCA function

This function were translated by Adrien Lagrange ([view his github page](https://github.com/Laadr)).

### Usage

```
Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
```

### Input variables

Y - matrix with dimensions L(channels) x N(pixels) each pixel is a linear mixture of R endmembers signatures Y = M x s, where 
- s = gamma x alpha
- gamma is a illumination perturbation factor and
- alpha are the abundance fractions of each endmember.
      
R - positive integer number of endmembers in the scene
    
### Output variables

Ae     - estimated mixing matrix (endmembers signatures)

indice - pixels that were chosen to be the most pure

Yp     - Data matrix Y projected.   
    
### Optional parameters
snr_input - (float) signal to noise ratio (dB)

v         - [True | False]
    
### Author, license and info

Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)

This code is a translation of a matlab code provided by Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt) available at http://www.lx.it.pt/~bioucas/code.htm under the GNU General Public License 2.0.

Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004)).


## SISAL function

### Usage

```
M,Up,my,sing_values = sisal(Y,p,**kwargs)
```

### Description

Simplex identification via split augmented Lagrangian (SISAL) estimates the vertices  M={m_1,...m_p} of the (p-1)-dimensional simplex of minimum volume containing the vectors [y_1,...y_N], under the assumption that y_i belongs to a (p-1)  dimensional affine set.
    
For details see José M. Bioucas-Dias, "A variable splitting augmented lagrangian approach to linear spectral unmixing", First IEEE GRSS Workshop on Hyperspectral Image and Signal Processing - WHISPERS, 2009. ([http://arxiv.org/abs/0904.4635v1](http://arxiv.org/abs/0904.4635v1))

### Input
    
Y - matrix with dimension  L(channels) x N(pixels). Each pixel is a linear mixture of p endmembers signatures Y = M*x + noise.

p - number of independent columns of M. Therefore, M spans a (p-1)-dimensional affine set. p is the number of endmembers.
    
### Optional input
    
mm_iters - Maximum number of constrained quadratic programs. Default: 80
    
tau - Regularization parameter in the problem

```
             Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
                   subject to np.ones((1,p))*Q=mq
             where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge" induced norm.
```
       Default: 1
    
mu - Augmented Lagrange regularization parameter. Default: 1
    
spherize - {True, False} Applies a spherization step to data such that the spherized data spans over the same range along any axis. Default: True
    
tolf - Tolerance for the termination test (relative variation of f(Q)). Default: 1e-2
    
M0 - Initial M, dimension L x p. Defaults is given by the VCA algorithm.
    
verbose - {0,1,2,3} 
- 0 - work silently
- 1 - display simplex volume
- 2 - display figures
- 3 - display SISAL information 
- 4 - display SISAL information and figures
Default: 1
    
### Output 
    
M - estimated endmember signature matrix L x p
    
Up - isometric matrix spanning the same subspace as M, imension is L x p
   
my - mean value of Y
    
sing_values - (p-1) eigenvalues of Cy = (y-my)*(y-my)/N. The dynamic range of these eigenvalues gives an idea of the  difficulty of the underlying problem
    
### Note 

The identified affine set is given by

```
{z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}
```    
    
### Author, license and info

Author: Etienne Monier (etienne.monier@enseeiht.fr)
    
This code is a translation of a matlab code provided by Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt) available at http://www.lx.it.pt/~bioucas/code.htm under the GNU General Public License 2.0.

Translation of last version at 20-April-2018 (Matlab version 2.1 (7-May-2004))


## SUNSAL function

### Usage
x = sunsal_v2(M,Y,**kwargs)
        
### Description 
        
SUNSAL (sparse unmixing via variable splitting and augmented Lagrangian methods) algorithm implementation. Accepted constraints are:
- 1. Positivity:  X >= 0
- 2. Addone:      np.sum(X,axis=0) = np.ones(N)
    
For details see J. Bioucas-Dias and M. Figueiredo, “Alternating direction algorithms for constrained sparse regression: Application to hyperspectral unmixing”, in 2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal Processing-WHISPERS'2010, Raykjavik, Iceland, 2010.
    
### Input 
    
M - endmember signature matrix with dimensions L(channels) x p(endmembers)
    
Y - matrix with dimensions L(channels) x N(pixels). Each pixel is a linear mixture of p endmembers signatures
    
### Optional input 
    
al_iters - Minimum number of augmented Lagrangian iterations. Default: 100
    
lambda_p - regularization parameter. lambda is either a scalar or a vector with N components (one per column of x). Default: 0
    
positivity - {True, False} Enforces the positivity constraint. Default: False
    
addone - {True, False} Enforces the addone constraint. Default: False
    
tol - tolerance for the primal and  dual residuals. Default: 1e-4
    
verbose = {True, False}
- False - work silently
- True - display iteration info
Default: True
    
### Output 

X - estimated abundance matrix of size p x N
    
### Author, license and info

Author: Etienne Monier (etienne.monier@enseeiht.fr)
    
This code is a translation of a matlab code provided by Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt) available at http://www.lx.it.pt/~bioucas/code.htm under the GNU General Public License 2.0.

Translation of last version at 20-April-2018 (Matlab version 2.1 (7-May-2004))

## Authors

Software translated from matlab to python by Etienne Monier (etienne.monier@enseeiht.fr), 2018.

Initial matlab author: Jose Bioucas-Dias, 2009

## License

This code is distributed under the terms of the GNU General Public License 2.0.
