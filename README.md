# PafnutyMatch
Experimental code for Chebyshev moment matching algorithm for private synthetic data

## Get setup with MOSEK
We use the MOSEK solver for the convex optimization problem. 

### Running in colab notebook
If you want to run in colab, add to top of your notebook:
```
! pip install cvxpy
! pip install Mosek
! export PYTHONPATH="$PYTHONPATH:/content"
from mosek.fusion import *
```
Then, add `mosek.lic` to `:/root/mosek/mosek.lic` in your colab VM. You can get a free academic license from [Mosek](https://www.mosek.com/products/academic-licenses/).

### Running locally
If you want to run locally, you can pip install `requirements.txt` and add `mosek.lic` to your local path, location may vary.