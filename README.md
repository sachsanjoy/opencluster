# Open cluster membership analysis tool

## requirements
```python
# python 3.12.1

! pip install numpy
! pip install astropy
! pip install astroquery
! pip install matplotlib
! pip install pandas
! pip install scipy
! pip install scikit-learn
! pip install gaiadr3-zeropoint
```
- simply install
```pip install -r requirements.txt```

#### Please check the notebook open_cluster.ipynb

# Includes
- Module to Download data from GAIA DR3 database


# Gaussian Mixture Model for membership analysis


Plot the confidence ellipsoids of a mixture of two Gaussians
obtained with Expectation Maximisation (``GaussianMixture`` class) and
Variational Inference (``BayesianGaussianMixture`` class models with
a Dirichlet process prior).

Both models have access to five components with which to fit the data. Note
that the Expectation Maximisation model will necessarily use all five
components while the Variational Inference model will effectively only use as
many as are needed for a good fit. Here we can see that the Expectation
Maximisation model splits some components arbitrarily, because it is trying to
fit too many components, while the Dirichlet Process model adapts it number of
state automatically.

Another advantage of the Dirichlet process model is that it can fit
full covariance matrices effectively even when there are less examples
per cluster than there are dimensions in the data, due to
regularization properties of the inference algorithm.



