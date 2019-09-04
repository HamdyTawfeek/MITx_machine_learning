"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    from scipy.stats import multivariate_normal

    mu_s, var_s, p_s = mixture.mu, mixture.var, mixture.p
    K,d = mu_s.shape
    n = X.shape[0]
    E = np.zeros((n,K))
    for i in range(n):
        x = X[i]
        for j,mu in enumerate(mu_s):
            var, p = var_s[j], p_s[j]
            norm = multivariate_normal.pdf(x, mean=mu, cov=var)
            E[i,j] = p*norm
    
    LL = np.log(E.sum(axis=1)).sum()
    E = E/np.sum(E, axis=1).reshape((-1,1))
    
    return E, LL


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    n,K = post.shape
    mu = np.zeros((K,d))
    var = np.zeros(K,)


    p = post.sum(axis=0)/n

    for j in range(K):
        for i in range(n):
            x = X[i]
            mu[j,:] += x*post[i,j]

        mu[j,:] /= (n*p[j])

    for j in range(K):
        sigma = np.zeros((d,d))
        for i in range(n):
            x = X[i]
            sigma += post[i,j]*np.dot(x-mu[j,:], x-mu[j,:])
        var[j] = sigma[0][0]/d
        var[j] /= (n*p[j])

    return GaussianMixture(mu, var, p)  


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    LL_old = None
    while True:
        post, LL = estep(X, mixture)
        mixture = mstep(X, post)
        
        if LL_old is None:
            LL_old = LL
            continue
        
        if (LL - LL_old) <= (10**-6)* np.abs(LL):
            break
        else:
            LL_old = LL
    
    return mixture, post, LL

