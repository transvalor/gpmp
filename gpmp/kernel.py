''' Helper functions for defining covariance functions in the gpmp

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)
'''
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import gammaln
from math import exp, sqrt

# -- distance


def scale(x, invrho):
    ''' Scale x'''
    return invrho * x


@jax.jit
def distance(x, y, alpha=1e-8):
    '''Compute a distance matrix

    Inputs
      * x: numpy array n x dim
      * y: numpy array m x dim or None
      * alpha: a small number to prevent auto-differentation problems
        with the derivative of sqrt at zero

    If y is None, it is assumed y is x

    NB: in practice however, it seems that it makes no performance
    improvement; FIXME: investigate memory and CPU usage

    Output
      * distance matrix d of size n x m such that
        d_{i,j} = (alpha + sum_{k=1}^dim (x_{i,k} - y_{i,k})^2)^(1/2)

    '''
    if y is None:
        y = x

    y2 = jnp.sum(y**2, axis=1)

    # Debug: check if x is y
    # print("&x = {}, &y = {}".format(hex(id(x)), hex(id(y))))
    
    if x is y:
        d = jnp.sqrt(alpha + jnp.reshape(y2, [-1, 1]) + y2 -
                     2 * jnp.inner(x, y))
    else:
        x2 = jnp.reshape(jnp.sum(x**2, axis=1), [-1, 1])
        d = jnp.sqrt(alpha + x2 + y2 - 2 * jnp.inner(x, y))

    return d


# -- kernels


def exponential_kernel(h):
    ''' exponential kernel'''
    return jnp.exp(-h)


def matern32_kernel(h):
    ''' Matérn 3/2 kernel'''
    nu = 3 / 2
    c = 2 * sqrt(nu)
    t = c * h

    return (1 + t) * jnp.exp(-t)


def maternp_kernel(p, h):
    ''' Matérn kernel with half-integer regularity nu = p + 1/2'''
    c = 2 * jnp.sqrt(p + 1 / 2)
    polynomial = 0
    for i in range(p + 1):
        polynomial = polynomial + (2 * c * h) ** (p - i) \
            * exp(gammaln(p + 1) - gammaln(2 * p + 1)
                  + gammaln(p + i + 1) - gammaln(i + 1) - gammaln(p - i + 1))
    return jnp.exp(-c * h) * polynomial


@partial(jax.jit, static_argnums=2)
def maternp_covariance(x, y, p, param):
    '''Matérn covariance function with half-integer regularity nu = p + 1/2

    * parameters
      - x : ndarray nx x dim
      - y : ndarray ny x dim
      - p : integer
      - param : [log(sigma2) log(1/rho_1) log(1/rho_2) ...]

    NB: an isotropic covariance is obtained if param = [log(sigma2) log(1/rho)]
    (only one length scale parameter)

    * output : covariance matrix nx x ny
    '''
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1:])
    nugget = 10 * jnp.finfo(jnp.float64).eps

    xs = scale(x, invrho)

    if y is x:
        K = distance(xs, xs)
        K = sigma2 * maternp_kernel(p, K) + nugget * jnp.eye(K.shape[0])
    else:
        ys = scale(y, invrho)
        K = distance(xs, ys)
        K = sigma2 * maternp_kernel(p, K)

    return K


# -- parameters


def anisotropic_parameters_initial_guess_with_zero_mean(model, xi, zi):
    '''initialization strategy based on
       Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.

    '''
    rho = jnp.std(xi, axis=0)
    covparam = jnp.concatenate((jnp.array([jnp.log(1.0)]), -jnp.log(rho)))
    n = xi.shape[0]
    sigma2_GLS = 1 / n * model.norm_k_sqrd_with_zero_mean(xi, zi, covparam)

    return jnp.concatenate((jnp.array([jnp.log(sigma2_GLS)]), -jnp.log(rho)))

def anisotropic_parameters_initial_guess(model, xi, zi):
    '''initialization strategy based on
       Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.

    '''
    rho = jnp.std(xi, axis=0)
    covparam = jnp.concatenate((jnp.array([jnp.log(1.0)]), -jnp.log(rho)))
    n = xi.shape[0]
    sigma2_GLS = 1 / n * model.norm_k_sqrd(xi, zi, covparam)

    return jnp.concatenate((jnp.array([jnp.log(sigma2_GLS)]), -jnp.log(rho)))


def autoselect_parameters(p0, criterion, gradient):

    # scipy.optimize.minimize cannot use jax arrays
    if isinstance(p0, jax.numpy.ndarray):
        p0 = jnp.asarray(p0)
    gradient_asnumpy = lambda p: np.array(jnp.asarray(gradient(p)))

    r = minimize(criterion,
                 p0,
                 args=(),
                 method='L-BFGS-B',
                 jac=gradient_asnumpy,
                 bounds=None,
                 tol=None,
                 callback=None,
                 options={
                     'disp': True,
                     'maxcor': 20,
                     'ftol': 1e-06,
                     'gtol': 1e-05,
                     'eps': 1e-08,
                     'maxfun': 15000,
                     'maxiter': 15000,
                     'iprint': -1,
                     'maxls': 40,
                     'finite_diff_rel_step': None
                 })

    best = r.x

    return best


def print_sigma_rho(covparam):
    print("sigma      : {}".format(jnp.exp(0.5 * covparam[0])))
    rho_str = "rho [ {:2d} ] : {}".format(0, jnp.exp(-covparam[1]))
    for i in range(covparam.size - 2):
        rho_str += "\n    [ {:2d} ] : {}".format(i + 1,
                                                 jnp.exp(-covparam[i + 2]))
    print(rho_str)
