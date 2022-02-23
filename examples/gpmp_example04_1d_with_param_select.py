'''Plot and optimize the restricted negative log-likelihood

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

'''
import math
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp

# -- dataset


def generate_data():
    '''
    Data generation
    (xt, zt): target
    (xi, zi): input dataset
    '''
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    ind = [10, 45, 100, 130, 160]
    xi = xt[ind]
    zi = zt[ind]

    return xt, zt, xi, zi


xt, zt, xi, zi = generate_data()

# -- model specification


def constant_mean(x, param):
    return np.ones((x.shape[0], 1))


def kernel(x, y, covparam):
    p = 1
    return gp.kernel.maternp_covariance(x, y, p, covparam)


meanparam = None
covparam0 = None

model = gp.core.Model(constant_mean, kernel, meanparam, covparam0)

# -- automatic selection of parameters using REML

covparam0 = gp.kernel.anisotropic_parameters_initial_guess(model, xi, zi)

nlrl, dnlrl = model.make_reml_criterion(xi, zi)

covparam_reml = gp.kernel.autoselect_parameters(covparam0, nlrl, dnlrl)

model.covparam = covparam_reml

gp.kernel.print_sigma_rho(covparam_reml)


# -- plot likelihood profile

n = 50
sigma = np.logspace(-0.6, 1, n)
rho = np.logspace(-1.25, 0.5, n)

sigma_mesh, rho_mesh = np.meshgrid(sigma, rho)

nlrl_values = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        covparam = np.array(
            [math.log(sigma_mesh[i, j]**2), math.log(1 / rho_mesh[i, j])])
        nlrl_values[i, j] = nlrl(covparam)

plt.contourf(np.log10(sigma_mesh), np.log10(rho_mesh), np.log10(nlrl_values))
plt.plot(np.log10(np.exp(covparam_reml[0])), -
         np.log10(np.exp(covparam_reml[1])), 'ro')
plt.xlabel('sigma (log10)')
plt.ylabel('rho (log10)')
plt.title('log10 of the negative log restricted likelihood')
plt.colorbar()
plt.show()

# -- prediction

(zpm, zpv) = model.predict(xi, zi, xt)

zpv = np.maximum(zpv, 0)  # zeroes negative variances

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zt, 'C2', linewidth=0.5)
fig.plot(xi, zi, 'rs')
fig.plotgp(xt, zpm, zpv)
fig.ax.set_xlabel('$x$')
fig.ax.set_ylabel('$z$')
fig.ax.set_title('Posterior GP with parameters selected by ReML')
fig.show()
