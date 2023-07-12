import math
import os
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import optimize
from scipy.spatial import distance_matrix
from d2l import torch as d2l

from gaussian_process.gpy_torch import ExactGPModel

d2l.set_figsize()

def data_maker1(x, sig):
    """
    y = sin(x) + 0.5sin(4x) + eps, eps (0, sig^2)
    """
    return np.sin(x) + 0.5 * np.sin(4 * x) \
        + np.random.randn(x.shape[0]) * sig

def rbfkernel(x1, x2, ls=4.0):
    """
    k(xi, xj) = a^2 * exp(-(0.5/l^2) * ||x-x'||^2)
    """
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1.0 / ls / 2) * (dist ** 2))

def neg_MLL(pars):
    """
    n = train_x.shape[0] = train_y.shape[0]
    log(P(y|X)) = kernel_term + logdet + const
    kernel_term = -0.5 * y.T * (K + sig^2*I)^-1 *y
    logdet = -0.5 * log(det((K + sig^2*I)^-1))
    const = -n/2 * log(2pi)
    """
    K = rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(
        np.linalg.det(K + pars[1] ** 2 * np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    return -(kernel_term + logdet + const)


if __name__ == "__main__":
    # dataset
    sig = 0.25
    train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
    train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

    d2l.plt.scatter(train_x, train_y)
    d2l.plt.plot(test_x, test_y)
    d2l.plt.xlabel("x", fontsize=20)
    d2l.plt.ylabel("Observations y", fontsize=20)
    d2l.plt.show()

    # GP prior, f(x) ~ gp(m, k), m(x) = 0, k is rbf
    mean = np.zeros(test_x.shape[0])
    cov = rbfkernel(test_x, test_x, ls=0.2)

    prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
    d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
    d2l.plt.plot(test_x, mean, linewidth=2.)
    d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov),
                         alpha=0.25)
    d2l.plt.show()

    # posterior inference
    ell_est = 0.4
    post_sig_est = 0.5

    learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est, post_sig_est]),
                                       bounds=((0.01, 10.), (0.01, 10.)))
    ell = learned_hypers.x[0]
    post_sig_est = learned_hypers.x[1]
    print(ell, post_sig_est)

    """
    n = x.shape[0], m = x*.shape[0]
    K(x,x): (n,n); K(x,x*): (n,m); K(x*,x*): (m,m)
    post_mean = K(x,x*).T * (K(x,x) + sig^2*I)^-1 * y
    post_cov = K(x*,x*) - K(x,x*).T * (K(x,x) + sig^2*I)^-1 * K(x,x*)
    """
    K_x_xstar = rbfkernel(train_x, test_x, ls=ell)
    K_x_x = rbfkernel(train_x, train_x, ls=ell)
    K_xstar_xstar = rbfkernel(test_x, test_x, ls=ell)

    post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x +
                        post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
    post_cov = K_xstar_xstar - K_x_xstar.T @ \
               np.linalg.inv((K_x_x + post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

    lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
    up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

    d2l.plt.scatter(train_x, train_y)
    d2l.plt.plot(test_x, test_y, linewidth=2.)
    d2l.plt.plot(test_x, post_mean, linewidth=2.)
    d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
    d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
    d2l.plt.show()

    post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
    d2l.plt.scatter(train_x, train_y)
    d2l.plt.plot(test_x, test_y, linewidth=2.)
    d2l.plt.plot(test_x, post_mean, linewidth=2.)
    d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
    d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
    plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
    d2l.plt.show()


    """**********************************************************************************"""
    # gpytorch
    # First let's convert our data into tensors for use with PyTorch
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    # Initialize Gaussian likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    training_iter = 50
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer, includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # Set our loss as the negative log GP marginal likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if i % 10 == 0:
            print(f'Iter {i + 1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
                  f'squared lengthscale: '
                  f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
                  f'noise variance: {model.likelihood.noise.item():.3f}')
        optimizer.step()

    # test
    # Get into evaluation (predictive posterior) mode
    test_x = torch.tensor(test_x)
    model.eval()
    likelihood.eval()
    observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        # Initialize plot
        f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
        # Get upper and lower bounds for 95\% credible set (in this case, in
        # observation space)
        lower, upper = observed_pred.confidence_region()
        ax.scatter(train_x.numpy(), train_y.numpy())
        ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
        ax.set_ylim([-1.5, 1.5])
        ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
                   '95% Credible Set'])























