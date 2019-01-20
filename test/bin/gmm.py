import logging

import click

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from leitmotiv.algorithms.gmm import AdaptiveGMM, GMM


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[1]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[:, n], eig_vals[0], eig_vals[1],
                                  180 + angle)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)

        ell = mpl.patches.Ellipse(means[:, n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_facecolor('none')
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)


@click.command()
@click.option('--algorithm', type=click.Choice(['normal', 'adaptive']),
              default='normal', help='Select the GMM variant to run.')
@click.option('--random-seed', is_flag=True,
              help='Use a random seed; default is fixed.')
@click.option('--clusters', default=2, help='Number of GMM clusters.')
@click.option('--points', default=500, help='Number of points to use.')
def main(algorithm, random_seed, clusters, points):
    '''Test the leitmotive SOFM class.

    The SOFM is trained on uniformly distributed noise in order to visualize
    how it evolves over time.  All logging is enabled so the SOFM output can be
    easily inspected.
    '''
    logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

    # Initialize the data.
    if not random_seed:
        np.random.seed(10)

    d1 = np.random.multivariate_normal((-2, 2), 5*np.eye(2), size=points//3)
    d2 = np.random.multivariate_normal((5, -3), 4*np.array([[1.5, .4],
                                                            [.4, .3]]),
                                       size=points//3)
    d3 = np.random.multivariate_normal((-3, -2), 4*np.array([[1.3, .5],
                                                             [.5, 2]]),
                                       size=points//3)
    data = np.concatenate((d1, d2, d3)).T

    # Initialize the GMM object.
    if algorithm == 'adaptive':
        click.secho('Running AdaptiveGMM', bold=True)
        gmm = AdaptiveGMM(0.01, clusters)
    elif algorithm == 'normal':
        click.secho('Running Standard GMM', bold=True)
        gmm = GMM(clusters)
    gmm.train(data)

    if algorithm == 'adaptive':
        gmm.prune()
        click.secho('Obtained %d models.' % len(gmm), bold=True)

    weights = gmm.weights
    means = gmm.means
    covars = gmm.covariances

    plt.scatter(data[0, :], data[1, :])
    plot_ellipses(plt.gca(), weights, means, covars)

    # display predicted scores by the model as a contour plot
    x = np.linspace(data[0, :].min(), data[0, :].max())
    y = np.linspace(data[1, :].min(), data[1, :].max())
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()])
    Z = -np.log(gmm.likelihoods(XX))
    Z = Z.reshape(X.shape)

    plt.contour(X, Y, Z, norm=LogNorm(), levels=np.logspace(0, 2, 15))
    plt.title('Algorithm Solution')

    plt.figure()
    plt.plot(gmm.negloglikelihood)
    plt.title('Negative Log-likelihood')

    plt.figure()
    plt.bar(np.arange(len(weights)), weights)
    plt.title('Model Weights')

    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    main()
