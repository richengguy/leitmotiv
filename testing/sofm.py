import logging
import logging.config

import click
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from leitmotiv.algorithms import SOFM


def plot_sofm(sofm):
    '''Plot the contents of an initialized SOFM.

    Parameters
    ----------
    sofm: leitmotiv.algorithms.SOFM
        an initialized, though not necessarily trained, SOFM object
    '''
    sz = sofm.gridsz
    ndim = sofm.dimensionality

    pts = np.zeros((ndim, sz[0]*sz[1]))
    for i in range(sz[0]*sz[1]):
        pts[:, i] = sofm.node_value(i)

    lines = []
    for y in range(sz[1]):
        for x in range(1, sz[0]):
            x0 = sofm.node_value(x-1, y)
            x1 = sofm.node_value(x, y)
            lines.append([x0, x1])

    for y in range(1, sz[1]):
        for x in range(sz[0]):
            y0 = sofm.node_value(x, y-1)
            y1 = sofm.node_value(x, y)
            lines.append([y0, y1])

    plt.scatter(pts[0, :], pts[1, :], marker='x')
    ax = plt.gca()
    ax.add_collection(LineCollection(lines, colors='k'))


@click.command()
@click.option('--random-seed', is_flag=True,
              help='Use a random seed; default is fixed.')
@click.option('--points', default=250, help='Number of points to use.')
@click.option('--distribution', default='uniform',
              type=click.Choice(['uniform', 'gaussian', 'clusters']),
              help='Type of distribution.')
def main(random_seed, points, distribution):
    '''Test the leitmotive SOFM class.

    The SOFM is trained on uniformly distributed noise in order to visualize
    how it evolves over time.  All logging is enabled so the SOFM output can be
    easily inspected.
    '''
    logging.basicConfig(level=logging.DEBUG, format=logging.BASIC_FORMAT)

    # Initialize the data.
    if not random_seed:
        np.random.seed(10)

    if distribution == 'uniform':
        data = np.random.uniform(low=-1, high=1, size=(2, points))
    elif distribution == 'gaussian':
        data = np.random.multivariate_normal((-2, 2), np.eye(2), size=points)
        data = data.T
    elif distribution == 'clusters':
        d1 = np.random.multivariate_normal((-2, 2), np.eye(2), size=points//2)
        d2 = np.random.multivariate_normal((5, -3), 0.75*np.eye(2),
                                           size=points//2)
        data = np.concatenate((d1, d2)).T
        print(data.shape)

    # Initialize the SOFM object.
    sofm = SOFM((10, 10))
    sofm._initialize(data)

    plt.scatter(data[0, :], data[1, :])
    plot_sofm(sofm)
    plt.title('Initial SOFM')

    # Train the SOFM
    sofm.train(data)

    plt.figure()
    plt.scatter(data[0, :], data[1, :])
    plot_sofm(sofm)
    plt.title('Final SOFM')

    plt.show()


if __name__ == '__main__':
    main()
