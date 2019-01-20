import logging

import click
import numpy as np

import matplotlib.pyplot as plt

from leitmotiv.features import ImageGMM
from leitmotiv.io import imread, to_ndarray


@click.command()
@click.option('--random-seed', is_flag=True,
              help='Use a random seed; default is fixed.')
@click.option('--colour-space', type=click.Choice(['rgb', 'yuv', 'lab']),
              default='rgb', help='Colour space the GMM is trained on.')
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
def main(random_seed, colour_space, image):
    '''Test the leitmotive ImageGMM class.

    The ImageGMM is trained on the provided image in order to create a
    representation of that image.  The results of that representation are then
    visualized.
    '''
    logging.basicConfig(level=logging.DEBUG, format=logging.BASIC_FORMAT)

    # Initialize the data.
    if not random_seed:
        np.random.seed(10)

    img = imread(image)
    igmm = ImageGMM(img, alpha=0.01, max_clusters=50, samples=5000,
                    colour_space=colour_space, sigma=3)

    plt.figure()
    plt.bar(np.arange(len(igmm.weights)), igmm.weights)
    plt.title('Cluster Weights')

    plt.figure()
    plt.imshow(to_ndarray(img))
    plt.title('Original Image')

    plt.figure()
    pts = igmm.sample(5000, True)
    plt.scatter(pts[0, :], pts[1, :], c=pts[2:, :].T/255)
    plt.axis('equal')
    plt.gca().invert_yaxis()

    plt.figure()
    igmm.visualize()
    plt.show()


if __name__ == '__main__':
    main()
