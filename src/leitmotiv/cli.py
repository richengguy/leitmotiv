import itertools
import logging
import logging.config

import click

import mkl

from ruamel.yaml import YAML

import torch
from torch.utils.data import DataLoader

from leitmotiv import actions
from leitmotiv import library
from leitmotiv import models
from leitmotiv.processor import Processor

import torchvision
import matplotlib.pyplot as plt

__LOGGER_OPTIONS = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'brief': {
            'format': '-- %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
            'level': 'INFO'
        }
    },
    'loggers': {
        'leitmotiv': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}


def _process_images(config, images):
    '''Batch process a set of images.

    Adding images or directories are essentially just the same operations: find
    a list of images and then do something with them.  This just puts
    everything in the same spot to avoid duplicating code.

    Parameters
    ----------
    config : dict-like object
        runtime configuration
    images : list
        list of images
    '''
    processor = Processor(nproc=config['num_workers'])

    processor.set_message('Inserting images')
    hashes = processor.mapreduce(images, actions.InsertImages(config))

    processor.set_message('Extracting features')
    processor.submit_batch(hashes, actions.ExtractFeatures(config))


@click.group()
@click.option('-c', '--config', type=click.Path(dir_okay=False),
              help='Path to application configuration file.',
              default='./config.yml')
@click.option('-d', '--debug', is_flag=True,
              help='Enable all debugging output.')
@click.pass_context
def main(ctx, config, debug):
    '''leitmotiv - Extract Trends from Image Collections'''
    if debug:
        __LOGGER_OPTIONS['handlers']['console']['level'] = 'DEBUG'
        __LOGGER_OPTIONS['loggers']['leitmotiv']['level'] = 'DEBUG'
        __LOGGER_OPTIONS['loggers']['peewee'] = {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        }
        logging.config.dictConfig(__LOGGER_OPTIONS)

    yaml = YAML()
    with open(config) as f:
        config = yaml.load(f)['leitmotiv']

    mkl.set_num_threads(config['num_threads'])
    library.LIBRARY_PATH = config['library']['path']
    ctx.obj = config


@main.command('add-image')
@click.argument('image', type=click.Path(exists=True, dir_okay=False),
                nargs=-1)
@click.pass_obj
def add_image(config, image):
    '''Add either a single or multiple images into the library.

    Any files that are passed in must be valid image files.  Any files that do
    not have a valid image extension are ignored.
    '''
    click.secho('Importing Files\n----', bold=True)
    _process_images(config, image)
    click.echo('Done...', nl=False)
    click.secho('\u2713', fg='green', bold=True)


@main.command('add-directory')
@click.argument('directory', type=click.Path(exists=True, file_okay=False),
                nargs=-1)
@click.pass_obj
def add_directory(config, directory):
    '''Add one or more directories into the library.

    The contents of the directories are recursively scanned to find new images.
    Feature extraction is performed immediately following the ingestion on all
    new images.
    '''
    click.secho('Importing from:', bold=True)
    for d in directory:
        click.echo('  - %s' % d)
    click.echo()

    with library.Library() as lib:
        images = lib.scan_directory(directory[0], *directory[1:])

    click.echo('Found %d images in %d directories.' % (len(images),
                                                       len(directory)))

    click.secho('Processing Files\n----', bold=True)
    _process_images(config, images)
    click.echo('Done...', nl=False)
    click.secho('\u2713', fg='green', bold=True)


@main.command('build-index')
@click.option('--skip-distances', is_flag=True,
              help='Skip computing the pairwise distances.')
@click.pass_obj
def build_index(config, skip_distances):
    '''Generate the image distance index.

    This computes the set of pairwise distances for all images in the database.
    Running this command will cause a pre-existing index to be regenerated from
    scratch.
    '''
    click.secho('Building Index\n----', bold=True)

    if not skip_distances:
        with library.Library() as lib:
            hashes = lib.get_hashes(sort_by_date=False)
            pairs = itertools.combinations(hashes, 2)
            lib.clear_distances()

        processor = Processor(desc='Computing image-image distances')
        processor.mapreduce(pairs, actions.PairwiseDistance(config))

    with library.Library() as lib:
        from leitmotiv.features import ImageGMM
        from leitmotiv.library import DistanceType

        if 'index' in config and 'knn' in config['index']:
            knn = config['index']['knn']
        else:
            knn = None

        distances, hashes = lib.distance_matrix(ImageGMM.ftype(),
                                                DistanceType.JENSEN_SHANNON,
                                                knn=knn)

        with lib.datastore(mode='w') as hdf5:
            hdf5.create_array('/', 'distances', distances)

            grp = hdf5.create_group('/', 'hashes')
            hdf5.create_array(grp, 'keys', list(hashes.keys()))
            hdf5.create_array(grp, 'indices', list(hashes.values()))


@main.command('train-model')
@click.option('--epochs', '-e', type=click.INT, default=1000, metavar='EPOCHS',
              help='Number of training epochs.')
@click.option('--batch-size', '-b', type=click.INT, default=1, metavar='BATCH',
              help='Training batch size.')
@click.pass_obj
def train_model(config, epochs, batch_size):
    '''Command for training the VAE.'''
    with library.Library() as lib:
        click.echo('Training Device: ', nl=False)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            click.secho('CUDA', bold=True)
        else:
            click.secho('CPU', bold=True)

        dataset = models.Dataset(lib, img_dim=256, to_gpu=use_gpu)
        model = models.VariationalAutoencoder(dataset.img_dim, sigma=0.9)
        trainer = models.ModelTrainer(batch_size, epochs, split=None,
                                      verbose=True, use_gpu=use_gpu)
        losses, validation = trainer.train(model, dataset)

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        for i, imgs in enumerate(dataloader):
            mu, _ = model.infer(imgs)
            torchvision.utils.save_image(imgs, 'inputs-%d.png' % i)
            torchvision.utils.save_image(model.generate(mu), 'outputs-%d.png' % i)  # noqa: E501
            break

        plt.figure()
        plt.plot(losses['elbo'])
        plt.plot(validation['elbo'])
        plt.title('Loss Function')
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')
        plt.legend(('Training', 'Validation'))
        plt.savefig('training-losses.png')


if __name__ == '__main__':
    main()
