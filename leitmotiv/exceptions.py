class LeitmotivError(Exception):
    '''Top-level application exception.'''


class DatabaseConnectionError(LeitmotivError):
    '''Indicates an error occurred when using the database.'''


class ImageNotFoundError(LeitmotivError):
    '''The requested image was not in the leitmotiv database.'''
    def __init__(self, iid=None, imghash=None):
        '''Initialize the ImageNotFoundError exception.

        Parameters
        ----------
        iid: int
            the image ID that triggered the error
        imghash: str
            the hash that triggered the error
        '''
        if iid is not None:
            super().__init__('No image with ID %d.' % (iid,))
        elif imghash is not None:
            super().__init__('No image with hash "%s".' % (imghash,))
        else:
            super().__init__('Could not find the requested image.')


class AlgorithmUntrainedError(LeitmotivError):
    '''The particiular ML algorithm being used wasn't trained.

    This is called if a machine learning algorithm used by leitmotiv hasn't
    been trained prior to the model parameters being retrieved.
    '''


class LinearAlgebraError(LeitmotivError):
    '''One of the linear algebra methods exhibit some fault.

    This is raised by some of the lower-level classes in the case that there
    was an unrecoverable numerical issue.
    '''
