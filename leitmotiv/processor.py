from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import signal

from tqdm import tqdm

__all__ = [
    'Processor'
]


class SignalWrapper(object):
    '''Temporarily disables a signal interrupt handler.'''
    def __init__(self):
        self._hndl = None

    def __enter__(self):
        self._hndl = signal.signal(signal.SIGINT, signal.SIG_IGN)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGINT, self._hndl)
        return False


class Processor(object):
    '''A processor that performs batch image operations.

    The :class:`Processor` performs computationally expensive batch image
    processing operations by splitting the operations across multiple
    processes.  It can operate in either single image or multi-image mode.
    '''
    def __init__(self, nproc=None, quiet=False, desc=None):
        '''
        Parameters
        ----------
        nproc : int
            number of processes to use; the default is to take the number of
            processors and divide by two
        quiet : bool
            if ``True`` then do not show any output onto the command line
        desc : str
            if provided then this will be shown as part of the command line
            output
        '''
        self._nproc = nproc
        if self._nproc is None:
            self._nproc = os.cpu_count() // 2
        self._tqdm_args = {
            'disable': quiet,
            'desc': 'Processing jobs' if desc is None else desc,
            'unit': 'Images',
            'ascii': True
        }

    def set_message(self, msg):
        '''Set the message shown by the progress bar.

        Parameters
        ----------
        msg : str
            message to display when the progress bar is enabled
        '''
        self._tqdm_args['desc'] = msg

    def set_units(self, unit):
        '''Set the units shown on the progress bar.

        Parameters
        ----------
        unit : str
            displayed units
        '''
        self._tqdm_args['unit'] = unit

    def submit_batch(self, items, fn, block=True):
        '''Batch process a set of items using the provided functor.

        The batch processing assumes that the functor is something that
        performs a single unit of work in isolation.  It only needs to accept a
        single argument and should not return anything.

        Batch processing can be done either synchronously or asynchronously.
        Running in synchronous mode simply means that this function blocks
        until the all of the jobs have completed.  This will produce output to
        the console if CLI mode is enabled.  Otherwise, a list of
        :class:`concurrent.future.Future` objects will be returned that can be
        used to determine job completion.

        Parameters
        ----------
        items : list
            a list of items that will be processed in parallel
        fn : callable
            the function that processes the list
        block : bool
            if ``True`` then the function blocks until all jobs are complete

        Returns
        -------
        list of futures or ``None``
            depending on whether or not 'block' was set to ``True``; the
            list will be in the same order as the input items
        '''
        pool = ProcessPoolExecutor(max_workers=self._nproc)
        if block:
            with pool:
                with SignalWrapper():
                    futures = [pool.submit(fn, itm) for itm in items]
                try:
                    for f in tqdm(as_completed(futures), total=len(futures),
                                  **self._tqdm_args):
                        if f.exception():
                            pool.shutdown()
                            raise f.exception()
                except KeyboardInterrupt:
                    if not self._tqdm_args['disable']:
                        tqdm.write('Cancelling jobs...please wait.')
                    for f in futures:
                        f.cancel()
        else:
            futures = [pool.submit(fn, itm) for itm in items]
            pool.shutdown(wait=False)
            return futures

    def mapreduce(self, items, mapper):
        '''Applies a map-reduce-like operation onto a list of hashes.

        The map function accepts a list of items where some operation applied
        to them in parallel.  The output from the map function will then be
        processed by the reduce function.  While not required, the reduce
        function *may* be made to be stateful if it is useful to cache the
        outputs from the mapping function.

        The method expects a single callable object that will be used to
        perform the mapping operation.  The object is expected to have a
        ``reduce`` attribute that will process the output of the mapping
        operation.

        The method blocks until all of the processing has finished.

        Parameters
        ----------
        hashes : list
            a list of images hashes; this may also be a list of hash tuples if
            the map function operates on multiple images at once
        mapper : callable
            a callable object that accepts an entry of 'hashes' and returns
            some value; it must contain a ``reduce`` attribute

        Returns
        -------
        list
            a list containing any outputs from the reduce function, in the
            order of their completion

        Raises
        ------
        ValueError
            if the mapper does not have a ``reduce`` attribute
        '''
        if not hasattr(mapper, 'reduce'):
            raise AttributeError(
                'Mapper callable must have a "reduce" attribute.')

        pool = ProcessPoolExecutor(max_workers=self._nproc)

        results = []
        with pool:
            with SignalWrapper():
                futures = [pool.submit(mapper, itm) for itm in items]
            try:
                for f in tqdm(as_completed(futures), total=len(futures),
                              **self._tqdm_args):
                    # Quite processing if an exception occurred.
                    if f.exception():
                        pool.shutdown()
                        raise f.exception()

                    # Send the result to the reduce function.
                    output = mapper.reduce(f.result())
                    if output is not None:
                        results.append(output)

                # Inform the mapper object that the parallel processing is
                # done.
                try:
                    mapper.done()
                except AttributeError:
                    pass
            except KeyboardInterrupt:
                if not self._tqdm_args['disable']:
                        tqdm.write('Cancelling jobs...please wait.')
                for f in futures:
                    f.cancel()

        return results
