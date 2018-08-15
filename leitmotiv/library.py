import enum
import logging
import os.path
import pathlib

import numpy as np

import tables

from peewee import JOIN

from tqdm import tqdm

from leitmotiv import database
from leitmotiv.exceptions import ImageNotFoundError
from leitmotiv.features import ImageDescriptor, DESCRIPTORS


__all__ = [
    'Library',
    'LIBRARY_PATH'
]


LIBRARY_PATH = None


def _expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


class DistanceType(enum.IntEnum):
    '''Specifies the available distance types.'''
    EUCLIDEAN = enum.auto()
    JENSEN_SHANNON = enum.auto()
    KL_DIVERGENCE = enum.auto()
    SYMMETRIC_KL_DIVERGENCE = enum.auto()

    def to_string(self):
        '''Obtain a string reprsentation of the distance type.

        Returns
        -------
        str
            string representation
        '''
        return {
            DistanceType.EUCLIDEAN: 'euclidean',
            DistanceType.JENSEN_SHANNON: 'js',
            DistanceType.KL_DIVERGENCE: 'kl',
            DistanceType.SYMMETRIC_KL_DIVERGENCE: 'symmetric_kl',
        }[self]

    def is_symmetric(self):
        return {
            DistanceType.EUCLIDEAN: True,
            DistanceType.JENSEN_SHANNON: True,
            DistanceType.KL_DIVERGENCE: False,
            DistanceType.SYMMETRIC_KL_DIVERGENCE: True,
        }[self]

    @staticmethod
    def from_string(val):
        '''Convert a string into the distance type.

        Parameters
        ----------
        val : str
            string to inspect

        Returns
        -------
        :class:`DistanceType`
            enumerated type

        Raises
        ------
        ValueError
            if the string is invalid
        '''
        try:
            return {
                'euclidean': DistanceType.EUCLIDEAN,
                'js': DistanceType.JENSEN_SHANNON,
                'kl': DistanceType.KL_DIVERGENCE,
                'symmetric_kl': DistanceType.SYMMETRIC_KL_DIVERGENCE
            }[val]
        except KeyError as e:
            raise ValueError('Invalid enum string.') from e


class DistanceMeasure(object):
    '''Wraps a distance measure for storage and retrieval.

    This mirrors the database :class:`~leitmotiv.database.DistanceMeasure` in
    order to abstract some of the underlying mechanisms.

    All attributes are read-only after the objects initialization.

    Attributes
    ----------
    source : str
        hash of the source object
    target : str
        hash of the target object
    distance : double
        distance between the source and the target
    feature_type : int
        feature type ID
    distance_type : :class:`DistanceType`
        the distance type
    is_similarity : bool
        distance is a similarity measure, not a distance
    symmetric : bool
        distance is symmetric
    '''
    def __init__(self, source, target, distance, ftype, dtype):
        '''
        Parameters
        ----------
        source : str
            source hash
        target : str
            target hash
        distance : double
            distance from source to target
        ftype : int
            feature type ID
        dtype : :class:`DistanceType`
            the distance type
        '''
        self._source = source
        self._target = target
        self._distance = distance
        self._ftype = ftype
        self._dtype = dtype

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def distance(self):
        return self._distance

    @property
    def feature_type(self):
        return self._ftype

    @property
    def distance_type(self):
        return self._dtype

    @property
    def is_similarity(self):
        return False

    @property
    def symmetric(self):
        return self._dtype.is_symmetric()


class Library(object):
    '''The leitmotiv image library.

    The image library does not store images directly.  Rather, the database
    maintains references to all of the images that the application is currently
    aware of.  This avoids having the library keep a copy of all of the images
    itself.  The allows leitmotiv to be used both as a local application and
    web server so that database is independent of how the images are actually
    stored.

    The :class:`Library` is a high-level interface into the actual database.
    It provides the mechanisms that allow images to be added and retrieved
    without explicitly having to access the underlying database model.

    Attributes
    ----------
    path : path-like object
        resolved path to where the library is stored; this is only valid when
        used within a context manager and will be ``None`` if not in one
    '''
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info('Opening connection to application database.')
        self._libpath = None
        if LIBRARY_PATH is None:
            raise ValueError(
                'Module-wide LIBRARY_PATH parameter must be set.')

    def __enter__(self):
        new_library = not os.path.exists(LIBRARY_PATH)

        libpath = pathlib.Path(LIBRARY_PATH)
        libpath.resolve()

        if new_library:
            libpath.mkdir(parents=True)

        dbpath = libpath / 'database.sdb'

        # Connect to the database.
        database.db.init(dbpath.as_posix())
        database.db.connect()

        # Create the tables if the database hadn't existed already.
        if not database.Image.table_exists():
            self._logger.info('Initializing SQLite database.')
            database.create_tables()

        self._libpath = libpath
        return self

    def __exit__(self, *args):
        database.db.close()
        self._libpath = None
        return False

    @property
    def path(self):
        return self._libpath

    def number_of_images(self):
        '''Number of images stored in the database.'''
        return database.Image.select().count()

    def insert_image(self, path, image_hash, dt, thumb):
        '''Insert an image into the database.

        Image insertion is a "cheap" operation in that all it will do is open
        the image and then generate a thumbnail from it.

        Parameters
        ----------
        path : path-like object
            path to where the image is located
        image_hash : str
            image hash
        dt : :class:`datetime.datetime`
            time when the image was taken
        thumb : bytes
            image thumbnail

        Raises
        ------
        IOError
            if the image could not be opened

        Returns
        -------
        str or ``None``
            the hash if the image was added or ``None`` if not
        '''
        path = _expand_path(path)
        with database.db.atomic():
            if database.Image.get_path(image_hash) is None:
                entry = database.Image()
                entry.path = path
                entry.hash = image_hash
                entry.thumb = thumb.getvalue()

                if dt:
                    entry.datetime = dt
                else:
                    self._logger.debug('Image has no date time.')

                entry.save()
                self._logger.debug('Inserted %s into the database.', path)
                return image_hash
            else:
                self._logger.debug('%s already exists in the database.', path)

        return None

    def scan_directory(self, path, *extra, quiet=False):
        '''Scan a directory for images to add.

        Given one or more top-level paths, this will walk through the directory
        trees to find all image files that should be added into the library.

        Parameters
        ----------
        path : path-like object
            path to an image folder
        extra : list of path-like objects
            extra paths to investigate
        quiet : bool
            suppress CLI output

        Returns
        -------
        list
            a list of image paths for all images not currently in the database;
            this doesn't perform a hash check so some duplicates may still be
            discovered
        '''
        paths = [path] + list(extra)

        # Scan for images.
        imgfiles = []
        for path in tqdm(paths, desc='Scanning directories', disable=quiet):
            self._logger.info('Scanning %s for images.', path)
            path = pathlib.Path(path)
            for root, dirs, files in os.walk(path):
                rootpath = pathlib.Path(root)
                for f in files:
                    if f[0] == '.':
                        continue
                    imgpath = rootpath / pathlib.Path(f)
                    if self.has_path(imgpath):
                        self._logger.debug('%s is already in the database.')
                        continue
                    ext = imgpath.suffix.lower()[1:]
                    if ext == 'jpg' or ext == 'jpeg':
                        imgfiles.append(imgpath.as_posix())

        return imgfiles

    def get_hashes(self, sort_by_date=True, page=None, count=None):
        '''Return a list of image hashes.

        The images may be sorted by date and paginated to make the output a bit
        easier to work with.

        Parameters
        ----------
        sort_by_date : bool
            sort the images by capture date
        page : int
            starting page
        count : int
            number of items on a particular page

        Returns
        -------
        list
            a list of image hashes, in the requested order
        '''
        query = database.Image.select(database.Image.hash)
        if sort_by_date:
            query = query.order_by(database.Image.datetime.asc())

        if page is not None and count is not None:
            query = query.paginate(page, count)

        return iter(row.hash for row in query)

    def get_thumbnail(self, query):
        '''Obtain an image thumbnail.

        Parameters
        ----------
        query : str
            the image hash to retrieve

        Returns
        -------
        bytes
            JPEG-encoded thumbnail

        Raises
        ------
        :exc:`~leitmotiv.exceptions.ImageNotFoundError`
            if the image hash is not in the database
        '''
        thumb = database.Image.get_thumbnail(query)
        if thumb is None:
            raise ImageNotFoundError(imghash=query)
        return thumb

    def get_path(self, query):
        '''Obtain a path to a full-resolution image.

        Parameters
        ----------
        query : str
            the image hash to retrieve

        Returns
        -------
        path-like object
            path to the full-resolution image

        Raises
        ------
        :exc:`~leitmotiv.exceptions.ImageNotFoundError`
            if the image hash is not in the database
        '''
        path = database.Image.get_path(query)
        if path is None:
            raise ImageNotFoundError(imghash=query)
        return pathlib.Path(path)

    def has_path(self, path):
        '''Check to see if an image path exists in the database.

        Parameters
        ----------
        path : path-like object
            candidate image path

        Returns
        -------
        bool
            ``True`` if the filename is in the database
        '''
        try:
            path = path.as_posix()
        except AttributeError:
            pass
        return not database.Image.get_hash(path) is None

    def append_feature(self, query, feature):
        '''Add (append) an extracted feature for an image.

        Parameters
        ----------
        query : str
            image hash
        feature : :class:`~leitmotiv.feature.ImageDescriptor` subclass
            a serializable image descriptor

        Raises
        ------
        :exc:`leitmotiv.exceptions.ImageNotFoundError`
            if the query hash is not in the database
        TypeError
            if the feature object is not a proper
            :class:`~leitmotiv.feature.ImageDescriptor` subclass
        '''
        if not isinstance(feature, ImageDescriptor):
            raise TypeError(
                'Feature object must be a subclass of ImageDescriptor')

        try:
            image = database.Image.select() \
                                  .where(database.Image.hash == query) \
                                  .get()
            with database.db.atomic():
                database.Feature.create(image=image,
                                        feature_type=feature.ftype(),
                                        model=feature.serialize())
        except database.Image.DoesNotExist as e:
            raise ImageNotFoundError(imghash=query) from e

    def get_features(self, query, ftype=None):
        '''Retrieve all features associated with a particular image hash.

        When no feature type is provided then all features are returned from
        the database.  Otherwise the list only contains the selected feature
        type and is usually just a single entry.

        Parameters
        ----------
        query : str
            image hash
        ftype : int
            feature type identifier

        Returns
        -------
        list of :class:`~leitmotiv.features.ImageDescriptor`
            a list of feature objects

        Raises
        ------
        :exc:`~leitmotiv.exceptions.ImageNotFoundError`
            if the query hash is not in the database
        '''
        try:
            image = database.Image.select() \
                                  .where(database.Image.hash == query) \
                                  .get()
        except database.Image.DoesNotExist as e:
            raise ImageNotFoundError(imghash=query) from e

        dbquery = database.Feature.select() \
                                  .where(database.Feature.image == image)
        if ftype is not None:
            dbquery = dbquery.where(database.Feature.feature_type == ftype)

        return [DESCRIPTORS[entry.feature_type].from_bytes(entry.model)
                for entry in dbquery]

    def clear_distances(self):
        '''Clears the distance table within the library.

        This will delete all of the entries within the distance table.  The
        table will need to be regenerated following a call to this method.
        '''
        with database.db.atomic():
            q = database.DistanceMeasure.delete()
            q.execute()

    def get_distances(self, query, ftype, dtype):
        '''Return a list of distances for a particular image hash.

        The distance type must be specified as comparisons with other distance
        or similarity measures is not necessarily meaningful.

        Parameters
        ----------
        query : str
            query of the hash image
        ftype : int
            feature type ID
        dtype : :class:`DistanceType`
            distance type to retrieve

        Returns
        -------
        list of ``(hash, distance)`` tuples
            the first entry is always the query hash with a distance of
            ``None``, with the remainder sorted in descending or ascending
            order depending on whether or not the value is a similarity or
            dissimilarity measure
        '''
        DM = database.DistanceMeasure.alias()
        feature = (
            database.Feature.select(database.Feature, database.Image)
                            .join(database.Image)
                            .where(database.Image.hash == query)
                            .where(database.Feature.feature_type == ftype)
                            .get()
        )

        distances = DM.select(DM).where(DM.source == feature)
        if dtype.is_symmetric():
            is_target = DM.select(DM).where(DM.target == feature)
            distances = distances | is_target

        distances = distances.order_by(DM.distance.asc())

        def sel(row):
            if row.source.image.hash == query:
                return row.target.image.hash
            else:
                return row.source.image.hash

        out = [(query, None)]
        out.extend((sel(row), row.distance) for row in distances)

        return out

    def distance_matrix(self, ftype, dtype, quiet=False, knn=None):
        '''Obtain a distance matrix from the set of feature distances.

        The structure of the matrix will depend on the type of feature and the
        distance measure used to compare them.  For example, a proper metric,
        such as Euclidean distance, will produce a symmetric matrix with zeros
        along the main diagonal.  Other "distances", such as Kullbeck-Leibler
        Divergence, are not symmetric but still have zeros along the main
        diagonal.

        This method will return a matrix for the entire database.  As this is a
        conventional Numpy array, a second list is returned that contains the
        hashes in the order that they are stored in the array.

        Parameters
        ----------
        ftype : int
            feature type ID
        dtype : :class:`DistanceType`
            distance type to retrieve
        quiet : bool
            do not show the progress bar
        knn : int
            number of k-nearest neighbours to retrieve; if ``None`` then all
            distances are returned to produce a fully-connected graph

        Returns
        -------
        distances : numpy.ndarray
            an NxN Numpy matrix
        hashes : list
            an N-length string containing the image hashes
        '''
        query = database.Image.select(database.Image.hash)
        hashes = {row.hash: id for id, row in enumerate(query)}

        if knn is None:
            distances = self._fully_connected_graph(hashes, ftype, dtype,
                                                    quiet)
        else:
            distances = self._knn_graph(hashes, ftype, dtype, quiet, knn)

        return distances, hashes

    def _fully_connected_graph(self, hashes, ftype, dtype, quiet):
        '''Generate a fully-connected distance graph.

        Parameters
        ----------
        hashes : dict
            dictionary mapping hashes to numerical IDs
        ftype : int
            feature type ID
        dtype : :class:`DistanceType`
            distance type to retrieve
        quiet : bool
            enable/disable verbose output

        Returns
        -------
        numpy.ndarray
            distance matrix
        '''
        query = (
            database.DistanceMeasure.select(database.DistanceMeasure,
                                            database.Feature.id,
                                            database.Feature.image,
                                            database.Image.hash)
                                    .join(database.Feature, JOIN.LEFT_OUTER, database.DistanceMeasure.source)    # noqa: E501
                                    .join(database.Image)
                                    .where((database.DistanceMeasure.measure == dtype.to_string()) &    # noqa: E501
                                           (database.Feature.feature_type == ftype))    # noqa: E501
        )

        distances = np.zeros((len(hashes), len(hashes)))
        for row in tqdm(query, desc='Generating distance matrix',
                        disable=quiet, ascii=True, unit='Entry'):
            src = hashes[row.source.image.hash]
            tgt = hashes[row.target.image.hash]

            distances[src, tgt] = row.distance
            if row.symmetric:
                distances[tgt, src] = row.distance

        return distances

    def _knn_graph(self, hashes, ftype, dtype, quiet, knn):
        '''Generate a kNN (or epsilon) graph.

        Parameters
        ----------
        hashes : dict
            dictionary mapping hashes to numerical IDs
        ftype : int
            feature type ID
        dtype : :class:`DistanceType`
            distance type to retrieve
        quiet : bool
            enable/disable verbose output

        Returns
        -------
        numpy.ndarray
            distance matrix; entries that are not a kNN will be set to 'inf'
            and must be accounted for
        '''
        def query(src):
            return (
                database.DistanceMeasure.select(database.DistanceMeasure,
                                                database.Feature.id,
                                                database.Feature.image,
                                                database.Image.hash)
                                        .join(database.Feature,
                                              JOIN.LEFT_OUTER,
                                              database.DistanceMeasure.source)
                                        .join(database.Image)
                                        .where(database.DistanceMeasure.source == src.source)  # noqa: E501
                                        .order_by(database.DistanceMeasure.distance)  # noqa: E501
                                        .limit(knn)
            )

        sources = (
            database.DistanceMeasure.select(database.DistanceMeasure.source,
                                            database.DistanceMeasure.measure,
                                            database.Feature.id,
                                            database.Feature.feature_type,
                                            database.Image.hash)
                                    .join(database.Feature, JOIN.LEFT_OUTER,
                                          database.DistanceMeasure.source)
                                    .join(database.Image)
                                    .where(database.DistanceMeasure.measure == dtype.to_string())  # noqa: E501
                                    .where(database.Feature.feature_type == ftype)  # noqa: E501
        )

        distances = np.zeros((len(hashes), len(hashes)))
        distances[:] = np.inf
        distances[np.diag_indices(len(hashes))] = 0

        for src in tqdm(sources, desc='Generating distance matrix',
                        disable=quiet, ascii=True, unit='Entry'):
            dists = query(src)
            for row in dists:
                src = hashes[row.source.image.hash]
                tgt = hashes[row.target.image.hash]

                distances[src, tgt] = row.distance
                if row.symmetric:
                    distances[tgt, src] = row.distance

        return distances

    def update_distances(self, distances):
        '''Update the set of distances stored in the library.

        Parameters
        ----------
        distances : list of :class:`DistanceMeasure`
            list of distances to add or update into the library; this may also
            be a generator object
        '''
        def get_feature(query, ftype):
            return (
                database.Feature.select(database.Feature.id,
                                        database.Feature.image,
                                        database.Feature.feature_type,
                                        database.Image.hash)
                                .join(database.Image)
                                .where((database.Image.hash == query) &
                                       (database.Feature.feature_type == ftype)
                                       )
                                .get()
            )

        def next_pair():
            for dm in distances:
                source = get_feature(dm.source, dm.feature_type)
                target = get_feature(dm.target, dm.feature_type)
                yield {
                    'source': source,
                    'target': target,
                    'distance': dm.distance,
                    'measure': dm.distance_type.to_string(),
                    'symmetric': dm.symmetric,
                    'similarity': dm.is_similarity
                }

        with database.db.atomic():
            insert = iter(pair for pair in next_pair())
            for ins in database.db.batch_commit(insert, 250):
                database.DistanceMeasure.create(**ins)

    def datastore(self, name='data.h5', mode='a'):
        '''Opens up an HDF5-backed data store.

        This is a thin wrapper around PyTables to open an HDF5 file.  The
        default mode is "a", meaning that if the file doesn't exist then it is
        created; otherwise, it is opened for reading and writing.  Please see
        the PyTables documentation for the `open_file()`_ function for more
        options.

        The actual HDF5 file is stored within the library's internal directory.
        It will exist alongside the SQLite database that contains the image and
        thumbnail metadata.  Modifying the HDF5 has no impact on the database.

        Parameters
        ----------
        name : str
            name of the HDF5 file
        mode : str
            file access mode

        Returns
        -------
        ``tables.File``:
            a PyTables `File`_ object

        .. open_file(): http://www.pytables.org/usersguide/libref/top_level.html#tables.open_file
        .. File: http://www.pytables.org/usersguide/libref/file_class.html#the-file-class
        '''  # noqa: E501
        hdf5path = self._libpath / name
        return tables.open_file(hdf5path.as_posix(), mode=mode)
