import peewee

db = peewee.SqliteDatabase(None, pragmas=(('foreign_keys', 'on'),))


class BaseModel(peewee.Model):
    '''Base model for all database objects.'''
    class Meta:
        database = db


class Image(BaseModel):
    '''Store information about images.

    Attributes
    ----------
    path : str
        path to an image stored somewhere on disk
    datetime : str
        time when the original image was taken
    hash : str
        image file hash
    thumb : bytes
        JPEG-encoded image thumbnail
    '''
    path = peewee.TextField(unique=True)
    datetime = peewee.DateTimeField()
    hash = peewee.CharField(unique=True, max_length=64)
    thumb = peewee.BlobField()

    class Meta:
        table_name = 'images'

    @classmethod
    def size(cls):
        '''Obtain the number of images in the database.

        Returns
        -------
        int
            the number of images in the database
        '''
        return Image.select().count()

    @classmethod
    def get_hash(cls, path):
        '''Return the hash of a particular image, given its file name.

        Parameters
        ----------
        path : str
            image path

        Returns
        -------
        str
            the associated hash or ``None`` if it does not exist
        '''
        try:
            image = Image.select(Image.hash).where(Image.path == path).get()
        except Image.DoesNotExist:
            return None
        return image.hash

    @classmethod
    def get_path(cls, query):
        '''Return the path to a particular image.

        Parameters
        ----------
        query: str
            a 64-character sha256 hash

        Returns
        -------
        str or ``None``
            the image path or ``None`` if it does not exist
        '''
        try:
            image = Image.select(Image.path).where(Image.hash == query).get()
        except Image.DoesNotExist:
            return None
        return image.path

    @classmethod
    def get_thumbnail(cls, query):
        '''Retrieve a thumbnail given an image hash.

        Parameters
        ----------
        query : str
            a 64-character sha256 hash

        Returns
        -------
        bytes or ``None``
            a JPEG-encoded thumbnail; will be ``None`` if the hash does not
            exist in the database
        '''
        try:
            image = Image.select().where(Image.hash == query).get()
        except Image.DoesNotExist:
            return None

        return image.thumb


class Feature(BaseModel):
    '''A set of features that describe an image.

    This can be any number of entities, from a statistical model trained on a
    particular image to a set of features (e.g. ORB, HOG, etc.) that have been
    pulled from the image.  The entry will store a reference to the image,
    the feature type and the associated data.  An image may have multiple
    descriptors associated with it.

    Attributes
    ----------
    image : :class:`Image`
        the image that the feature is associated with
    feature_type : int
        feature type ID
    model : blob
        the extracted feature data; the application is responsible for
        properly interpreting the contents of this blob
    '''
    image = peewee.ForeignKeyField(Image)
    feature_type = peewee.IntegerField()
    model = peewee.BlobField()

    class Meta:
        table_name = 'features'

    @classmethod
    def for_image(cls, query):
        '''Obtain all features for some image.

        Parameters
        ----------
        query : str
            expected sha256 hash

        Returns
        -------
        list of :class:`Feature`
            a list of features for that image, in no particular order
        '''
        return Feature.select().join(Image).where(Image.hash == query)


class FeatureVisualization(BaseModel):
    '''Stores a visualization of a particular feature type.

    Certain features may be easy to visualize, even after extracting them from
    an image.  The best way to do this depends entirely on the nature of that
    feature.  For instance, a histogram is best visualized as plot while a GMM
    may be better visualized as a set of ellipses.  The database model makes no
    assumptions on the nature of the visualization; it assumes that it is an
    image of some known type.  A feature may have multiple representations if
    so appropriate.

    Attributes
    ----------
    feature : :class:`Feature`
        feature being visualized
    path : str
        path to where the image is stored
    content_type : str
        image content-type for HTTP requests
    viz_type : int
        application-defined ID used to indicate what type of visualization is
        being shown
    '''
    feature = peewee.ForeignKeyField(Feature)
    content_type = peewee.TextField(null=False)
    path = peewee.TextField(null=False)
    viz_type = peewee.IntegerField(null=False)

    class Meta:
        indexes = (
            (('feature', 'viz_type'), True),
        )


class DistanceMeasure(BaseModel):
    '''Stores the "distance" between any two sets of features in the database.

    A distance, in this sense, is just measure of dissimilarity between two
    entities, where a distance may only be zero if the two entities are
    identical.  This is not necessarily the formal definition of a distance,
    which describes a metric on some space.  For example, a divergence between
    two probability distributions is a measure of dissimilarity but is not
    necessarily symmetric.

    Attributes
    ----------
    source : :class:`Feature`
        the first element
    target : :class:`Feature`
        the second element
    distance : double
        the distance from the source to the target
    measure : str
        type of measure (e.g. Euclidean distance between image feature vectors)
    symmetric : bool
        the distance measure is symmetric, meaning that the distance from A to
        B is the same as the distance from B to A
    similarity : bool
        stored values are a measure of *similarity* rather than dissimilarity
    '''
    source = peewee.ForeignKeyField(Feature)
    target = peewee.ForeignKeyField(Feature)
    distance = peewee.DoubleField()
    measure = peewee.TextField()
    symmetric = peewee.BooleanField()
    similarity = peewee.BooleanField()

    class Meta:
        indexes = (
            (('source', 'target'), True),
        )


def create_tables():
    '''Utility function to create the database tables.

    This only needs to be called when the database is first created.
    '''
    with db.atomic():
        db.create_tables([
            Image, Feature, FeatureVisualization, DistanceMeasure
        ])
