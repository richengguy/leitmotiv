import abc
import struct


MAGIC_NUMBER = 0xFD


class ImageDescriptor(abc.ABC):
    '''Defines the top-level class for all image feature descriptors.

    This interface must be implemented by any feature descriptors used by the
    application.  It provides the necessary infrastructure for storing
    descriptors in a database and then recreating them.  The majority of these
    are static methods as there are potentially an infinite number of
    descriptor instances but they all can be serialized and deserialized.
    '''
    @abc.abstractmethod
    def serialize(self):
        '''Serialize the descriptor object.

        A subclass will need to implement this if the data it contains needs to
        be exported.

        Returns
        -------
        bytes
            a bytes object containing the serialized form of the descriptor
        '''

    @staticmethod
    @abc.abstractmethod
    def from_bytes(buffer):
        '''Generate a new ImageDescriptor from a byte array.

        This has no preferred implementation and it is the subclass'
        responsibility to provide an implementation.  The only requirement is
        that the buffer is converted into an :class:`ImageDescriptor` object.

        Parameters
        ----------
        buffer: bytes
            byte array containing the serialized descriptor

        Returns
        -------
        ImageDescriptor
            the initialized descriptor
        '''

    @staticmethod
    @abc.abstractmethod
    def ftype():
        '''Return the type's identifier.

        Returns
        -------
        int
            a unique ID value
        '''

    @staticmethod
    @abc.abstractmethod
    def _hdrfmt():
        '''Return the header format used for storing descriptors.

        Returns
        -------
        str
            a ``struct``-compatible string
        '''

    @classmethod
    def _pack_header(cls, *data):
        '''Generate the header necessary for serialization.

        The method will take in a list of arguments that are used by Python's
        ``struct`` module.

        Returns
        -------
        bytes
            an array of bytes containing the generated header
        '''
        return struct.pack('!BB' + cls._hdrfmt(), MAGIC_NUMBER, cls.ftype(),
                           *data)

    @classmethod
    def _unpack_header(cls, buffer):
        '''Parse a header using the provided header format.

        This will emit an error if the first byte is not ``0xFD`` to indicate
        that this is a feature descriptor.

        Parameters
        ----------
        buffer: bytes
            the buffer to unpack

        Returns
        -------
        ftype: int
            the feature type ID
        header: tuple
            a tuple containing the unpacked header, minus the magic number
        offset: int
            the offset to the start of the descriptor data
        '''
        hdrfmt = '!BB' + cls._hdrfmt()

        nbytes = struct.calcsize(hdrfmt)
        data = struct.unpack_from(hdrfmt, buffer)

        if data[0] != MAGIC_NUMBER:
            raise ValueError('Descriptor does not start with 0xFD')

        return data[1], data[2:], nbytes
