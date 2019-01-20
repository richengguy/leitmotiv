LEITMOTIV_VERSION_MAJOR = 0
LEITMOTIV_VERSION_MINOR = 2
LEITMOTIV_VERSION_REVISION = 0

LEITMOTIV_VERSION_STRING = '%d.%d.%d' % (LEITMOTIV_VERSION_MAJOR,
                                         LEITMOTIV_VERSION_MINOR,
                                         LEITMOTIV_VERSION_REVISION)

def version_string():
    # NOTE: This is will be removed in the future.
    return LEITMOTIV_VERSION_STRING
