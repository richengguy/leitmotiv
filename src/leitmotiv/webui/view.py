import flask

import numpy as np

import sklearn.cluster

from leitmotiv.math import graph
from leitmotiv.features import ImageGMM
from leitmotiv.library import Library, DistanceType


view = flask.Blueprint('view', __name__)


@view.route('/')
def index():
    with Library() as library:
        hashes = iter((h, None) for h in library.get_hashes())
        return flask.render_template('index.html', images=hashes)


@view.route('/clusters')
def clusters():
    with Library() as library:
        with library.datastore(mode='r') as hdf5:
            distances = np.array(hdf5.root.distances)
            hashes = dict(zip(hdf5.root.hashes.indices, hdf5.root.hashes.keys))
            similarities = graph.distance_to_similarity(distances)
            centres, labels = sklearn.cluster.affinity_propagation(
                similarities, preference=similarities.min())

    clusters = []
    for i in range(len(centres)):
        items = []
        for ind, lbl in enumerate(labels):
            if lbl == i:
                items.append((distances[centres[i], ind], hashes[ind]))
        items.sort(key=lambda x: x[0])
        clusters.append(items)

    centres = [hashes[ind] for ind in centres]
    return flask.render_template('cluster.html', centres=centres,
                                 hashes=hashes, clusters=clusters)


@view.route('/thumb/<hash>')
def thumb(hash):
    with Library() as library:
        resp = flask.make_response(library.get_thumbnail(hash))
        resp.headers['Content-Type'] = 'image/jpeg'
        return resp


@view.route('/ranked/<hash>')
def ranked(hash):
    with Library() as library:
        query = library.get_distances(hash, ImageGMM.ftype(),
                                      DistanceType.JENSEN_SHANNON)
        return flask.render_template('index.html', images=query, sorted=True)
