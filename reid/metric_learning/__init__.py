from __future__ import absolute_import


from .euclidean import Euclidean

__factory = {
    'euclidean': Euclidean
}


def get_metric(algorithm, *args, **kwargs):
    if algorithm not in __factory:
        raise KeyError("Unknown metric:", algorithm)
    return __factory[algorithm](*args, **kwargs)
