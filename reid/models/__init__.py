from __future__ import absolute_import

from .newresnet import *

__factory = {
    'ft_net': ft_net
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
