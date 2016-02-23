import numpy as np


def slice_entity(stash, field, slidx, keys=None, key=None):
    key = np.random.choice(list(keys)) if keys else key
    entity = stash.get(key)
    entity.slice(field, slidx)
    return True


def touch_entity(stash, keys=None, key=None):
    key = np.random.choice(list(keys)) if keys else key
    entity = stash.get(key)
    np.asarray(entity.data)
    return True


def slice_npz_load(field, fpaths=None, fpath=None):
    fpath = np.random.choice(fpaths) if fpaths else fpath
    arc = np.load(fpath)
    np.asarray(arc[field])
    return True


def touch_npz_load(field, fpaths=None, fpath=None):
    fpath = np.random.choice(fpaths) if fpaths else fpath
    arc = np.load(fpath)
    np.asarray(arc[field])
    return True


def touch_next_npz(field, fpaths):
    fpath = fpaths.pop(0)
    arc = np.load(fpath)
    np.asarray(arc[field])
    fpaths.append(fpath)
    return True
