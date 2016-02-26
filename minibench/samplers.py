import numpy as np
import pescador


def random_slices(arr_shape, slc_shape, max_count=None, seed=None):
    """Generate slice objects within the valid limits of the given array shape.

    TODO: Move to utils?

    Parameters
    ----------
    arr_shape : tuple, len=n
        Dimensions of the object to slice.

    scl_shape : tuple, len=n
        Dimensions of the slice to extract, or full if [None]*len(arr_shape).

    Yields
    ------
    slc : slice
        Slice object to use for indexing.
    """
    if len(arr_shape) != len(slc_shape):
        raise ValueError("shapes must have same length.")
    max_dims = [(x - y + 1) for x, y in zip(arr_shape, slc_shape)]
    rng = np.random.RandomState(seed)
    max_count = np.inf if max_count is None else max_count
    while max_count:
        offsets = [rng.randint(x) for x in max_dims]
        slcs = [slice(start, start + dim)
                for start, dim in zip(offsets, slc_shape)]
        yield tuple(slcs)
        max_count -= 1


# ---------------
#  Might be worth keeping around to get a handle on the overhead introduced by
#  pescador.
# ---------------
def touch_npy_load(fpaths=None, fpath=None):
    fpath = np.random.choice(fpaths) if fpaths else fpath
    arr = np.load(fpath)
    # Force the mem-copy to the namespace.
    np.array(arr)
    return True

# @deprecated
# def touch_npz_load(field, fpaths=None, fpath=None):
#     fpath = np.random.choice(fpaths) if fpaths else fpath
#     arc = np.load(fpath)
#     np.asarray(arc[field])
#     return True
#
#
# @deprecated
# def touch_next_npz(field, fpaths):
#     fpath = fpaths.pop(0)
#     arc = np.load(fpath)
#     np.asarray(arc[field])
#     fpaths.append(fpath)
#     return True
#
#
# @deprecated
# def slice_entity(stash, field, slidx, keys=None, key=None):
#     key = np.random.choice(list(keys)) if keys else key
#     entity = stash.get(key)
#     entity.slice(field, slidx)
#     return True
#
#
# @deprecated
# def touch_entity(stash, keys=None, key=None):
#     key = np.random.choice(list(keys)) if keys else key
#     entity = stash.get(key)
#     np.asarray(entity.data)
#     return True
#
#
# @deprecated
# def slice_npz_load(field, shape, fpaths=None, fpath=None):
#     """writeme
#
#     Parameters
#     ----------
#     field : str
#         Name of the ndarray in the npz archive to slice
#
#     shape : tuple
#         Shape of the slice to extract
#     """
#     fpath = np.random.choice(fpaths) if fpaths else fpath
#     arc = np.load(fpath)
#     raise NotImplementedError("writemesucka")
#     np.asarray(arc[field])
#     return True


def one_npy_random_slice(fpath, shape, mmap_mode=None, **kwargs):
    """Extract random slices from an NPY file, using np.load.

    Parameters
    ----------
    fpath : str
        Path to an NPY file to load.

    shape : tuple
        Shape of the random slice to extract.

    mmap_mode : [None, 'r+', 'r', 'w', 'c'], default=None
        Memory mapping mode; see np.memmap for more details on the modes.

    kwargs : dict
        Arguments to forward on to the random_slices

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    np_data = np.load(fpath, mmap_mode)
    data_shape = np.shape(np_data)
    # Generate a slice in the bounds of this data.
    for new_slice in random_slices(data_shape, shape, **kwargs):
        yield {'X': np_data[new_slice]}


def one_npz_random_slice(fpath, field, shape, **kwargs):
    """Extract random slices from an NPZ archive.

    IOW: I yield observations from a bag of correlated data.

    Parameters
    ----------
    fpath : str
        Path to an NPZ file to load.

    field : str
        Key in the NPZ archive to access.

    shape : tuple
        Shape of the random slice to extract.

    kwargs : dict
        Arguments to forward on to the random_slices

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    arc = np.load(fpath)
    arr_shape = np.shape(arc[field])
    for new_slice in random_slices(arr_shape, shape, **kwargs):
        yield {'X': arc[field][new_slice]}


def one_h5py_random_slice(key, fp, shape, **kwargs):
    """Extract random slices from an h5py File.

    Parameters
    ----------
    fp : h5py.File
        An open h5py file. This may not be thread-safe / handle getting passed
        around.

    -- or --
    fpath : str
        A filepath to an h5py file. This will be slower than the above.

    key : str
        Full path into the h5py file, pointing to a dataset. In other words,
        a dataset name like '01/a3/stuff'.

    shape : tuple
        Shape of the random slice to extract.

    kwargs : dict
        Arguments to forward on to the random_slices

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    dset = fp[key]
    arr_shape = dset.shape
    for new_slice in random_slices(arr_shape, shape, **kwargs):
        yield {'X': dset[new_slice]}


def one_biggie_random_slice(key, stash, field, shape, **kwargs):
    """Extract random slices from a biggie Stash.

    Parameters
    ----------
    stash : biggie.Stash
        An instantiated biggie Stash. Note that stash objects have init args
        for keeping / reconnecting to an h5py file under the hood.

    key : str
        Pointer to an entity in the stash.

    field : str
        Name of the field (array) to slice.

    shape : tuple
        Shape of the random slice to extract.

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    entity = stash.get(key)
    arr_shape = entity[field].shape
    for new_slice in random_slices(arr_shape, shape, **kwargs):
        yield {'X': entity[field].slice(new_slice)}


def mux_random_slice(sampler, collec, shape, working_size=10, lam=25,
                     pool_weights=None, with_replacement=True, n_samples=None,
                     prune_empty_seeds=True, revive=False, **kwargs):
    """Sample random slices from a collection of stuff.

    Parameters
    ----------
    sampler : func
        A bag generator, from above.

    collec : iterable
        An iterable collection of items, over which samplers will be created.
        NOTE: This must be the first (positional) argument of sampler, or baby
        kittens will die.

    shape : tuple
        Shape of the random slice to extract.

    working_size : int, default=10, > 0
        Number of generators to keep alive at any point in time.

    pescador.mux parameters
    -------------------------
    lam : scalar, default=25
    pool_weights : array_like, default=None
    with_replacement : bool, default=True
    prune_empty_seeds : bool=True
    revive : bool=False
        See pescador.mux for more details.

    kwargs : dict
        Key-value map for the `sampler` generator.

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    npz_pool = [pescador.Streamer(sampler, item, shape=shape, **kwargs)
                for item in collec]

    return pescador.mux(seed_pool=npz_pool, n_samples=n_samples,
                        k=working_size, lam=lam, pool_weights=pool_weights,
                        with_replacement=with_replacement,
                        prune_empty_seeds=prune_empty_seeds, revive=revive)


def zmq_random_slice(**kwargs):
    """Thin wrapper on mux_random_slice which just adds zmq streaming to it."""
    streamer = pescador.Streamer(mux_random_slice(**kwargs))
    return pescador.zmq_stream(streamer)
    # return streamer
