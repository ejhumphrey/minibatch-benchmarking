import numpy as np
import pescador


def random_slice_idx(arr_shape, slc_shape, seed=None):
    """Return a slice object within the valid limits of the given array shape.

    TODO:
      * Maybe as a generator? Or a generator version of the same idea.
      * Move to utils?

    Parameters
    ----------
    arr_shape : tuple
        Dimensions of the object to slice.

    scl_shape : tuple
        Dimensions of the slice to extract, or full if [None]*len(arr_shape).

    Returns
    -------
    slc : slice
        Slice object to use for indexing.
    """
    pass


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


def one_npy_load_random_slice(fpath, shape):
    """Extract random slices from an NPY file, using np.load.

    Parameters
    ----------
    fpath : str
        Path to an NPY file to load.

    shape : tuple
        Shape of the random slice to extract.

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    pass


def one_npy_memmap_random_slice(fpath, shape):
    """Extract random slices from an NPY file, using np.memmap.

    Parameters
    ----------
    fpath : str
        Path to an NPY file to load.

    shape : tuple
        Shape of the random slice to extract.

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    pass


def one_npz_load_random_slice(fpath, field, shape):
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

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    arc = np.load(fpath)
    arr_shape = np.shape(arc[field])
    while True:
        yield arc[field][random_slice_idx(arr_shape, shape)]

    # Alternatively
    # for slc in random_slice_idx(arr_shape, shape):
    #     yield arc[field][slc]


def one_h5py_random_slice(key, fp, shape):
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

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    pass


def one_biggie_random_slice(key, stash, field, shape):
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
    pass


def mux_random_slice(sampler, collec, shape, working_size=10, lam=25,
                     **kwargs):
    """Sample random slices from a collection of stuff.

    IOW: I yield observations from a consolidated stream of generators.

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

    lam : scalar, > 0
        Poisson parameter, used for dropping / replacing a given generator.

    kwargs : dict
        Key-value map for the `sampler` generator.

    Yields
    ------
    obs : np.ndarray
        Extracted observation with size `shape`.
    """
    npz_pool = [pescador.Streamer(sampler, item, shape=shape, **kwargs)
                for item in collec]

    return pescador.mux(seed_pool=npz_pool, n_samples=None, k=working_size,
                        lam=lam, pool_weights=None, with_replacement=True,
                        prune_empty_seeds=True, revive=False)
