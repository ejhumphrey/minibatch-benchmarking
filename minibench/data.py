import numpy as np
import os
import uuid


def _basename(fpath):
    """Return the file's base name, e.g. '/x/y.z' -> 'y'

    Parameters
    ----------
    fpath : str
        Filepath to parse

    Returns
    -------
    fbase : str
        Basename of the file.
    """
    return os.path.splitext(os.path.basename(fpath))[0]


def random_ndarrays(shape, num_items=None, loc=0, scale=1.0,
                    dtype=np.float64, seed=12345):
    """Produce a number of key-value, normally distributed ndarrays.

    Parameters
    ----------
    shape : array_like
        Shape of the ndarrays to produce.

    num_items : int, default=None
        Number of ndarrays to produce. Infinite generator if None.

    loc, scale : scalars, defaults=(0, 1.0)
        Mean and standard deviation for the normal RNG.

    dtype : type, default=np.float64
        Datatype of the ndarrays.

    seed : int, default=12345
        Seed for the random number generator.

    Yields
    ------
    key, ndarray : str, np.ndarray
        Unique key and random value ndarray.
    """
    rng = np.random.RandomState(seed)
    count = 0
    while True:
        if num_items is not None and count >= num_items:
            break
        yield uuid.uuid4(), rng.normal(loc, scale, shape).astype(dtype=dtype)
        count += 1


def create_npy_collection(shape, num_items, output_dir, **kwargs):
    """Create a number of NPY files.

    Parameters
    ----------
    shape : array_like
        Shape of the ndarrays to produce.

    num_items : int
        Number of ndarrays to produce.

    output_dir : str
        Path under which to write data.

    kwargs : other args to pass to `random_ndarrays`.

    Returns
    -------
    new_files = list of str
        Paths to the created set of files.
    """
    data_gen = random_ndarrays(shape, num_items=num_items, **kwargs)
    new_files = []
    for key, value in data_gen:
        fpath = os.path.join(output_dir, "{}.npy".format(key))
        np.save(fpath, value)
        if not os.path.exists(fpath):
            raise ValueError("Expected output doesn't exist?? fpath={}"
                             "".format(fpath))
        new_files.append(fpath)

    return new_files


def convert_npys_to_npzs(npy_files, arr_key, output_dir):
    """Create a number of NPY files.

    Parameters
    ----------
    npy_files = list of str
        Paths to the created set of files.

    arr_key : str
        Name to write the array under in the npz archive.

    output_dir : str
        Path under which to write data.

    Returns
    -------
    npz_files : list of str
        Newly created NPZ files.
    """
    npz_files = []
    for fpath in npy_files:
        data = {arr_key: np.load(fpath)}
        npz_path = os.path.join(output_dir, "{}.npz".format(_basename(fpath)))
        np.savez(npz_path, **data)
        npz_files.append(npz_path)

    return npz_files


def convert_npys_to_h5py(npy_files, fpath):
    """Convert a collection of NPY files into h5py.

    Note: It will (should?) do this in a flat manner. This is suspected to be
    suboptimal (hence biggie).

    Parameters
    ----------
    npy_files = list of str
        Paths to the created set of files.

    fpath : str
        Filepath to write h5py file.

    Returns
    -------
    success : bool
        True if `fpath` exists, else False.
    """
    return os.path.exists(fpath)


def convert_npzs_to_biggie(npz_files, fpath):
    """Convert a collection of NPZ files into a biggie stash.

    Parameters
    ----------
    npz_files = list of str
        Paths to the created set of files.

    fpath : str
        Filepath to write biggie stash file.

    Returns
    -------
    success : bool
        True if `fpath` exists, else False.
    """
    return os.path.exists(fpath)
