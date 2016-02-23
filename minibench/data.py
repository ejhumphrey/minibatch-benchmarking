import numpy as np
import os
import uuid


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
