import atexit
import numpy as np
import os
import shutil
import tempfile

import minibench.data

WORKSPACE = tempfile.mkdtemp()


def test_random_ndarrays():
    shape = (20, 20)
    seed = 123
    num_items = 5
    gen = minibench.data.random_ndarrays(shape, num_items=num_items,
                                         loc=0, scale=1.0, seed=seed)
    count = 0
    for key, arr in gen:
        assert arr.shape == shape
        count += 1

    assert count == num_items


def test_create_npy_collection():
    shape = (20, 20)
    seed = 123
    num_items = 5

    npy_files = minibench.data.create_npy_collection(shape, num_items,
                                                     WORKSPACE, seed=seed)

    assert len(npy_files) == num_items

    data = np.load(npy_files[0])
    assert data.shape == shape


def test_convert_npys_to_npzs():
    shape = (20, 20)
    seed = 123
    num_items = 5
    arr_key = 'data'

    npy_files = minibench.data.create_npy_collection(
        shape, num_items, WORKSPACE, seed=seed)
    assert len(npy_files) == num_items
    npz_files = minibench.data.convert_npys_to_npzs(
        npy_files, arr_key, WORKSPACE)

    arc = np.load(npz_files[0])
    assert arc[arr_key].shape == shape
    np.testing.assert_array_equal(arc[arr_key], np.load(npy_files[0]))


def test_convert_npys_to_h5py():
    assert False


def test_convert_npzs_to_biggie():
    assert False


def cleanup():
    """Be sure to clear out the temp directory."""
    if os.path.exists(WORKSPACE):
        shutil.rmtree(WORKSPACE)

atexit.register(cleanup)
