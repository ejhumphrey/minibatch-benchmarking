import atexit
import h5py
import numpy as np
import os
import pytest
import shutil
import tempfile

import minibench.data

# Could use a py2 fallback? py3 does this cleanly.
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


@pytest.fixture()
def npy_files():
    shape = (20, 20)
    seed = 123
    num_items = 5
    return minibench.data.create_npy_collection(
        shape, num_items, WORKSPACE, seed=seed)


def test_convert_npys_to_npzs(npy_files):
    arr_key = 'data'
    npz_files = minibench.data.convert_npys_to_npzs(
        npy_files, arr_key, WORKSPACE)

    for npy, npz in zip(npy_files, npz_files):
        arc = np.load(npz)
        arr = np.load(npy)
        # TODO: Use yield?
        np.testing.assert_array_equal(arc[arr_key], arr)


def test_convert_npys_to_h5py(npy_files):
    fpath = os.path.join(WORKSPACE, "test_h5py.hdf5")
    success = minibench.data.convert_npys_to_h5py(
        npy_files, fpath)

    assert success
    fhandle = h5py.File(fpath)
    for npy in npy_files:
        key = minibench.data.filebase(npy)
        dset = fhandle[key]
        arr = np.load(npy)
        # TODO: Use yield?
        np.testing.assert_array_equal(dset.value, arr)


def test_convert_npzs_to_biggie():
    assert False


def cleanup():
    """Be sure to clear out the temp directory."""
    if os.path.exists(WORKSPACE):
        shutil.rmtree(WORKSPACE)

atexit.register(cleanup)
