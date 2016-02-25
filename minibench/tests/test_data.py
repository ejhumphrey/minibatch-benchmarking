"""
Important Development Info:
The data @fixtures live in the conftest.py, making them global
to the modules in this folder.
"""

import biggie
import h5py
import numpy as np
import os

import minibench.data


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


def test_create_npy_collection(workspace):
    shape = (20, 20)
    seed = 123
    num_items = 5

    npy_files = minibench.data.create_npy_collection(shape, num_items,
                                                     workspace, seed=seed)

    assert len(npy_files) == num_items

    data = np.load(npy_files[0])
    assert data.shape == shape


def test_convert_npys_to_npzs(npy_files, workspace):
    arr_key = 'data'
    npz_files = minibench.data.convert_npys_to_npzs(
        npy_files, arr_key, workspace)

    for npy, npz in zip(npy_files, npz_files):
        arc = np.load(npz)
        arr = np.load(npy)
        # TODO: Use yield?
        np.testing.assert_array_equal(arc[arr_key], arr)


def test_convert_npys_to_h5py(npy_files, workspace):
    fpath = os.path.join(workspace, "test_h5py.hdf5")
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


def test_convert_npzs_to_biggie(npz_files, workspace):
    fpath = os.path.join(workspace, "test_biggie.hdf5")
    success = minibench.data.convert_npzs_to_biggie(npz_files, fpath)

    assert success
    stash = biggie.Stash(fpath)
    for npz in npz_files:
        key = minibench.data.filebase(npz)
        entity = stash.get(key)
        arc = np.load(npz)
        # TODO: Use yield?
        for field in entity.keys():
            np.testing.assert_array_equal(
                # Update to `entity.get(field)` when biggie:#
                getattr(entity, field),
                arc[field])
