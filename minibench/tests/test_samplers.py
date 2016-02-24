"""
Important Development Info:
The data @fixtures live in the conftest.py, making them global
to the modules in this folder.
"""
import biggie
import h5py
import numpy as np
import pytest

import minibench.data
import minibench.samplers


def __asseq(a, b):
    assert a == b


def test_random_slices():
    n, m = 8, 5
    max_count = 3
    x = np.arange(n*m)
    slice_shape = (3,)
    randomize_slices = minibench.samplers.random_slices(
        x.shape, slice_shape, max_count=max_count)
    for slices in randomize_slices:
        x_sub = x[slices]
        yield __asseq, x_sub.shape, slice_shape

    x = x.reshape(n, m)
    slice_shape = (3, 2)
    randomize_slices = minibench.samplers.random_slices(
        x.shape, slice_shape, max_count=max_count)
    for slices in randomize_slices:
        x_sub = x[slices]
        yield __asseq, x_sub.shape, slice_shape


@pytest.mark.skipif(True, reason='todo')
def test_touch_npy_load():
    assert False


def test_one_npy_load_random_slice(npy_files):
    max_count = 3

    for fpath in npy_files:
        # Get a few of one kind of shape
        slice_shape = (3, 2)
        sampler = minibench.samplers.one_npy_random_slice(
            fpath, slice_shape, mmap_mode=None, max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape

        # Get a few of a different shape
        slice_shape = (1, 20)
        sampler = minibench.samplers.one_npy_random_slice(
            fpath, slice_shape, mmap_mode=None, max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape


def test_one_npy_memmap_random_slice(npy_files):
    max_count = 3

    for fpath in npy_files:
        # Get a few of one kind of shape
        slice_shape = (3, 2)
        sampler = minibench.samplers.one_npy_random_slice(
            fpath, slice_shape, mmap_mode='r', max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape

        # Get a few of a different shape
        slice_shape = (1, 20)
        sampler = minibench.samplers.one_npy_random_slice(
            fpath, slice_shape, mmap_mode='r', max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape


def test_one_npz_load_random_slice(npz_files):
    max_count = 3
    field = 'data'
    for fpath in npz_files:
        # Get a few of one kind of shape
        slice_shape = (3, 2)
        slicer = minibench.samplers.one_npz_random_slice(
            fpath, field, slice_shape, max_count=max_count)

        for rand_slice in slicer:
            assert rand_slice.shape == slice_shape

        # Get a few of a different shape
        slice_shape = (1, 20)
        slicer = minibench.samplers.one_npz_random_slice(
            fpath, field, slice_shape, max_count=max_count)

        for rand_slice in slicer:
            assert rand_slice.shape == slice_shape


def test_one_h5py_random_slice(h5py_file):
    max_count = 3

    fp = h5py.File(h5py_file)
    for key in fp:
        # Get a few of one kind of shape
        slice_shape = (3, 2)
        sampler = minibench.samplers.one_h5py_random_slice(
            key, fp, slice_shape, max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape

        # Get a few of a different shape
        slice_shape = (1, 20)
        sampler = minibench.samplers.one_h5py_random_slice(
            key, fp, slice_shape, max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape


def test_one_biggie_random_slice(stash_file):
    max_count = 3
    field = 'data'

    stash = biggie.Stash(stash_file)
    for key in stash.keys():
        # Get a few of one kind of shape
        slice_shape = (3, 2)
        sampler = minibench.samplers.one_biggie_random_slice(
            key, stash, field, slice_shape, max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape

        # Get a few of a different shape
        slice_shape = (1, 20)
        sampler = minibench.samplers.one_biggie_random_slice(
            key, stash, field, slice_shape, max_count=max_count)

        for rand_slice in sampler:
            assert rand_slice.shape == slice_shape


def test_mux_random_slice(npy_files):
    slice_shape = (3, 2)
    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files, shape=slice_shape, mmap_mode='r',
        max_count=3, with_replacement=False)

    # Run the sampler to exhaustion.
    for rand_slice in sampler:
        assert rand_slice.shape == slice_shape
