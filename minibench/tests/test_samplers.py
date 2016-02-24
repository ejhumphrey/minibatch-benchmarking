import numpy as np
import os
import pytest
import tempfile
import uuid

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


@pytest.mark.skipif(True, reason='todo')
def test_one_npy_load_random_slice():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_one_npy_memmap_random_slice():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_one_npz_load_random_slice():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_one_h5py_random_slice():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_one_biggie_random_slice():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_mux_random_slice():
    assert False
