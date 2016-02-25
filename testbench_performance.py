"""Collection of benchmarking routines for tasks related to minibatch
generation.

Sample Calls
------------
To run the tests with a bit of verbosity...

  $ py.test -vs testbench_performance.py \
    --benchmark-min-rounds=100 \
    --benchmark-sort=mean

Optionally, you can direct the test to preserve the temporary data generated
during the tests...

  $ py.test -vs --no-clean testbench_performance.py


You need to run it with the --benchmark-save=foobar parameter
for to successfully generate a json output file.

  $ py.test -vs testbench_performance.py --benchmark-save=bench1
"""
import biggie
import h5py
import logging
import pytest

import minibench


# TODO: Add verbosity to the config options.
logging.basicConfig(level=logging.INFO)

# logging.info("Using {} for workspace".format(workspace()))


# def test_touch_npy_load_random(benchmark, npys_params):
#     """Stress test random-access loads on saved NPY arrays."""
#     npy_files = npys_params[0]
#     assert benchmark(minibench.samplers.touch_npy_load, fpaths=npy_files)


def test_npy_load(benchmark, npys_params):
    npy_files, params = npys_params

    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=params['slice'],
        n_samples=None,
        with_replacement=True)
    obs = benchmark(next, sampler)
    assert obs.shape == tuple(params['slice'])


def test_npy_memmap(benchmark, npys_params):
    npy_files, params = npys_params
    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=params['slice'],
        n_samples=None,
        with_replacement=True,
        mmap_mode='r')

    obs = benchmark(next, sampler)
    assert obs.shape == tuple(params['slice'])


def test_npz_load(benchmark, npzs_params):
    npz_files, params = npzs_params
    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_npz_random_slice,
        collec=npz_files,
        shape=params['slice'],
        n_samples=None,
        with_replacement=True,
        field='data')

    obs = benchmark(next, sampler)
    assert obs.shape == tuple(params['slice'])


def test_h5py(benchmark, h5py_params):
    h5py_file, params = h5py_params

    fp = h5py.File(h5py_file)
    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_h5py_random_slice,
        collec=fp.keys(),
        shape=params['slice'],
        n_samples=None,
        with_replacement=True,
        fp=fp)

    obs = benchmark(next, sampler)
    assert obs.shape == tuple(params['slice'])


def test_biggie(benchmark, stash_params):
    stash_file, params = stash_params

    stash = biggie.Stash(stash_file)
    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_biggie_random_slice,
        collec=stash.keys(),
        shape=params['slice'],
        n_samples=None,
        with_replacement=True,
        stash=stash,
        field='data')

    obs = benchmark(next, sampler)
    assert obs.shape == tuple(params['slice'])
