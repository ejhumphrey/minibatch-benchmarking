"""Collection of benchmarking routines for tasks related to minibatch
generation.

Sample Calls
------------
To run the tests with a bit of verbosity...

  $ py.test -vs testbench_performance.py


Optionally, you can direct the test to preserve the temporary data generated
during the tests...

  $ py.test -vs --no-clean testbench_performance.py


You need to run it with the --benchmark-save=foobar parameter
for to successfully generate a json output file.

  $ py.test -vs testbench_performance.py --benchmark-save=bench1
"""

import logging
import pytest

import minibench


# TODO: Add verbosity to the config options.
logging.basicConfig(level=logging.INFO)

# TODO: Can we still do this with fixtures?
# logging.info("Using {} for workspace".format(workspace))


# def test_touch_npy_load_random(benchmark, npys_params):
#     """Stress test random-access loads on saved NPY arrays."""
#     npy_files = npys_params[0]
#     assert benchmark(minibench.samplers.touch_npy_load, fpaths=npy_files)


def test_npy_load(benchmark, npys_params):
    npy_files, params = npys_params
    # Todo / question: this doens't necessarily and maybe
    #  shouldn't be the same as the size of the data generated.
    slice_shape = params['n_samples']
    # What to set this to?
    # We could also do separate experiments for sampling full
    # files vs parts?
    n_samples = params['n_samples']
    assert benchmark(minibench.samplers.mux_random_slice,
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=slice_shape,
        n_samples=n_samples,
        with_replacement=False)


def test_npy_memmap(benchmark, npys_params):
    npy_files, params = npys_params
    slice_shape = params['n_samples']
    n_samples = params['n_samples']
    assert benchmark(minibench.samplers.mux_random_slice,
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=slice_shape,
        n_samples=n_samples,
        with_replacement=False,
        mmap_mode='r')


def test_npz_load(benchmark, npzs_params):
    npz_files, params = npzs_params
    slice_shape = params['n_samples']
    n_samples = params['n_samples']
    assert benchmark(minibench.samplers.mux_random_slice,
        sampler=minibench.samplers.one_npz_random_slice,
        collec=npz_files,
        shape=slice_shape,
        n_samples=n_samples,
        with_replacement=False,
        field='data')


def test_h5py(benchmark, h5py_file):
    pass


def test_biggie(benchmark, stash_file):
    pass
