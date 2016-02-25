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


def test_touch_npy_load_random(benchmark, npy_files):
    """Stress test random-access loads on saved NPY arrays."""
    assert benchmark(minibench.samplers.touch_npy_load, fpaths=npy_files)


def test_npy_load(benchmark, npy_files):
    pass


def test_npy_memmap(benchmark, npy_files):
    pass


def test_npz(benchmark, npz_files):
    pass


def test_h5py(benchmark, h5py_file):
    pass


def test_biggie(benchmark, stash_file):
    pass
