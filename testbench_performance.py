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

import atexit
import json
import logging
import os
import shutil
import pytest
import tempfile as tmp

import minibench

# TODO: Yank these two from the config options.
WORKSPACE = tmp.mkdtemp()
DATA_PARAMS = json.load(open("./params.json"))

# TODO: Add verbosity to the config options.
logging.basicConfig(level=logging.INFO)

logging.info("Using {} for workspace".format(WORKSPACE))


@pytest.fixture(params=DATA_PARAMS,
                ids=["{}".format(p) for p in DATA_PARAMS],
                scope="module")
def npy_files(request):
    """Populate a temporary stash file with data."""
    fset = minibench.data.create_npy_collection(
        shape=request.param['shape'],
        num_items=request.param['num_items'],
        output_dir=WORKSPACE)
    return fset


def test_touch_npy_load_random(benchmark, npy_files):
    """Stress test random-access loads on saved NPY arrays."""
    assert benchmark(minibench.getters.touch_npy_load, fpaths=npy_files)


# Don't call this `teardown`; reserved function name.
def cleanup():
    no_clean = pytest.config.getoption("--no-clean")
    if os.path.exists(WORKSPACE) and not no_clean:
        logging.info("Removing {} and its contents.".format(WORKSPACE))
        shutil.rmtree(WORKSPACE)

atexit.register(cleanup)
