import json
import os
import pytest
import tempfile
import shutil

import minibench.data


def pytest_addoption(parser):
    parser.addoption(
        "--param_file", type=str, default='./params.json',
        help="JSON File of data parameters to use for benchmarking.")
    parser.addoption(
        "--workspace", type=str, default=None,
        help="Directory to use for generating data. Will attempt t "
             "temp directory if one is not provided.")
    parser.addoption(
        "--no-clean", action='store_true',
        help="If provided, will not annihilate the data generated.")


@pytest.fixture(scope="module")
def workspace(request):
    # TODO: Use pytest.config.option('--workspace')
    test_workspace = tempfile.mkdtemp()

    def fin():
        # TODO: Add `if not no-clean`
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)

    request.addfinalizer(fin)

    # Could use a py2 fallback? py3 does this cleanly.
    return test_workspace


# TODO: Ideally it'd be something like the following.
# param_file = pytest.config.getoption("--param_file")
# @pytest.fixture(scope='module')
# def data_params():
#     return json.load(open("./params.json"))
data_params = json.load(open("./params.json"))


@pytest.fixture(params=data_params,
                ids=["{}".format(p) for p in data_params],
                scope="module")
def npy_files(request, workspace):
    """Populate a temporary stash file with data."""
    return minibench.data.create_npy_collection(
        shape=request.param['shape'],
        num_items=request.param['num_items'],
        output_dir=workspace)
