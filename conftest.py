import json
import os
import pytest
import tempfile
import shutil
import uuid

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


@pytest.fixture()
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
                ids=["{}".format(p) for p in data_params])
def npys_params(request, workspace):
    """Populate a temporary stash file with data.

    Returns
    -------
    npy_files : list of str
        Collection of npy files on disk.

    param : obj
        Each data param item.
    """
    return minibench.data.create_npy_collection(
        shape=request.param['shape'],
        num_items=request.param['num_items'],
        output_dir=workspace), \
        request.param


@pytest.fixture(params=data_params,
                ids=["{}".format(p) for p in data_params])
def npzs_params(request, workspace):
    npz_files = minibench.data.convert_npys_to_npzs(
        minibench.data.create_npy_collection(
                shape=request.param['shape'],
                num_items=request.param['num_items'],
                output_dir=workspace), 'data', workspace)
    return npz_files, request.param


@pytest.fixture(params=data_params,
                ids=["{}".format(p) for p in data_params])
def h5py_params(request, workspace):
    fpath = os.path.join(workspace, "{}.hdf5".format(str(uuid.uuid4())))
    npy_files = minibench.data.create_npy_collection(
        shape=request.param['shape'],
        num_items=request.param['num_items'],
        output_dir=workspace)
    minibench.data.convert_npys_to_h5py(npy_files, fpath)
    return fpath, request.param


@pytest.fixture(params=data_params,
                ids=["{}".format(p) for p in data_params])
def stash_params(request, workspace):
    fpath = os.path.join(workspace, "{}.hdf5".format(str(uuid.uuid4())))
    npz_files = minibench.data.convert_npys_to_npzs(
        minibench.data.create_npy_collection(
                shape=request.param['shape'],
                num_items=request.param['num_items'],
                output_dir=workspace), 'data', workspace)
    minibench.data.convert_npzs_to_biggie(npz_files, fpath)
    return fpath, request.param
