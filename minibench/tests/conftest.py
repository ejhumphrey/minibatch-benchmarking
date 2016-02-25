import os
import pytest
import shutil
import tempfile

import minibench.data


@pytest.fixture(scope="module")
def workspace(request):
    test_workspace = tempfile.mkdtemp()

    def fin():
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)

    request.addfinalizer(fin)

    # Could use a py2 fallback? py3 does this cleanly.
    return test_workspace


@pytest.fixture()
def npy_files(workspace):
    shape = (20, 20)
    seed = 123
    num_items = 5
    return minibench.data.create_npy_collection(
        shape, num_items, workspace, seed=seed)


@pytest.fixture()
def npz_files(npy_files, workspace):
    return minibench.data.convert_npys_to_npzs(
        npy_files, 'data', workspace)


@pytest.fixture()
def h5py_file(npy_files, workspace):
    fpath = os.path.join(workspace, "test_h5py.hdf5")
    minibench.data.convert_npys_to_h5py(npy_files, fpath)
    return fpath


@pytest.fixture()
def stash_file(npz_files, workspace):
    fpath = os.path.join(workspace, "test_biggie.hdf5")
    minibench.data.convert_npzs_to_biggie(npz_files, fpath)
    return fpath
