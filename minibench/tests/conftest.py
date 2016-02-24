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
