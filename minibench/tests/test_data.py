import numpy as np
import tempfile

import minibench.data


def test_random_ndarrays():
    shape = (20, 20)
    seed = 123
    num_items = 5
    gen = minibench.data.random_ndarrays(shape, num_items=num_items,
                                         loc=0, scale=1.0, seed=seed)
    count = 0
    for key, arr in gen:
        assert arr.shape == shape
        count += 1

    assert count == num_items


def test_create_npy_collection():
    shape = (20, 20)
    seed = 123
    num_items = 5

    tempdir = tempfile.mkdtemp()
    new_files = minibench.data.create_npy_collection(shape, num_items,
                                                     tempdir, seed=seed)

    assert len(new_files) == num_items

    data = np.load(new_files[0])
    assert data.shape == shape
