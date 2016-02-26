"""Collection of benchmarking routines for tasks related testing
minibatch generation with Pescador.

Sample Calls
------------
To run the tests with a bit of verbosity...

  $ py.test -vs testbench_gil.py \
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
import numpy as np
import pescador
import pytest

import theano
import theano.tensor as T

import minibench

logging.basicConfig(level=logging.INFO)


def theano_test_fx(n_dots):
    # input vector.
    x = T.matrix('x')
    # A fake weight matrix.
    w = T.matrix('w')

    result = x
    for i in range(n_dots):
        result = T.dot(result, w)
        # Some more computation, and keep the value
        # from exploding (although who cares, really).
        if i % 3 == 0:
            result = T.sqrt(result)
    result = T.mean(result)
    return theano.function([x, w], [result])


@pytest.fixture
def npy_sampler(benchmark, npys_params):
    npy_files, params = npys_params

    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=params['slice'],
        n_samples=200,
        lam=params['lam'],
        working_size=params['working_size'],
        with_replacement=True)
    return sampler


@pytest.fixture
def 


def test_theano_test_fx():
    """Make some data and make sure the function runs"""
    samples = np.random.random([12, 16])
    weights = np.random.random((samples.shape[1],)*2)

    train = theano_test_fx(n_dots=10)
    err = train(samples, weights)


def test_pescador_same_thread():
    pass


def test_zmq_sampling_no_copy():
    pass


def test_zmq_sampling_copy():
    pass


# def test_npy_load(benchmark, npys_params):
#     npy_files, params = npys_params

#     sampler = minibench.samplers.mux_random_slice(
#         sampler=minibench.samplers.one_npy_random_slice,
#         collec=npy_files,
#         shape=params['slice'],
#         n_samples=None,
#         lam=params['lam'],
#         working_size=params['working_size'],
#         with_replacement=True)
#     obs = benchmark(next, sampler)
#     assert obs.shape == tuple(params['slice'])


# def test_npy_memmap(benchmark, npys_params):
#     npy_files, params = npys_params
#     sampler = minibench.samplers.mux_random_slice(
#         sampler=minibench.samplers.one_npy_random_slice,
#         collec=npy_files,
#         shape=params['slice'],
#         n_samples=None,
#         lam=params['lam'],
#         working_size=params['working_size'],
#         with_replacement=True,
#         mmap_mode='r')

#     obs = benchmark(next, sampler)
#     assert obs.shape == tuple(params['slice'])

