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
import logging
import numpy as np
import pytest

import theano
import theano.tensor as T

import minibench

logging.basicConfig(level=logging.INFO)


def theano_test_fx(n_dots, w_shape):
    # input vector.
    x = T.matrix('x')
    # A fake weight matrix.
    w = theano.shared(np.random.random(w_shape), name='w')

    result = x
    for i in range(n_dots):
        result = T.dot(result, w)
        # Some more computation, and keep the value
        # from exploding (although who cares, really).
        if i % 3 == 0:
            result = T.sqrt(result)
    result = T.mean(result)
    return theano.function(inputs=[x],
                           outputs=result,
                           updates=[(w, w-.0001)])


@pytest.fixture
def npy_sampler_params(benchmark, npys_params):
    npy_files, params = npys_params

    sampler = minibench.samplers.mux_random_slice(
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=params['slice'],
        n_samples=100,
        lam=params['lam'],
        working_size=params['working_size'],
        with_replacement=True)
    return sampler, params


@pytest.fixture
def zmq_sampler_params(benchmark, npys_params):
    npy_files, params = npys_params

    sampler = minibench.samplers.zmq_random_slice(
        sampler=minibench.samplers.one_npy_random_slice,
        collec=npy_files,
        shape=params['slice'],
        n_samples=100,
        lam=params['lam'],
        working_size=params['working_size'],
        with_replacement=True)
    return sampler, params


def run_theano_fx(train_fx, sampler):
    """Make some data and make sure the function runs"""
    errs = []
    for sample in sampler:
        # the np.array is required here because pescador is producing
        # unaligned numpy arrays :(
        errs += [train_fx(np.array(sample['X']))]
    return np.mean(errs)


def test_pescador_same_thread(benchmark, npy_sampler_params):
    sampler, params = npy_sampler_params

    train_fx = theano_test_fx(n_dots=5, w_shape=(params['slice'][-1],)*2)

    # run_theano_fx(train_fx, sampler, weights)
    assert benchmark(run_theano_fx, train_fx, sampler)


def test_zmq_sampling_no_copy(benchmark, zmq_sampler_params):
    sampler, params = zmq_sampler_params

    train_fx = theano_test_fx(n_dots=5, w_shape=(params['slice'][-1],)*2)

    # run_theano_fx(train_fx, sampler, weights)
    # Forcing a copy in the run_ function, so this is actually
    #  doing the copy all the time, even though this says no copy.
    assert benchmark(run_theano_fx, train_fx, sampler)


# def test_zmq_sampling_copy():
#     pass
