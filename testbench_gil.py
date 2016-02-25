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
import pytest

import minibench

logging.basicConfig(level=logging.INFO)


def heavy_cpu_fx(x):
    for n in range(1000):
        np.power(np.sqrt(np.abs(np.fft(x))), 2)
        # You could sleep here or not just for fun / extra cycles.
    return x


def example_training_loop(sampler, train):
    """
    Parameters
    ----------
    sampler : generator
        Generator which yields np.ndarrays

    train : function
        A do-the-thing funciton that uses CPU cycles.
    """
    # Get some data to start with
    working_data = next(sampler)

    for next_data in sampler:
        err = train(working_data)

        working_data = next_data

    return err




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

