# Required import
import pytest


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
