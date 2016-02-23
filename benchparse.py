"""Tools for handling pytest benchmarking json files.

Examples
--------
To load / parse a benchmarking file.

    > fn = ".benchmarks/Darwin-CPython-2.7-64bit/0001_foobar.json"
    > import benchparse
    > benchmarks = benchparse.load(fn)

Now that you have your benchmarks in memory, access them usefully
by getting a pandas data frame out of them:

    > df = benchmarks.to_df()

However, you may wish separate out the tests by their parameters.
A helper is provided to do this; calling to_df() in the following way
will return a dict of DataFrames, where the keys are the different
parameters.

    > df = benchmarks.to_df(split_on_params=True)
    > df.keys()

    [u'{"shape": [2048, 256], "num_items": 100}', u'{"shape": [64, 64], "num_items": 100}', u'{"shape": [64, 64], "num_items": 10}', u'{"shape": [2048, 256], "num_items": 10}']
"""

import collections
import json
import logging
import os
import pandas

logger = logging.getLogger(__name__)

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), ".benchmarks")
BENCHMARK_STR = "'py.test -vs testbench_performance.py --benchmark-save=bench1'"


def parse_benchmark_name(name):
    """Parse a benchmark name of the form:

    test_touch_npy_load_random[{u'shape': [64, 64], u'num_items': 100}

    and return the name, and the dict of parameters.

    Parameters
    ----------
    name : str
        Test name with parameterization.

    Returns
    -------
    name : str
        The name of the test ran.

    params : str
        JSON serializable string containing the parameters for this
        function.
        This is left as a string so it is easily hashable.
    """
    name_end_idx = name.find('[')
    test_name = name[:name_end_idx if name_end_idx != -1 else None]

    params_str = ""
    if name_end_idx != -1:
        params_end_idx = name.rfind(']')
        if params_end_idx != -1:
            params_str = name[name_end_idx+1:params_end_idx]
            # Now we have to do some stupid magic because the json
            #  parser is kind of dumb.
            # Make sure all the "'"s are '"'s, and that there aren't marked
            #  unicode strings (ie u")
            # Yeah, this is gross, I know.
            params_str = params_str.replace("'", '"').replace('u"', '"')

    return test_name, params_str


class PytestBenchmarkFile(object):
    """Thin wrapper on a py.test benchmarking json file with utilities for
    dealing with data stored in them."""

    def __init__(self, file_path):
        """Load the benchmark file in.

        Parameters
        ----------
        file_path : str
            File path to the benchmark file.
        """
        self._path = file_path
        with open(file_path, 'r') as fh:
            self.data = json.load(fh)

    def __contains__(self, key):
        return key in self.data

    def __getattr__(self, key):
        return self.data[key]

    def __repr__(self):
        return "PytestBenchmarkFile(file_path='{}', n_benchmarks={})".format(
            self._path, len(self.data['benchmarks']))

    def to_df(self, split_on_params=False):
        """Return the benchmarks from this file as a dataframe.

        Parameters
        ----------
        split_on_params : bool
            If true, parses the labels, and returns a list of dataframes
            with one for each parameter set.

        Returns
        -------
        df_result : pandas.DataFrame or dict
            If split_on_params is False, returns a single DataFrame
            with all results.

            If True, returns a dictionary where the keys are the
            parameter strings, and the values are dataframes
            indexed by the test name.
        """
        if not split_on_params:
            labels = [x["name"] for x in self.data['benchmarks']]
            stats = [x["stats"] for x in self.data['benchmarks']]

            return pandas.DataFrame(stats, index=labels)
        else:
            labels = collections.defaultdict(list)
            stats = collections.defaultdict(list)
            # Collect the data
            for benchmark in self.data['benchmarks']:
                name, params = parse_benchmark_name(benchmark['name'])
                labels[params] += [name]
                stats[params] += [benchmark['stats']]
            # make a dataframe for each params value and return it as
            # a dict.
            dataframes = {}
            for key in labels.keys():
                dataframes[key] = pandas.DataFrame(
                    stats[key],
                    index=labels[key])
            return dataframes


def last_benchmark():
    """Get the most recent benchmarking file produced by the py.test
    benchmarking script.

    TODO: Untested for more than one configuration.

    Returns
    -------
    benchmarks : PytestBenchmarkFile
    """
    if not os.path.exists(BENCHMARKS_DIR):
        logger.warning("No Benchmarks directory; have you run the "
                       "benchmarking script yet? \nTry:{}"
                       .format(BENCHMARK_STR))
        return None

    # Get all the json files
    file_ref = {}
    for configuration in os.listdir(BENCHMARKS_DIR):
        for filename in os.listdir(os.path.join(
                                   BENCHMARKS_DIR, configuration)):
            fp_split = filename.split('_')
            index = int(fp_split[0])
            file_ref[index] = (BENCHMARKS_DIR, configuration, filename)

    # Get the latest one. Should be the max index.
    if file_ref:
        path = os.path.join(*file_ref[max(file_ref.keys())])
        return load(path)
    # If it hasn't been run yet, there are no files available.
    else:
        logger.warning("No benchmark json files exist; try running:\n{}"
                       .format(BENCHMARK_STR))
        return None


def load(file_path):
    """Load a benchmark json file given a path.

    Parameters
    ----------
    file_path : str
        Path to a pytest benchmarking json file.

    Returns
    -------
    benchmark_data : PytestBenchmarkFile
        Loaded benchmark object with the data loaded into memory.
    """
    return PytestBenchmarkFile(file_path)
