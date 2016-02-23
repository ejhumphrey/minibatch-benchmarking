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

"""

import collections
import json
import logging
import pandas

logger = logging.getLogger(__name__)


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
        with open(file_path, 'r') as fh:
            self.data = json.load(fh)

    def __contains__(self, key):
        return key in self.data

    def __getattr__(self, key):
        return self.data[key]

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
