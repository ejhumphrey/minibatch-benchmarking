import os

import minibench.parse as benchparse


def test_parse_benchmark_name():
    def __test(thing, expected_thing):
        assert thing == expected_thing, "{} != {}".format(
            thing, expected_thing)

    # Try the basic case
    test_name = "test_touch_npy_load_random[{u'shape': [64, 64], " \
                "u'num_items': 100}]"
    name, params = benchparse.parse_benchmark_name(test_name)
    yield __test, name, "test_touch_npy_load_random"
    yield __test, params, '{"shape": [64, 64], "num_items": 100}'

    test_name = "test_touch_npy_load_random"
    name, params = benchparse.parse_benchmark_name(test_name)
    yield __test, name, "test_touch_npy_load_random"
    yield __test, params, ""

    test_name = "test_touch_npy_load_random[boofsaas lj t"
    name, params = benchparse.parse_benchmark_name(test_name)
    yield __test, name, "test_touch_npy_load_random"
    yield __test, params, ""


def test_load_benchfile():
    bench_fixture_path = os.path.join(os.path.dirname(__file__),
                                      "fixtures", "benchfixture.json")

    def __test_exists(thing):
        assert thing is not None

    def __test_is(thing, test_type):
        assert isinstance(thing, test_type)

    def __test_has(thing, key):
        assert key in thing

    benchmarks = benchparse.load(bench_fixture_path)
    # Make sure we got something back
    yield __test_exists, benchmarks

    # Make sure the thing is of a sane type
    yield __test_is, benchmarks, benchparse.PytestBenchmarkFile

    yield __test_has, benchmarks, "version"
    yield __test_has, benchmarks, "commit_info"
    yield __test_has, benchmarks, "benchmarks"

    yield __test_exists, benchmarks.to_df()
    df_data = benchmarks.to_df(split_on_params=True)
    yield __test_exists, df_data
    yield __test_is, df_data, dict
