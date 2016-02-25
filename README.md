# minibatch-benchmarking

Benchmarking tools and visualizations for minibatch building in Python.

[![Build Status](https://travis-ci.org/ejhumphrey/minibench-benchmarking.svg?branch=master)](https://travis-ci.org/ejhumphrey/minibench-benchmarking)
[![Coverage Status](https://coveralls.io/repos/github/ejhumphrey/minibatch-benchmarking/badge.svg?branch=master)](https://coveralls.io/github/ejhumphrey/minibatch-benchmarking?branch=master)

## Goals

Have you ever had to train an on-line learning algorithm and though, "gee, what's the best way to prepare, access, and sample my data so I'm not wasting clock cycles?"

We have. One too many times, in fact. And so, this is an effort to get to the bottom of it, once and for all.


## Dependencies

- pytest
- pytest-benchmarking
- numpy
- biggie
- pescador


## Running

This is very much a your mileage may vary kind of operation, and will be tightly coupled to your hardware configuration and data of interest. To get an idea of what you're up against, call the primary testbench routine from the cloned repository:

```
git clone https://github.com/ejhumphrey/minibatch-benchmarking.git
cd minibatch-benchmarking
pip install -e ./  # Pick up yr deps
py.test -vs testbench_performance.py \
    --benchmark-min-rounds=100 \
    --benchmark-sort=mean \
    --benchmark-save=my-stats
```

Afterwards, you'll be able to pull your `my-stats` report into the VisualizeBenchmarks notebook and get some handy analysis specific to your machine configuration.
