import imp
from setuptools import setup

description = \
    """Minibatch Benchmarking -- Comparison of data storage and sampling
    strategies for training online learning algorithms."""

version = imp.load_source('minibench.version', 'minibench/version.py')
url = 'http://github.com/ejhumphrey/minibatch-benchmarking'
setup(
    name='minibench',
    version=version.version,
    description=description,
    author='Eric J. Humphrey',
    author_email='humphrey.eric@gmail.com',
    url=url,
    download_url=url + '/releases',
    packages=['minibench'],
    package_data={},
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5"
    ],
    keywords='data',
    license='ISC',
    install_requires=[
        'numpy >= 1.8.0',
        'pescador >= 0.1.2',
        'pytest',
        'pytest-benchmark',
        'six'
    ]
)
