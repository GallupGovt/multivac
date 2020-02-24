#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    req_path = 'requirements.txt'
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(name='multivac',
      version='0.1',
      description="Software for DARPA’s Information Innovation Office’s "
                  "Automating Scientific Knowledge Extraction (ASKE) program",
      long_description=readme(),
      maintainer='Benjamin Ryan',
      maintainer_email='ben_ryan@gallup.com',
      url='https://github.com/GallupGovt/multivac',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      include_package_data=True,
      zip_safe=False,
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
      python_requires='>=3.6',
      install_requires=requirements(),
      )
