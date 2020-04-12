"""Pip installation script for `formable`."""

import os
import re
from setuptools import find_packages, setup


def get_version():

    ver_file = '{}/_version.py'.format('formable')
    with open(ver_file) as handle:
        ver_str_line = handle.read()

    ver_pattern = r'^__version__ = [\'"]([^\'"]*)[\'"]'
    match = re.search(ver_pattern, ver_str_line, re.M)
    if match:
        ver_str = match.group(1)
    else:
        msg = 'Unable to find version string in "{}"'.format(ver_file)
        raise RuntimeError(msg)

    return ver_str


def get_long_description():

    readme_file = 'README.md'
    with open(readme_file, encoding='utf-8') as handle:
        contents = handle.read()

    return contents


setup(
    name='formable',
    version=get_version(),
    description=('Formability analysis in materials science.'),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Adam J. Plowman',
    author_email='adam.plowman@manchester.ac.uk',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'plotly',
        'vecmaths',
        'pyvista',
        'itkwidgets',  # for pyvista in Jupyter notebooks
    ],
    project_urls={
        'Github': 'https://github.com/LightForm-group/formable',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
    ],
)
