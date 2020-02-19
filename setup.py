"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='embehEvalData',
    version='1.0',
    packages=['embehEvalData'],
    install_requires=['numpy', 'matplotlib', 'xrayutilities', 'scipy', 'uncertainties', 'pyEvalData'],  # Optional
    license='',
    author='Martin Borchert',
    author_email='martin.b@robothek.de',
    description='Python Modul to evaluate SPEC data together with pyEvalData',  # Required
)