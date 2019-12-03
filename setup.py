# setup.py
from setuptools import setup, find_packages
setup(
    name = "lstm",
    version = "0.0.1",
    packages = find_packages(),
    data_files=[('.', ['__main__.py', '__init__.py'])]
)