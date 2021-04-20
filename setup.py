from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = """Utility functions for plotting seismic data
                 and images"""
LONG_DESCRIPTION = """Utility functions for plotting seismic data
                      and images. Provides both static and interactive
                      plotting utilities. All functions use either matplotlib
                      or vedo."""

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="sivu",
    version=VERSION,
    author="Joseph Jennings",
    author_email="<joseph29@sep.stanford.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    keywords=['python', 'plotting', 'seismic', 'image'],
    classifiers=[
        "Intended Audience :: Seismic processing/imaging",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux ",
    ],
)
