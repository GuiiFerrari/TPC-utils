# TPC-utils

 Useful functions for TPC data analysis.

## Installation

    python3 setup.py install


## Requirements

Python >= 3.8

Numpy >= 1.22

## Usage

The package contains a number of functions for TPC data analysis. For spectrum analysis, the following functions are available:

backgound: Calculate the background spectrum from a signal;

deconvolution: Deconvolve a spectrum from a background and find the peaks.

All of this functions are present in the TSpectrum module, in ROOT.

For the pointcloud analysis, the following functions are available:

pRansac: Perform a RANSAC algorithm to find the best 3D lines of a pointcloud;

fit_3d: Fit a 3D line to a pointcloud.
