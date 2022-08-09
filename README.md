# TPC-utils

 Useful functions for TPC data analysis.

## Installation

On the same directory as the package, run:

    python3 -m pip install .

## Requirements

Python >= 3.8

Numpy >= 1.22

## Usage

The package contains a number of functions for TPC data analysis. For spectrum analysis, the following functions are available:

 - backgound: Calculate the background spectrum from a signal (a single one or an array of signals);

- deconvolution: Deconvolve a spectrum (a single one or an array of signals) from a background and find the peaks.

All of this functions are present in the TSpectrum module, in ROOT.

For the pointcloud analysis, the following functions are available:

 - pRansac: Perform a RANSAC algorithm to find the best 3D lines of a pointcloud;

 - fit_3d: Fit a 3D line to a pointcloud.

For TPC_CNN auxiliar functions, the following functions are available:

 - get_peaks: Get the peaks of a spectrum after the segmentation cnn.
