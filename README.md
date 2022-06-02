# kornmo-master-thesis

Git repository for our masters thesis. Our experiments are located in the folders grain_classification, vegetation_indices and yield_predictions.
This repository uses https://github.com/putetrekk/kornmo.git as a submodule and uses much of this code as dependencies. 

## data_handling:
data_handling consists of files and scripts for general data processing such as mask creation for satellite images, calculating farmers coordinates and creating small 16x16 satellite images from large 100x100 images. It also contains general data classes for loading, editing and working with masks, images and gemoetry files.

## frost:
frost contains all filed related to acquiring and processing meteorological data from the Frost API. 
From weather station values, all the way to interpolated daily farmer value. 
This codebase initially started as the frost code from Kornmo, located here: https://github.com/putetrekk/kornmo/tree/master/frost.
But were over time edited to handle new features and optimized to consecutively download multiple features from many years. 
The specific API keys have been removed from the code.

## grain_classification:
grain_classification contains all classification experiments, including raw satellite images, vegetation indices and MIL.

## satellite_images:
satellite_images contains some functions for extracting and reading satellite images and soil quality from .h5 files, and a file purely for satellite image exploration.

## vegetation_indices:
vegetation_indices contains the scrips used to calculate all vegetation indices, their average values and the plotting of these.

## yield_prediction:
yield_prediction contains all prediction experiments, including adding new features and labels, and early predictions.
