# Forecasting unprecedented critical events in ecological fluctuations

One Paragraph of project description goes here

## System requirements

This project was tested and develped in python 3.7 environment managed by anaconda 2019.03 on a Ubuntu 18.04 operating system.

### Package dependencies

```
python 3.7
numpy 1.17.3
scipy 1.3.1
matplotlib 3.1.1

```

### Installing

If necessary download Anaconda package manager and create a new virtual environment. Activate the environment and install the necessaryu packages using the following commands

```
conda install numpy
conda install scipy
conda install matplotlib
```

The git repository can by cloned or downloaded to the local machine and files run from their directory after extraction.

Typical install time: ~5 min

## Demo

### Generate simulated data (~5min)

Within the Demo folder, there are two folders containing example data and necessary functions. To generate simulated data from the linear response and intertidal models please run the files 'linearResponseModel.ipynb' and 'musselCommunityModel.ipynb' within the data folder respectively. 

### Example analysis (~5min)

Within the Demo Folder, open the jupyter notebook 'demo.ipynb. This file demonstrates all functions and analysis developed and used in the corresponding paper using the baltic sea plankton data set as a default example. Other data sets can be run by changing the appropriate variables indicated within. Running through this file will:

** Load and plot appropriate dataset
** Extract avalanche size and durations
** Perform AIC testing to confirm power law probability distribution of size and duration
** Extract average normalized avalanche shape
** Demonstrate ASE extrapolation of historical data
** perform s-map based forecasting with and without ASE-augmented training data

To examine the implementation of each function, see the appropriate .py files in '/DEMO/functions/'

### Reproduction

The majority of the paper's results can be replicated by replacing the data and appropriate parameters in the demo notebook. This and the corresponding files in the function folder comprise a well-documented assembly of the project code. For the benchmarking sweeps, see the completeAnalysis folder in the github repository at https://github.com/samuelbray32/ecologicalAvalanches and the secondary README contained within.


## Authors

* **Samuel Bray** 
* **Bo Wang**


## License

This project is licensed under GNUV3

## Basic Support

For questions, please contact sambray@stanford.edu

