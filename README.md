See the <a href='https://nubs01.github.io/blechpy'>full documentation</a> here.

- [blechpy](#blechpy)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
  * [Starting wit a raw dataset](#starting-wit-a-raw-dataset)
    + [Create dataset](#create-dataset)
    + [Initialize Parameters](#initialize-parameters)
    + [Basic Processing](#basic-processing)
    + [Viewing a Dataset](#viewing-a-dataset)
  * [Loading an existing dataset](#loading-an-existing-dataset)
  * [Import processed dataset into dataset framework](#import-processed-dataset-into-dataset-framework)
- [Experiments](#experiments)
  * [Creating an experiment](#creating-an-experiment)
  * [Editing recordings](#editing-recordings)
  * [Held unit detection](#held-unit-detection)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# blechpy
This is a package to extract, process and analyze electrophysiology data recorded with Intan or OpenEphys recording systems. This package is customized to store experiment and analysis metadata for the BLECh Lab (Katz lab) @ Brandeis University, but can readily be used and customized for other labs.

# Installation
I recommend installing miniconda to handle your virtual environments
Create a miniconda environment with: 
```bash
conda create -n blechpy python==3.7.13
conda activate blechpy
```
Now you can install this package simply with pip:
```bash
pip install blechpy
```

If you want to update blechpy to the latest version:
```bash
pip install blechpy -U
```

Now you can deal with all of your data from within an ipython terminal:
`ipython`

```python
import blechpy
```

### Ubuntu 20.04 LTS+
With Ubuntu 20 or higher, you will get a segmentation fault when importing blechpy because numba version 0.48 installed through pip is corrupted. You will need to reinstall it via conda

```bash
conda install numba=0.48.0
```

# Usage
blechpy handles experimental metadata using data_objects which are tied to a directory encompassing some level of data. Existing types of data_objects include:
* dataset
    * object for a single recording session
* experiment
    * object encompasing an ordered set of recordings from a single animal
    * individual recordings must first be processed as datasets
* project
    * object that can encompass multiple experiments & data groups and allow analysis or group differences

# Datasets
Right now this pipeline is only compatible with recordings done with Intan's 'one file per channel' or 'one file per signal type' recordings settings.

## Starting with a raw dataset
### Create dataset
With a brand new *shiny* recording you can initilize a dataset with:
```python
dat = blechpy.dataset('path/to/recording/directory')
# or
dat = blechpy.dataset()  # for user interface to select directory
```
This will create a new dataset object and setup basic file paths.
If you're working via SSH or just want a command-line interface instead of a GUI you can use the keyword argument `shell=True`
You should only do this when starting data processing for the first time. If you use it on a processed dataset, it will get overwritten.
Use blechpy.load_dataset() instead to load an existing dataset (see below)

### Initialize Parameters
```python
dat.initParams() 
```
Initalizes all analysis parameters with a series of prompts.
See prompts for optional keyword params.
Primarily setups parameters for:
* Flattening Port & Channel in Electrode designations
* Common average referencing
* Labelling areas of electrodes
* Labelling digital inputs & outputs
* Labelling dead electrodes
* Clustering parameters
* Spike array creation
* PSTH creation
* Palatability/Identity Responsiveness calculations

Initial parameters are pulled from default json files in the dio subpackage.
Parameters for a dataset are written to json files in a *parameters* folder in the recording directory

Useful dat.initParams() arguments:
* data_quality='hp' -increases strictness of clustering, total # of clusters, and spike-sorting window to -0.75 to 1s.
* CAR_keyword = 'bilateral64' -auto assigns channel mapping to match the Omnetics-connector open ephys 64 channel EIB with 2-site implantation
* CAR_keyword = '2site_OE64' -auto assigns channel mapping to match Hirose-connector Open Ephys 64 channel EIB with 2-site implantation
* shell = True -bypasses GUI interface in favor of shell interface, useful if working over SSH or GUI is broken
### Basic Processing

The most basic data extraction workflow would be:
```python
dat = blechpy.dataset('/path/to/data/dir/')
dat.initParams()            # See fucntion docstring, lots of optional parameters to eliminate need for user interaction
dat.extract_data()          # Extracts raw data into HDF5 store
dat.create_trial_list()     # Creates table of digital input triggers
dat.mark_dead_channels()    # View traces and label electrodes as dead, or just pass list of dead channels
dat.mark_dead_channels([dead channel indices]) #alternatively, if you already know which chanels are dead, you can pass them as an argument
dat.common_average_reference() # Use common average referencing on data. Repalces raw with referenced data in HDF5 store
dat.detect_spikes()
dat.blech_clust_run()       # Cluster data using GMM
dat.blech_clust_run(data_quality='noisy') # alternative: re-run clustering with less strict parameters
dat.sort_spikes(electrode_number)        # Split, merge and label clusters as units
```
check blechpy/datastructures/dataset.py to see what functions are available

### Preferred Workflow:

This workflow uses some parameters with defualts which makes the workflow more convenient. 
```python
dat = blechpy.dataset('/path/to/data/dir/')
dat.initParams(data_quality = 'hp', channel_mapping = '2site_OE64')	# 'hp' parameter for stricter clustering criteria, '2site_OE64' automatically maps channels to hirose-connector 64ch OEPS EIB in 2-site implantation
dat.extract_data()          
dat.create_trial_list()    
dat.mark_dead_channels([channel numbers])	# pass a list of dead channels (i.e. [1,2,3]) to bypass GUI marking of dead channels. Requires that you note them during drive building &/ recording
dat.common_average_reference() 
dat.detect_spikes()
dat.blech_clust_run(umap=True)	# Cluster with UMAP instead of GMM, supposedly better clustering
dat.sort_spikes(electrode_number)	# Split, merge and label clusters as units
```

### Checking processing progress:

```python
dat.processing_status
```
Can provide an overview of basic data extraction and processing steps that need to be taken.

### Viewing a Dataset
Experiments can be easily viewed wih: `print(dat)`
A summary can also be exported to a text with: `dat.export_to_text()`

## Loading an existing dataset
```python
dat = blechpy.load_dataset() # load an existing dataset from .p file
# or
dat = blechpy.load_dataset('path/to/recording/directory') 
# or
dat = blechpy.load_dataset('path/to/dataset/save/file.p')
```

## Import processed dataset into dataset framework
```python
dat = blechpy.port_in_dataset()
# or
dat = blechpy.port_in_dataset('/path/to/recording/directory')
```

# Experiments
## Creating an experiment
```python
exp = blechpy.experiment('/path/to/dir/encasing/recordings')
# or
exp = blechpy.experiment()
```
This will initalize an experiment with all recording folders within the chosen directory.

## Editing recordings
```python
exp.add_recording('/path/to/new/recording/dir/')    # Add recording
exp.remove_recording('rec_label')                   # remove a recording dir 
```
Recordings are assigned labels when added to the experiment that can be used to easily reference exerpiments.

## Held unit detection
```python
exp.detect_held_units()
```
Uses raw waveforms from sorted units to determine if units can be confidently classified as "held". Results are stored in exp.held_units as a pandas DataFrame.
This also creates plots and exports data to a created directory:
/path/to/experiment/experiment-name_analysis

# Analysis
The `blechpy.analysis` module has a lot of useful tools for analyzing your data.
Most notable is the `blechpy.analysis.poissonHMM` module which will allow fitting of the HMM models to your data. See tutorials. 
