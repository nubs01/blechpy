- [blechpy](#blechpy)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
  * [Getting Started](#getting-started)
    + [With a raw dataset](#with-a-raw-dataset)
      - [Create dataset](#create-dataset)
      - [Initialize Parameters](#initialize-parameters)
      - [Basic Processing](#basic-processing)
    + [Viewing a Dataset](#viewing-a-dataset)
  * [Loading an existing dataset](#loading-an-existing-dataset)
  * [Import processed dataset into dataset framework](#import-processed-dataset-into-dataset-framework)
- [Experiments](#experiments)
    + [Creating an experiment](#creating-an-experiment)
    + [Editing recordings](#editing-recordings)
    + [Held unit detection](#held-unit-detection)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# blechpy
This is a package to extract, process and analyze electrophysiology data recorded with Intan or OpenEphys recording systems. This package is customized to store experiment and analysis metadata for the BLECh Lab (Katz lab) @ Brandeis University, but can readily be used and customized for other labs.

# Installation
If is set this up correctly you can install with pip:
`pip install blechpy`

If you are setting up from source you can create a compatible conda environment with: 
`conda env create --name blech -f=conda_environment.yml`

Can  then handle all data from within an ipython terminal
`conda activate blech`
`ipython`

```python
import blechpy
```

# Usage
blechpy handles experimental metadata using data_objects which are tied to a directory encompassing some level of data. Existing types of data_objects include:
* dataset
    * object for a single recording session
* experiment
    * object encompasing an ordered set of recordings from a single animal
    * individual recordings must first be processes as datasets
* project
    * object that can encompass multiple experiments & data groups and allow analysis or group differences

# Datasets
## Getting Started
Right now this pipeline is only compatible with recordings done with Intan's 'one file per channel' or 'one file per signal type' recordings settings.

### With a raw dataset
#### Create dataset
With a brand new *shiny* recording you can initilize a dataset with:
```python
dat = blechpy.dataset('path/to/recording/directory')
# or
dat = blechpy.dataset()  # for user interface to select directory
```
This will create a new dataset object and setup basic file paths.
If you're working via SSH or just want a command-line interface instead of a GUI you can use the keyword argument `shell=True`

#### Initialize Parameters
```python
dat.initParams() 
# or
dat.initParams(shell=True)
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

#### Basic Processing
```python
dat.processing_status
```
Can provide an overview of basic data extraction and processing steps that need to be taken.


An example data extraction workflow would be:
```python
dat = blechpy.dataset('/path/to/data/dir/')
dat.initParams()
dat.extract_data()          # Extracts raw data into HDF5 store
dat.create_trial_list()     # Creates table of digital input triggers
dat.mark_dead_channels()    # View traces and label electrodes as dead
dat.common_average_reference() # Use common average referencing on data. 
                               # Repalces raw with referenced data in HDF5 store
dat.blech_clust_run()       # Cluster data using GMM
dat.blech_clust_run(data_quality='noisy') # re-run clustering with less strict parameters

dat.sort_units()        # Split, merge and label clusters as units
```

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
##Getting Started
### Creating an experiment
```python
exp = blechpy.experiment('/path/to/dir/encasing/recordings')
# or
exp = blechpy.experiment()
```
This will initalize an experiment with all recording folders within the chosen directory.

### Editing recordings
```python
exp.add_recording('/path/to/new/recording/dir/')    # Add recording
exp.remove_recording('rec_label')                   # remove a recording dir 
```
Recordings are assigned labels when added to the experiment that can be used to easily reference exerpiments.

### Held unit detection
```python
exp.detect_held_units()
```
Uses raw waveforms from sorted units to determine if units can be confidently classified as "held". Results are stored in exp.held_units as a pandas DataFrame.
This also creates plots and exports data to a created directory:
/path/to/experiment/experiment-name_analysis
