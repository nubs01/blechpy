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

# Requirements
### Operating system:
Currently, blechpy is only developed and validated to work properly on Linux operating systems. It is possible to use blechpy on Mac, but some GUI features may not work properly. It is not possible to use blechpy on Windows. 

### Virtual environments
Because blechpy depends on a very specific mix of package versions, it is required to install blechpy in a virtual environment. *We highly recommend using miniconda to handle your virtual environments. You can download miniconda here: https://docs.conda.io/en/latest/miniconda.html*

### Hardware
We recommend using a computer with at least 32gb of ram and a muti-core processor. The more cores and memory, the better. Memory usage scales with core usage--memory needed to run without overflow errors as you increase the number of cores used. It is possible to run memory-intensive functions with fewer cores to avoid overflow errors, but this will increase processing time. It is also possible to re-run memory-intensive functions after an overflow error, and the function will pick up where it left off.

### Data
Right now this pipeline is only compatible with recordings done with Intan's 'one file per channel' or 'one file per signal type' recordings settings.


# Installation + maintainance
### Installation

Create a miniconda environment with: 
```bash
conda create -n blechpy python==3.7.13
conda activate blechpy
```
Now you can install the package with pip:
```bash
pip install blechpy
```

### Activation

Once you have installed blechpy, you will need to perform some steps to "activate" blechpy, whenever you want to use it. 

1) open up a bash terminal (control+alt+t on ubuntu)
2) activate your miniconda environment with the following command:
```bash
conda activate blechpy
```
3) start an ipython console by typing into the terminal
```bash 
ipython
```
4) You will now be in an ipython console. Now, import blechpy. Simply type:
```python
import blechpy
```
Now, you can use blechpy functions in your ipython console.

### Updating
To update blechpy, open up a bash terminal and type:
```bash
conda activate blechpy #activate your blechpy virtual environment
pip install blechpy -U #install updated version of blechpy
```

### Troubleshooting Segmentation Fault: Only applies if you are using Ubuntu version 20.XX LTS
If your operating system is Ubuntu version 20.XX LTS, "import blechpy" may throw a "segmentation fault" error. This is because numba version 0.48 available via pip-install is corrupted. You can fix this issue by reinstalling numba via conda, by entering the following command in your bash terminal:

```bash
conda install numba=0.48.0
```

# Blechpy Overview
blechpy handles experimental metadata using data_objects which are tied to a directory encompassing some level of data. Existing types of data_objects include:
* dataset
    * object for a single recording session
    * to create a dataset, you will need to have recording files from a single recording in its own "dataset folder". The path to this folder is the "recording directory"
    * The dataset processing pipleine creates 2 critical files that will live alongside your recording files in the dataset folder: the .h5 file and the .p file. The .h5 file contains the actual processed data, along with some metadata. The .p file contains additional critical metadata. 
    * code lives in blechpy/datastructures/dataset.py
* experiment
    * object encompasing an ordered set of recordings from a single animal
    * individual recordings must first be processed as datasets
    * to create an experiment, you will need to have all the dataset folders from a single animal in its own "experiment folder". The path to this folder is the "experiment directory"
    * code lives in blechpy/datastructures/experiment.py
* project
    * object that can encompass multiple experiments & data groups and allow analysis or group differences
    * to create a project, you will need to have all the experiment folders from a single project in its own "project folder". The path to this folder is the "project directory"
    * code lives in blechpy/datastructures/project.py
* HMMHandler
  * object that can be used to set up and run hidden markov model analysis on a dataset
  * HMMHandler objects are created on the level of the dataset. You will need to have a fully processed dataset to create an HMMHandler object from it.

# Dataset Processing (start here if you have a raw recording)

### Basic dataset processing pipeline:

With a brand new *shiny* dataset, the most basic recommended data extraction workflow would be:
```python
dat = blechpy.dataset('/path/to/data/dir/') #create dataset object. Path to data dir should be your recording directory
# IMPORTANT: only run blechpy.dataset ONCE on a dataset, unless you want to overwrite the existing dataset and your preprocessing
# to load an existing dataset, use dat = blechpy.load_dataset('/path/to/data/dir/') instead
dat.initParams(data_quality='hp') # follow GUI prompts. 
dat.extract_data()          # Extracts raw data into HDF5 store
dat.create_trial_list()     # Creates table of digital input triggers
dat.mark_dead_channels()    # View traces and label electrodes as dead, or just pass list of dead channels
dat.common_average_reference() # Use common average referencing on data. Repalces raw with referenced data in HDF5 store
dat.detect_spikes()        # Detect spikes in data. Replaces raw data with spike data in HDF5 store
dat.blech_clust_run(umap=True)       # Cluster data using GMM
dat.sort_spikes(electrode_number) # Split, merge and label clusters as units. Follow GUI prompts. Perform this for every electrode
dat.post_sorting() #run this after you finish sorting all electrodes
dat.make_PSTH_plots() #optional: make PSTH plots for all units 
dat.make_raster_plots() #optional: make raster plots for all units
```

### troubleshooting common error:
It is common to get the following error after running the functions `dat.detect_spikes() ` or `dat.blech_clust_run()`:
```python
TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.
```
**If you encounter this error, simply re-run the function which caused the error, and it will pick-up where it left off.** Occasionally, you may have to do this several times before the function completes.

The reason for this error is that these functions multi-process across channels, but underlying libraries like scipy also parallelize their own operations. This makes it impossible for the program to know how much memory will be used to automatically constrain the number of multi-processes used. 

## Explainers and useful substitutions: 
### blechpy.dataset(): make a NEW dataset
blechpy.dataset() makes a NEW dataset or to OVERWRITES an existing dataset. DO NOT use it on an existing dataset unless you want to overwrite the existing dataset and lose you preprocessing progress with it. 
```python
dat = blechpy.dataset('path/to/recording/directory') # replace quoted text with the filepath to the folder where your recording files are
# or
dat = blechpy.dataset()  # for user interface to select directory
```
This will create a new dataset object and setup basic file paths. You should only do this when starting data processing for the first time. If you use it on a processed dataset, it will get overwritten.

### blechpy.load_dataset(): LOAD an existing dataset
If you already have a dataset and want to pick up where you left off, use blechpy.load_dataset() instead of blechpy.dataset(). 
```python
dat = blechpy.load_dataset('/path/to/recording/directory')  # create dataset object
# or
dat = blechpy.load_dataset()  # for user interface to select directory
# or
dat = blechpy.load_dataset('path/to/dataset/save/file.p')
```

### initParams(): initialize parameters
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
Parameters for a dataset are written to json files in a *parameters* folder in the recording directory. 

#### useful presets:
```python
dat.initParams(data_quality='noisy') # alternative: less strict clustering parameters
dat.initParams(car_keyword='2site_OE64') # automatically map channels to hirose-connector 64ch OEPS EIB in 2-site implantation
dat.initParams(car_keyword='bilateral64') # automatically map channels to omnetics-connector 64ch EIB in 2-site implantation
dat.initParams(shell=True) # alternative: bypass GUI interface in favor of shell interface, useful if working over SSH or GUI is broken
#remember that you can chain any combination of valid keyword arguments together, eg.:
dat.initParams(data_quality='hp', car_keyword='bilateral64', shell=True)
```

### mark_dead_channels(): mark dead channels for exclusion from common average refrencing and clustering
```python
dat.mark_dead_channels() # opens GUI to view traces and label dead channels
```
Marking dead channels is critical for good common average referencing, since dead channels typically have a signal that differs a lot from the "true" average voltage at the electrode tips.

#### HIGHLY RECOMMENDED preset: 
If you already know your dead channels a-priori, you can pass them to mark_dead_channels() as a list of integers:
```python
dat.mark_dead_channels([dead channel indices]) # dead channel indices eg. : [1,2,3]
```

### blech_clust_run(): run clustering
blech_clust_run's keywords can change the clustering algorithm and/or parameters 
```python
dat.blech_clust_run(data_quality='noisy') # alternative: re-run clustering with less strict parameters
dat.blech_clust_run(umap=True) # alternative: cluster with UMAP instead of GMM, improves clustering
dat.blech_clust_run() # default uses PCA instead of UMAP, which is faster, but lower quality clustering
```

## Other useful functions:
### dat._change_root() for moving a dataset:
If you want to move a dataset folder, it is critical you perform the following steps:
1) move the dataset folder to the desired location 
2) copy the new dataset folder directory (right click on folder, select copy)
3) in the ipython console, run the following commands:
```Python
new_directory = 'path/to/new/dataset/folder' # You can paste the directory by right clicking and selecting 'paste filename' 
dat = blechpy.load_dataset(new_directory) # load the dataset
dat._change_root(new_directory) # change the root directory of the dataset to the new directory
dat.save() # save the new directory to the dataset file
```
### Checking processing progress:
```python
dat.processing_status
```
Can provide an overview of basic data extraction and processing steps that need to be taken.

### Viewing a Dataset
Experiments can be easily viewed wih: `print(dat)`
A summary can also be exported to a text with: `dat.export_to_text()`


## Import processed dataset into dataset framework (in development)
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
