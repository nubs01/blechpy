In order to use first create a compatible conda environment with:
`conda env create --name blech -f=conda_environment.yml`

Can  then handle all data from within an ipython terminal
`conda activate blech`
`ipython`

```python
import blechpy
```

### Load dataset
```python
dat = blechpy.dataset() # create new dataset object, popup to select file directory
dat = blechpy.dataset('path/to/recording/directory') # no popup

dat = blechpy.load_dataset() # load an existing dataset from .p file
dat = blechpy.load_dataset('path/to/recording/directory') # or
dat = blechpy.load_dataset('path/to/dataset/save/file.p')
```
### View dataset object
```python
print(dat)
```

### Extract and cluster data
Works with 'one file per channel' and 'one file per signal type' recordings
Though extracting 'one file per signal type' data uses quite a bit of memory

```python
dat.initParams()                 # initialize all parameters, uses standard
                                 # defaults, can pass data_quality='clean'
                                 # or data_quality='noisy' for useful defaults 

dat.edit_clustering_parameters() # optional if parameters need editing

dat.extract_data()               # extracts raw data from files into hdf5 store

dat.create_trial_list()          # creates pandas dataframe of trial order,
                                 # adds to hdf5 store

dat.common_average_reference() # common average references data, deletes raw
                               # data and stores referenced data in /referenced
                               # node in hdf5 store

dat.blech_clust_run()          # runs GMM clustering 

dat.save()                     # Saves dataset object in recording directory

dat.extract_and_cluster()      # bundles all of the above steps
```

### Extract and cluster without any prompts
So unless you're doing 32ch bilateral implants (16ch per side), then you can't
currently do this without ANY prompts since it'll need help determining common
average groups, but I can add defaults if I know what yours are
```python
dat.extract_and_cluster(data_quality='clean',num_CAR_groups='bilateral32',
                        shell=True,
                        dig_in_names=['Water','Quinine','NaCl','Citric Acid'],
                        emg_port=False)

# If you have an EMG
dat.extract_and_cluster(data_quality='clean',num_CAR_groups='bilateral32',
                        shell=True,
                        dig_in_names=['Water','Quinine','NaCl','Citric Acid'],
                        emg_port='C',emg_channels=[31])
```

### Spike Sorting
```python
dat.cleanup_clustering()        # Deletes raw & referenced data, consolidates
                                # clustering memory logs, sets up HDF5 store
                                # for spike sorting and repacks

dat.sort_units()                # Series of input GUIs to sort and label units
                                # Clusters can now be merged and split over and
                                # over until satisfied

#or

dat.sort_units(shell=True)      # For command-line interface, though you still
                                # gotta see the plots if clusters are split
```

