import os
import tables
import numpy as np
import easygui
import ast
import pylab as plt
from sklearn.mixture import GaussianMixture
import blech_waveforms_datashader
from blechpy.dio import h5io
from blechpy.dio import particles
import re


def sort_units(file_dir,shell=False):
    '''Allows user to sort clustered units 
    '''
    hf5_name = h5io.get_h5_filename(file_dir)
    hf5_file = os.path.join(file_dir,hf5_name)

def label_single_unit(hf5_file,electrode_num,spike_times,spike_waveforms,
        single_unit=0,reg_spiking=0,fast_spiking=0):
    '''Adds a sorted unit to the hdf5 store
    adds unit info to unit_descriptor table and spike times and waveforms to
    sorted_units

    Parameters
    ----------
    hf5_file : str, full path to .h5 file
    electrode_num : int, electrode number
    spike_times : numpy.array, 1D array of spike times corresponding to this unit
    spike_waveforms : numpy.array, 
        array containing waveforms of spikes for this unit with each row
        corresponding to rows in spike_times
    single_unit : {0,1},
        0 (default) is for multi-unit activity, 1 to indicate single unit
    reg_spiking : {0,1},
        0 (default) is if multi-unit or not regular-spiking pyramidal cell
    fast_spiking : {0,1},
        0 (default) is if multi-unit or not fast-spiking interneuron
    '''
    unit_name = get_next_unit_name(hf5_file)
    with table.open_file(hf5_file,'r+') as hf5:
        table = hf5.root.unit_descriptor
        unit_descrip = table.row
        unit_descrip['electrode_number'] = electrode_num
        unit_descrip['single_unit'] = int(single_unit)
        unit_descrip['regular_spiking'] = int(reg_spiking)
        unit_descrip['fast_spiking'] = int(fast_spiking)

        waveforms = hf5.create_array('/sorted_units/%s' % unit_name,
                                    'waveforms',spike_waveforms)
        times = hf5.create_array('/sorted_units/%s' % unit_name,
                                'times',spike_times)
        unit_descrip.append()
        table.flush()
        hf5.flush()

def split_cluster():
    pass

def get_unit_metrics(file_dir,electrode_num,solution_num,cluster_num):
    '''Grab all metrics for a specfic cluster from a clustering solution for an
    electrode

    Parameters
    ----------
    file_dir : str, full path for data directory
    electrode_num : int, electrode number
    solution_num : int,  number of clusters in solution
    cluster_num : int, number of desired cluster within solution
    
    Returns
    -------
    numpy.array, numpy.array, numpy.array,numpy.array, numpy.array, numpy.array
    spike_waveforms, spike_times, pca_slices, energy, amplitudes, predictions
    '''
    pass

def get_clustering_metrics(file_dir,electrode_num,solution_num):
    '''Grab all saved metrics associated with a particular clustering solution
    for an electrode

    Parameters
    ----------
    file_dir : str, path to data directory
    electrode_num : int, electrode number
    solution_num : int, number of clusters in desired solution

    Returns
    -------
    numpy.array, numpy.array, numpy.array,numpy.array, numpy.array, numpy.array
    spike_waveforms, spike_times, pca_slices, energy, amplitudes, predictions
    '''
    waveform_dir = os.path.join(file_dir,'spike_waveforms',
                                    'electrode%i' % electrode_num)
    spike_time_file = os.path.join(file_dir,'spike_times',
                                    'electrode%i' % electrode_num,
                                    'spike_times.npy')
    predictions_file = os.path.join(file_dir,'clustering_results',
                                    'electrode%i' % electrode_num,
                                    'clusters%i' % solution_num,
                                    'predictions.npy')
    spike_times = np.load(spike_time_file)
    predictions = np.load(predictions_file)
    spike_waveforms = np.load(os.path.join(waveform_dir,'spike_waveforms.npy'))
    pca_slices = np.load(os.path.join(waveform_dir,'pca_waveforms.npy'))
    energy = np.load(os.path.join(waveform_dir,'energy.npy'))
    amplitudes = np.load(os.path.join(waveform_dir,'spike_amplitudes.npy'))

    return spike_waveforms,spike_times,pca_slices,energy,amplitudes,predictions

def get_unit_numbers(hf5_file):
    '''returns a list of sorted unit numbers from the hdf5 store

    Parameters
    ----------
    hf5_file : str, full path to h5 file

    Returns
    -------
    list of ints
    '''
    pattern = 'unit(\d*)'
    parser = re.compile(pattern)
    out = []
    with tables.open_file(hf5_file,'r') as hf5:
        if '/sorted_units' in hf5:
            node_list = hf5.list_nodes('/sorted_units')
            if node_list == []:
                return []
            out = [int(parser.match(node)[1]) for node in node_list]
            return out
        else:
            return []

def get_next_unit_name(hf5_file):
    '''returns node name for next sorted unit

    Parameters
    ----------
    hf5_file : str, full path to h5 file

    Returns
    -------
    str , name of next unit in sequence ex: "unit01"
    '''
    units = get_unit_numbers(hf5_file)
    if units == []:
        out = 'unit%03d' % 0
    else:
        out = 'unit%03d' % int(max(units)+1)
    return out
