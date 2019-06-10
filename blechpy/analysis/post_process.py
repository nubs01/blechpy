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
from blechpy.widgets import userIO
import re


def sort_units(file_dir,fs,shell=False):
    '''Allows user to sort clustered units 
    '''
    hf5_name = h5io.get_h5_filename(file_dir)
    hf5_file = os.path.join(file_dir,hf5_name)

def label_single_unit(hf5_file,cluster,sorting_log=None):
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
    if sorting_log is None:
        sorting_log = h5_file.replace('.h5','_sorting.log')
    unit_name = get_next_unit_name(hf5_file)
    with open(sorting_log,'a+') as log:
        print('%s sorted on %s' % (unit_name,dt.datetime.today().strftime('%m/%d/%y %H:%M')),
                file=log)
        print('Cluster info:\n----------',file=log)
        print_clust = deepcopy(cluster)
        # Get rid of data arrays in output clister
        for k,v in cluster.items():
            if isinstance(v,numpy.array):
                print_clust.pop(k)
        print(dp.print_dict(print_clust),file=log)
        print('--------------',file=log)

    with table.open_file(hf5_file,'r+') as hf5:
        table = hf5.root.unit_descriptor
        unit_descrip = table.row
        unit_descrip['electrode_number'] = int(cluster['electrode'])
        unit_descrip['single_unit'] = int(cluster['single_unit'])
        unit_descrip['regular_spiking'] = int(cluster['regular_spiking'])
        unit_descrip['fast_spiking'] = int(cluster['fast_spiking'])

        waveforms = hf5.create_array('/sorted_units/%s' % unit_name,
                                    'waveforms',cluster['spike_waveforms'])
        times = hf5.create_array('/sorted_units/%s' % unit_name,
                                'times',cluster['spike_times'])
        unit_descrip.append()
        table.flush()
        hf5.flush()

    # Save metrics for sorted unit

def split_cluster(cluster,fs,params=None,shell=True):
    '''Use GMM to re-cluster a single cluster into a number of sub clusters

    Parameters
    ----------
    cluster : dict, cluster metrics and data
    params : dict (optional), parameters for reclustering
        with keys: 'Number of Clusters', 'Max Number of Iterations',
                    'Convergence Criterion','GMM random restarts'
    shell : bool , optional, whether to use shell or GUI input (defualt=True)

    Returns
    -------
    clusters : list of dicts
        resultant clusters from split
    '''
    if params is None:
        clustering_params = {'Number of Clusters':2,
                            'Max Number of Iterations':1000,
                            'Convergence Criterion':0.00001,
                            'GMM random restarts':10}
        params = userIO.fill_dict(clustering_params,shell)

    n_clusters = int(params['Number of Clusters'])
    n_iter = int(params['Max Number of Iterations'])
    thresh = float(params['Convergence Critereon'])
    n_restarts = int(params['GMM random restarts'])

    g = GaussianMixture(n_components=n_clusters, covariance_type='full', \
            tol=thresh, max_iter = n_iter, n_init = n_restarts)
    g.fit(cluster['data'])
    
    out_clusters = []
    if g.converged_:
        predictions = g.predict(data)
        for c in range(n_clusters):
            clust_idx = np.where(predictions==c)
            tmp_clust = deepcopy(cluster)
            clust_id = str(cluster['cluster_id']) + \
                    bytes([b'A'[0]+c]).decode('utf-8')
            clust_name = 'E%iS%i_cluster%s' % \
                    (cluster['electrode'],cluster['solution'],clust_id)
            clust_waveforms = spike_waveforms[clust_idx]
            clust_times = spike_times[clust_idx]
            clust_data = data[clust_idx,:]
            clust_log = cluster['manipulations'] + \
                    '\nSplit %s with parameters:' \
                    '\n%s\nCluster %i from split results. Named %s' \
                    % (cluster['Cluster Name'],dp.print_dict(params),c,clust_name)
            tmp_clust['Cluster Name'] = clust_name
            tmp_clust['cluster_id'] = clust_id
            tmp_clust['data'] = clust_data
            tmp_clust['spike_times'] = clust_times
            tmp_clust['spike_waveforms'] = clust_waveforms

            tmp_isi, tmp_violations1, tmp_violations2 = \
                    get_ISI_and_violations(clust_times,fs)
            tmp_clust['ISI'] = tmp_isi
            tmp_clust['1ms_violations'] = tmp_violations1
            tmp_clust['2ms_violations'] = tmp_violations2
            out_clusters.append(deepcopy(tmp_clust))

    return out_clusters


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
    spike_waveforms,spike_times,pca_slices,energy,amplitudes,predictions = \
            get_clustering_metrics(file_dir,electrode_num,solution_num)

    this_cluster = np.where(predictions==cluster_num)[0]
    spike_waveforms = spike_waveforms[this_cluster,:]
    spike_times = spike_times[this_cluster]
    pca_slices = pca_slices[this_cluster,:]
    energy = energy[this_cluster]
    amplitudes = amplitudes[this_cluster]
    predictions = predictions[this_cluster]

    return spike_waveforms,spike_times,pca_slices,energy,ampltitudes,predictions

def get_cluster_data(file_dir,electrode_num,solution_num,cluster_id,fs):
    '''Grab all metrics for a specific clustering solution and return in a data
    matrix that can be used for re-clustering

    Parameters
    ----------
    file_dir : str, full path for data directory
    electrode_num : int, electrode number
    solution_num : int,  number of clusters in solution
    cluster_num : int, number of desired cluster within solution
    fs : float, sampling rate of the data in Hz
    
    Returns
    -------
    cluster : dict
        with keys: Cluster Name, data, spike_times, spike_waveforms, manipulations

        Cluster Name: str, Unique identifier for cluster
        electrode : int
        solution : int
        cluster_id : str, unique cluster id, updated on splits and merges
        data : numpy.array 
            with columns: normalized energy, normalized amplitudes, PC1, PC2, PC3
        spike_times : numpy.array
        spike_waveforms : numpy.array
        manipulations : str, log of actions taken on this cluster
    '''
    spike_waveforms,spike_times,pca_slices,energy,amplitudes,predictions = \
            get_unit_metrics(file_dir,electrode_num,solution_num,cluster_num)
    
    data = np.zeros(len(spike_times),n_pc+2)
    data[:,0] = energy/np.max(energy)
    data[:,1] = np.abs(ampltitudes)/np.max(amplitudes)
    data[:,2:] = pca_slices[:,:n_pc]

    ISI,violations1,violations2 = get_ISI_and_violations(spike_times,fs)
    cluster = {'Cluster Name':'E%iS%i_cluster%i' % (electrode_num,
                                                    solution_num,
                                                    cluster_num),
                'electrode':electrode_num,
                'solution':solution_num,
                'cluster_id':str(cluster_num),
                'data':data,
                'spike_times':,spike_times,
                'spike_waveforms':spike_waveforms,
                'ISI':ISI,
                '1ms_violations':violations1,
                '2ms_violations':violations2,
                'single_unit':0,
                'fast_spiking':0,
                'regular_spiking':0,
                'manipulations':''}

    return deepcopy(cluster)

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

def get_ISI_and_violations(spike_times,fs):
    '''returns array of ISIs and percent of 1 & 2 ms ISI violations

    Parameters
    ----------
    spike_time  numpy.array
    fs : float, sampling rate in Hz
    '''
    fs = float(fs/1000.0)
    ISIs = np.ediff1d(np.sort(spike_times))/fs
    violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(spike_times))
    violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(spike_times))

    return ISIs, violations1, violations2

def merge_clusters(clusters,fs):
    '''Merge 2 or more clusters into a single cluster

    Parameters
    ----------
    clusters : list of dict, list of cluster dictionaries
    fs : float, sampling rate in Hz

    Returns
    -------
    cluster : dict, merged cluster
    '''
    clust = deppcopy(clusters.pop(0))
    for c in clusters:
        clust['cluster_id'] = clust['cluster_id']+c['cluster_id']
        clust['spike_times'] = np.concatenate((clust['spike_times'],c['spike_times']))
        clust['data'] = np.concatenate((clust['data'],c['data']))
        clust['spike_waveforms'] = np.concatenate((clust['spike_waveforms'],
            c['spike_waveforms']))
        new_name = 'E%iS%i_cluster%s' % (clust['electrode'],clust['solution'],
                clust['cluster_id'] )
        clust['manipulations'] += ('Cluster %s merged with cluster %s.\n'
                'New Cluster named: %s') % (clust['Cluster Name'],c['Cluster Name'],
                    new_name)
    idx = np.argsort(clust['spike_times'])
    clust['spike_times'] = clust['spike_times'][idx]
    clust['spike_waveforms'] = clust['spike_waveforms'][idx,:]
    clust['data'] = clust['data'][idx,:]
    ISI,v1,v2 = get_ISI_and_violations(clust['spike_times'],fs)
    clust['ISI'] = ISI
    clust['1ms_violations'] = v1
    clust['2ms_violations'] = v2
    clust['regular_spiking'] = 0
    clust['single_unit'] = 0
    clust['fast_spiking'] = 0

    return deepcopy(clust)

def plot_cluster(cluster):
    '''Plots a cluster with isi and violation info for viewing

    Parameters
    ----------
    cluster : dict with cluster info

    '''
    fix,ax = blech_waveforms_datashader(cluster['spike_waveforms'])
    ax.set_xlabel('Sample (30 samples per ms)')
    ax.set_ylabel('Voltage (microvolts)')
    ax.set_title(('Cluster Name: {:s}\n2ms Violations={:.1f},'
        '1ms Violations={:.1f}\nNumber of Waveforms={:d}').format( \
                cluster['Cluster Name'],cluster['2ms_violations'],
                cluster['1ms_violations'],cluster['spike_times'].shape[0]))
    return fig,ax
