import os
import tables
import ast
import re
import shutil
import numpy as np
import easygui as eg
import pylab as plt
import datetime as dt
from sklearn.mixture import GaussianMixture
from blechpy.plotting import blech_waveforms_datashader
from blechpy.dio import h5io, particles
from blechpy.data_print import data_print as dp
from blechpy.widgets import userIO
from copy import deepcopy


def sort_units(file_dir, fs, shell=False):
    '''Allows user to sort clustered units

    Parameters
    ----------
    file_dir : str, path to recording directory
    fs : float, sampling rate in Hz
    shell : bool
        True for command-line interface, False for GUI (default)
    '''
    hf5_name = h5io.get_h5_filename(file_dir)
    hf5_file = os.path.join(file_dir, hf5_name)
    sorting_log = hf5_file.replace('.h5', '_sorting.log')
    metrics_dir = os.path.join(file_dir, 'sorted_unit_metrics')
    if not os.path.exists(metrics_dir):
        os.mkdir(metrics_dir)
    quit_flag = False

    # Start loop to label a cluster
    clust_info = {'Electrode': 0, 'Clusters in solution': 7,
                  'Cluster Numbers': [], 'Edit Clusters': False}
    clust_query = userIO.dictIO(clust_info, shell=shell)
    print(('Beginning spike sorting for: \n\t%s\n'
          'Sorting Log written to: \n\t%s') % (hf5_file, sorting_log))
    print(('To select multiple clusters, simply type in the numbers of each\n'
           'cluster comma-separated, however, you MUST check Edit Clusters in'
           'order\nto split clusters, merges will be assumed'
           ' from multiple cluster selection.'))
    print(('\nIf using GUI input, hit cancel at any point to abort sorting'
           ' of current cluster.\nIf using shell interface,'
           ' type abort at any time.\n'))
    while not quit_flag:
        clust_info = clust_query.fill_dict()
        if clust_info is None:
            quit_flag = True
            break
        clusters = []
        for c in clust_info['Cluster Numbers']:
            tmp = get_cluster_data(file_dir, clust_info['Electrode'],
                                   clust_info['Clusters in solution'],
                                   int(c), fs)
            clusters.append(tmp)
        if len(clusters) == 0:
            quit_flag = True
            break
        if clust_info['Edit Clusters'] or len(clusters) > 1:
            clusters = edit_clusters(clusters, fs, shell)
            if isinstance(clusters, dict):
                clusters = [clusters]
            if clusters is None:
                quit_flag = True
                break

        cell_types = get_cell_types([x['Cluster Name'] for x in clusters],
                                    shell)
        if cell_types is None:
            quit_flag = True
            break
        else:
            for x, y in zip(clusters, cell_types):
                x.update(y)
        for unit in clusters:
            label_single_unit(hf5_file, unit, fs, sorting_log, metrics_dir)

        q = input('Do you wish to continue sorting units? (y/n) >> ')
        if q == 'n':
            quit_flag = True


def edit_clusters(clusters, fs, shell=False):
    '''Handles editing of a cluster group until a user has a single cluster
    they are satisfied with

    Parameters
    ----------
    clusters : list of dict
        list of dictionaries each defining clusters of spikes
    fs : float
        sampling rate in Hz
    shell : bool
        set True if command-line interface is desires, False for GUI (default)

    Returns
    -------
    dict
        dict representing the resulting cluster from manipulations, None if
        aborted
    '''
    clusters = deepcopy(clusters)
    quit_flag = False
    while not quit_flag:
        if len(clusters) == 1:
            # One cluster, ask if they want to keep or split
            if shell:
                q = input('Do you want to split %s?\n(y/n/blank to abort) >> '
                          % clusters[0]['Cluster Name'])
                if q == 'n':
                    return clusters[0]
                elif q == '':
                    return None
            else:
                q = eg.ynbox('Do you want to split %s?'
                             % clusters[0]['Cluster Name'])
                if not q:
                    return clusters[0]

            clusters = split_cluster(clusters[0], fs, shell=shell)
            if clusters is None or clusters == []:
                return None
            figs = []
            for i, c in enumerate(clusters):
                tmp_fig = plot_cluster(c, i)
                figs.append(tmp_fig)
            plt.show()
            query = {'Clusters to keep (indices)': []}
            query = userIO.dictIO(query, shell=shell)
            ans = query.fill_dict()
            if ans is None:
                return None
            ans = [int(x) for x in ans['Clusters to keep (indices)']]
            new_clusters = [clusters[x] for x in ans]
            del clusters
            clusters = new_clusters
            del new_clusters
        else:
            # Automatically merge multiple clusters
            cluster = merge_clusters(clusters, fs)
            print('%i clusters merged into %s'
                  % (len(clusters), cluster['Cluster Name']))
            clusters = [cluster]


def get_cell_types(cluster_names, shell=True):
    '''Queries user to identify cluster as multiunit vs single-unit, regular vs
    fast spiking

    Parameters
    ----------
    shell : bool (optional),
        True if command-line interface desired, False for GUI (default)

    Returns
    -------
    dict
        with keys 'Single unit', 'Regular spiking', 'Fast spiking' and values
        are 0 or 1 for each  key
    '''
    query = {'Single Unit': False, 'Regular Spiking': False,
             'Fast Spiking': False}
    new_query = {}
    for name in cluster_names:
        new_query[name] = query.copy()

    query = userIO.dictIO(new_query, shell=shell)
    ans = query.fill_dict()
    if ans is None:
        return None
    out = []
    for name in cluster_names:
        c = {}
        c['single_unit'] = int(ans[name]['Single Unit'])
        c['regular_spiking'] = int(ans[name]['Regular Spiking'])
        c['fast_spiking'] = int(ans[name]['Fast Spiking'])
        out.append(c.copy())
    return out


def label_single_unit(hf5_file, cluster, fs, sorting_log=None,
                      metrics_dir=None):
    '''Adds a sorted unit to the hdf5 store
    adds unit info to unit_descriptor table and spike times and waveforms to
    sorted_units
    and saves metrics for unit into sorted_units folder

    Parameters
    ----------
    hf5_file : str, full path to .h5 file
    electrode_num : int, electrode number
    spike_times : numpy.array
        1D array of spike times corresponding to this unit
    spike_waveforms : numpy.array,
        array containing waveforms of spikes for this unit with each row
        corresponding to rows in spike_times
    single_unit : {0, 1},
        0 (default) is for multi-unit activity, 1 to indicate single unit
    reg_spiking : {0, 1},
        0 (default) is if multi-unit or not regular-spiking pyramidal cell
    fast_spiking : {0, 1},
        0 (default) is if multi-unit or not fast-spiking interneuron
    '''
    if sorting_log is None:
        sorting_log = hf5_file.replace('.h5', '_sorting.log')
    if metrics_dir is None:
        file_dir = os.path.dirname(hf5_file)
        metrics_dir = os.path.join(file_dir, 'sorted_unit_metrics')

    unit_name = get_next_unit_name(hf5_file)
    metrics_dir = os.path.join(metrics_dir, unit_name)
    if not os.path.exists(metrics_dir):
        os.mkdir(metrics_dir)

    with open(sorting_log, 'a+') as log:
        print('%s sorted on %s'
              % (unit_name,
                 dt.datetime.today().strftime('%m/%d/%y %H: %M')),
              file=log)
        print('Cluster info: \n----------', file=log)
        print_clust = deepcopy(cluster)
        # Get rid of data arrays in output clister
        for k, v in cluster.items():
            if isinstance(v, np.ndarray):
                print_clust.pop(k)
        print(dp.print_dict(print_clust), file=log)
        print('Saving metrics to %s' % metrics_dir, file=log)
        print('--------------', file=log)

    with tables.open_file(hf5_file, 'r+') as hf5:
        table = hf5.root.unit_descriptor
        unit_descrip = table.row
        unit_descrip['electrode_number'] = int(cluster['electrode'])
        unit_descrip['single_unit'] = int(cluster['single_unit'])
        unit_descrip['regular_spiking'] = int(cluster['regular_spiking'])
        unit_descrip['fast_spiking'] = int(cluster['fast_spiking'])

        hf5.create_group('/sorted_units', unit_name, title=unit_name)
        waveforms = hf5.create_array('/sorted_units/%s' % unit_name,
                                     'waveforms', cluster['spike_waveforms'])
        times = hf5.create_array('/sorted_units/%s' % unit_name,
                                 'times', cluster['spike_times'])
        unit_descrip.append()
        table.flush()
        hf5.flush()

    # Save metrics for sorted unit
    energy = cluster['data'][:, 0]
    amplitudes = cluster['data'][:, 1]
    pca_slices = cluster['data'][:, 2:]

    np.save(os.path.join(metrics_dir, 'spike_times.npy'),
            cluster['spike_times'])
    np.save(os.path.join(metrics_dir, 'spike_waveforms.npy'),
            cluster['spike_waveforms'])
    np.save(os.path.join(metrics_dir, 'energy.npy'), energy)
    np.save(os.path.join(metrics_dir, 'amplitudes.npy'), amplitudes)
    np.save(os.path.join(metrics_dir, 'pca_slices.npy'), pca_slices)
    clust_info_file = os.path.join(metrics_dir, 'cluster.info')
    with open(clust_info_file, 'a+') as log:
        print('%s sorted on %s'
              % (unit_name,
                 dt.datetime.today().strftime('%m/%d/%y %H: %M')),
              file=log)
        print('Cluster info: \n----------', file=log)
        print(dp.print_dict(print_clust), file=log)
        print('Saved metrics to %s' % metrics_dir, file=log)
        print('--------------', file=log)

    print('Added %s to hdf5 store as %s'
          % (cluster['Cluster Name'], unit_name))
    print('Saved metrics to %s' % metrics_dir)


def split_cluster(cluster, fs, params=None, shell=True):
    '''Use GMM to re-cluster a single cluster into a number of sub clusters

    Parameters
    ----------
    cluster : dict, cluster metrics and data
    params : dict (optional), parameters for reclustering
        with keys: 'Number of Clusters', 'Max Number of Iterations',
                    'Convergence Criterion', 'GMM random restarts'
    shell : bool , optional, whether to use shell or GUI input (defualt=True)

    Returns
    -------
    clusters : list of dicts
        resultant clusters from split
    '''
    if params is None:
        clustering_params = {'Number of Clusters': 2,
                             'Max Number of Iterations': 1000,
                             'Convergence Criterion': 0.00001,
                             'GMM random restarts': 10}
        params_filler = userIO.dictIO(clustering_params, shell=shell)
        params = params_filler.fill_dict()

    if params is None:
        return None

    n_clusters = int(params['Number of Clusters'])
    n_iter = int(params['Max Number of Iterations'])
    thresh = float(params['Convergence Criterion'])
    n_restarts = int(params['GMM random restarts'])

    g = GaussianMixture(n_components=n_clusters, covariance_type='full',
                        tol=thresh, max_iter=n_iter, n_init=n_restarts)
    g.fit(cluster['data'])

    out_clusters = []
    if g.converged_:
        spike_waveforms = cluster['spike_waveforms']
        spike_times = cluster['spike_times']
        data = cluster['data']
        predictions = g.predict(data)
        for c in range(n_clusters):
            clust_idx = np.where(predictions == c)[0]
            tmp_clust = deepcopy(cluster)
            clust_id = str(cluster['cluster_id']) + \
                bytes([b'A'[0]+c]).decode('utf-8')
            clust_name = 'E%iS%i_cluster%s' % \
                (cluster['electrode'], cluster['solution'], clust_id)
            clust_waveforms = spike_waveforms[clust_idx]
            clust_times = spike_times[clust_idx]
            clust_data = data[clust_idx, :]
            clust_log = cluster['manipulations'] + \
                '\nSplit %s with parameters: ' \
                '\n%s\nCluster %i from split results. Named %s' \
                % (cluster['Cluster Name'], dp.print_dict(params),
                   c, clust_name)
            tmp_clust['Cluster Name'] = clust_name
            tmp_clust['cluster_id'] = clust_id
            tmp_clust['data'] = clust_data
            tmp_clust['spike_times'] = clust_times
            tmp_clust['spike_waveforms'] = clust_waveforms

            tmp_isi, tmp_violations1, tmp_violations2 = \
                get_ISI_and_violations(clust_times, fs)
            tmp_clust['ISI'] = tmp_isi
            tmp_clust['1ms_violations'] = tmp_violations1
            tmp_clust['2ms_violations'] = tmp_violations2
            out_clusters.append(deepcopy(tmp_clust))

    return out_clusters


def get_unit_metrics(file_dir, electrode_num, solution_num, cluster_num):
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
    np.array, np.array, np.array, np.array, np.array, np.array
    spike_waveforms, spike_times, pca_slices, energy, amplitudes, predictions
    '''
    spike_waveforms, spike_times, pca_slices, energy, amplitudes, predictions \
        = get_clustering_metrics(file_dir, electrode_num, solution_num)

    this_cluster = np.where(predictions == cluster_num)[0]
    spike_waveforms = spike_waveforms[this_cluster, :]
    spike_times = spike_times[this_cluster]
    pca_slices = pca_slices[this_cluster, :]
    energy = energy[this_cluster]
    amplitudes = amplitudes[this_cluster]
    predictions = predictions[this_cluster]

    return spike_waveforms, spike_times, pca_slices, \
        energy, amplitudes, predictions


def get_cluster_data(file_dir, electrode_num, solution_num, cluster_num, fs,
                     n_pc=3):
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
        with keys: Cluster Name, data, spike_times, spike_waveforms,
        manipulations

        Cluster Name: str, Unique identifier for cluster
        electrode : int
        solution : int
        cluster_id : str, unique cluster id, updated on splits and merges
        data : numpy.array
            with columns: normalized energy, normalized amplitudes, PC1, PC2,
            PC3
        spike_times : numpy.array
        spike_waveforms : numpy.array
        manipulations : str, log of actions taken on this cluster
    '''
    spike_waveforms, spike_times, pca_slices, energy, amplitudes, predictions \
        = get_unit_metrics(file_dir, electrode_num, solution_num, cluster_num)

    data = np.zeros((len(spike_times), n_pc+2))
    data[:, 0] = energy/np.max(energy)
    data[:, 1] = np.abs(amplitudes)/np.max(amplitudes)
    data[:, 2:] = pca_slices[:, : n_pc]

    ISI, violations1, violations2 = get_ISI_and_violations(spike_times, fs)
    cluster = {'Cluster Name': 'E%iS%i_cluster%i' % (electrode_num,
                                                     solution_num,
                                                     cluster_num),
               'electrode': electrode_num,
               'solution': solution_num,
               'cluster_id': str(cluster_num),
               'data': data,
               'spike_times': spike_times,
               'spike_waveforms': spike_waveforms,
               'ISI': ISI,
               '1ms_violations': violations1,
               '2ms_violations': violations2,
               'single_unit': 0,
               'fast_spiking': 0,
               'regular_spiking': 0,
               'manipulations': ''}

    return deepcopy(cluster)


def get_clustering_metrics(file_dir, electrode_num, solution_num):
    '''Grab all saved metrics associated with a particular clustering solution
    for an electrode

    Parameters
    ----------
    file_dir : str, path to data directory
    electrode_num : int, electrode number
    solution_num : int, number of clusters in desired solution

    Returns
    -------
    np.array, np.array, np.array, np.array, np.array, np.array
    spike_waveforms, spike_times, pca_slices, energy, amplitudes, predictions
    '''
    waveform_dir = os.path.join(file_dir, 'spike_waveforms',
                                'electrode%i' % electrode_num)
    spike_time_file = os.path.join(file_dir, 'spike_times',
                                   'electrode%i' % electrode_num,
                                   'spike_times.npy')
    predictions_file = os.path.join(file_dir, 'clustering_results',
                                    'electrode%i' % electrode_num,
                                    'clusters%i' % solution_num,
                                    'predictions.npy')
    spike_times = np.load(spike_time_file)
    predictions = np.load(predictions_file)
    spike_waveforms = np.load(os.path.join(waveform_dir,
                                           'spike_waveforms.npy'))
    pca_slices = np.load(os.path.join(waveform_dir, 'pca_waveforms.npy'))
    energy = np.load(os.path.join(waveform_dir, 'energy.npy'))
    amplitudes = np.load(os.path.join(waveform_dir, 'spike_amplitudes.npy'))

    return spike_waveforms, spike_times, pca_slices, energy, \
        amplitudes, predictions


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
    with tables.open_file(hf5_file, 'r') as hf5:
        if '/sorted_units' in hf5:
            node_list = hf5.list_nodes('/sorted_units')
            if node_list == []:
                return []
            out = [int(parser.match(node._v_name)[1]) for node in node_list]
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


def get_ISI_and_violations(spike_times, fs):
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


def merge_clusters(clusters, fs):
    '''Merge 2 or more clusters into a single cluster

    Parameters
    ----------
    clusters : list of dict, list of cluster dictionaries
    fs : float, sampling rate in Hz

    Returns
    -------
    cluster : dict, merged cluster
    '''
    clusters = deepcopy(clusters)
    clust = clusters.pop(0)
    for c in clusters:
        clust['cluster_id'] = clust['cluster_id']+c['cluster_id']
        clust['spike_times'] = np.concatenate((clust['spike_times'],
                                               c['spike_times']))
        clust['data'] = np.concatenate((clust['data'], c['data']))
        clust['spike_waveforms'] = np.concatenate((clust['spike_waveforms'],
                                                   c['spike_waveforms']))
        new_name = 'E%iS%i_cluster%s' % (clust['electrode'], clust['solution'],
                                         clust['cluster_id'])
        clust['manipulations'] += ('Cluster %s merged with cluster %s.\n'
                                   'New Cluster named: %s') \
            % (clust['Cluster Name'], c['Cluster Name'], new_name)
    idx = np.argsort(clust['spike_times'])
    clust['Cluster Name'] = 'E%iS%i_cluster%s' % (clust['electrode'],
                                                  clust['solution'],
                                                  clust['cluster_id'])
    clust['spike_times'] = clust['spike_times'][idx]
    clust['spike_waveforms'] = clust['spike_waveforms'][idx, :]
    clust['data'] = clust['data'][idx, :]
    ISI, v1, v2 = get_ISI_and_violations(clust['spike_times'], fs)
    clust['ISI'] = ISI
    clust['1ms_violations'] = v1
    clust['2ms_violations'] = v2
    clust['regular_spiking'] = 0
    clust['single_unit'] = 0
    clust['fast_spiking'] = 0

    return deepcopy(clust)


def plot_cluster(cluster, index=None):
    '''Plots a cluster with isi and violation info for viewing

    Parameters
    ----------
    cluster : dict with cluster info

    '''
    fig, ax = blech_waveforms_datashader.waveforms_datashader(
        cluster['spike_waveforms'])
    ax.set_xlabel('Sample (30 samples per ms)')
    ax.set_ylabel('Voltage (microvolts)')
    title_str = (('Cluster Name: {: s}\n2ms Violations={: .1f}%, '
                  '1ms Violations={: .1f}%\nNumber of Waveforms'
                  '={: d}').format(
                                   cluster['Cluster Name'],
                                   cluster['2ms_violations'],
                                   cluster['1ms_violations'],
                                   cluster['spike_times'].shape[0]))
    if index is not None:
        title_str = 'Index: %i %s, ' % (index, title_str)

    ax.set_title(title_str)
    return fig, ax


def make_unit_plots(file_dir, fs):
    '''Makes waveform plots for sorted unit in unit_waveforms_plots

    Parameters
    ----------
    file_dir : str, full path to recording directory
    fs : float, smapling rate in Hz
    '''
    h5_name = h5io.get_h5_filename(file_dir)
    h5_file = os.path.join(file_dir, h5_name)

    plot_dir = os.path.join(file_dir, 'unit_waveforms_plots')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir, ignore_errors=True)
    os.mkdir(plot_dir)
    fs_str = '(%g samples per ms)' % (fs/1000.0)
    unit_numbers = get_unit_numbers(h5_file)
    with tables.open_file(h5_file, 'r+') as hf5:
        units = hf5.list_nodes('/sorted_units')
        for i, unit in zip(unit_numbers, units):
            # plot all waveforms
            waveforms = unit.waveforms[:]
            descriptor = hf5.root.unit_descriptor[i]
            fig, ax = blech_waveforms_datashader.waveforms_datashader(
                waveforms)
            ax.set_xlabel('Samples (%s)' % fs_str)
            ax.set_ylabel('Voltage (microvolts)')
            unit_title = (('Unit %i, total waveforms = %i\nElectrode: %i, '
                           'Single Unit: %i, RSU: %i, FSU: %i') %
                          (i, waveforms.shape[0],
                           descriptor['electrode_number'],
                           descriptor['single_unit'],
                           descriptor['regular_spiking'],
                           descriptor['fast_spiking']))
            ax.set_title(unit_title)
            fig.savefig(os.path.join(plot_dir, 'Unit%i.png' % i))
            plt.close('all')

            # Plot mean and SEM of waveforms
            # Downsample by 10 to remove upsampling from de-jittering
            fig = plt.figure()
            mean_wave = np.mean(waveforms[:, ::10], axis=0)
            std_wave = np.std(waveforms[:, ::10], axis=0)
            mean_x = np.arange(mean_wave.shape[0]) + 1
            plt.plot(mean_x, mean_wave, linewidth=4.0)
            plt.fill_between(mean_x, mean_wave - std_wave,
                             mean_wave + std_wave, alpha=0.4)
            plt.xlabel('Samples (%s)' % fs_str)
            plt.ylabel('Voltage (microvolts)')
            plt.title(unit_title)
            fig.savefig(os.path.join(plot_dir, 'Unit%i_mean_sd.png' % i))
            plt.close('all')


def delete_unit(file_dir, unit_num):
    '''Delete a sorted unit and re-label all following units. Also relabel all
    associated plots and data in sorted_unit_metrics and unit_waveform_plots

    Parameters
    ----------
    file_dir : str, full path to recording directory
    unit_num : int, number of unit to delete
    '''
    h5_name = h5io.get_h5_filename(file_dir)
    h5_file = os.path.join(file_dir, h5_name)
    unit_numbers = get_unit_numbers(h5_file)
    unit_name = 'units%03d' % unit_num
    change_units = [x for x in unit_numbers if x > unit_num]
    new_units = [x-1 for x in change_units]
    new_names = ['unit%03d' % x for x in new_units]
    old_names = ['unit%03d' % x for x in change_units]
    old_prefix = ['Unit%i' % x for x in change_units]
    new_prefix = ['Unit%i' % x for x in new_units]

    metrics_dir = os.path.join(file_dir, 'sorted_unit_metrics')
    plot_dir = os.path.join(file_dir, 'unit_waveforms_plots')

    # Remove metrics
    shutil.rmtree(os.path.join(metrics_dir, unit_name))

    # remove unit from hdf5 store
    with tables.open_file(h5_file, 'r+') as hf5:
        hf5.remove_node('/sorted_units', name=unit_name)
        table = hf5.root.unit_descriptor
        table.remove_row(unit_num)
        # rename rest of units in hdf5 and metrics folders
        for x, y in zip(old_names, new_names):
            hf5.rename_node('/sorted_units', newname=y, name=x)
            os.rename(os.path.join(metrics_dir, x),
                      os.path.join(metrics_dir, y))
        hf5.flush()

    # delete and rename plot files
    plot_files = os.listdir(plot_dir)
    for x in plot_files:
        if x.startswith('Unit%i' % unit_num):
            os.remove(os.path.join(plot_dir, x))
        elif any([x.startswith(y) for y in old_prefix]):
            pre = [b for a, b in zip(old_prefix, new_prefix)
                   if x.startswith(a)]
            old_file = os.path.join(plot_dir, x)
            new_file = os.path.join(plot_dir, x.replace(pre[0][0], pre[0][1]))
            os.rename(old_file, new_file)

    # Compress and repack
    h5io.compress_and_repack(h5_file)


def make_spike_arrays(h5_file, params):
    '''Makes stimulus triggered spike array for all sorted units

    Parameters
    ----------
    h5_file : str, full path to hdf5 store
    params : dict
        Parameters for arrays with fields:
            dig_ins_to_use : list of int, which dig_in channels for arrays
            laser_channels : list of int or None if no lasers
            sampling_rate : float, sampling rate of data in Hz
            pre_stimulus: : int, ms before stimulus to include in array
            post_stimulus : int, ms after stimulus to include in array
    '''
    dig_in_ch = params['dig_ins_to_use']
    laser_ch = params['laser_channels']
    fs = params['sampling_rate']
    pre_stim = int(params['pre_stimulus'])
    post_stim = int(params['post_stimulus'])
    pre_idx = int(pre_stim * (fs/1000))
    post_idx = int(pre_stim * (fs/1000))
    n_pts = pre_stim + post_stim

    if dig_in_ch is None or dig_in_ch == []:
        raise ValueError('Must provide dig_ins_to_use in params in '
                         'order to make spike arrays')
    # get digital input table
    dig_in_table = h5io.read_trial_data_table(h5_file, 'in',
                                              dig_in_ch)

    # Get laser input table
    if laser_ch is not None or laser_ch == []:
        laser_table = h5io.read_trial_data_table(h5_file, 'in',
                                                 laser_ch)
        lasers = True
        n_lasers = len(laser_ch)
    else:
        laser_table = None
        lasers = False
        n_lasers = 1

    exp_table = h5io.read_trial_data_table(h5_file, 'in', [-1])
    exp_end_idx = exp_table['off_index'][0]
    exp_end_time = exp_end_idx/fs

    # Use last spike time to determine end of experiment in case headstage fell
    # off

    with tables.open_file(h5_file, 'r+') as hf5:
        n_units = len(hf5.list_nodes('/sorted_units'))

        # Get experiment end time from last spike time in case headstage fell
        # off
        exp_end_idx = 0
        for unit in hf5.root.sorted_units:
            tmp = np.max(unit.times)
            if tmp > exp_end_idx:
                exp_end_idx = tmp

        exp_end_time = exp_end_idx/fs

        if '/spike_trains' in hf5:
            hf5.remove_node('/', 'spike_trains', recursive=True)

        hf5.create_group('/', 'spike_trains')

        for i in dig_in_ch:

            # grab trials for dig_in_ch that end more than post_stim ms before
            # the end of the experiment
            tmp_trials = dig_in_table.query('channel == @i')
            trial_cutoff_idx = exp_end_idx - post_idx

            # get the end indices for those trials
            off_idx = np.array(tmp_trials['off_index'])
            off_idx.sort()
            n_trials = len(off_idx)

            # loop through trials and get spike train array for each
            spike_train = []
            cond_array = np.zeros(n_trials)
            laser_start = np.zeros(n_trials)
            laser_single = np.zeros((n_trials, n_lasers))

            for ti, trial_off in enumerate(off_idx):
                if trial_off >= trial_cutoff_idx:
                    cond_array[ti] = -1
                    continue

                window = (trial_off - pre_idx, trial_off + post_idx)
                spike_shift = pre_idx - trial_off
                spike_array = np.zeros((n_units, n_pts))

                # loop through units
                for unit in hf5.root.sorted_units:
                    unit_num = int(unit._v_name[-3:])
                    spike_idx = np.where((unit.times[:] >= window[0]) &
                                         (unit.times[:] <= window[1]))[0]
                    spike_times = unit.times[spike_idx]

                    # Shift to align to trial window and convert to ms
                    spike_times = (spike_times + spike_shift) / (fs/1000)
                    spike_times = spike_times.astype(int)
                    spike_array[unit_num, spike_times] = 1

                spike_train.append(spike_array)

                if lasers:
                    # figure out which laser trial matches with this dig_in
                    # trial and get the duration and onset lag
                    for li, l in enumerate(laser_ch):
                        tmp_trial = laser_table.query('abs(off_index - '
                                                      '@trial_off) <= '
                                                      '@post_idx')
                        if not tmp_trial.empty:
                            # Mark which laser was on
                            laser_single[ti, li] = 1.0

                            # Get duration of laser
                            duration = (tmp_trial['off_index'] -
                                        tmp_trial['on_index']) / (fs/1000)
                            # round duration down to nearest multiple of 10ms
                            cond_array[ti] = 10*int(duration.iloc[0]/10)

                            # Get onset lag of laser, time between laser start
                            # and end of the trial
                            lag = (tmp_trial['on_index'] -
                                   trial_off) / (fs/1000)
                            # Round lag down to nearest multiple of 10ms
                            laser_start[ti] = 10*int(lag.iloc[0]/10)

            hf5.create_group('/spike_trains', 'dig_in_%i' % i)
            tmp = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                   'spike_array', np.array(spike_train))
            hf5.flush()

            if lasers:
                ld = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                      'laser_durations', cond_array)
                ls = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                      'laser_onset_lag', laser_start)
                ol = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                      'on_laser', laser_single)
                hf5.flush()
