from blechpy.dio import rawIO, h5io
from blechpy.utils.print_tools import println
import numpy as np
import configparser
import os
import h5py
import shutil
import subprocess
import pylab as plt
from blechpy.plotting import data_plot as dplt
from blechpy.analysis import clustering as clust, spike_sorting as ss
from blechpy.analysis import blech_clustering as blcust
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULTS_DIR = os.path.join(SCRIPT_DIR, 'defaults')

def set_config_parameter(config_file, heading, parameter, value):
    '''Allows editing of a single parameter in a config.params file for spyking
    circus

     Parameters
     ----------
     config_file : str
     heading : str
     parameter: str
     value : str
     '''
    parser = configparser.ConfigParser()
    with open(config_file, 'r') as f:
        parser.read_file(f)
        if not parser.has_section(heading):
            raise ValueError('%s is not a valid heading the the config file' % heading)
        elif not parser.has_option(heading, parameter):
            raise ValueError('%s not found under heading %s' % (parameter, heading))
        else:
            parser.set(heading, parameter, str(value))

    with open(config_file, 'w') as f:
        parser.write(f)


def read_config_parameter(config_file, heading, parameter):
    '''Returns a parameter values from a config.params file for spyking circus

    Parameters
    ----------
    config_file : str
    heading : str
    parameter : str
    '''
    parser = configparser.ConfigParser()
    with open(config_file, 'r') as f:
        parser.read_file(f)
        if not parser.has_section(heading):
            raise ValueError('%s is not a valid heading the the config file' % heading)
        elif not parser.had_option(heading, parameter):
            raise ValueError('%s not found under heading %s' % (parameter, heading))
        else:
            return parser.get(heading, parameter)


class circus_clust(object):
    def __init__(self, file_dir, data_name, sampling_rate, emap):
        self._data_name = data_name
        self._rec_dir = file_dir
        self._fs = sampling_rate
        self._emap = emap
        self._circus_dir = os.path.join(file_dir, 'spyking_circus')
        self._param_file = os.path.join(self._circus_dir, data_name+'.params')
        self._probe_file = self._param_file.replace('.params','.prb')
        self._results_dir = self._param_file.replace('.params','')
        self._data_file = self._param_file.replace('.params','.npy')
        self._cluster_file = os.path.join(self._results_dir, data_name + '.clusters.hdf5')
        self._circus_map_file = os.path.join(self._circus_dir, 'channel_map.json')

        if not os.path.isdir(self._circus_dir):
            os.mkdir(self._circus_dir)

    def prep_for_circus(self):
        '''Prepares 'one file per channel' data from intan recording for spyking
        circus

        Parameters
        ----------
        file_dir : str, recording directory
        out_dir : str (optional), directory for spyking circus data
        '''
        # Copy default config and probe files
        if not os.path.isfile(self._param_file):
            print('Creating spyking circus param file')
            default_config = os.path.join(DEFAULTS_DIR, 'default_config.params')
            shutil.copyfile(default_config, self._param_file)
        else:
            print('Spyking circus param file already exists. Keeping.')

        if not os.path.isfile(self._probe_file):
            print('Creating circus probe file from electrode mapping')
            self._make_probe_file()
        else:
            print('Probe file already exists. Keeping.')

        # Adjust config parameters
        set_config_parameter(self._param_file, 'data','mapping',os.path.basename(self._probe_file))
        set_config_parameter(self._param_file, 'data','sampling_rate',self._fs)
        emap = self._emap
        dead_channels = (emap[emap['dead']]
                         .groupby('CAR_group')['Electrode']
                         .unique().to_dict())
        dead_channels = {int(i+1) : list(j) for i,j in dead_channels.items()}
        set_config_parameter(self._param_file, 'detection', 'dead_channels', repr(dead_channels))

        # Get referenced data into npy
        '''
        if os.path.isfile(self._data_file):
            print('Numpy circus datafile already exists. Keeping.')
        else:
            print('Creating referenced circus numpy datafile...')
            self._make_referenced_circus_npy()
        '''
        # ALTERNATIVE: Make raw data into npy and enable filtering in circus
        if os.path.isfile(self._data_file):
            print('Numpy circus datafile already exists. Keeping.')
        else:
            print('Creating raw circus numpy datafile...')
            self._make_raw_circus_npy()

        # Re-number electrodes to match spyking-circus output after dead channels dropped
        if os.path.isfile(self._circus_map_file):
            print('Reading circus channel mapping from json file.')
            self._circus_channel_map = pd.read_json(self._circus_map_file, orient='records')
        else:
            new_map = (emap[emap['dead']==False].reset_index(drop=True)
                       .reset_index().rename(columns={'index': 'circus_channel'}))
            self._circus_channel_map = new_map
            new_map.to_json(self._channel_map_file, orient='records')


    def _make_probe_file(self, radius=100):
        '''Make prb file for spyking circus from electrode_mapping
        This assumes single-channel electrodes and the probe file defines geometry
        such that all electrodes are farther than radius away from each other so
        that spikes do not interact.

        Parameters
        ----------
        radius : int
            radius around eletrode from which spikes can be detected in microns
            100 is good for in vivo extracellular recordings
        '''
        emap = self._emap
        out_str = ["'''",
                   "Custom probe layout generated from electrode_mapping",
                   "%s" % self._data_name,
                   "'''"]
        out_str.append('total_nb_channels = %i' % len(emap))
        out_str.append('radius = %i' % radius)

        bundles = emap.groupby('CAR_group')['Electrode'].unique().to_dict()
        group_nums = list(bundles.keys())

        channel_group = {int(i+1) : {'channels': list(bundles[i]), 'graph': [],
                                     'geometry': {j : [i*2*radius, k*2*radius]
                                                  for k,j in enumerate(bundles[i])}}
                         for i in group_nums}

        out_str.append('channel_groups = %s' % repr(channel_group))

        with open(self._probe_file, 'w') as f:
            f.write('\n'.join(out_str))


    def _make_referenced_circus_npy(self):
        emap = self._emap
        file_dir = self._rec_dir
        all_data = None
        for i, row in emap.iterrows():
            el = row['Electrode']
            if row['dead']:
                trace = h5io.get_raw_trace(file_dir, el, emap)
            else:
                trace = h5io.get_referenced_trace(file_dir, el)

            if trace is None:
                raise ValueError('Unable to obtain data trace for electrode %i' % el)

            if all_data is None:
                all_data = trace
            else:
                all_data = np.vstack((all_data, trace))

        np.save(self._data_file, all_data)

    def _make_raw_circus_npy(self):
        emap = self._emap
        file_dir = self._rec_dir
        all_data = None
        for i, row in emap.iterrows():
            el = row['Electrode']
            trace = h5io.get_raw_trace(file_dir, el, emap)

            if trace is None:
                raise ValueError('Unable to obtain data trace for electrode %i' % el)

            if all_data is None:
                all_data = trace
            else:
                all_data = np.vstack((all_data, trace))

        np.save(self._data_file, all_data)

    def start_the_show(self):
        fn = os.path.basename(self._data_file)
        curr_dir = os.getcwd()
        os.chdir(self._circus_dir)
        subprocess.call(['spyking-circus', fn])
        os.chdir(curr_dir)

    def plot_cluster_waveforms(self):
        plot_dir = os.path.join(self._circus_dir, 'Cluster_Plots')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)
        fs = self._fs

        for i, row in self._circus_channel_map.iterrows():
            channel = row['circus_channel']
            electrode_num = row['Electrode']
            cluster_data = self._get_clusters(channel)
            if sum((x['spike_waveforms'].shape[0] for x in cluster_data)) < 10:
                continue

            # Plot pca
            fig, axs = dplt.plot_cluster_pca(cluster_data)
            fig.savefig(os.path.join(plot_dir, 'Electrode_%i_pca.png' % electrode_num))
            plt.close('all')

            for clust in cluster_data:
                fig, ax = dplt.plot_waveforms(clust['spike_waveforms'])
                fn = os.path.join(plot_dir, 'E%i_Ch%i_C%s_waveforms.png'
                                  % (electrode_num, channel, clust['cluster_id']))
                fig.savefig(fn)
                plt.close('all')

    def _get_clusters(self, channel_num=None, electrode_num=None):
        if channel_num is None and electrode_num is None:
            raise ValueError('Must provide channel or electrode')

        fs = self._fs
        ch_map = self._circus_channel_map

        if electrode_num is None:
            electrode_num = ch_map[ch_map['circus_channel'] == channel_num]['Electrode'].values[0]

        if channel_num is None:
            channel_num = ch_map[ch_map['Electrode'] == electrode_num]['circus_channel'].values[0]

        out = []
        with h5py.File(self._cluster_file, 'r') as f:
            clusters = f['clusters_%i' % channel_num][:]
            all_times = f['times_%i' % channel_num][:]
            ref_el = h5io.get_referenced_trace(self._rec_dir, electrode_num)
            if ref_el is None:
                raise ValueError('Referenced Trace for electrode %i not found' % electrode_num)

            for c in np.unique(clusters):
                cluster_name = 'E%i_cluster%i' % (electrode_num, c)
                idx = np.where(clusters == c)[0]
                spike_times = all_times[idx]
                spike_waveforms, new_fs = clust.get_waveforms(ref_el, spike_times)
                ISI, violations1, violations2 = bclust.get_ISI_and_violations(spike_times, fs)
                # TODO: Actually make cluster data matrix
                data = None
                cluster = {'Cluster_Name': cluster_name,
                           'electrode': electrode_num,
                           'solution': channel_num,
                           'cluster_id': str(c),
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
                out.append(cluster)

        return out
