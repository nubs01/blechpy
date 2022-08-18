import pandas as pd
import numpy as np
import datetime as dt
import pickle
import os
import shutil
import sys
from joblib import Parallel, delayed, cpu_count
import subprocess
from tqdm import tqdm
from copy import deepcopy
from blechpy.utils import print_tools as pt, write_tools as wt, userIO
from blechpy.utils.decorators import Logger
from blechpy.analysis import palatability_analysis as pal_analysis
from blechpy.analysis import spike_sorting as ss, spike_analysis, circus_interface as circ
from blechpy.analysis import blech_clustering as clust
from blechpy.plotting import palatability_plot as pal_plt, data_plot as datplt
from blechpy import dio
from blechpy.datastructures.objects import data_object
from blechpy.utils import spike_sorting_GUI as ssg
from scipy.ndimage import gaussian_filter1d

class dataset(data_object):
    '''Stores information related to an intan recording directory, allows
    executing basic processing and analysis scripts, and stores parameters data
    for those analyses

    Parameters
    ----------
    file_dir : str (optional)
        absolute path to a recording directory, if left empty a filechooser
        will popup
    '''
    PROCESSING_STEPS = ['initialize parameters',
                        'extract_data', 'create_trial_list',
                        'mark_dead_channels',
                        'common_average_reference', 'spike_detection',
                        'spike_clustering', 'cleanup_clustering',
                        'sort_units', 'make_unit_plots',
                        'units_similarity', 'make_unit_arrays',
                        'make_psth_arrays', 'plot_psths',
                        'palatability_calculate', 'palatability_plot',
                        'overlay_psth']

    def __init__(self, file_dir=None, data_name=None, shell=False):
        '''Initialize dataset object from file_dir, grabs basename from name of
        directory and initializes basic analysis parameters

        Parameters
        ----------
        file_dir : str (optional), file directory for intan recording data

        Throws
        ------
        ValueError
            if file_dir is not provided and no directory is chosen
            when prompted
        NotADirectoryError : if file_dir does not exist
        '''
        super().__init__('dataset', file_dir, data_name=data_name, shell=shell)
        h5_file = dio.h5io.get_h5_filename(self.root_dir)
        if h5_file is None:
            h5_file = os.path.join(self.root_dir, '%s.h5' % self.data_name)
            print(f'No existing h5 file found. New h5 will be created at {h5_file}.')
        else:
            print(f'Existing h5 file found. Using {h5_file}.')

        self.h5_file = h5_file

        self.dataset_creation_date = dt.datetime.today()

        # Outline standard processing pipeline and status check
        self.processing_steps = dataset.PROCESSING_STEPS.copy()
        self.process_status = dict.fromkeys(self.processing_steps, False)

    def _change_root(self, new_root=None):
        old_root = self.root_dir
        new_root = super()._change_root(new_root)
        self.h5_file = self.h5_file.replace(old_root, new_root)
        return new_root

    @Logger('Initializing Parameters')
    def initParams(self, data_quality='clean', 
                   emg_port=None, emg_channels=None, 
                   car_keyword=None, car_group_areas=None,
                   shell=False, dig_in_names=None, dig_out_names=None,
                   accept_params=False):
        '''
        Initalizes basic default analysis parameters and allows customization
        of parameters

        Parameters (all optional)
        -------------------------
        data_quality : {'clean', 'noisy', 'hp'}
            keyword defining which default set of parameters to use to detect
            headstage disconnection during clustering
            default is 'clean'. Best practice is to run blech_clust as 'clean'
            and re-run as 'noisy' if too many early cutoffs occurr. 
            Alternately run as 'hp' (high performance)
            default parameter sets found in dio.defualts.clustering_params.json
        emg_port : str
            Port ('A', 'B', 'C') of EMG, if there was an EMG. None (default)
            will query user. False indicates no EMG port and not to query user
        emg_channels : list of int
            channel or channels of EMGs on port specified
            default is None
        car_keyword : str
            Specifes default common average reference groups
            defaults are found in dio.defaults.CAR_params.json
            'bilateral32' and 'bilateral64' are available keywords 
            If left as None (default) user will be queries to select common
            average reference groups
        shell : bool
            False (default) for GUI. True for command-line interface
        dig_in_names : list of str
            Names of digital inputs. Must match number of digital inputs used
            in recording.
            None (default) queries user to name each dig_in
        dig_out_names : list of str
            Names of digital outputs. Must match number of digital outputs in
            recording.
            None (default) queries user to name each dig_out
        accept_params : bool
            True automatically accepts default parameters where possible,
            decreasing user queries
            False (default) will query user to confirm or edit parameters for
            clustering, spike array and psth creation and palatability/identity
            calculations
        '''
        # Get parameters from info.rhd
        file_dir = self.root_dir
        rec_info = dio.rawIO.read_rec_info(file_dir, shell=shell)
        ports = rec_info.pop('ports')
        channels = rec_info.pop('channels')
        sampling_rate = rec_info['amplifier_sampling_rate']
        self.rec_info = rec_info
        self.sampling_rate = sampling_rate

        # Get default parameters from files
        # Get default parameters from files
        clustering_params = dio.params.load_params('clustering_params', file_dir,
                                                   default_keyword=data_quality)
        spike_array_params = dio.params.load_params('spike_array_params', file_dir)
        psth_params = dio.params.load_params('psth_params', file_dir)
        pal_id_params = dio.params.load_params('pal_id_params', file_dir)
        spike_array_params['sampling_rate'] = sampling_rate
        clustering_params['file_dir'] = file_dir
        clustering_params['sampling_rate'] = sampling_rate
        self.spike_array_params = spike_array_params

        # Setup digital input mapping
        if rec_info.get('dig_in'):
            self._setup_digital_mapping('in', dig_in_names, shell)
            dim = self.dig_in_mapping.copy()
        else:
            self.dig_in_mapping = None

        if rec_info.get('dig_out'):
            q = userIO.ask_user('Your info.rhd suggests you have digital '
                                'outputs. Is this True?', shell=shell)
            if q == 1:
                self._setup_digital_mapping('out', dig_out_names, shell)
                dom = self.dig_out_mapping.copy()
            else:
                _ = rec_info.pop('dig_out')
                self.dig_out_mapping = None
        else:
            self.dig_out_mapping = None

        # Setup electrode and emg mapping
        self._setup_channel_mapping(ports, channels, emg_port,
                                    emg_channels, shell=shell)

        # Set CAR groups
        self._set_CAR_groups(group_keyword=car_keyword, group_areas=car_group_areas, shell=shell)

        # Confirm parameters
        if not accept_params:
            conf = userIO.confirm_parameter_dict
            clustering_params = conf(clustering_params,
                                     'Clustering Parameters', shell=shell)
            self.edit_spike_array_params(shell=shell)
            psth_params = conf(psth_params,
                               'PSTH Parameters', shell=shell)
            pal_id_params = conf(pal_id_params,
                                 'Palatability/Identity Parameters\n'
                                 'Valid unit_type is Single, Multi or All',
                                 shell=shell)

        # Store parameters
        self.clustering_params = clustering_params
        self.pal_id_params = pal_id_params
        self.psth_params = psth_params
        self._write_all_params_to_json()
        self.process_status['initialize parameters'] = True
        self.save()

    def _set_CAR_groups(self, group_keyword=None, shell=False, group_areas=None):
        '''Sets that electrode groups for common average referencing and
        defines which brain region electrodes eneded up in

        Parameters
        ----------
        group_keyword : str or int
            Keyword corresponding to a preset electrode grouping in CAR_params.json
            Or integer indicating number of CAR groups
        shell : bool
            True for command-line interface, False (default) for GUI
        '''
        if not hasattr(self, 'electrode_mapping'):
            raise ValueError('Set electrode mapping before setting CAR groups')

        em = self.electrode_mapping.copy()

        car_param_file = os.path.join(self.root_dir, 'analysis_params',
                                      'CAR_params.json')
        if os.path.isfile(car_param_file):
            tmp = dio.params.load_params('CAR_params', self.root_dir)
            if tmp is not None:
                group_electrodes = tmp
            else:
                raise ValueError('CAR_params file exists in recording dir, but is empty')

        else:
            if group_keyword is None:
                group_keyword = userIO.get_user_input(
                    'Input keyword for CAR parameters or number of CAR groups',
                    shell=shell)

                if group_keyword is None:
                    ValueError('Must provide a keyword or number of groups')

            if group_keyword.isnumeric():
                num_groups = int(group_keyword)
                group_electrodes = dio.params.select_CAR_groups(num_groups, em,
                                                                shell=shell)
            else:
                group_electrodes = dio.params.load_params('CAR_params',
                                                          self.root_dir,
                                                          default_keyword=group_keyword)

        num_groups = len(group_electrodes)
        if group_areas is not None and len(group_areas) == num_groups:
            for i, x in enumerate(zip(group_electrodes, group_areas)):
                em.loc[x[0], 'area'] = x[1]
                em.loc[x[0], 'CAR_group'] = i

        else:
            group_names = ['Group %i' % i for i in range(num_groups)]
            area_dict = dict.fromkeys(group_names, '')
            area_dict = userIO.fill_dict(area_dict, 'Set Areas for CAR groups',
                                         shell=shell)
            for k, v in area_dict.items():
                i = int(k.replace('Group', ''))
                em.loc[group_electrodes[i], 'area'] = v
                em.loc[group_electrodes[i], 'CAR_group'] = i

        self.CAR_electrodes = group_electrodes
        self.electrode_mapping = em.copy()
        if os.path.isfile(self.h5_file):
            dio.h5io.write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)

    @Logger('Re-labelling CAR group areas')
    def set_electrode_areas(self, areas):
        '''sets the electrode area for each CAR group.

        Parameters
        ----------
        areas : list of str
            number of elements must match number of CAR groups

        Throws
        ------
        ValueError
        '''
        em = self.electrode_mapping.copy()
        if len(em['CAR_group'].unique()) != len(areas):
            raise ValueError('Number of items in areas must match number of CAR groups')

        em['area'] = em['CAR_group'].apply(lambda x: areas[int(x)])
        self.electrode_mapping = em.copy()
        dio.h5io.write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)
        self.save()

    def _setup_digital_mapping(self, dig_type, dig_in_names=None, shell=False):
        '''sets up dig_in_mapping dataframe  and queries user to fill in columns

        Parameters
        ----------
        dig_in_names : list of str (optional)
        shell : bool (optional)
            True for command-line interface
            False (default) for GUI
        '''
        rec_info = self.rec_info
        df = pd.DataFrame()
        df['channel'] = rec_info.get('dig_%s' % dig_type)
        n_dig_in = len(df)
        # Names
        if dig_in_names:
            df['name'] = dig_in_names
        else:
            df['name'] = ''

        # Parameters to query
        if dig_type == 'in':
            df['palatability_rank'] = 0
            df['laser'] = False
            df['spike_array'] = True

        df['exclude'] = False
        # Re-format for query
        idx = df.index
        df.index = ['dig_%s_%i' % (dig_type, x) for x in df.channel]
        dig_str = dig_type + 'put'
        # Query for user input
        prompt = ('Digital %s Parameters\nSet palatability ranks from 1 to %i'
                  '\nor blank to exclude from pal_id analysis') % (dig_str, len(df))
        tmp = userIO.fill_dict(df.to_dict(), prompt=prompt, shell=shell)
        # Reformat for storage
        df2 = pd.DataFrame.from_dict(tmp)
        df2 = df2.sort_values(by=['channel'])
        df2.index = idx
        if dig_type == 'in':
            df2['palatability_rank'] = df2['palatability_rank'].fillna(-1).astype('int')

        if dig_type == 'in':
            self.dig_in_mapping = dim = df2.copy()
            self.spike_array_params['laser_channels'] = dim.channel[dim['laser']].to_list()
            self.spike_array_params['dig_ins_to_use'] = dim.channel[dim['spike_array']].to_list()
            wt.write_params_to_json('spike_array_params', self.root_dir,
                                    self.spike_array_params)
        else:
            self.dig_out_mapping = df2.copy()

        if os.path.isfile(self.h5_file):
            dio.h5io.write_digital_map_to_h5(self.h5_file, self.dig_in_mapping, dig_type)

    def _setup_channel_mapping(self, ports, channels, emg_port, emg_channels, shell=False):
        '''Creates electrode_mapping and emg_mapping DataFrames with columns:
        - Electrode
        - Port
        - Channel

        Parameters
        ----------
        ports : list of str, item corresponing to each channel
        channels : list of int, channels on each port
        emg_port : str
        emg_channels : list of int
        '''
        if emg_port is None:
            q = userIO.ask_user('Do you have an EMG?', shell=shell)
            if q==1:
                emg_port = userIO.select_from_list('Select EMG Port:',
                                                   ports, 'EMG Port',
                                                   shell=shell)
                emg_channels = userIO.select_from_list(
                    'Select EMG Channels:',
                    [y for x, y in
                     zip(ports, channels)
                     if x == emg_port],
                    title='EMG Channels',
                    multi_select=True, shell=shell)

        el_map, em_map = dio.params.flatten_channels(ports, channels,
                                                     emg_port, emg_channels)
        self.electrode_mapping = el_map
        self.emg_mapping = em_map
        if os.path.isfile(self.h5_file):
            dio.h5io.write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)

    def edit_spike_array_params(self, shell=False):
        '''Edit spike array parameters and adjust dig_in_mapping accordingly

        Parameters
        ----------
        shell : bool, whether to use CLI or GUI
        '''
        if not hasattr(self, 'dig_in_mapping'):
            self.spike_array_params = None
            return

        sa = deepcopy(self.spike_array_params)
        tmp = userIO.fill_dict(sa, 'Spike Array Parameters\n(Times in ms)',
                               shell=shell)
        if tmp is None:
            return

        dim = self.dig_in_mapping
        dim['spike_array'] = False
        if tmp['dig_ins_to_use'] != ['']:
            tmp['dig_ins_to_use'] = [int(x) for x in tmp['dig_ins_to_use']]
            dim.loc[[x in tmp['dig_ins_to_use'] for x in dim.channel],
                    'spike_array'] = True

        dim['laser'] = False
        if tmp['laser_channels'] != ['']:
            tmp['laser_channels'] = [int(x) for x in tmp['laser_channels']]
            dim.loc[[x in tmp['laser_channels'] for x in dim.channel],
                    'laser'] = True

        self.spike_array_params = tmp.copy()
        wt.write_params_to_json('spike_array_params',
                                self.root_dir, tmp)
        if os.path.isfile(self.h5_file):
            dio.h5io.write_digital_map_to_h5(self.h5_file, self.dig_in_mapping, 'in')

        self.save()

    def edit_clustering_params(self, shell=False):
        '''Allows user interface for editing clustering parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        '''
        tmp = userIO.fill_dict(self.clustering_params,
                               'Clustering Parameters\n(Times in ms)',
                               shell=shell)
        if tmp:
            self.clustering_params = tmp
            wt.write_params_to_json('clustering_params', self.root_dir, tmp)

        self.save()

    def edit_psth_params(self, shell=False):
        '''Allows user interface for editing psth parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        '''
        tmp = userIO.fill_dict(self.psth_params,
                               'PSTH Parameters\n(Times in ms)',
                               shell=shell)
        if tmp:
            self.psth_params = tmp
            wt.write_params_to_json('psth_params', self.root_dir, tmp)

        self.save()

    def edit_pal_id_params(self, shell=False):
        '''Allows user interface for editing palatability/identity parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        '''
        tmp = userIO.fill_dict(self.pal_id_params,
                               'Palatability/Identity Parameters\n(Times in ms)',
                               shell=shell)
        if tmp:
            self.pal_id_params = tmp
            wt.write_params_to_json('pal_id_params', self.root_dir, tmp)

        self.save()

    def __str__(self):
        '''Put all information about dataset in string format

        Returns
        -------
        str : representation of dataset object
        '''
        out1 = super().__str__()
        out = [out1]
        out.append('\nObject creation date: '
                   + self.dataset_creation_date.strftime('%m/%d/%y'))

        if hasattr(self, 'raw_h5_file'):
            out.append('Deleted Raw h5 file: '+self.raw_h5_file)

        out.append('h5 File: '+self.h5_file)
        out.append('')

        out.append('--------------------')
        out.append('Processing Status')
        out.append('--------------------')
        out.append(pt.print_dict(self.process_status))
        out.append('')

        if not hasattr(self, 'rec_info'):
            return '\n'.join(out)

        info = self.rec_info

        out.append('--------------------')
        out.append('Recording Info')
        out.append('--------------------')
        out.append(pt.print_dict(self.rec_info))
        out.append('')

        out.append('--------------------')
        out.append('Electrodes')
        out.append('--------------------')
        out.append(pt.print_dataframe(self.electrode_mapping))
        out.append('')

        if hasattr(self, 'CAR_electrodes'):
            out.append('--------------------')
            out.append('CAR Groups')
            out.append('--------------------')
            headers = ['Group %i' % x for x in range(len(self.CAR_electrodes))]
            out.append(pt.print_list_table(self.CAR_electrodes, headers))
            out.append('')

        if not self.emg_mapping.empty:
            out.append('--------------------')
            out.append('EMG')
            out.append('--------------------')
            out.append(pt.print_dataframe(self.emg_mapping))
            out.append('')

        if info.get('dig_in'):
            out.append('--------------------')
            out.append('Digital Input')
            out.append('--------------------')
            out.append(pt.print_dataframe(self.dig_in_mapping))
            out.append('')

        if info.get('dig_out'):
            out.append('--------------------')
            out.append('Digital Output')
            out.append('--------------------')
            out.append(pt.print_dataframe(self.dig_out_mapping))
            out.append('')

        out.append('--------------------')
        out.append('Clustering Parameters')
        out.append('--------------------')
        out.append(pt.print_dict(self.clustering_params))
        out.append('')

        out.append('--------------------')
        out.append('Spike Array Parameters')
        out.append('--------------------')
        out.append(pt.print_dict(self.spike_array_params))
        out.append('')

        out.append('--------------------')
        out.append('PSTH Parameters')
        out.append('--------------------')
        out.append(pt.print_dict(self.psth_params))
        out.append('')

        out.append('--------------------')
        out.append('Palatability/Identity Parameters')
        out.append('--------------------')
        out.append(pt.print_dict(self.pal_id_params))
        out.append('')

        return '\n'.join(out)

    @Logger('Writing parameters to JSON')
    def _write_all_params_to_json(self):
        '''Writes all parameters to json files in analysis_params folder in the
        recording directory
        '''
        print('Writing all parameters to json file in analysis_params folder...')
        clustering_params = self.clustering_params
        spike_array_params = self.spike_array_params
        psth_params = self.psth_params
        pal_id_params = self.pal_id_params
        CAR_params = self.CAR_electrodes

        rec_dir = self.root_dir
        wt.write_params_to_json('clustering_params', rec_dir, clustering_params)
        wt.write_params_to_json('spike_array_params', rec_dir, spike_array_params)
        wt.write_params_to_json('psth_params', rec_dir, psth_params)
        wt.write_params_to_json('pal_id_params', rec_dir, pal_id_params)
        wt.write_params_to_json('CAR_params', rec_dir, CAR_params)

    @Logger('Extracting Data')
    def extract_data(self, filename=None, shell=False):
        '''Create hdf5 store for data and read in Intan .dat files. Also create
        subfolders for processing outputs

        Parameters
        ----------
        data_quality: {'clean', 'noisy'} (optional)
            Specifies quality of data for default clustering parameters
            associated. Should generally first process with clean (default)
            parameters and then try noisy after running blech_clust and
            checking if too many electrodes as cutoff too early
        '''
        if self.rec_info['file_type'] is None:
            raise ValueError('Unsupported recording type. Cannot extract yet.')

        if filename is None:
            filename = self.h5_file

        print('\nExtract Intan Data\n--------------------')
        # Create h5 file
        tmp = dio.h5io.create_empty_data_h5(filename, shell)
        if tmp is None:
            return

        # Create arrays for raw data in hdf5 store
        dio.h5io.create_hdf_arrays(filename, self.rec_info,
                                   self.electrode_mapping, self.emg_mapping)

        # Read in data to arrays
        dio.h5io.read_files_into_arrays(filename,
                                        self.rec_info,
                                        self.electrode_mapping,
                                        self.emg_mapping)

        # Write electrode and digital input mapping into h5 file
        # TODO: write EMG and digital output mapping into h5 file
        dio.h5io.write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)
        if self.dig_in_mapping is not None:
            dio.h5io.write_digital_map_to_h5(self.h5_file, self.dig_in_mapping, 'in')

        if self.dig_out_mapping is not None:
            dio.h5io.write_digital_map_to_h5(self.h5_file, self.dig_in_mapping, 'out')

        # update status
        self.h5_file = filename
        self.process_status['extract_data'] = True
        self.save()
        print('\nData Extraction Complete\n--------------------')
        
        

    @Logger('Creating Trial List')
    def create_trial_list(self):
        '''Create lists of trials based on digital inputs and outputs and store
        to hdf5 store
        Can only be run after data extraction
        '''
        if self.rec_info.get('dig_in'):
            in_list = dio.h5io.create_trial_data_table(
                self.h5_file,
                self.dig_in_mapping,
                self.sampling_rate,
                'in')
            self.dig_in_trials = in_list
        else:
            print('No digital input data found')

        if self.rec_info.get('dig_out'):
            out_list = dio.h5io.create_trial_data_table(
                self.h5_file,
                self.dig_out_mapping,
                self.sampling_rate,
                'out')
            self.dig_out_trials = out_list
        else:
            print('No digital output data found')

        self.process_status['create_trial_list'] = True
        self.save()

    @Logger('Marking Dead Channels')
    def mark_dead_channels(self, dead_channels=None, shell=False):
        '''Plots small piece of raw traces and a metric to help identify dead
        channels. Once user marks channels as dead a new column is added to
        electrode mapping

        Parameters
        ----------
        dead_channels : list of int, optional
            if this is specified then nothing is plotted, those channels are
            simply marked as dead
        shell : bool, optional
        '''
        print('Marking dead channels\n----------')
        em = self.electrode_mapping.copy()
        if dead_channels is None:
            userIO.tell_user('Making traces figure for dead channel detection...',
                             shell=True)
            save_file = os.path.join(self.root_dir, 'Electrode_Traces.png')
            fig, ax = datplt.plot_traces_and_outliers(self.h5_file, save_file=save_file)
            if not shell:
                # Better to open figure outside of python since its a lot of
                # data on figure and matplotlib is slow
                subprocess.call(['xdg-open', save_file])
            else:
                userIO.tell_user('Saved figure of traces to %s for reference'
                                 % save_file, shell=shell)

            choice = userIO.select_from_list('Select dead channels:',
                                             em.Electrode.to_list(),
                                             'Dead Channel Selection',
                                             multi_select=True,
                                             shell=shell)
            dead_channels = list(map(int, choice))

        print('Marking eletrodes %s as dead.\n'
              'They will be excluded from common average referencing.'
              % dead_channels)
        em['dead'] = False
        em.loc[dead_channels, 'dead'] = True
        self.electrode_mapping = em
        if os.path.isfile(self.h5_file):
            dio.h5io.write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)

        self.process_status['mark_dead_channels'] = True
        self.save()
        return dead_channels

    @Logger('Common Average Referencing')
    def common_average_reference(self):
        '''Define electrode groups and remove common average from  signals

        Parameters
        ----------
        num_groups : int (optional)
            number of CAR groups, if not provided
            there's a prompt
        '''
        if not hasattr(self, 'CAR_electrodes'):
            raise ValueError('CAR_electrodes not set')

        if not hasattr(self, 'electrode_mapping'):
            raise ValueError('electrode_mapping not set')

        car_electrodes = self.CAR_electrodes
        num_groups = len(car_electrodes)
        em = self.electrode_mapping.copy()

        if 'dead' in em.columns:
            dead_electrodes = em.Electrode[em.dead].to_list()
        else:
            dead_electrodes = []

        # Gather Common Average Reference Groups
        print('CAR Groups\n')
        headers = ['Group %i' % x for x in range(num_groups)]
        print(pt.print_list_table(car_electrodes, headers))

        # Reference each group
        for i, x in enumerate(car_electrodes):
            tmp = list(set(x) - set(dead_electrodes))
            dio.h5io.common_avg_reference(self.h5_file, tmp, i)

        # Compress and repack file
        dio.h5io.compress_and_repack(self.h5_file)

        self.process_status['common_average_reference'] = True
        self.save()

    @Logger('Running Spike Detection')
    def detect_spikes(self, data_quality=None, multi_process=True, n_cores=None):
        '''Run spike detection on each electrode. Prepares for clustering with
        BlechClust. Works for both single recording clustering or
        multi-recording clustering

        Parameters
        ----------
        data_quality : {'clean', 'noisy', None (default)}
            set if you want to change the data quality parameters for cutoff
            and spike detection before running clustering. These parameters are
            automatically set as "clean" during initial parameter setup
        n_cores : int (optional)
            number of cores to use for parallel processing. default is max-1.
        '''
        if data_quality:
            tmp = dio.params.load_params('clustering_params', self.root_dir,
                                         default_keyword=data_quality)
            if tmp:
                self.clustering_params = tmp
                wt.write_params_to_json('clustering_params', self.root_dir, tmp)
            else:
                raise ValueError('%s is not a valid data_quality preset. Must '
                                 'be "clean" or "noisy" or None.')

        print('\nRunning Spike Detection\n-------------------')
        print('Parameters\n%s' % pt.print_dict(self.clustering_params))

        # Create folders for saving things within recording dir
        data_dir = self.root_dir

        em = self.electrode_mapping
        if 'dead' in em.columns:
            electrodes = em.Electrode[em['dead'] == False].tolist()
        else:
            electrodes = em.Electrode.tolist()



        if multi_process:
            spike_detectors = [clust.SpikeDetection(data_dir, x,
                                                    self.clustering_params)
                               for x in electrodes]

            if n_cores is None or n_cores > cpu_count():
                n_cores = cpu_count() - 1

            # results = Parallel(n_jobs=n_cores, verbose=10,
            #                    backend='multiprocessing')(delayed(run_joblib_process)
            #                                               (sd) for sd in spike_detectors)
            results = Parallel(n_jobs=n_cores)(delayed(run_joblib_process)(sd)
                                                for sd in spike_detectors)
        else:
            results = [(None, None, None)] * (max(electrodes)+1)
            spike_detectors = [clust.SpikeDetection(data_dir, x,
                                                    self.clustering_params)
                               for x in electrodes]
            for sd in tqdm(spike_detectors):
                res = sd.run()
                results[res[0]] = res

        print('Electrode    Result    Cutoff (s)')
        cutoffs = {}
        clust_res = {}
        clustered = []
        for x, y, z in results:
            if x is None:
                continue

            clustered.append(x)
            print('  {:<13}{:<10}{}'.format(x, y, z))
            cutoffs[x] = z
            clust_res[x] = y

        print('1 - Sucess\n0 - No data or no spikes\n-1 - Error')

        em = self.electrode_mapping.copy()
        em['cutoff_time'] = em['Electrode'].map(cutoffs)
        em['clustering_result'] = em['Electrode'].map(clust_res)
        self.electrode_mapping = em.copy()
        self.process_status['spike_detection'] = True
        dio.h5io.write_electrode_map_to_h5(self.h5_file, em)
        self.save()
        print('Spike Detection Complete\n------------------')
        return results

    @Logger('Running Blech Clust')
    def blech_clust_run(self, data_quality=None, multi_process=True, n_cores=None, umap=False):
        '''Write clustering parameters to file and
        Run blech_process on each electrode using GNU parallel

        Parameters
        ----------
        data_quality : {'clean', 'noisy', None (default)}
            set if you want to change the data quality parameters for cutoff
            and spike detection before running clustering. These parameters are
            automatically set as "clean" during initial parameter setup
        accept_params : bool, False (default)
            set to True in order to skip popup confirmation of parameters when
            running
        '''
        if self.process_status['spike_detection'] == False:
            raise FileNotFoundError('Must run spike detection before clustering.')

        if data_quality:
            tmp = dio.params.load_params('clustering_params', self.root_dir,
                                         default_keyword=data_quality)
            if tmp:
                self.clustering_params = tmp
                wt.write_params_to_json('clustering_params', self.root_dir, tmp)
            else:
                raise ValueError('%s is not a valid data_quality preset. Must '
                                 'be "clean" or "noisy" or None.')

        print('\nRunning Blech Clust\n-------------------')
        print('Parameters\n%s' % pt.print_dict(self.clustering_params))

        # Get electrodes, throw out 'dead' electrodes
        em = self.electrode_mapping
        if 'dead' in em.columns:
            electrodes = em.Electrode[em['dead'] == False].tolist()
        else:
            electrodes = em.Electrode.tolist()


        if not umap:
            clust_objs = [clust.BlechClust(self.root_dir, x, params=self.clustering_params)
                          for x in electrodes]
        else:
            clust_objs = [clust.BlechClust(self.root_dir, x,
                                           params=self.clustering_params,
                                           data_transform=clust.UMAP_METRICS,
                                           n_pc=5)
                          for x in electrodes]

        if multi_process:
            if n_cores is None or n_cores > cpu_count():
                n_cores = cpu_count() - 1

            results = Parallel(n_jobs=n_cores, verbose=10)(delayed(run_joblib_process)
                                                          (co) for co in clust_objs)
          
                               
        else:
            results = []
            for x in clust_objs:
                res = x.run()
                results.append(res)

        self.process_status['spike_clustering'] = True
        self.process_status['cleanup_clustering'] = False
        dio.h5io.write_electrode_map_to_h5(self.h5_file, em)
        self.save()
        print('Clustering Complete\n------------------')

    @Logger('Cleaning up clustering memory logs. Removing raw data and setting'
            'up hdf5 for unit sorting')
    def cleanup_clustering(self):
        '''Consolidates memory monitor files, removes raw and referenced data
        and setups up hdf5 store for sorted units data
        '''
        if self.process_status['cleanup_clustering']:
            return

        h5_file = dio.h5io.cleanup_clustering(self.root_dir, h5_file=self.h5_file)
        self.h5_file = h5_file
        self.process_status['cleanup_clustering'] = True
        self.save()

    def sort_spikes(self, electrode=None, shell=False):
        if electrode is None:
            electrode = userIO.get_user_input('Electrode #: ', shell=shell)
            if electrode is None or not electrode.isnumeric():
                return

            electrode = int(electrode)

        if not self.process_status['spike_clustering']:
            raise ValueError('Must run spike clustering first.')

        if not self.process_status['cleanup_clustering']:
            self.cleanup_clustering()

        sorter = clust.SpikeSorter(self.root_dir, electrode=electrode, shell=shell)
        if not shell:
            root, sorting_GUI = ssg.launch_sorter_GUI(sorter)
            return root, sorting_GUI
        else:
            # TODO: Make shell UI
            # TODO: Make sort by table
            print('No shell UI yet')
            return

        self.process_status['sort_units'] = True

    @Logger('Calculating Units Similarity')
    def units_similarity(self, similarity_cutoff=50, shell=False):
        if 'SSH_CONNECTION' in os.environ:
            shell= True

        metrics_dir = os.path.join(self.root_dir, 'sorted_unit_metrics')
        if not os.path.isdir(metrics_dir):
            raise ValueError('No sorted unit metrics found. Must sort units before calculating similarity')

        violation_file = os.path.join(metrics_dir,
                                      'units_similarity_violations.txt')
        violations, sim = ss.calc_units_similarity(self.h5_file,
                                                   self.sampling_rate,
                                                   similarity_cutoff,
                                                   violation_file)
        if len(violations) == 0:
            userIO.tell_user('No similarity violations found!', shell=shell)
            self.process_status['units_similarity'] = True
            return violations, sim

        out_str = ['Units Similarity Violations Found:']
        out_str.append('Unit_1    Unit_2    Similarity')
        for x,y in violations:
            u1 = dio.h5io.parse_unit_number(x)
            u2 = dio.h5io.parse_unit_number(y)
            out_str.append('   {:<10}{:<10}{}\n'.format(x, y, sim[u1][u2]))

        out_str.append('Delete units with dataset.delete_unit(N)')
        out_str = '\n'.join(out_str)
        userIO.tell_user(out_str, shell=shell)
        self.process_status['units_similarity'] = True
        self.save()
        return violations, sim

    @Logger('Deleting Unit')
    def delete_unit(self, unit_num, confirm=False, shell=False):
        if isinstance(unit_num, str):
            unit_num = dio.h5io.parse_unit_number(unit_num)

        if unit_num is None:
            print('No unit deleted')
            return

        if not confirm:
            q = userIO.ask_user('Are you sure you want to delete unit%03i?' % unit_num,
                                choices = ['No','Yes'], shell=shell)
        else:
            q = 1

        if q == 0:
            print('No unit deleted')
            return
        else:
            tmp = dio.h5io.delete_unit(self.root_dir, unit_num, h5_file=self.h5_file)
            if tmp is False:
                userIO.tell_user('Unit %i not found in dataset. No unit deleted'
                                 % unit_num, shell=shell)
            else:
                userIO.tell_user('Unit %i sucessfully deleted.' % unit_num,
                                 shell=shell)

        self.save()

    @Logger('Making Unit Arrays')
    def make_unit_arrays(self):
        '''Make spike arrays for each unit and store in hdf5 store
        '''
        params = self.spike_array_params

        print('Generating unit arrays with parameters:\n----------')
        print(pt.print_dict(params, tabs=1))
        ss.make_spike_arrays(self.h5_file, params)
        self.process_status['make_unit_arrays'] = True
        self.save()

    @Logger('Making Unit Plots')
    def make_unit_plots(self):
        '''Make waveform plots for each sorted unit
        '''
        unit_table = self.get_unit_table()
        save_dir = os.path.join(self.root_dir, 'unit_waveforms_plots')
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        os.mkdir(save_dir)
        for i, row in unit_table.iterrows():
            datplt.make_unit_plots(self.root_dir, row['unit_name'], save_dir=save_dir)

        self.process_status['make_unit_plots'] = True
        self.save()

    @Logger('Making PSTH Arrays')
    def make_psth_arrays(self):
        '''Make smoothed firing rate traces for each unit/trial and store in
        hdf5 store
        '''
        params = self.psth_params
        dig_ins = self.dig_in_mapping.query('spike_array == True')
        for idx, row in dig_ins.iterrows():
            spike_analysis.make_psths_for_tastant(self.h5_file,
                                                  params['window_size'],
                                                  params['window_step'],
                                                  row['channel'])

        self.process_status['make_psth_arrays'] = True
        self.save()
        
    def make_raster_plots(self):
        '''make raster plots with electrode noise for each unit  
        '''
        
        unit_table = self.get_unit_table()
        save_dir = os.path.join(self.root_dir, 'unit_raster_plots')
        
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        for i, row in unit_table.iterrows():
            spike_times, _, _ = dio.h5io.get_unit_spike_times(self.root_dir, row['unit_name'], h5_file = self.h5_file) 
            
            waveforms, _, _ = dio.h5io.get_unit_waveforms(self.root_dir, row['unit_name'], h5_file = self.h5_file)
            save_file = os.path.join(save_dir, row['unit_name']+'_raster')
            datplt.plot_spike_raster([spike_times], [waveforms], save_file = save_file)
            
        self.save()

    def make_ensemble_raster_plots(self):
        save_dir = os.path.join(self.root_dir, 'raster_plots')
        name = self.data_name
        save_file = os.path.join(save_dir, name+'_ensemble_raster')
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir) 
        os.mkdir(save_dir)     
        
        datplt.plot_ensemble_raster(self,save_file)
        
        
    @Logger('Calculating Palatability/Identity Metrics')
    def palatability_calculate(self, shell=False):
        pal_analysis.palatability_identity_calculations(self.root_dir,
                                                        params=self.pal_id_params)
        self.process_status['palatability_calculate'] = True
        self.save()

    @Logger('Plotting Palatability/Identity Metrics')
    def palatability_plot(self, shell=False):
        pal_plt.plot_palatability_identity([self.root_dir], shell=shell)
        self.process_status['palatability_plot'] = True
        self.save()

    @Logger('Removing low-spiking units')
    def cleanup_lowSpiking_units(self, min_spikes=100):
        unit_table = self.get_unit_table()
        remove = []
        spike_count = []
        for unit in unit_table['unit_num']:
            waves, descrip, fs = dio.h5io.get_unit_waveforms(self.root_dir, unit, h5_file=self.h5_file)
            if waves.shape[0] < min_spikes:
                spike_count.append(waves.shape[0])
                remove.append(unit)

        for unit, count in zip(reversed(remove), reversed(spike_count)):
            print('Removing unit %i. Only %i spikes.' % (unit, count))
            userIO.tell_user('Removing unit %i. Only %i spikes.'
                             % (unit, count), shell=True)
            self.delete_unit(unit, confirm=True, shell=True)

        userIO.tell_user('Removed %i units for having less than %i spikes.'
                         % (len(remove), min_spikes), shell=True)

    def get_unit_table(self):
        '''Returns a pandas dataframe with sorted unit information

        Returns
        --------
        pandas.DataFrame with columns:
            unit_name, unit_num, electrode, single_unit,
            regular_spiking, fast_spiking
        '''
        unit_table = dio.h5io.get_unit_table(self.root_dir, h5_file=self.h5_file)
        return unit_table
    
    def edit_unit_descriptor(self, unit_num, descriptor_key,descriptor_val):
        '''
        use this to edit unit table, i.e. if you made a mistake labeling a neuron in spike sorting
        unit_num takes integers, corresponds to unit_num in get_unit_table()
        descriptor_key takes string, can be "single_unit", "regular_spiking", or "fast_spiking"
        descriptor_val takes boolean, can be True or False
        '''
        dio.h5io.edit_unit_descriptor(self.root_dir, unit_num, descriptor_key, descriptor_val, self.h5_file)
        print("descriptor edit success")

    def circus_clust_run(self, shell=False):
        circ.prep_for_circus(self.root_dir, self.electrode_mapping,
                             self.data_name, self.sampling_rate)
        circ.start_the_show()

    def pre_process_for_clustering(self, shell=False, dead_channels=None):
        status = self.process_status
        if not status['initialize parameters']:
            self.initParams(shell=shell)

        if not status['extract_data']:
            self.extract_data(shell=True)

        if not status['create_trial_list']:
            self.create_trial_list()

        if not status['mark_dead_channels'] and dead_channels != False:
            self.mark_dead_channels(dead_channels=dead_channels, shell=shell)

        if not status['common_average_reference']:
            self.common_average_reference()

        if not status['spike_detection']:
            self.detect_spikes()

    def extract_and_circus_cluster(self, dead_channels=None, shell=True):
        print('Extracting Data...')
        self.extract_data()
        print('Marking dead channels...')
        self.mark_dead_channels(dead_channels, shell=shell)
        print('Common average referencing...')
        self.common_average_reference()
        print('Initiating circus clustering...')
        circus = circ.circus_clust(self.root_dir, self.data_name,
                                   self.sampling_rate, self.electrode_mapping)
        print('Preparing for circus...')
        circus.prep_for_circus()
        print('Starting circus clustering...')
        circus.start_the_show()
        print('Plotting cluster waveforms...')
        circus.plot_cluster_waveforms()

    def post_sorting(self):
        self.make_unit_plots()
        self.make_unit_arrays()
        self.units_similarity(shell=True)
        self.make_psth_arrays()
        self.make_raster_plots()


def run_joblib_process(process):
    res = process.run()
    return res


def port_in_dataset(rec_dir=None, shell=False):
    '''Import an existing dataset into this framework
    '''
    if rec_dir is None:
        rec_dir = userIO.get_filedirs('Select recording directory', shell=shell)
        if rec_dir is None:
            return None

    dat = dataset(rec_dir, shell=shell)
    # Check files that will be overwritten: log_file, save_file
    if os.path.isfile(dat.save_file):
        prompt = '%s already exists. Continuing will overwrite this. Continue?' % dat.save_file
        q = userIO.ask_user(prompt, shell=shell)
        if q == 0:
            print('Aborted')
            return None

    if os.path.isfile(dat.log_file):
        prompt = '%s already exists. Continuing will append to this. Continue?' % dat.log_file
        q = userIO.ask_user(prompt, shell=shell)
        if q == 0:
            print('Aborted')
            return None

    with open(dat.log_file, 'a') as f:
        print('\n==========\nPorting dataset into blechpy format\n==========\n', file=f)
        print(dat, file=f)

    # Check for info.rhd file or query needed info
    info_rhd = os.path.join(dat.root_dir, 'info.rhd')
    if os.path.isfile(info_rhd):
        dat.initParams(shell=shell)
    else:
        raise FileNotFoundError(f'{info_rhd} is required for proper dataset creation') 

    status = dat.process_status

    user_status = status.copy()
    user_status = userIO.fill_dict(user_status,
                                   'Which processes have already been '
                                   'done to the data?', shell=shell)

    status.update(user_status)
    # if h5 exists data must have been extracted

    if not os.path.isfile(dat.h5_file) or status['extract_data'] == False:
        dat.save()
        return dat

    # write eletrode map and digital input & output maps to hf5
    node_list = dio.h5io.get_node_list(dat.h5_file)

    if 'electrode_map' not in node_list:
        dio.h5io.write_electrode_map_to_h5(dat.h5_file, dat.electrode_mapping)

    if dat.rec_info.get('dig_in') is not None and 'digital_input_map' not in node_list:
        dio.h5io.write_digital_map_to_h5(dat.h5_file, dat.dig_in_mapping, 'in')

    if dat.rec_info.get('dig_out') is not None and 'digital_output_map' not in node_list:
        dio.h5io.write_digital_map_to_h5(dat.h5_file, dat.dig_out_mapping, 'out')


    if ('trial_info' not in node_list) and ('digital_in' in node_list):
        dat.create_trial_list()
    else:
        status['create_trial_list'] == True

    dat.save()

    if status['spike_clustering'] and not status['sort_units']:
        # Move files into correct structure to support spike sorting
        for i, row in dat.electrode_mapping.iterrows():
            el = row['Electrode']
            src = [os.path.join(dat.root_dir, 'clustering_results', f'electrode{el}'),
                   os.path.join(dat.root_dir, 'Plots', f'{el}', 'Plots'),
                   os.path.join(dat.root_dir, 'Plots', f'{el}', 'Plots', 'cutoff_time.png'),
                   os.path.join(dat.root_dir, 'Plots', f'{el}', 'Plots', 'pca_variance.png'),
                   os.path.join(dat.root_dir, 'spike_waveforms', f'electrode{el}'),
                   os.path.join(dat.root_dir, 'spike_times', f'electrode{el}', 'spike_times.npy')]
            clust_dir = os.path.join(dat.root_dir, 'BlechClust', f'electrode_{el}')
            detect_dir = os.path.join(dat.root_dir, 'spike_detection', f'electrode_{el}')
            dest = [os.path.join(clust_dir, 'clustering_results'),
                    os.path.join(clust_dir, 'plots'),
                    os.path.join(detect_dir, 'plots'),
                    os.path.join(detect_dir, 'plots'),
                    os.path.join(detect_dir, 'data'),
                    os.path.join(detect_dir, 'data')]
            for s,d in zip(src, dest):
                if not os.path.exists(s):
                    continue

                if not os.path.isdir(os.path.dirname(d)):
                    os.makedirs(os.path.dirname(d))

                shutil.copytree(s,d)

            # Make params files
            params = dat.clustering_params.copy()

            sd_fn = os.path.join(dat.root_dir, 'analysis_params', 'spike_detection_params.json')
            if not os.path.isfile(sd_fn):
                sd_params = {}
                sd_params['voltage_cutoff'] = params['data_params']['V_cutoff for disconnected headstage']
                sd_params['max_breach_rate'] = params['data_params']['Max rate of cutoff breach per second']
                sd_params['max_secs_above_cutoff'] = params['data_params']['Max allowed seconds with a breach']
                sd_params['max_mean_breach_rate_persec'] = params['data_params']['Max allowed breaches per second']
                band_lower = params['bandpass_params']['Lower freq cutoff']
                band_upper = params['bandpass_params']['Upper freq cutoff']
                sd_params['bandpass'] = [band_lower, band_upper]
                snapshot_pre = params['spike_snapshot']['Time before spike (ms)']
                snapshot_post = params['spike_snapshot']['Time after spike (ms)']
                sd_params['spike_snapshot'] = [snapshot_pre, snapshot_post]
                sd_params['sampling_rate'] = params['sampling_rate']
                wt.write_dict_to_json(sd_params, sd_fn)

            c_fn = os.path.join(clust_dir, 'BlechClust_params.json')
            if not os.path.isfile(c_fn):
                c_params = params.copy()
                c_params['max_clusters'] = params['clustering_params']['Max Number of Clusters']
                c_params['max_iterations'] = params['clustering_params']['Max Number of Iterations']
                c_params['threshold'] = params['clustering_params']['Convergence Criterion']
                c_params['num_restarts'] = params['clustering_params']['GMM random restarts']
                c_params['wf_amplitude_sd_cutoff'] = params['data_params']['Intra-cluster waveform amp SD cutoff']
                wt.write_dict_to_json(c_params, c_fn)

            # To make: clust_dir/clustering_results/ clustering_results.json, rec_key.json, spike_id.npy
            # To make: detect_dir/data/cutoff_time.txt and detection_threshold.txt
            sd = clust.SpikeDetection(dat.root_dir, el, overwrite=False)
            sd.run() # should only filter referenced electrode trace and get cutoff and threshold
            bc = clust.BlechClust(dat.root_dir, el)


    # Add array_time to spike_arrays/dig_in_#
    if 'spike_trains' in node_list:
        digs = set(x.split('.')[1] for x in node_list if 'spike_trains.' in x)
        params = dat.spike_array_params
        array_time = np.arange(-params['pre_stimulus'], params['post_stimulus'], 1)
        for x in digs:
            if f'spike_trains.{x}.array_time' not in node_list:
                dio.h5io.write_array_to_hdf5(dat.h5_file, f'/spike_trains/{x}',
                                             'array_time', array_time)

        for x in dat.processing_steps:
            status[x] = True
            if x == 'make_unit_arrays':
                break

        dat.save()

    return dat



def validate_data_integrity(rec_dir, verbose=False):
    '''incomplete
    '''
    # TODO: Finish this
    print('Raw Data Validation\n' + '-'*19)
    test_names = ['file_type', 'recording_info', 'files', 'dropped_packets', 'data_length']
    number_names = ['sample_rate', 'dropped_packets', 'missing_files', 'recording_length']
    tests = dict.fromkeys(test_names, 'NOT TESTED')
    numbers = dict.fromkeys(number_names, -1)
    file_type = dio.rawIO.get_recording_filetype(rec_dir)
    if file_type is None:
        file_type_check = 'UNSUPPORTED'
    else:
        tests['file_type'] = 'PASS'

    # Check info.rhd integrity
    info_file = os.path.join(rec_dir, 'info.rhd')
    try:
        rec_info = dio.rawIO.read_rec_info(rec_dir, shell=True)
        with open(info_file, 'rb') as f:
            info = dio.load_intan_rhd_format.read_header(f)

        tests['recording_info'] = 'PASS'
    except FileNotFoundError:
        tests['recording_info'] = 'MISSING'
    except Exception as e:
        info_size = os.path.getsize(os.path.join(rec_dir, 'info.rhd'))
        if info_size == 0:
            tests['recording_info'] = 'EMPTY'
        else:
            tests['recording_info'] = 'FAIL'

        print(pt.print_dict(tests, tabs=1))
        return tests, numbers

    counts = {x : info(x) for x in info.keys() if 'num' in x}
    numbers.update(counts)
    fs = info['sample_rate']
    # Check all files needed are present
    files_expected = ['time.dat']
    if file_type == 'one file per signal type':
        files_expected.append('amplifier.dat')
        if rec_info.get('dig_in') is not None:
            files_expected.append('digitalin.dat')

        if rec_info.get('dig_out') is not None:
            files_expected.append('digitalout.dat')

        if info['num_auxilary_input_channels'] > 0:
            files_expected.append('auxiliary.dat')

    elif file_type == 'one file per channel':
        for x in info['amplifier_channels']:
            files_expected.append('amp-' + x['native_channel_name'] + '.dat')

        for x in info['board_dig_in_channels']:
            files_expected.append('board-%s.dat' % x['native_channel_name'])

        for x in info['board_dig_out_channels']:
            files_expected.append('board-%s.dat' % x['native_channel_name'])

        for x in info['aux_input_channels']:
            files_expected.append('aux-%s.dat' % x['native_channel_name'])


    missing_files = []
    file_list = os.listdir(rec_dir)
    for x in files_expected:
        if x not in file_list:
            missing_files.append(x)

    if len(missing_files) == 0:
        tests['files'] = 'PASS'
    else:
        tests['files'] = 'MISSING'
        numbers['missing_files'] = missing_files

    # Check time data for dropped packets
    time = dio.rawIO.read_time_dat(rec_dir, sampling_rate=1)  # get raw timestamps
    numbers['n_samples'] = len(time)
    numbers['recording_length'] = float(time[-1])/fs
    expected_time = np.arange(time[0], time[-1]+1, 1)
    missing_timestamps = np.setdiff1d(expected_time, time)
    missing_times = np.array([float(x)/fs for x in missing_timestamps])
    if len(missing_timestamps) == 0:
        tests['dropped_packets'] = 'PASS'
    else:
        tests['dropped_packets'] = '%i' % len(missing_timestamps)
        numbers['dropped_packets'] = missing_times

    # Check recording length of each trace
    tests['data_traces'] = 'FAIL'
    if file_type == 'one file per signal type':
        try:
            data = dio.rawIO.read_amplifier_dat(rec_dir)
            if data is None:
                tests['data_traces'] = 'UNREADABLE'
            elif data.shape[0] == numbers['n_samples']:
                tests['data_traces'] = 'PASS'
            else:
                tests['data_traces'] = 'CUTOFF'
                numbers['data_trace_length (s)'] = data.shape[0]/fs

        except:
            tests['data_traces'] = 'UNREADABLE'

    elif file_type == 'one file per channel':
        chan_info = pd.DataFrame(columns=['port', 'channel', 'n_samples'])
        lengths = []
        min_samples = numbers['n_samples']
        max_samples = numbers['n_samples']
        for x in info['amplifier_channels']:
            fn = os.path.join(rec_dir, 'amp-%s.dat' % x['native_channel_name'])
            if os.path.basename(fn) in missing_files:
                continue

            data = dio.rawIO.read_one_channel_file(fn)
            lengths.append((x['native_channel_name'], data.shape[0]))
            if data.shape[0] < min_samples:
                min_samples = data.shape[0]

            if data.shape[0] > max_samples:
                max_samples = data.shape[0]

        if min_samples == max_samples:
            tests['data_traces'] = 'PASS'

        else:
            tests['data_traces'] = 'CUTOFF'

        numbers['max_recording_length (s)'] = max_samples/fs
        numbers['min_recording_length (s)'] = min_samples/fs
