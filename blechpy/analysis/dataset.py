import easygui as eg
import pandas as pd
from blechpy import dio
from blechpy.widgets import userIO
from blechpy.analysis import spike_sorting as ss, taste_palatability_testing as tpt
from blechpy.analysis import spike_analysis
from blechpy.plotting import data_plot as datplt, palatability_plot as pal_plt
import datetime as dt
from blechpy.data_print import data_print as dp
import pickle
import os
import shutil
import sys
import multiprocessing
import subprocess
import pylab as plt
from copy import deepcopy


def Logger(heading):
    def real_logger(func):
        def wrapper(*args, **kwargs):
            old_out = sys.stdout
            sys.stdout.write(heading+'...')
            sys.stdout.flush()
            if hasattr(args[0], 'log_file'):
                log_file = args[0].log_file
                with open(log_file, 'a') as f:
                    sys.stdout = f
                    func(*args, **kwargs)
                    sys.stdout = old_out
            else:
                func(*args, **kwargs)
                print('Done!')

        return wrapper
    return real_logger


def load_dataset(file_name=None):
    '''Loads dataset processing metadata object from dataset.p file

    Parameters
    ----------
    file_name : str (optional), absolute path to file, if not given file
                chooser is displayed

    Returns
    -------
    blechpy.analysis.dataset : processing metadata object

    Throws
    ------
    FileNotFoundError : if file_name is not a file
    '''
    if file_name is None:
        file_name = eg.fileopenbox('Choose dataset.p file', \
                                    'Choose file',filetypes=['.p'])
    if os.path.isdir(file_name):
        ld = os.listdir(file_name)
        fn = [os.path.join(file_name,x) for x in ld if x.endswith('.p')]
        file_name = fn[0]
    if not os.path.isfile(file_name):
        raise FileNotFoundError('%s is not a valid filename' % file_name)

    with open(file_name,'rb') as f:
        dat = pickle.load(f)
    return dat


class dataset(object):
    '''Stores information related to an intan recording directory and allows
    running of basic analysis script
    Only works for 'one file per channel' recording type

    Parameters
    ----------
    file_dir : str (optional)
        directory of intan recording data, if left blank
        a filechooser will popup
    '''

    def __init__(self, file_dir=None):
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
        # Get file directory is not given
        if file_dir is None:
            file_dir = eg.diropenbox(title='Select dataset directory')
            if file_dir is None:
                raise ValueError('Dataset cannot be initialized without a '
                                 'directory')

        if not os.path.isdir(file_dir):
            raise NotADirectoryError('Could not find folder %s' % file_dir)

        # Get basename of dataset from as name of file_dir
        tmp = os.path.basename(file_dir)
        if tmp == '':
            file_dir = file_dir[:-1]
            tmp = os.path.basename(file_dir)

        self.data_name = tmp
        self.data_dir = file_dir

        # Make paths for analysis log file, dataset object savefile and hdf5
        self.log_file = os.path.join(file_dir, '%s_processing.log' % tmp)
        self.save_file = os.path.join(file_dir, '%s_dataset.p' % tmp)
        h5_name = dio.h5io.get_h5_filename(file_dir)
        if h5_name is None:
            h5_file = os.path.join(file_dir, '%s.h5' % tmp)
        else:
            h5_file = os.path.join(file_dir, h5_name)

        self.h5_file = h5_file

        self.dataset_creation_date = dt.datetime.today()

        # Outline standard processing pipeline and status check
        self.processing_steps = ['extract_data', 'create_trial_list',
                                 'mark_dead_channels',
                                 'common_average_reference', 'blech_clust_run',
                                 'cleanup_clustering',
                                 'sort_units', 'make_unit_plots',
                                 'units_similarity', 'make_unit_arrays',
                                 'make_psth_arrays', 'plot_psths',
                                 'palatability_calculate', 'palatability_plot',
                                 'overlay_psth']
        self.process_status = dict.fromkeys(self.processing_steps, False)

    def __str__(self):
        '''Put all information about dataset in string format

        Returns
        -------
        str : representation of dataset object
        '''
        out = [self.data_name]
        out.append('Data directory:  '+self.data_dir)
        out.append('Object creation date: '
                   + self.dataset_creation_date.strftime('%m/%d/%y'))
        out.append('Dataset Save File: ' + self.save_file)

        if hasattr(self, 'raw_h5_file'):
            out.append('Deleted Raw h5 file: '+self.raw_h5_file)
            out.append('h5 File: '+self.h5_file)
            out.append('')

        out.append('--------------------')
        out.append('Processing Status')
        out.append('--------------------')
        out.append(dp.print_dict(self.process_status))
        out.append('')

        if not hasattr(self, 'rec_info'):
            return '\n'.join(out)

        info = self.rec_info

        out.append('--------------------')
        out.append('Recording Info')
        out.append('--------------------')
        out.append(dp.print_dict(self.rec_info))
        out.append('')

        out.append('--------------------')
        out.append('Electrodes')
        out.append('--------------------')
        out.append(dp.print_dataframe(self.electrode_mapping))
        out.append('')

        if hasattr(self, 'CAR_electrodes'):
            out.append('--------------------')
            out.append('CAR Groups')
            out.append('--------------------')
            headers = ['Group %i' % x for x in range(len(self.CAR_electrodes))]
            out.append(dp.print_list_table(self.CAR_electrodes, headers))
            out.append('')

        if not self.emg_mapping.empty:
            out.append('--------------------')
            out.append('EMG')
            out.append('--------------------')
            out.append(dp.print_dataframe(self.emg_mapping))
            out.append('')

        if info.get('dig_in'):
            out.append('--------------------')
            out.append('Digital Input')
            out.append('--------------------')
            out.append(dp.print_dataframe(self.dig_in_mapping))
            out.append('')

        if info.get('dig_out'):
            out.append('--------------------')
            out.append('Digital Output')
            out.append('--------------------')
            out.append(dp.print_dataframe(self.dig_out_mapping))
            out.append('')

        out.append('--------------------')
        out.append('Clustering Parameters')
        out.append('--------------------')
        out.append(dp.print_dict(self.clust_params))
        out.append('')

        out.append('--------------------')
        out.append('Spike Array Parameters')
        out.append('--------------------')
        out.append(dp.print_dict(self.spike_array_params))
        out.append('')

        out.append('--------------------')
        out.append('PSTH Parameters')
        out.append('--------------------')
        out.append(dp.print_dict(self.psth_params))
        out.append('')

        out.append('--------------------')
        out.append('Palatability/Identity Parameters')
        out.append('--------------------')
        out.append(dp.print_dict(self.pal_id_params))
        out.append('')

        return '\n'.join(out)

    def save(self):
        '''Saves dataset object to dataset.p file in recording directory
        '''
        with open(self.save_file, 'wb') as f:
            pickle.dump(self, f)
            print('Saved dataset processing metadata to %s' % self.save_file)

    @Logger('Initializing Parameters')
    def initParams(self, data_quality='clean', emg_port=None,
                   emg_channels=None, car_keyword=None,
                   shell=False, dig_in_names=None,
                   dig_out_names=None, accept_params=False):
        '''
        Initializes basic default analysis parameters that can be customized
        before running processing methods
        Can provide data_quality as 'clean' or 'noisy' to preset some
        parameters that are useful for the different types. Best practice is to
        run as clean (default) and to re-run as noisy if you notice that a lot
        of electrodes are cutoff early
        '''

        # Get parameters from info.rhd
        file_dir = self.data_dir
        rec_info = dio.rawIO.read_rec_info(file_dir, shell)
        ports = rec_info.pop('ports')
        channels = rec_info.pop('channels')
        sampling_rate = rec_info['amplifier_sampling_rate']
        self.rec_info = rec_info
        self.sampling_rate = sampling_rate

        # Get default parameters from files
        clustering_params = dio.params.load_params('clustering_params', file_dir,
                                                   default_keyword=data_quality)
        spike_array_params = dio.params.load_params('spike_array_params', file_dir)
        psth_params = dio.params.load_params('psth_params', file_dir)
        pal_id_params = dio.params.load_params('pal_id_params', file_dir)
        spike_array_params['sampling_rate'] = sampling_rate
        clustering_params['file_dir'] = file_dir
        clustering_params['sampling_rate'] = sampling_rate

        # Setup digital input mapping
        #TODO: Setup digital output mapping...ignoring for now
        if rec_info.get('dig_in'):
            self._setup_din_mapping(rec_info, dig_in_names, shell)
            dim = self.dig_in_mapping.copy()
            spike_array_params['laser_channels'] = dim.channel[dim['laser']].to_list()
            spike_array_params['dig_ins_to_use'] = dim.channel[dim['spike_array']].to_list()

        # Setup electrode and emg mapping
        self._setup_channel_mappings(ports, channels, emg_port,
                                     emg_channels, shell=shell)

        # Set CAR groups
        self._set_CAR_groups(group_keyword=car_keyword, shell=shell)

        # Confirm parameters
        if not accept_params:
            conf = userIO.confirm_parameter_dict
            clustering_params = conf(clustering_params,
                                     'Clustering Parameters', shell=shell)
            self.spike_array_params = spike_array_params
            self._edit_spike_array_params(shell=shell)
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
        self.save()

    def _edit_spike_array_params(self, shell=False):
        '''Edit spike array parameters and adjust dig_in_mapping accordingly
        '''
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

        dim['laser_channels'] = False
        if tmp['laser_channels'] != ['']:
            tmp['laser_channels'] = [int(x) for x in tmp['laser_channels']]
            dim.loc[[x in tmp['laser_channels'] for x in dim.channel],
                    'laser'] = True

        self.spike_array_params = tmp.copy()
        dio.params.write_params_to_json('spike_array_params',
                                        self.data_dir, tmp)

    def _set_CAR_groups(self, group_keyword=None, shell=False):
        '''Sets that electrode groups for common average referencing and
        defines which brain region electrodes eneded up in

        Parameters
        ----------
        group_keyword : str
            Keyword corresponding to a preset electrode grouping in CAR_params.json
        group_electrodes: list of list of int
            Shadowed by keyword. list of lists of electrodes in each group
        group_areas: list of str
            list of brain regions for each group
        num_groups: int
            number of CAR groups. Needed if no other arguments are passed
        '''
        if not hasattr(self, 'electrode_mapping'):
            raise ValueError('Set electrode mapping before setting CAR groups')

        em = self.electrode_mapping.copy()

        car_param_file = os.path.join(self.data_dir, 'analysis_params',
                                      'CAR_params.json')
        if os.path.isfile(car_param_file):
            group_electrodes = dio.params.load_params('CAR_params',
                                                      self.data_dir)
        else:
            if group_keyword is None:
                group_keyword = userIO.get_user_input(
                    'Input keyword for CAR parameters or number of CAR groups',
                    shell=shell)

            if group_keyword is None:
                ValueError('Must provide a keyword or number of groups')
            elif group_keyword.isnumeric():
                num_groups = int(group_keyword)
                group_electrodes = dio.params.select_CAR_groups(num_groups, em,
                                                                shell=shell)
            else:
                group_electrodes = dio.params.load_params('CAR_params',
                                                          self.data_dir,
                                                          default_keyword=group_keyword)

        num_groups = len(group_electrodes)
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

    def _setup_din_mapping(self, rec_info, dig_in_names=None, shell=False):
        '''sets up dig_in_mapping dataframe  and queries user to fill in columns
        Parameters
        ----------
        rec_info : dict,
        requires 'dig_in' key with list of int corresponding to digital input
        channels
        dig_in_names : list of str (optional)
        shell : bool (optional)
        True for command-line interface
        False (default) for GUI
        '''
        df = pd.DataFrame()
        df['channel'] = rec_info.get('dig_in')
        n_dig_in = len(df)
        # Names
        if dig_in_names:
            df['name'] = dig_in_names
        else:
            df['name'] = ''

        # Parameters to query
        df['palatability_rank'] = 0
        df['laser'] = False
        df['spike_array'] = True
        df['exclude'] = False
        # Re-format for query
        idx = df.index
        df.index = ['dig_in_%i' % x for x in df.channel]
        # Query for user input
        prompt = ('Digital Input Parameters\nSet palatability ranks from 1 to %i'
                  '\nor blank to exclude from pal_id analysis') % len(df)
        tmp = userIO.fill_dict(df.to_dict(), prompt=prompt, shell=shell)
        # Reformat for storage
        df2 = pd.DataFrame.from_dict(tmp)
        df2 = df2.sort_values(by=['channel'])
        df2.index = idx
        self.dig_in_mapping = df2.copy()

    def _setup_channel_mappings(self, ports, channels, emg_port, emg_channels, shell=False):
        '''Creates electrode_mapping and emg_mapping DataFrames with columns:
        - Electrode
        - Port
        - Channel
        - Dead
        Parameters
        ----------
        ports : list of str, item corresponing to each channel
        channels : list of int, channels on each port
        emg_port : str
        emg_channels : list of int
        '''
        if emg_port is None:
            q = userIO.ask_user('Do you have an EMG?', shell=shell)
            if q==0:
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

    def _write_all_params_to_json(self):
        '''Writes all parameters to json files in analysis_params folder in the
        recording directory
        '''
        clustering_params = self.clustering_params
        spike_array_params = self.spike_array_params
        psth_params = self.psth_params
        pal_id_params = self.pal_id_params
        CAR_params = self.CAR_electrodes

        rec_dir = self.data_dir
        dio.params.write_params_to_json('clustering_params', rec_dir, clustering_params)
        dio.params.write_params_to_json('spike_array_params', rec_dir, spike_array_params)
        dio.params.write_params_to_json('psth_params', rec_dir, psth_params)
        dio.params.write_params_to_json('pal_id_params', rec_dir, pal_id_params)
        dio.params.write_params_to_json('CAR_params', rec_dir, CAR_params)

    @Logger('Calculating Unit Similarity')
    def units_similarity(self, similarity_cutoff=50):
        '''Go through sorted units and compute similarity
        Creates a matrix in HDF5 store labelled unit_distances that is an ixj
        matrix with values corresponding to the % of unit i spikes that are
        within 1ms of any unit j spikes

        Parameters
        ----------
        similarity_cutoff : float (optional)
            percentage (0-100) above which to classify units as violating
            similarity. Default is 50
        '''
        print('Assessing unit similarity with similarity cutoff at %i%%'
              % similarity_cutoff)
        violation_file = os.path.join(self.data_dir,
                                      'unit_similarity_violations.txt')
        ss.calc_units_similarity(self.h5_file, self.sampling_rate,
                                 similarity_cutoff, violation_file)
        self.process_status['units_similarity'] = True
        self.save()

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
        dio.h5io.create_empty_data_h5(filename, shell)

        # Create arrays for raw data in hdf5 store
        dio.h5io.create_hdf_arrays(filename, self.rec_info,
                                   self.electrode_mapping, self.emg_mapping)

        # Read in data to arrays
        dio.h5io.read_files_into_arrays(filename,
                                        self.rec_info,
                                        self.electrode_mapping,
                                        self.emg_mapping)

        # update status
        self.h5_file = filename
        self.process_status['extract_data'] = True
        self.save()
        print('\nData Extraction Complete\n--------------------')

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
        if dead_channels is None:
            userIO.tell_user('Making traces figure for dead channel detection...',
                             shell=True)
            em = self.electrode_mapping.copy()
            fig, ax = datplt.plot_traces_and_outliers(self.h5_file)

            save_file = os.path.join(self.data_dir, 'Electrode_Traces.png')
            fig.savefig(save_file)
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
            plt.close('all')
            dead_channels = list(map(int, choice))

        em['dead'] = False
        em.loc[dead_channels, 'dead'] = True
        self.electrode_mapping = em
        return dead_channels

    @Logger('Running blech_clust')
    def blech_clust_run(self, data_quality=None, accept_params=False,
                        shell=False):
        '''
        Write clustering parameters to file and
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
        if data_quality:
            tmp = dio.params.load_params('clustering_params', self.data_dir,
                                         default_keyword=data_quality)
            if tmp:
                self.clustering_params = tmp
            else:
                raise ValueError('%s is not a valid data_quality preset. Must '
                                 'be "clean" or "noisy" or None.')

        print('\nRunning Blech Clust\n-------------------')
        print('Parameters\n%s' % dp.print_dict(self.clustering_params))

        # Write parameters into .params file
        self.param_file = os.path.join(self.data_dir, self.data_name+'.params')
        dio.params.write_clustering_params(self.param_file, self.clustering_params)

        # Create folders for saving things within recording dir
        data_dir = self.data_dir
        directories = ['spike_waveforms', 'spike_times',
                       'clustering_results',
                       'Plots', 'memory_monitor_clustering']
        for d in directories:
            tmp_dir = os.path.join(data_dir, d)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            os.mkdir(tmp_dir)

        # Set file for clusting log
        self.clustering_log = os.path.join(data_dir, 'results.log')
        if os.path.exists(self.clustering_log):
            os.remove(self.clustering_log)

        process_path = os.path.realpath(__file__)
        process_path = os.path.join(os.path.dirname(process_path),
                                    'blech_process.py')
        em = self.electrode_mapping
        if 'dead' in em.columns:
            electrodes = em.Electrode[em['dead'] == False].tolist()
        else:
            electrodes = em.Electrode.tolist()

        my_env = os.environ
        my_env['OMP_NUM_THREADS'] = '1'  # possibly not necesary
        cpu_count = int(multiprocessing.cpu_count())-1
        process_call = ['parallel', '-k', '-j', str(cpu_count), '--noswap',
                        '--load', '100%', '--progress', '--memfree', '4G',
                        '--retry-failed', '--joblog', self.clustering_log,
                        'python', process_path, '{1}', self.data_dir, ':::']
        process_call.extend([str(x) for x in electrodes])
        subprocess.call(process_call, env=my_env)
        self.process_status['blech_clust_run'] = True
        self.save()
        print('Clustering Complete\n------------------')

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
        print(dp.print_list_table(car_electrodes, headers))

        # Reference each group
        for i, x in enumerate(car_electrodes):
            tmp = list(set(x) - set(dead_electrodes))
            dio.h5io.common_avg_reference(self.h5_file, tmp, i)

        # Compress and repack file
        dio.h5io.compress_and_repack(self.h5_file)

        self.process_status['common_average_reference'] = True
        self.save()

    @Logger('Creating Trial Lists')
    def create_trial_list(self):
        '''Create lists of trials based on digital inputs and outputs and store
        to hdf5 store
        Can only be run after data extraction
        '''
        if not self.process_status['extract_data']:
            eg.exceptionbox('Must extract data before creating trial list',
                            'Data Not Extracted')
            return

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

    @Logger('Cleaning up memory logs, removing raw data and setting up hdf5 '
            'for unit sorting')
    def cleanup_clustering(self):
        '''Consolidates memory monitor files, removes raw and referenced data
        and setups up hdf5 store for sorted units data
        '''
        h5_file = dio.h5io.cleanup_clustering(self.data_dir)
        self.h5_file = h5_file
        self.process_status['cleanup_clustering'] = True
        self.save()

    @Logger('Making Unit Arrays')
    def make_unit_arrays(self, shell=False):
        params = self.spike_array_params

        print('Generating unit arrays with parameters:\n----------')
        print(dp.print_dict(params, tabs=1))
        ss.make_spike_arrays(self.h5_file, params)
        self.process_status['make_unit_arrays'] = True
        self.save()

    @Logger('Making Unit Plots')
    def make_unit_plots(self):
        ss.make_unit_plots(self.data_dir, self.sampling_rate)
        self.process_status['make_unit_plots'] = True
        self.save()

    @Logger('Making PSTH Arrays')
    def make_psth_arrays(self, shell=False):
        params = self.psth_params
        dig_ins = self.dig_in_mapping
        for idx, row in dig_ins.iterrows():
            spike_analysis.make_psths_for_tastant(self.h5_file,
                                                  params['window_size'],
                                                  params['window_step'],
                                                  row['channel'])

        self.process_status['make_psth_arrays'] = True
        self.save()

    def edit_clustering_parameters(self, shell=False):
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
            dio.params.write_params_to_json('clustering_params', self.data_dir, tmp)

    def edit_psth_parameters(self, shell=False):
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
            dio.params.write_params_to_json('psth_params', self.data_dir, tmp)

    def edit_pal_id_parameters(self, shell=False):
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
            dio.params.write_params_to_json('pal_id_params', self.data_dir, tmp)

    def sort_units(self, shell=False):
        '''Begins processes to allow labelling of clusters as sorted units

        Parameters
        ----------
        shell : bool
            True if command-line interfaced desired, False for GUI (default)
        '''
        fs = self.sampling_rate
        ss.sort_units(self.data_dir, fs, shell)
        self.process_status['sort_units'] = True
        self.save()

    def get_unit_table(self):
        '''Returns a pandas dataframe with sorted unit information

        Returns
        --------
        pandas.DataFrame with columns:
            unit_name, unit_num, electrode, single_unit,
            regular_spiking, fast_spiking
        '''
        unit_table = dio.h5io.get_unit_table(self.data_dir)
        return unit_table

    @Logger('Calculating Palatability/Identity Metrics')
    def palatability_calculate(self, shell=False):
        tpt.palatability_identity_calculations(self.data_dir,
                                               params=self.pal_id_params)
        self.process_status['palatability_calculate'] = True
        self.save()

    @Logger('Plotting Palatability/Identity Metrics')
    def palatability_plot(self, shell=False):
        pal_plt.plot_palatability_identity([self.data_dir], shell=shell)
        self.process_status['palatability_plot'] = True
        self.save()

    def extract_and_cluster(self, data_quality='clean',
                            num_CAR_groups='bilateral32', shell=False,
                            dig_in_names=None, dig_out_names=None,
                            emg_port=None, emg_channels=None):
        '''Runs data from raw to clustered with no fuss

        Parameters (all optional)
        ----------
        data_quality : {'clean', 'noisy'}
            sets defaults for clustering parameters, best to start with 'clean'
            (default) and re-cluster with dat.blech_clust_run('noisy') if too
            many electrodes have early cutoffs
        num_CAR_groups : int,
            number of common average reference groups
            can also pass 'bilateral32' keyword to automatically set 2 CAR
            groups with 16 channels each. This is currently the default so be
            sure to change if this is not true
        shell : bool
            True if you want to use command-line for everything, False for GUI
            (default)
        dig_in_names : list of str
            Names of digital inputs, must match number of digital inputs or it
            gets mad default is None which means you can enter them when to
            program ask Can set this to False if no digital inputs in order to
            skip questions
        dig_out_names : list of str
            Same as dig_in_names but for digital outputs
        emg_port : str
            port that emg is on, if no EMG you can set this to False and skip
            questions default is None, will prompt you about EMG later
        emg_channels : list of int
            any emg channel numbers on emg_port
        '''
        self.initParams(data_quality=data_quality, shell=shell)
        self.extract_data(shell=shell)
        self.mark_dead_channels(shell=shell)
        self.create_trial_list()
        self.common_average_reference(num_CAR_groups)
        self.blech_clust_run(data_quality, accept_params=shell, shell=shell)
        self.save()

    @Logger('Post Sorting Processing')
    def post_sorting(self):
        print('Post sorting processing')
        print('Making unit plots...')
        self.make_unit_plots()
        print('Computing units similarity...')
        self.units_similarity()
        print('Making spike arrays...')
        self.make_unit_arrays()
        print('Making psth arrays...')
        self.make_psth_arrays()
        self.save()


