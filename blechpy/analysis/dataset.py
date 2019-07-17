import easygui as eg
import pandas as pd
from blechpy import dio
from blechpy.widgets import userIO
from blechpy.analysis import spike_sorting as ss
from blechpy.analysis import spike_analysis
import datetime as dt
from blechpy.data_print import data_print as dp
import pickle
import os
import shutil
import sys
import multiprocessing
import subprocess
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

        if hasattr(self, 'car_electrodes'):
            out.append('--------------------')
            out.append('CAR Groups')
            out.append('--------------------')
            headers = ['Group %i' % x for x in range(len(self.car_electrodes))]
            out.append(dp.print_list_table(self.car_electrodes, headers))
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

        return '\n'.join(out)

    def save(self):
        '''Saves dataset object to dataset.p file in recording directory
        '''
        with open(self.save_file, 'wb') as f:
            pickle.dump(self, f)
            print('Saved dataset processing metadata to %s' % self.save_file)

    @Logger('Initializing Parameters')
    def initParams(self, data_quality='clean', emg_port=None,
                   emg_channels=None, shell=False, dig_in_names=None,
                   dig_out_names=None,
                   spike_array_params=None,
                   psth_params=None,
                   confirm_all=False):
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

        # Get default parameters for blech_clust
        clustering_params = deepcopy(dio.params.clustering_params)
        data_params = deepcopy(dio.params.data_params[data_quality])
        bandpass_params = deepcopy(dio.params.bandpass_params)
        spike_snapshot = deepcopy(dio.params.spike_snapshot)
        if spike_array_params is None:
            spike_array_params = deepcopy(dio.params.spike_array_params)
        if psth_params is None:
            psth_params = deepcopy(dio.params.psth_params)

        # Ask for emg port & channels
        if emg_port is None and not shell:
            q = eg.ynbox('Do you have an EMG?', 'EMG')
            if q:
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

        elif emg_port is None and shell:
            print('\nNo EMG port given.\n')

        electrode_mapping, emg_mapping = dio.params.flatten_channels(
            ports,
            channels,
            emg_port=emg_port,
            emg_channels=emg_channels)
        self.electrode_mapping = electrode_mapping
        self.emg_mapping = emg_mapping

        # Get digital input names and spike array parameters
        if rec_info.get('dig_in'):
            if dig_in_names is None:
                dig_in_names = dict.fromkeys(['dig_in_%i' % x
                                              for x in rec_info['dig_in']])
                name_filler = userIO.dictIO(dig_in_names, shell=shell)
                dig_in_names = name_filler.fill_dict('Enter names for '
                                                     'digital inputs:')
                if dig_in_names is None or \
                   any([x is None for x in dig_in_names.values()]):
                    raise ValueError('Must name all dig_ins')

                dig_in_names = list(dig_in_names.values())

            if spike_array_params['laser_channels'] is None:
                laser_dict = dict.fromkeys(['dig_in_%i' % x
                                            for x in rec_info['dig_in']],
                                           False)
                laser_filler = userIO.dictIO(laser_dict, shell=shell)
                laser_dict = laser_filler.fill_dict('Select any lasers:')
                if laser_dict is None:
                    laser_channels = []
                else:
                    laser_channels = [i for i, v
                                      in zip(rec_info['dig_in'],
                                             laser_dict.values()) if v]

                spike_array_params['laser_channels'] = laser_channels

            if spike_array_params['dig_ins_to_use'] is None:
                di = [x for x in rec_info['dig_in']
                      if x not in laser_channels]
                dn = [dig_in_names[x] for x in di]
                spike_dig_dict = dict.fromkeys(dn, True)
                filler = userIO.dictIO(spike_dig_dict, shell=shell)
                spike_dig_dict = filler.fill_dict('Select digital inputs '
                                                  'to use for making spike'
                                                  ' arrays:')
                if spike_dig_dict is None:
                    spike_dig_ins = []
                else:
                    spike_dig_ins = [x for x, y in
                                     zip(di, spike_dig_dict.values())
                                     if y]

                spike_array_params['dig_ins_to_use'] = spike_dig_ins

            self.dig_in_mapping = pd.DataFrame([(x, y) for x, y in
                                                zip(rec_info['dig_in'],
                                                    dig_in_names)],
                                               columns=['dig_in', 'name'])

        # Get digital output names
        if rec_info.get('dig_out'):
            if dig_out_names is None:
                dig_out_names = dict.fromkeys(['dig_out_%i' % x
                                              for x in rec_info['dig_out']])
                name_filler = userIO.dictIO(dig_out_names, shell=shell)
                dig_out_names = name_filler.fill_dict('Enter names for '
                                                      'digital outputs:')
                if dig_out_names is None or \
                   any([x is None for x in dig_out_names.values()]):
                    raise ValueError('Must name all dig_outs')

                dig_out_names = list(dig_out_names.values())

            self.dig_out_mapping = pd.DataFrame([(x, y) for x, y in
                                                 zip(rec_info['dig_out'],
                                                     dig_out_names)],
                                                columns=['dig_out', 'name'])

        # Store clustering parameters
        self.clust_params = {'file_dir': file_dir,
                             'data_quality': data_quality,
                             'sampling_rate': sampling_rate,
                             'clustering_params': clustering_params,
                             'data_params': data_params,
                             'bandpass_params': bandpass_params,
                             'spike_snapshot': spike_snapshot}

        # Store and confirm spike array parameters
        spike_array_params['sampling_rate'] = sampling_rate
        self.spike_array_params = spike_array_params
        self.psth_params = psth_params
        if not confirm_all:
            prompt = ('\n----------\nSpike Array Parameters\n----------\n'
                      + dp.print_dict(spike_array_params) +
                      '\nAre these parameters good?')
            q_idx = userIO.ask_user(prompt, ('Yes', 'Edit'), shell=shell)
            if q_idx == 1:
                self.edit_spike_array_parameters(shell=shell)

            # Edit and store psth parameters
            prompt = ('\n----------\nPSTH Parameters\n----------\n'
                      + dp.print_dict(psth_params) +
                      '\nAre these parameters good?')
            q_idx = userIO.ask_user(prompt, ('Yes', 'Edit'), shell=shell)
            if q_idx == 1:
                self.edit_psth_parameters(shell=shell)

        self.save()

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
    def extract_data(self, shell=False):
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

        print('\nExtract Intan Data\n--------------------')
        # Create h5 file
        dio.h5io.create_empty_data_h5(self.h5_file, shell)

        # Create arrays for raw data in hdf5 store
        dio.h5io.create_hdf_arrays(self.h5_file, self.rec_info,
                                   self.electrode_mapping, self.emg_mapping)

        # Read in data to arrays
        dio.h5io.read_files_into_arrays(self.h5_file,
                                        self.rec_info,
                                        self.electrode_mapping,
                                        self.emg_mapping)

        # update status
        self.process_status['extract_data'] = True
        self.save()
        print('\nData Extraction Complete\n--------------------')

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
            tmp = deepcopy(dio.params.data_params.get(data_quality))
            if tmp:
                self.clust_params['data_params'] = tmp
            else:
                raise ValueError('%s is not a valid data_quality preset. Must '
                                 'be "clean" or "noisy" or None.')

        # Check if they are OK with the parameters that will be used
        if not accept_params:
            if not shell:
                q = eg.ynbox(dp.print_dict(self.clust_params)
                             + '\n Are these parameters OK?',
                             'Check Extraction and Clustering Parameters')
            else:
                q = input(dp.print_dict(self.clust_params)
                          + '\n Are these paramters OK? (y/n):  ')
                if q == 'y':
                    q = True
                else:
                    False

            if not q:
                return

        print('\nRunning Blech Clust\n-------------------')
        print('Parameters\n%s' % dp.print_dict(self.clust_params))

        # Write parameters into .params file
        self.param_file = os.path.join(self.data_dir, self.data_name+'.params')
        dio.params.write_params(self.param_file, self.clust_params)

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
        electrodes = self.electrode_mapping['Electrode'].tolist()

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
    def common_average_reference(self, num_groups=None):
        '''Define electrode groups and remove common average from  signals

        Parameters
        ----------
        num_groups : int (optional)
            number of CAR groups, if not provided
            there's a prompt
        '''
        # Gather Common Average Reference Groups
        if num_groups is None:
            num_groups = eg.enterbox('Enter number of common average '
                                     'reference groups (integer only):',
                                     'CAR Groups')
            if num_groups.isnumeric():
                num_groups = int(num_groups)

        num_groups, car_electrodes = dio.params.get_CAR_groups(
            num_groups,
            self.electrode_mapping)

        self.car_electrodes = car_electrodes
        print('CAR Groups\n')
        headers = ['Group %i' % x for x in range(num_groups)]
        print(dp.print_list_table(car_electrodes, headers))

        # Reference each group
        for i, x in enumerate(car_electrodes):
            dio.h5io.common_avg_reference(self.h5_file, x, i)

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
        query = ('\n----------\nParameters for Spike Array Creation'
                 '\n----------\ntimes in ms\n%s\nWould you like to'
                 ' continue with these parameters?') % dp.print_dict(params)
        q_idx = userIO.ask_user(query, choices=('Continue', 'Abort', 'Edit'),
                                shell=shell)
        if q_idx == 1:
            return
        elif q_idx == 2:
            params = self.edit_spike_array_parameters(shell=shell)
            if params is None:
                return

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
                                                  row['dig_in'])

        self.process_status['make_psth_arrays'] = True
        self.save()

    def edit_spike_array_parameters(self, shell=False):
        params = self.spike_array_params
        param_filler = userIO.dictIO(params, shell=shell)
        new_params = param_filler.fill_dict(prompt=('Input desired parameters'
                                                    ' (Times are in ms'))
        if new_params is None:
            return None
        else:
            new_params['dig_ins_to_use'] = [int(x) for x in
                                            new_params['dig_ins_to_use']
                                            if x != '']
            new_params['laser_channels'] = [int(x) for x in
                                            new_params['laser_channels']
                                            if x != '']
            self.spike_array_params = new_params
            return new_params

    def edit_clustering_parameters(self, shell=False):
        '''Allows user interface for editing clustering parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        '''
        param_filler = userIO.dictIO(self.clust_params, shell=shell)
        tmp = param_filler.fill_dict()
        if tmp:
            self.clust_params = tmp

    def edit_psth_parameters(self, shell=False):
        '''Allows user interface for editing psth parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        '''
        param_filler = userIO.dictIO(self.psth_params, shell=shell)
        tmp = param_filler.fill_dict('Edit params for making PSTHs\n'
                                     'All times are in ms')
        if tmp:
            self.psth_params = tmp

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

    def extract_and_cluster(self, data_quality='clean',
                            num_CAR_groups='bilateral32', shell=False,
                            dig_in_names=None, dig_out_names=None,
                            emg_port=None, emg_channels=None,
                            spike_array_params=None, psth_params=None):
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
        if shell:
            # Initialize Initial Parameters
            if dig_in_names is None:
                num_ins = int(input('Number of digital inputs : '))
                if dig_out_names is None:
                    num_outs = int(input('Number of digital outputs : '))

            if emg_port is False:
                emg = False
                emg_port = None
            elif emg_port is None:
                emg = bool(int(input('EMG (0 or 1)? : ')))
            elif isinstance(emg_port, str) and isinstance(emg_channels, list):
                emg = False
            else:
                raise ValueError('emg_port and emg_channels must be both set '
                                 'or both left empty')

            if emg:
                emg_port = input('EMG Port? : ')
                emg_channels = input('EMG Channels (comma-separated) : ')
                emg_channels = [int(x) for x in emg_channels.split(', ')]

            if dig_in_names is None:
                dig_in_names = []
                if num_ins > 0:
                    print('Digital Input Names\n----------\n')
                    for i in range(num_ins):
                        tmp = input('dig_in_%i : ' % i)
                        dig_in_names.append(tmp)

                if dig_in_names == []:
                    dig_in_names = None

            if dig_out_names is None:
                dig_out_names = []
                if num_outs > 0:
                    print('Digital Output Names\n----------\n')
                    for i in range(num_outs):
                        tmp = input('dig_out_%i : ' % i)
                        dig_out_names.append(tmp)

                if dig_out_names == []:
                    dig_out_names = None

            if dig_out_names is False:
                dig_out_names = None

            if dig_in_names is False:
                dig_in_names = None

            self.initParams(data_quality, emg_port, emg_channels,
                            shell, dig_in_names, dig_out_names,
                            spike_array_params,psth_params,True)
        else:
            # Initialize default parameters
            self.initParams(shell=shell)

        self.extract_data(shell=shell)
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
