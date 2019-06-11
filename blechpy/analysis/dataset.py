import easygui as eg, pandas as pd
from  blechpy import dio
from blechpy.widgets import userIO
from blechpy.analysis import post_process as post
import datetime as dt
from blechpy.data_print import data_print as dp
import pickle, os, shutil, sys
import multiprocessing
import subprocess
from copy import deepcopy

def Logger(heading):
    def real_logger(func):
        def wrapper(*args,**kwargs):
            old_out = sys.stdout
            sys.stdout.write(heading+'...')
            sys.stdout.flush()
            if hasattr(args[0],'log_file'):
                log_file = args[0].log_file
                with open(log_file,'a') as f:
                    sys.stdout = f
                    func(*args,**kwargs)
                    sys.stdout = old_out
            else:
                func(*args,**kwargs)
            print('Done!')
        return wrapper
    return real_logger

class dataset(object):
    '''Stores information related to an intan recording directory and allows
    running of basic analysis script
    Only works for 'one file per channel' recording type

    Parameters
    ----------
    file_dir : str (optional), directory of intan recording data, if left blank
                               a filechooser will popup
    '''

    def __init__(self,file_dir=None):
        '''Initialize dataset object from file_dir, grabs basename from name of
        directory and initializes basic analysis parameters

        Parameters
        ----------
        file_dir : str (optional), file directory for intan recording data

        Throws
        ------
        ValueError : if file_dir is not provided and no directory is chosen
                     when prompted
        NotADirectoryError : if file_dir does not exist
        '''
        # Get file directory is not given
        if file_dir is None:
            file_dir = eg.diropenbox(title='Select dataset directory')
            if file_dir is None:
                raise ValueError('Dataset cannot be initialized without a directory')

        if not os.path.isdir(file_dir):
            raise NotADirectoryError('Could not find folder %s' % file_dir)

        # Get basename of dataset from as name of file_dir
        tmp = os.path.basename(file_dir)
        if tmp=='':
            file_dir = file_dir[:-1]
            tmp = os.path.basename(file_dir)
        self.data_name = tmp
        self.data_dir = file_dir

        # Make paths for analysis log file, dataset object savefile and hdf5
        self.log_file = os.path.join(file_dir,'%s_processing.log' % tmp)
        self.save_file = os.path.join(file_dir,'%s_dataset.p' % tmp)
        h5_name = dio.h5io.get_h5_filename(file_dir)
        if h5_name is None:
            h5_file = os.path.join(file_dir,'%s.h5' % tmp)
        else:
            h5_file = os.path.join(file_dir,h5_name)
        self.h5_file = h5_file

        self.dataset_creation_date = dt.datetime.today()

        # Outline standard processing pipeline and status check
        self.processing_steps = ['extract_data','create_trial_list',
                'common_average_reference','blech_clust_run','cleanup_clustering',
                'mark_units','gather_unit_plots',
                'units_similarity','make_unit_arrays','make_psth',
                'palatability_calculate','palatability_plot',
                'overlay_psth']
        self.process_status = dict.fromkeys(self.processing_steps,False)
        

    @Logger('Initializing Parameters')
    def initParams(self,data_quality='clean',emg_port=None,emg_channels=None,shell=False,dig_in_names=None,dig_out_names=None):
        '''
        Initializes basic default analysis parameters that can be customized
        before running processing methods
        Can provide data_quality as 'clean' or 'noisy' to preset some parameters
        that are useful for the different types. Best practice is to run as
        clean (default) and to re-run as noisy if you notice that a lot of
        electrodes are cutoff early
        '''

        # Get parameters from info.rhd
        file_dir = self.data_dir
        rec_info = dio.rawIO.read_rec_info(file_dir,shell)
        ports = rec_info.pop('ports')
        channels = rec_info.pop('channels')
        sampling_rate = rec_info['amplifier_sampling_rate']
        self.rec_info = rec_info

        # Get default parameters for blech_clust
        clustering_params = deepcopy(dio.params.clustering_params)
        data_params = deepcopy(dio.params.data_params[data_quality])
        bandpass_params = deepcopy(dio.params.bandpass_params)
        spike_snapshot = deepcopy(dio.params.spike_snapshot)

        # Ask for emg port & channels
        if emg_port is None and not shell:
            q = eg.ynbox('Do you have an EMG?','EMG')
            if q:
                emg_port = dio.params.select_from_list('Select EMG Port:',
                        'EMG Port',ports)
                emg_channels = dio.params.select_from_list('Select EMG Channels:',
                        'EMG Channels',
                        [y for x,y in zip(ports,channels) if x==emg_port],
                        multi_select=True)
        elif emg_port is None and shell:
            print('\nNo EMG port given.\n')

        electrode_mapping,emg_mapping = dio.params.flatten_channels(ports,channels,
                emg_port=emg_port,emg_channels=emg_channels)
        self.electrode_mapping = electrode_mapping
        self.emg_mapping = emg_mapping

        # Get digital input names
        if rec_info.get('dig_in'):
            if shell and dig_in_names is None:
                raise ValueError('dig_in_names must be provided if shell = True')
            elif not shell and dig_in_names is None:
                dig_in_names = eg.multenterbox('Give names for digital inputs:',
                        'Digital Input Names',
                        ['digital in %i' % x for x in rec_info['dig_in']])
            self.dig_in_mapping = pd.DataFrame( \
                    [(x,y) for x,y in  zip(rec_info['dig_in'],dig_in_names)],
                    columns=['dig_in','name'])

        # Get digital output names
        if rec_info.get('dig_out'):
            if shell and dig_out_names is None:
                raise ValueError('dig_out_names must be provided if shell = True')
            elif not shell and dig_out_names is None:
                dig_out_names = eg.multenterbox('Give names for digital outputs:',
                        'Digital Output Names',
                        ['digital out %i' % x for x in rec_info['dig_out']])
            self.dig_out_mapping = pd.DataFrame( \
                    [(x,y) for x,y in  zip(rec_info['dig_out'],dig_out_names)],
                    columns=['dig_out','name'])


        # Store clustering parameters
        self.clust_params = {'file_dir':file_dir,'data_quality':data_quality,
                             'sampling_rate':sampling_rate,
                             'clustering_params':clustering_params,
                             'data_params':data_params,'bandpass_params':bandpass_params,
                             'spike_snapshot':spike_snapshot}


    def edit_clustering_parameters(self,shell=False):
        '''Allows user interface for editing clustering parameters

        Parameters
        ----------
        shell : bool (optional), True if you want command-line interface, False for GUI (default)
        '''
        param_filler = userIO.dictIO(self.clust_params,shell)
        param_filler.fill_dict()
        self.clust_params = param_filler.get_dict()


    def __str__(self):
        '''Put all information about dataset in string format

        Returns
        -------
        str : representation of dataset object
        '''
        info = self.rec_info
        out = [self.data_name]
        out.append('Data directory:  '+self.data_dir)
        out.append('Object creation date: ' + self.dataset_creation_date.strftime('%m/%d/%y'))
        out.append('Dataset Save File: ' + self.save_file)
        if hasattr(self,'raw_h5_file'):
            out.append('Deleted Raw h5 file: '+self.raw_h5_file)
        out.append('h5 File: '+self.h5_file)
        out.append('')

        out.append('--------------------')
        out.append('Processing Status')
        out.append('--------------------')
        out.append(dp.print_dict(self.process_status))
        out.append('')

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
        
        if hasattr(self,'car_electrodes'):
            out.append('--------------------')
            out.append('CAR Groups')
            out.append('--------------------')
            headers = ['Group %i' % x for x in range(len(self.car_electrodes))]
            out.append(dp.print_list_table(self.car_electrodes,headers))
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

        return '\n'.join(out)

    def save(self):
        '''Saves dataset object to dataset.p file in recording directory
        '''
        with open(self.save_file,'wb') as f:
            pickle.dump(self,f)
            print('Saved dataset processing metadata to %s' % self.save_file)

    @Logger('Extracting Data')
    def extract_data(self,shell=False):
        '''Create hdf5 store for data and read in Intan .dat files. Also create
        subfolders for processing outputs

        Parameters
        ----------
        data_quality: {'clean','noisy'} (optional)
            Specifies quality of data for default clustering parameters
            associated. Should generally first process with clean (default)
            parameters and then try noisy after running blech_clust and
            checking if too many electrodes as cutoff too early

        '''
        if self.rec_info['file_type'] is None:
            raise ValueError('Unsupported recording type. Cannot extract yet.')


        print('\nExtract Intan Data\n--------------------')
        # Create h5 file
        fn = dio.h5io.create_empty_data_h5(self.h5_file,shell)


        # Create arrays for raw data in hdf5 store
        dio.h5io.create_hdf_arrays(self.h5_file,self.rec_info, \
                                    self.electrode_mapping,self.emg_mapping)

        # Read in data to arrays
        dio.h5io.read_files_into_arrays(self.h5_file,self.rec_info, \
                                        self.electrode_mapping,self.emg_mapping)

        # update status
        self.process_status['extract_data'] = True

        print('\nData Extraction Complete\n--------------------')

    @Logger('Running blech_clust')
    def blech_clust_run(self,data_quality=None,accept_params=False):
        '''
        Write clustering parameters to file and 
        Run blech_process on each electrode using GNU parallel

        Parameters
        ----------
        data_quality : {'clean','noisy',None (default)}, 
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
                raise ValueError('%s is not a valid data_quality preset. Must be "clean" or "noisy" or None.')

        # Check if they are OK with the parameters that will be used
        if not accept_params:
            q = eg.ynbox(dp.print_dict(self.clust_params)+'\n Are these parameters OK?',
                            'Check Extraction and Clustering Parameters')
            if not q:
                return

        print('\nRunning Blech Clust\n-------------------')
        print('Parameters\n%s' % dp.print_dict(self.clust_params))

        # Write parameters into .params file
        self.param_file = os.path.join(self.data_dir,self.data_name+'.params')
        dio.params.write_params(self.param_file,self.clust_params)

        # Create folders for saving things within recording dir
        data_dir = self.data_dir
        directories = ['spike_waveforms','spike_times','clustering_results',
                        'Plots','memory_monitor_clustering']
        for d in directories:
            tmp_dir = os.path.join(data_dir,d)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

        # Set file for clusting log
        self.clustering_log = os.path.join(data_dir,'results.log')
        if os.path.exists(self.clustering_log):
            os.remove(self.clustering_log)



        process_path = os.path.realpath(__file__)
        process_path = os.path.join(os.path.dirname(process_path),'blech_process.py')
        electrodes = self.electrode_mapping['Electrode'].tolist()

        my_env = os.environ
        my_env['OMP_NUM_THREADS'] = '1' # possibly not necesary, check blech_process, if needed use env_parallel
        cpu_count = int(multiprocessing.cpu_count())-1
        process_call = ['parallel','-k','-j',str(cpu_count),'--noswap','--load','100%',
                        '--progress','--memfree','4G','--retry-failed','--joblog',
                        self.clustering_log,'python',process_path,'{1}',self.data_dir,
                        ':::']
        process_call.extend([str(x) for x in electrodes])
        subprocess.call(process_call,env=my_env)
        self.process_status['blech_clust_run'] = True
        print('Clustering Complete\n------------------')

    @Logger('Common Average Referencing')
    def common_average_reference(self,num_groups=None):
        '''Define electrode groups and remove common average from  signals

        Parameters
        ----------
        num_groups : int (optional), number of CAR groups, if not provided
                                     there's a prompt
        '''
        # Gather Common Average Reference Groups
        if num_groups is None:
            num_groups = eg.enterbox('Enter number of common average reference groups (integers only):','CAR Groups')
            if num_groups.isnumeric():
                num_groups = int(num_groups)
        num_groups,car_electrodes = dio.params.get_CAR_groups(num_groups,self.electrode_mapping)
        self.car_electrodes = car_electrodes
        print('CAR Groups\n')
        headers = ['Group %i' %x for x in range(num_groups)]
        print(dp.print_list_table(car_electrodes,headers))

        # Reference each group
        for i,x in enumerate(car_electrodes):
            dio.h5io.common_avg_reference(self.h5_file,x,i)

        # Compress and repack file
        dio.h5io.compress_and_repack(self.h5_file)

        self.process_status['common_average_reference'] = True

    @Logger('Creating Trial Lists')
    def create_trial_list(self):
        '''Create lists of trials based on digital inputs and outputs and store to hdf5 store
        Can only be run after data extraction
        '''
        if not self.process_status['extract_data']:
            eg.exceptionbox('Must extract data before creating trial list','Data Not Extracted')
            return
        if self.rec_info.get('dig_in'):
            in_list = dio.h5io.create_trial_table(self.h5_file,self.dig_in_mapping,'in')
            self.dig_in_trials = in_list
        else:
            print('No digital input data found')
        if self.rec_info.get('dig_out'):
            out_list = dio.h5io.create_trial_table(self.h5_file,self.dig_out_mapping,'out')
            self.dig_out_trials = out_list
        else:
            print('No digital output data found')
        self.process_status['create_trial_list'] = True

    @Logger('Cleaning up memory logs, removing raw data and setting up hdf5 for unit sorting')
    def cleanup_clustering(self):
        '''Consolidates memory monitor files, removes raw and referenced data
        and setups up hdf5 store for sorted units data
        '''
        h5_file = dio.h5io.cleanup_clustering(self.data_dir)
        self.h5_file = h5_file
        self.process_status['cleanup_clustering'] = True

    def sort_units(self,shell=False):
        '''Begins processes to allow labelling of clusters as sorted units

        Parameters
        ----------
        shell : bool
            True if command-line interfaced desired, False for GUI (default)
        '''
        fs = self.rec_info['amplifier_sampling_rate']
        post.sort_units(self.data_dir,fs,shell)


    def extract_and_cluster(self,data_quality='clean',num_CAR_groups='bilateral32',shell=False,dig_in_names=None,dig_out_names=None,emg_port=None,emg_channels=None):
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
            elif isinstance(emg_port,str) and isinstance(emg_channels,list):
                emg = False
            else:
                raise ValueError('emg_port and emg_channels must be both set or both left empty')
            
            if emg:
                emg_port = input('EMG Port? : ')
                emg_channels = input('EMG Channels (comma-separated) : ')
                emg_channels = [int(x) for x in emg_channels.split(',')]

            if dig_in_names is None:
                dig_in_names = []
                if num_ins>0:
                    print('Digital Input Names\n----------\n')
                for i in range(num_ins):
                    tmp = input('dig_in_%i : ' % i)
                    dig_in_names.append(tmp)
                if dig_in_names == []:
                    dig_in_names = None

            if dig_out_names is None:
                dig_out_names = []
                if num_outs>0:
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

            self.initParams(data_quality,emg_port,emg_channels,
                    shell,dig_in_names,dig_out_names)
        else:
            # Initialize default parameters
            self.initParams(shell=shell)

        self.extract_data(shell=shell)
        self.create_trial_list()
        self.common_average_reference(num_CAR_groups)
        self.blech_clust_run(data_quality,accept_params=shell)
        self.save()
