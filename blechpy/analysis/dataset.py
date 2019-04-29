import easygui as eg
from  blechpy import dio
import datetime as dt
from blechpy.data_print import data_print as dp
import pickle, os, shutil



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
        self.log_file = os.path.join(file_dir,'%s_analysis.log' % tmp)
        self.save_file = os.path.join(file_dir,'%s_dataset.p' % tmp)
        h5_name = dio.h5io.get_h5_filename(file_dir)
        if h5_name is None:
            h5_file = os.path.join(file_dir,'%s.h5' % tmp)
        else:
            h5_file = os.path.join(file_dir,h5_name)
        self.h5_file = h5_file

        self.dataset_creation_date = dt.datetime.today()
        
        # Initialize default parameters
        self.initParams()

    def initParams(self,data_quality='clean',emg_port=None,emg_channels=None):
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
        rec_info = dio.rawIO.read_rec_info(file_dir)
        ports = rec_info.pop('ports')
        channels = rec_info.pop('channels')
        sampling_rate = rec_info['amplifier_sampling_rate']
        self.rec_info = rec_info

        # Get default parameters for blech_clust
        clustering_params = dio.params.clustering_params
        data_params = dio.params.data_params[data_quality]
        bandpass_params = dio.params.bandpass_params
        spike_snapshot = dio.params.spike_snapshot

        # Ask for emg port & channels
        if emg_port is None:
            q = eg.ynbox('Do you have an EMG?','EMG')
            if q:
                emg_port = dio.params.select_from_list('Select EMG Port:','EMG Port',ports)
                emg_channels = dio.params.select_from_list('Select EMG Channels:',
                                                            'EMG Channels',
                                                            [y for x,y in zip(ports,channels) if x==emg_port],
                                                            multi_select=True)

        electrode_mapping,emg_mapping = dio.params.flatten_channels(ports,channels,
                                                                    emg_port=emg_port,
                                                                    emg_channels=emg_channels)
        self.electrode_mapping = electrode_mapping
        self.emg_mapping = emg_mapping

        self.clust_params = {'file_dir':file_dir,'data_quality':data_quality,
                             'sampling_rate':sampling_rate,
                             'clustering_params':clustering_params,
                             'data_params':data_params,'bandpass_params':bandpass_params,
                             'spike_snapshot':spike_snapshot}

        # Outline standard processing pipeline and status check
        self.processing_steps = ['extract_data','common_avg_reference','blech_clust',
                                'blech_post_process','mark_units','gather_unit_plots',
                                'units_similarity','make_unit_arrays','make_psth',
                                'palatability_calculate','palatability_plot',
                                'overlay_psth']
        self.process_status = dict.fromkeys(self.processing_steps,False)

    def __str__(self):
        '''Put all information about dataset in string format

        Returns
        -------
        str : representation of dataset object
        '''
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
        
        out.append('--------------------')
        out.append('EMG')
        out.append('--------------------')
        out.append(dp.print_dataframe(self.emg_mapping))
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

    def extract_data(self,data_quality='clean'):
        '''
        Creates empty hdf5 store for data and makes directories for processed data
        '''
        if self.rec_info['file_type'] is None:
            raise ValueError('Unsupported recording type. Cannot extract yet.')

        # Check if they are OK with the parameters that will be used

        # Create h5 file
        fn = dio.h5io.create_empty_data_h5(self.h5_file)

        # Create folders for saving things within recording dir
        data_dir = self.data_dir
        directories = ['spike_waveforms','spike_times','clustering_results',
                        'Plots','memory_monitor_clustering']
        for d in directories:
            tmp_dir = os.path.join(data_dir,d)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

        # Create arrays for raw data in hdf5 store
        dio.h5io.create_hdf_arrays(self.h5_file,self.rec_info, \
                                    self.electrode_mapping,self.emg_mapping)

        # Read in data to arrays
        dio.h5io.read_files_into_arrays(self.h5_file,self.rec_info, \
                                        self.electrode_mapping,self.emg_mapping)

        # Write parameters into .params file
        self.param_file = os.path.join(data_dir,self.data_name+'.params')
        dio.params.write_params(self.param_file,self.clust_params)

        self.clustering_log = os.path.join(data_dir,'results.log')

        self.process_status['extract_data'] = True

    def blech_clust_run(self):
        '''
        Run blech_process on each electrode using GNU parallel
        '''
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

    def common_average_reference(self):
        pass
