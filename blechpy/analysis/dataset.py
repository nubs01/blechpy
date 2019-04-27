import easygui as eg
from  blechpy import dio
import datetime as dt
from blechpy.data_print import data_print as dp



class dataset(object):
    '''
    Stores information related to an intan recording directory and allows
    running of basic analysis script
    Only works for 'one file per channel' recording type
    '''
    def __init__(self,file_dir=None):
        '''
        Initialize dataset object from file_dir, grabs basename from name of
        directory and initializes basic analysis parameters
        '''
        if file_dir is None:
            file_dir = eg.diropenbox(title='Select dataset directory')
            if file_dir is None:
                raise ValueError('Dataset cannot be initialized without a directory')
        tmp = os.path.basename(file_dir)
        if tmp=='':
            file_dir = file_dir[:-1]
            tmp = os.path.basename(file_dir)
        self.data_name = tmp
        self.data_dir = file_dir
        self.log_file = os.path.join(file_dir,'%s_analysis.log' % tmp)
        self.save_file = os.path.join(file_dir,'%s_dataset.p' % tmp)
        h5_name = dio.params.get_h5_filename(file_dir)
        if h5_name is None:
            h5_file = os.path.join(file_dir,'%s.h5' % tmp)
        else:
            h5_file = os.path.join(file_dir,h5_name)
        self.h5_file = h5_file
        self.dataset_creation_date = dt.datetime.today()
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

        # Get default parameters for blech_clust
        file_dir = self.data_dir
        self.file_type = 'one file per channel'
        ports,channels = dio.params.parse_amp_files(file_dir)
        DIN = dio.params.get_din_channels(file_dir)
        sampling_rate = dio.params.get_sampling_rate(file_dir)
        clustering_params = dio.params.clustering_params
        data_params = dio.params.data_params[data_quality]
        bandpass_params = dio.params.bandpass_params
        spike_snapshot = dio.params.spike_snapshot

        # Ask for emg port & channels
        if emg_port is None:
            q = eg.ynbox('Do you have an EMG?','EMG')
            if q:
                emg_port = dio.params.select_from_list('Select EMG Port:','EMG Port',ports)
                tmp_idx = ports.index(emg_port)
                emg_channels = dio.params.select_from_list('Select EMG Channels:',
                                                            'EMG Channels',channels[tmp_idx],
                                                            multi_select=True)
        if emg_port is not  None:
            tmp_idx = ports.index(emg_port)
            trash = [channels[tmp_idx].pop(x) for x in emg_channels]

        electrode_mapping, emg_mapping = dio.params.flatten_channels(ports,channels,
                                                                    emg_port=emg_port,
                                                                    emg_channels=emg_channels)
        self.electrode_mapping = electrode_mapping
        self.emg_mapping = emg_mapping

        self.clust_params = {'file_dir':file_dir,'data_quality':data_quality,
                             'DIN':DIN,'sampling_rate':sampling_rate,
                             'emg_port':emg_port,'emg_channels':emg_channels,
                             'clustering_params':clustering_params,
                             'data_params':data_params,'bandpass_params':bandpass_params,
                             'spike_snapshot':spike_snapshot}

        # Outline standard processing pipeline and status check
        self.processing_steps = ['blech_clust_setup','common_avg_reference','blech_clust',,
                                'blech_post_process']
        self.process_status = dict.fromkeys(self.processing_steps,False)

    def __str__(self):
        '''
        print all information about dataset
        '''
        out = [self.data_name]
        out.append('Data directory:  '+self.data_dir)
        out.append('Object creation date: ' + self.dataset_creation_date.strftime('%m/%d/%y'))
        if hasattr(self,'raw_h5_file'):
            out.append('Deleted Raw h5 file: '+self.raw_h5_file)
        out.append('h5 File: '+self.h5_file)
        out.append('Recording File Type: '+self.file_type)
        out.append('')

        out.append('--------------------')
        out.append('Processing Status')
        out.append('--------------------')
        out.append(dp.print_dict(self.process_status))
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
        out.append('Parameters')
        out.append('--------------------')
        out.append(dp.print_dict(self.params))
        out.append('')
        return '\n'.join(out)

    def blech_clust_setup(self,data_quality='clean'):
        '''
        Creates empty hdf5 store for data and makes directories for processed data
        '''
        if self.file_type != 'one file per channel':
            raise ValueError('As of now, only "one file per channel" recording type is supported')

        # Check if they are OK with the parameters that will be used

        # Create h5 file
        fn = dio.h5io.create_empty_data_h5(self.h5_file)

        # Create folders for saving things
        data_dir = self.data_dir
        directories = ['spike_waveforms','spike_times','clustering_results',
                        'Plots','memory_monitor_clustering']
        for d in directories:
            os.mkdir(os.path.join(data_dir,d))

        # Create arrays for raw data in hdf5 store
        dio.h5io.create_hdf_arrays(self.h5_file,self.electrode_mapping,self.emg_mapping)

        # Read in data to arrays
        dio.h5io.read_files_into_arrays(self.h5_file,self.electrode_mapping,self.emg_mapping)

        # Write parameters into .params file
        self.param_file = os.path.join(data_dir,self.data_name+'.params')
        dio.params.write_params(self.param_file,self.clust_params)

        self.clustering_log = os.path.join(data_dir,'results.log')

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
