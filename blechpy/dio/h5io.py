import tables,os
import blechpy.dio.blech_params as params

def create_empty_data_h5(filename):
    '''
    create empty h5 store for blech data
    '''
    if not filename.endswith('.h5'):
        filename+='.h5'
    basename = os.path.basename(filename).replace('.h5','')
    with tables.open_file(filename,'w',title=basename) as hf5:
        hf5.create_group('/','raw')
        hf5.create_group('/', 'raw_emg')
        hf5.create_group('/', 'digital_in')
        hf5.create_group('/', 'digital_out')
    return filename

def get_h5_filename(file_dir):
    '''
    return the name of the h5 file found in file_dir
    asks for selection if multiple found
    returns None is no h5 found
    '''
    file_list = os.listdir(file_dir)
    h5_files = [f for f in file_list if f.endswith('.h5')]
    if len(h5_files)>1:
        choice = params.select_from_list('Choose which h5 file to load','Multiple h5 stores found',h5_files)
        if choice is None:
            return None
        else:
            h5_files = [choice]
    elif len(h5_files)==0:
        return None
    return h5_files[0]

def get_h5_object(file_dir):
    '''
    finds and opens the h5 file in file_dir, allows selection if multiple found
    returns tables file object with h5 data
    '''
    h5_file = get_h5_name(file_dir)
    hf5 = tables.open_file(os.path.join(file_dir,h5_file),'r+')
    return hf5


def create_hdf_arrays(file_name,electrode_mapping,emg_mapping,file_dir=None):
    '''
    creates empty data arrays in hdf5 store for storage of the intan recording
    data. For 'one file per channel' recording type
    '''
    if os.path.isabs(file_name):
        tmp_dir = os.path.dirname(file_name)
    elif file_dir is None:
        raise ValueError('Must provide absolute path to file or file_dir')
    else:
        file_name = os.path.join(file_dir,file_name)
    if file_dir is None:
        file_dir=tmp_dir

    din = params.get_din_channels(file_dir)
    atom = tables.IntAtom()
    electrodes = []
    with tables.open_file(file_name,'r+') as hf5:
        hf5.create_earray('/raw','amplifier_time',atom,(0,))
        for idx,row in electrode_mapping.iterrows():
            hf5.create_earray('/raw','electrode%i' % row['Electrode'],atom,(0,))

        if not emg_mapping.empty:
            for idx,row in emg_mapping:
                hf5.create_earray('/raw_emg','emg%i' % row['EMG'],atom,(0,))

        for x in din:
            hf5.create_earray('/digital_in','dig_in_%i' % x,atom,(0,))


def read_files_into_arrays(file_name,electrode_mapping,emg_mapping,file_dir=None):
    '''
    Read Intan data files into hdf5 store. Assumes 'one file per channel'
    recordings
    writes digital input and electrode data to h5 file
    can specify emg_port and emg_channels
    '''
    if os.path.isabs(file_name):
        tmp_dir = os.path.dirname(file_name)
    elif file_dir is None:
        raise ValueError('Must provide absolute path to file or file_dir')
    else:
        file_name = os.path.join(file_dir,file_name)
    if file_dir is None:
        file_dir=tmp_dir

    din = params.get_din_channels(file_dir)

    # Open h5 file and write in raw digital input, electrode and emg data
    with tables.open_file(file_name,'r+') as hf5:
        for i in din:
            inputs = np.fromfile('board-DIN-%02d.dat' % i,dtype=np.dtype('uint16'))
            exec('hf5.root.digital_in.dig_in_%i.append(inputs[:])' % i)

        for idx,row in electrode_mapping.iterrows():
            data = np.fromfile('amp-%s-%02d.dat' % (row['Port'],row['Channel']),dtype=np.dtype('int16'))
            exec('hf5.root.raw.electrodes%i.append(data[:])' % row['Electrode'])
        if not emg_mapping.empty:
            for idx,row in emg_mapping.iterrows():
                data = np.fromfile('amp-%s-%02d.dat' % (row['Port'],row['Channel']),dtype=np.dtype('int16'))
                exec('hf5.root.raw_emg.emg%i.append(data[:])' % row['EMG'])

        hf5.flush()
