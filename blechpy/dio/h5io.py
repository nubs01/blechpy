import tables,os
import blechpy.dio.blech_params as params
import blechpy.dio.rawIO as rawIO

def create_empty_data_h5(filename):
    '''Create empty h5 store for blech data with approriate data groups

    Parameters
    ----------
    filename : str, absolute path to h5 file for recording
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
    '''Return the name of the h5 file found in file_dir.
    Asks for selection if multiple found

    Parameters
    ----------
    file_dir : str, path to recording directory

    Returns
    -------
    str : filename of h5 file in directory (not full path), None if no file found
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

def get_h5_object(file_name):
    '''Finds and opens the h5 file in file_dir, allows selection if multiple found
    returns tables file object with h5 data

    Parameters
    ----------
    file_name : str, absolute path to h5 file OR path to recording directory
                (will detect h5 file and ask user if multiple)

    Returns
    -------
    tables.file.File : hdf5 object 

    Throws
    ------
    FileNotFoundError : if no h5 file in given directory
    NotADirectoryError : if provided file_name is neither a file nor directory str
    '''
    if not os.path.isfile(file_name):
        if os.path.isdir(file_name):
            h5_file = get_h5_filename(file_name)
            if h5_file is None:
                raise FileNotFoundError('No h5 file in %s' % file_name)
            file_name = os.path.join(h5_file,file_name)
        else:
            raise NotADirectoryError('%s is neither a valid h5 file path, nor directory path' % file_name)

    hf5 = tables.open_file(file_name,'r+')
    return hf5


def create_hdf_arrays(file_name,rec_info,electrode_mapping,emg_mapping,file_dir=None):
    '''Creates empty data arrays in hdf5 store for storage of the intan
    recording data.

    Parameters
    ----------
    file_name : str, absolute path to h5 file 
    rec_info : dict, recording info dict provided by
                     blechpy.rawIO.read_recording_info
    electrode_mapping : pandas.DataFrame, with colummns
                        Electrode, Port and Channels
    emg_mapping : pandas.Dataframe, with columns
                  EMG, Port and Channels (can be empty)
    file_dir : str (optional), path to recording directory if h5 is in
                               different folder

    Throws
    ------
    ValueError : if file_name is not absolute path to file and file_dir is not
                 provided

    '''
    if file_dir is None:
        file_dir = os.path.dirname(file_name)
    if file_dir is '':
        raise ValueError(('Must provide absolute path to file in a recording'
                          'directory or a file_dir argument'))
    if not os.path.isabs(file_name):
        file_name = os.path.join(file_dir,file_name)

    atom = tables.IntAtom()
    with tables.open_file(file_name,'r+') as hf5:

        # Create array for raw time vector
        hf5.create_earray('/raw','amplifier_time',atom,(0,))

        # Create arrays for each electrode
        for idx,row in electrode_mapping.iterrows():
            hf5.create_earray('/raw','electrode%i' % row['Electrode'],atom,(0,))

        # Create arrays for raw emg (if any exist)
        if not emg_mapping.empty:
            for idx,row in emg_mapping:
                hf5.create_earray('/raw_emg','emg%i' % row['EMG'],atom,(0,))

        # Create arrays for digital inputs (if any exist)
        if rec_info.get('dig_in'):
            for x in rec_info['dig_in']:
                hf5.create_earray('/digital_in','dig_in_%i' % x,atom,(0,))

        # Create arrays for digital outputs (if any exist)
        if rec_info.get('dig_out'):
            for x in rec_info['dig_out']:
                hf5.create_earray('/digital_out','dig_out_%i' % x,atom,(0,))


def read_files_into_arrays(file_name,rec_info,electrode_mapping,emg_mapping,file_dir=None):
    '''
    Read Intan data files into hdf5 store. Assumes 'one file per channel'
    recordings
    writes digital input and electrode data to h5 file
    can specify emg_port and emg_channels
    '''
    if file_dir is None:
        file_dir = os.path.dirname(file_name)
    if file_dir is '':
        raise ValueError(('Must provide absolute path to file in a recording'
                          'directory or a file_dir argument'))
    if not os.path.isabs(file_name):
        file_name = os.path.join(file_dir,file_name)

    file_type = rec_info['file_type']

    # Open h5 file and write in raw digital input, electrode and emg data
    with tables.open_file(file_name,'r+') as hf5:
        # Read in digital input data if it exists
        if rec_info.get('dig_in'):
            read_in_digital_signal(hf5,file_dir,file_type,rec_info['dig_in'],'in')

        if rec_info.get('dig_out'):
            read_in_digital_signal(hf5,file_dir,file_type,rec_info['dig_out'],'out')

        read_in_amplifier_signal(hf5,file_dir,file_type,rec_info['num_channels'], \
                                 electrode_mapping,emg_mapping)

def read_in_amplifier_signal(hf5,file_dir,file_type,num_channels,el_map,em_map):
    '''Read intan amplifier files into hf5 array. 
    For electrode and emg signals.
    Supported recording types:
        - one file per signal type
        - one file per channel

    Parameters
    ----------
    hf5 : tables.file.File, hdf5 object to write data into 
    file_dir : str, path to recording directory
    file_type : str, type of recording files to read in. Currently supported:
                        'one file per signal type' and 'one file per channel'
    num_channels: int, number of amplifier channels from info.rhd or
                       blechby.rawIO.read_recording_info
    el_map,em_map : pandas.DataFrame, dataframe mapping electrode or emg number
                    to port and channel numer. Must have columns Port and
                    Channel and either Electrode (el_map) or EMG (em_map)
    '''
    exec_str = 'hf5.root.%s.%s%i.append(data[:])'
    
    if file_type == 'one file per signal type':
        all_data = rawIO.read_amplifier_dat(file_dir,num_channels)

    # Read in electrode data
    for idx,row in el_map.iterrows():
        port = row['Port']
        channel = row['Channel']
        electrode = row['Electrode']
        if file_type == 'one file per signal type':
            data = all_data[channel]
        elif file_type == 'one file per channel':
            file_name = os.path.join(file_dir,'amp-%s-%02d.dat' % \
                                        (port,channel))
            data = rawIO.read_one_channel_file(file_name)
        tmp_str = exec_str % ('raw','electrode',electrode)
        print(tmp_str)
        exec(tmp_str)

    hf5.flush()

    # Read in emg data if it exists
    if not em_map.empty:
        for idx,row in em_map.iterrows():
            port = row['Port']
            channel = row['Channel']
            emg = row['EMG']
            if file_type == 'one file per signal type':
                data = all_data[channel]
            elif file_type == 'one file per channel':
                file_name = os.path.join(file_dir,'amp-%s-%02d.dat' % \
                                            (port,channel))
                data = rawIO.read_one_channel_file(file_name)
            tmp_str = exec_str % ('raw_emg','emg',emg)
            print(tmp_str)
            exec(tmp_str)
        hf5.flush() 


def read_in_digital_signal(hf5,file_dir,file_type,channels,dig_type='in'):
    '''Reads 'one file per signal type' or 'one file per signal' digital input
    or digital output into hf5 array

    Parameters
    ----------
    hf5 : tables.file.File, hdf5 object to write data into 
    file_dir : str, path to recording directory
    file_type : str, type of recording files to read in. Currently supported:
                        'one file per signal type' and 'one file per channel'
    channels : list, list of integer channel number of used digital
                     inputs/outputs
    dig_type : {'in','out'}
                Type of data being read (so it puts it in the right array in
                hdf5 store
    '''
    exec_str = 'hf5.root.digital_%s.dig_%s_%i.append(data[:])'
    if file_type == 'one file per signal type':
        all_data = rawIO.read_digital_dat(file_dir,channels,dig_type)
    for i,ch in enumerate(channels):
        if file_type == 'one file per signal type':
            data = all_data[i]
        elif file_type == 'one file per channel':
            file_name = os.path.join(file_dir,'board-D%s-%02d.dat' % \
                                        (dig_type.upper(),ch))
            data = rawIO.read_one_channel_file(file_name)
        tmp_str = exec_str % (dig_type,dig_type,ch)
        print(tmp_str)
        exec(tmp_str)
    hf5.flush()

