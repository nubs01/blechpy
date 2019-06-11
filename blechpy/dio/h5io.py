import tables,os, time, sys, subprocess
import easygui as eg, pandas as pd, numpy as np
from blechpy.dio import particles, rawIO, blech_params as params
from blechpy.data_print import data_print as dp

def Timer(heading):
    def real_timer(func):
        def wrapper(*args,**kwargs):
            start = time.time()
            print('')
            print('----------\n%s\n----------' % heading)
            result = func(*args,**kwargs)
            print('Done! Elapsed Time: %1.2f' % (time.time()-start))
            return result
        return wrapper
    return real_timer

def println(txt):
    '''Print inline without newline
    required due to how ipython doesn't work right with print(...,end='')
    '''
    sys.stdout.write(txt)
    sys.stdout.flush()

def create_empty_data_h5(filename,shell=False):
    '''Create empty h5 store for blech data with approriate data groups

    Parameters
    ----------
    filename : str, absolute path to h5 file for recording
    '''
    if not filename.endswith('.h5'):
        filename+='.h5'
    basename = os.path.basename(filename).replace('.h5','')

    # Check if file exists, and ask to delete if it does
    if os.path.isfile(filename):
        if shell:
            println('%s already exists. Deleting...' % filename)
            os.remove(filename)
            print('Done!')
        else:
            q = eg.ynbox('%s already exists. Would you like to delete?' % filename, 
                        'Delete existing file')
            if not q:
                return filename
            else:
                println('Deleting existing h5 file...')
                os.remove(filename)
                print('Done!')
    print('Creating empty HDF5 store with raw data groups')
    println('Writing %s.h5 ...' % basename)
    data_groups = ['raw','raw_emg','digital_in','digital_out','trial_info']
    with tables.open_file(filename,'w',title=basename) as hf5:
        for grp in data_groups:
            hf5.create_group('/',grp)
    print('Done!\n')
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
        choice = params.select_from_list('Choose which h5 file to load', \
                                         'Multiple h5 stores found',h5_files)
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

    println('Creating empty arrays in hdf5 store for raw data...')
    sys.stdout.flush()
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

    print('Done!')


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
    print(('Extracting Intan data to HDF5 Store:\n' 
        ' h5 file: %s' % file_name))
    print('')

    # Open h5 file and write in raw digital input, electrode and emg data
    with tables.open_file(file_name,'r+') as hf5:
        # Read in time data
        print('Reading time data...')
        time = rawIO.read_time_dat(file_dir,rec_info['amplifier_sampling_rate'])
        println('Writing time data...')
        hf5.root.raw.amplifier_time.append(time[:])
        print('Done!')

        # Read in digital input data if it exists
        if rec_info.get('dig_in'):
            read_in_digital_signal(hf5,file_dir,file_type,rec_info['dig_in'],'in')

        if rec_info.get('dig_out'):
            read_in_digital_signal(hf5,file_dir,file_type,rec_info['dig_out'],'out')

        read_in_amplifier_signal(hf5,file_dir,file_type,rec_info['num_channels'], \
                electrode_mapping,emg_mapping)

@Timer('Extracting Amplifier Signal Data')
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
        println('Reading all amplifier_dat...')
        all_data = rawIO.read_amplifier_dat(file_dir,num_channels)
        print('Done!')

    # Read in electrode data
    for idx,row in el_map.iterrows():
        port = row['Port']
        channel = row['Channel']
        electrode = row['Electrode']
        if file_type == 'one file per signal type':
            data = all_data[channel]
        elif file_type == 'one file per channel':
            file_name = os.path.join(file_dir,'amp-%s-%03d.dat' % \
                    (port,channel))
            println('Reading data from %s...' % os.path.basename(file_name))
            data = rawIO.read_one_channel_file(file_name)
            print('Done!')
        tmp_str = exec_str % ('raw','electrode',electrode)
        println('Writing data from port %s channel %i to electrode%i...' % \
                (port,channel,electrode))
        exec(tmp_str)
        print('Done!')
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
                file_name = os.path.join(file_dir,'amp-%s-%03d.dat' % \
                        (port,channel))
                println('Reading data from %s...' % os.path.basename(file_name))
                data = rawIO.read_one_channel_file(file_name)
                print('Done!')
            tmp_str = exec_str % ('raw_emg','emg',emg)
            println('Writing data from port %s channel %i to emg%i...' % \
                    (port,channel,emg))
            exec(tmp_str)
            print('Done!')
        hf5.flush() 

@Timer('Extracting Digital Signal Data')
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
        println('Reading all digital%s data...' % dig_type)
        all_data = rawIO.read_digital_dat(file_dir,channels,dig_type)
        print('Done!')
    for i,ch in enumerate(channels):
        if file_type == 'one file per signal type':
            data = all_data[i]
        elif file_type == 'one file per channel':
            file_name = os.path.join(file_dir,'board-D%s-%02d.dat' % \
                                        (dig_type.upper(),ch))
            println('Reading digital%s data from %s...' % \
                    (dig_type,os.path.basename(file_name)))
            data = rawIO.read_one_channel_file(file_name)
            print('Done!')
        tmp_str = exec_str % (dig_type,dig_type,ch)
        println('Writing data from ditigal %s channel %i to dig_%s_%i...' % \
                (dig_type,ch,dig_type,ch))
        exec(tmp_str)
        print('Done!')
    hf5.flush()


@Timer('Generating Trial List')
def create_trial_table(h5_file,digital_map,dig_type='in'):
    '''Gathers digital data from hf5 for channels in digital_map and
    creates and stores a list of trials  for easy viewing of trial order

    Parameters
    ----------
    h5_file : str, path to .h5 file that data is stored in and to write to
    digital_map : pandas.DataFrame, maps digital channel numbers to string names,
                  has columns 'dig_in' (or 'dig_out') and 'name'
    dig_type : str, {'in','out'}

    Returns
    -------
    pandas.DataFrame : listing trials in order with digital channel number and name
                        columns: 'Trial Num', 'channel','name'

    Throws
    ------
    ValueError : if dig_type is not 'in' or 'out'
    '''
    if dig_type not in ['in','out']:
        raise ValueError('Invalid digital type given.')

    with tables.open_file(h5_file,'r+') as hf5:
        # Grab relevant digital data from hf5
        tree = hf5.root['digital_'+dig_type]
        dig_str = 'dig_'+dig_type
        trial_map = []
        
        print('Generating trial list for digital %sputs: %s' % (dig_type,
                ', '.join([str(x) for x in digital_map[dig_str].tolist()])))

        # Loop through channels and get indices of digital signal onsets
        for i,row in digital_map.iterrows():
            println('Grabbing data for digital %sput %i...' % (dig_type,row[dig_str]))
            tmp = np.diff(tree[dig_str+'_'+str(row[dig_str])][:])>0
            tmp_idx = np.where(tmp)[0]
            trial_map.extend([(x,row[dig_str],row['name']) for x in tmp_idx])
            print('Done!')

        # Make dataframe and assign trial numbers
        println('Constructing DataFrame...')
        trial_df = pd.DataFrame(trial_map,columns=['idx',dig_str,'name'])
        trial_df = trial_df.sort_values(by=['idx']).reset_index(drop=True)
        trial_df = trial_df.reset_index(drop=False).rename(columns={'index':'Trial Num'})
        trial_df = trial_df.drop(columns=['idx'])
        print('Done!')

        # Make hf5 group and table
        println('Writing data to h5 file...')
        if not '/trial_info' in hf5:
            group = hf5.create_group("/",'trial_info','Trial Lists')
        table = hf5.create_table('/trial_info','digital_%s_trials' % dig_type,
                                 particles.trial_info_particle,
                                 'Trial List  for Digital %sputs' % dig_type)
        new_row = table.row
        for i,row in trial_df.iterrows():
            new_row['trial_num'] = row['Trial Num']
            new_row['name'] = row['name']
            new_row['channel'] = row[dig_str]
            new_row.append()
        hf5.flush()
        print('Done!')
    return trial_df

@Timer('Common Average Referencing')
def common_avg_reference(h5_file,electrodes,group_num):
    '''Computes and subtracts the common average for a group of electrodes 

    Parameters
    ----------
    h5_file : str, path to .h5 file with the raw data
    electrodes : list of int, electrodes to average
    group_num : int, number of common average group (for  storing common
                     average in hdf5 store)
    '''
    if not os.path.isfile(h5_file):
        raise FileNotFoundError('%s was not found.' % h5_file)

    print('Common Average Referencing Electrodes:\n'+ \
            ', '.join([str(x) for x in electrodes.copy()]))
    with tables.open_file(h5_file,'r+') as hf5:
        raw = hf5.root.raw
        n_samples = raw['electrode%i' % electrodes[0]].shape[0]

        # Calculate common average
        println('Computing common average...')
        common_avg = np.zeros((1,n_samples))[0]
        for x in electrodes:
            common_avg += raw['electrode%i' % x][:]
        common_avg /= float(len(electrodes))
        print('Done!')
        
        # Store common average 
        Atom = tables.Float64Atom()
        println('Storing common average signal...')
        if not '/common_average' in hf5:
            hf5.create_group('/','common_average','Common average electrodes and signals')
        if '/common_average/electrodes_group%i' % group_num in hf5:
            hf5.remove_node('/common_average/electrodes_group%i' % group_num)
        if '/common_average/common_average_group%i' % group_num in hf5:
            hf5.remove_node('/common_average/common_average_group%i' % group_num)
        hf5.create_array('/common_average','electrodes_group%i' % group_num,np.array(electrodes))
        hf5.create_earray('/common_average','common_average_group%i' % group_num,obj=common_avg)
        hf5.flush()
        print('Done!')

        # Replace raw data with referenced data
        println('Storing referenced signals...')
        for x in electrodes:
            referenced_data = raw['electrode%i' % x][:]-common_avg
            hf5.remove_node('/raw/electrode%i'  % x)
            if not '/referenced' in hf5:
                hf5.create_group('/','referenced','Common average referenced signals')
            if '/referenced/electrode%i' % x in hf5:
                hf5.remove_node('/referenced/electrode%i' % x)
            hf5.create_earray('/referenced','electrode%i' % x,obj=referenced_data)
            hf5.flush()
        print('Done!')

@Timer('Compressing and repacking h5 file')
def compress_and_repack(h5_file,new_file=None):
    '''Compress and repack the h5 file with ptrepack either to same name or new name

     Parameters
     ----------
     h5_file : str, path to h5 file
     new_file : str (optional), new path for h5_file 

     Returns
     -------
     str, new path to h5 file
     '''
    if new_file is None:
        new_file = os.path.join(os.path.dirname(h5_file),'tmp.h5')
        tmp = True
    else:
        tmp = False

    print('Repacking %s as %s...' % (h5_file,new_file))
    subprocess.call(['ptrepack','--chunkshape','auto','--propindexes','--complevel','9',
                     '--complib','blosc',h5_file,new_file])

    # Remove old  h5 file
    print('Removing old h5 file: %s' % h5_file)
    os.remove(h5_file)

    # If used a temporary rename to old file name
    if tmp:
        print('Renaming temporary file to %s' % h5_file)
        subprocess.call(['mv',new_file,h5_file])
        new_file = h5_file

    return new_file

def get_trial_info(h5_file):
    '''Opens the h5 file and returns the digital_in and digital_out trial info
    as pandas DataFrames

    Parameters
    ----------
    h5_file : str, path to h5_file

    Returns
    -------
    dict  of pandas.DataFrame values and table names as keys, one key for each
             table under /trial_info
    '''
    if not os.path.isfile(h5_file):
        raise FileNotFoundError('%s was not found' % h5_file)

    out = {}
    with tables.open_file(h5_file,'r') as hf5:
        if not '/trial_info' in hf5:
            return {}
        trial_nodes = h5_file.list_nodes('/trial_info')
        for node in trial_nodes:
            df = pd.DataFrame.from_records(node[:])
            df['name'] = df['name'].apply(lambda x: x.decode('utf-8'))
            out[node.name] = df

    return out

@Timer('Clustering Cleanup')
def cleanup_clustering(file_dir):
    '''Consolidate memory monitor files from clustering, remove raw and
    referenced data from hdf5 and repack

    Parameters
    ----------
    file_dir : str, path to recording directory

    Returns
    -------
    str, path to new hdf5 file
    '''
    # Check for memory_monitor_clustering files 
    # If found write all conents into memory_usage.txt and delete files
    println('Consolidating clustering memory usage logs...')
    mem_dir = os.path.join(file_dir,'memory_monitor_clustering') 
    mem_file = os.path.join(mem_dir,'memory_usage.txt')
    if not os.path.isfile(mem_file):
        file_list = os.listdir(mem_dir)
        with open(mem_file,'w') as write_file:
            for f in file_list:
                try:
                    mem_usage = np.loadtxt(os.path.join(mem_dir,f))
                    print('electrode%s\t%sMB' % (f.replace('.txt',''),str(mem_usage)),file=write_file)
                    os.remove(os.path.join(mem_dir,f))
                except:
                    pass
    print('Done!')

    # Grab h5 filename
    hdf5_name = get_h5_filename(file_dir)
    hdf5_file = os.path.join(file_dir,hdf5_name)

    # If raw and/or referenced data is still in h5
    # Remove raw/referenced data from hf5
    # Repack h5 as *_repacked.h5
    # Create sorted_units groups in h5, if it doesn't exist
    changes = False
    with tables.open_file(hdf5_file,'r+') as hf5:
        if '/raw' in hf5:
            println('Removing raw data from hdf5 store...')
            hf5.remove_node('/raw',recursive=1)
            changes = True
            print('Done!')
        if '/referenced' in hf5:
            println('Removing referenced data from hdf5 store...')
            hf5.remove_node('/referenced',recursive=1)
            changes = True
            print('Done!')
        if not '/sorted_units' in  hf5:
            hf5.create_group('/','sorted_units')
            changes = True
        if not '/unit_descriptor' in hf5:
            hf5.create_table('/','unit_descriptor',description = particles.unit_descriptor)
            changes = True
        else:
            hf5.remove_node('/unit_descriptor',recursive=1)
            hf5.create_table('/','unit_descriptor',description = particles.unit_descriptor)
            changes = True


    # Repack if any big changes were made to h5 store
    if changes:
        if hdf5_file.endswith('_repacked.h5'):
            new_fn = hdf5_file
        else:
            new_fn = hdf5_file.replace('.h5','_repacked.h5')
        new_h5 = compress_and_repack(hdf5_file,new_fn)
        return new_h5
    else:
        return hdf5_file
