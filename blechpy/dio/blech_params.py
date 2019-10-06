import os
import time
import json
import numpy as np
import pandas as pd
from blechpy.dio import rawIO
from blechpy.utils import print_tools as pt, userIO

SCRIPT_DIR = os.path.dirname(__file__)
PARAM_DIR = os.path.join(SCRIPT_DIR, 'defaults')
PARAM_NAMES = ['CAR_params', 'pal_id_params', 'data_cutoff_params',
               'clustering_params', 'bandpass_params', 'spike_snapshot',
               'psth_params']


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

def parse_amplifier_files(file_dir):
    '''
    parses the filenames of amp-*-*.dat files in file_dir and returns port and
    channel numbers
    for 'one file per channel' recordings

    deprecated: get ports and channels from rawIO.read_recording_info instead
    '''
    file_list = os.listdir(file_dir)
    ports = []
    channels = []
    for f in file_list:
        if f.startswith('amp'):
            tmp = f.replace('.dat','').split('-')
            if tmp[1] in ports:
                idx = ports.index(tmp[1])
                channels[idx].append(int(tmp[2]))
            else:
                ports.append(tmp[1])
                channels.append([int(tmp[2])])
    for c in channels:
        c.sort()
    return ports,channels

def parse_board_files(file_dir):
    '''
    parses board-*-*.dat files and returns lists of DIN and DOUT channels
    for 'one file per channel' type recordings

    deprecated: get DIN and DOUT from rawIO.read_recording_info instead
    '''
    file_list = os.listdir(file_dir)
    DIN = []
    DOUT = []
    for f in file_list:
        if f.startswith('board'):
            tmp = f.replace('.dat','').split('-')
            if tmp[1] == 'DIN':
                DIN.append(int(tmp[2]))
            elif tmp[1] == 'DOUT':
                DOUT.append(int(tmp[2]))
    return DIN,DOUT


def get_ports(file_dir):
    '''
    reads the data files in file_dir and returns a list of amplifier ports

    deprecated: get ports and channels from rawIO.read_recording_info instead
    '''
    ports,ch = parse_amplifier_files(file_dir)
    return ports

def get_channels_on_port(file_dir,port):
    '''
    reads files in file_dir to determine which amplifier channels are on port

    deprecated: get ports and channels from rawIO.read_recording_info instead
    '''
    ports,ch = parse_amplifier_files(file_dir)
    try:
        idx = ports.index(port)
    except ValueErrori as error:
        raise ValueError('Files for port %s not found in %s' % (port,file_dir)) from error
    return ch[idx]

def get_sampling_rate(file_dir):
    '''
    uses info.rhd in file_dir to get sampling rate of the data

    deprecated: get ports and channels from rawIO.read_recording_info instead
    '''
    sampling_rate = np.fromfile(os.path.join(file_dir,'info.rhd'), dtype = np.dtype('float32'))
    sampling_rate = int(sampling_rate[2])
    return sampling_rate

def get_din_channels(file_dir):
    '''
    returns a list of DIN channels read from filenames in file_dir

    deprecated: get ports and channels from rawIO.read_recording_info instead
    '''
    DIN,DOUT = parse_board_files(file_dir)
    return DIN

@Timer('Collecting parameters for common average referencing')
def select_CAR_groups(num_groups,electrode_mapping, shell=False):
    '''Returns a dict containing standard params for common average referencing
    Each dict field with fields, num groups, car_electrodes
    Can set num_groups to an integer or as unilateral or bilateral
    Settings as unilateral or bilateral will automatically assign channels to
    groups, setting to a number will allow choice of channels for each group
    unilateral: 1 CAR group, all channels on port
    bilateral: 2 CAR groups, [0-7,24-31] & [8-23], assumes same port for both

    Parameters
    ----------
    num_groups : int or 'bilateral', number of CAR groups, bilateral
                 autmatically assigns the first and last 8 electrodes to group 1 and the
                 middle 16 to group 2
    electrode_mapping : pandas.DataFrame, mapping electrode numbers to port and channel,
                        has columns: 'Electrode', 'Port' and 'Channel'

    Returns
    -------
    num_groups : int, number of CAR groups
    car_electrodes : list of lists of ints, list with a list of electrodes for
                     each CAR group 

    Throws
    ------
    ValueError : if num_groups is not a valid int (>0) or 'bilateral'
    '''
    electrodes = electrode_mapping['Electrode'].tolist()
    car_electrodes = []
    if num_groups==1:
        car_electrodes.append(electrodes)
    else:
        select_list = []
        for idx,row in electrode_mapping.iterrows():
            select_list.append(', '.join([str(x) for x in row]))
        for i in range(num_groups):
            tmp = userIO.select_from_list('Choose CAR electrodes for group %i'
                                          ': [Electrode, Port, Channel]' % i,
                                          select_list,
                                          title='Group %i Electrodes' % i,
                                          multi_select=True,
                                          shell=shell)
            if tmp is None:
                raise ValueError('Must select electrodes for CAR groups')
            car_electrodes.append([int(x.split(',')[0]) for x in tmp])

    # if 'dead' in electrode_mapping.columns:
    #     dead_ch = electrode_mapping['Electrode'][electrode_mapping['dead']]
    #     dead_ch = dead_ch.to_list()
    #     for group in car_electrodes:
    #         for dc in dead_ch:
    #             if dc in group:
    #                 group.remove(dc)


    return car_electrodes


def flatten_channels(ports,channels,emg_port=None,emg_channels=None):
    '''takes all ports and all channels and makes a dataframe mapping ports and
    channels to electrode numbers from 0 to N
    excludes emg_channels if given

    Parameters
    ----------
    ports : list, list of port names, length equal to channels
    channels : list, list of channels number, corresponding to elements of ports
    emg_port : str (optional), prefix of port with EMG channel. Default is None
    emg_channels: list (optional), list of channels on emg_port used for emg

    Returns
    -------
    electrode_mapping : pandas.DataFrame,
                        3 columns: Electrode, Port and Channel
    emg_mapping : pandas.DataFrame,
                    3 columns: EMG, Port, and Channel

    Throws
    ------
    ValueError : if length of ports is not equal to length of channels
    '''
    el_map = []
    em_map = []
    ports = ports.copy()
    channels = channels.copy()
    for idx,p in enumerate(zip(ports,channels)):
        if p[0]==emg_port and p[1] in emg_channels:
            em_map.append(p)
        else:
            el_map.append(p)

    map_df = pd.DataFrame(el_map,columns=['Port','Channel'])
    map_df.sort_values(by=['Port','Channel'],ascending=True,inplace=True)
    map_df.reset_index(drop=True,inplace=True)
    map_df = map_df.reset_index(drop=False).rename(columns={'index':'Electrode'})

    emg_df = pd.DataFrame(em_map,columns=['Port','Channel'])
    emg_df.sort_values(by=['Port','Channel'],ascending=True,inplace=True)
    emg_df.reset_index(drop=True,inplace=True)
    emg_df = emg_df.reset_index(drop=False).rename(columns={'index':'EMG'})
    return map_df, emg_df


def write_dict_to_json(dat, save_file):
    '''writes a dict to a json file

    Parameters
    ----------
    dat : dict
    save_file : str
    '''
    with open(save_file, 'w') as f:
        json.dump(dat, f, indent=True)


def read_dict_from_json(save_file):
    '''reads dict from json file

    Parameters
    ----------
    save_file : str
    '''
    with open(save_file, 'r') as f:
        out = json.load(f)

    return out


def load_params(param_name, rec_dir=None, default_keyword=None):
    '''checks rec_dir (if provided) for parameters in the analysis_params
    folder, if params do not exist then this  loads and returns the defaults
    from the package defaults folder.

    Parameters
    ----------
    param_name : basename of param file, i.e. CAR_params or pal_id_params
    rec_dir : str (optional)
        recording dir containing an analysis_params folder
        if not provided or None (default) then defaults are loaded
    default_keyword : str (optional)
        if provided and a rec specific param file does not exists then  this
        will to used to a grab a subset of params from the default file
        This is if multiple defaults are in a single default file
    '''
    if not param_name.endswith('.json'):
        param_name += '.json'

    default_file = os.path.join(PARAM_DIR, param_name)
    if rec_dir is not None:
        rec_file = os.path.join(rec_dir, 'analysis_params', param_name)
    else:
        rec_file = None

    if rec_file is not None and os.path.isfile(rec_file):
        out = read_dict_from_json(rec_file)
    elif os.path.isfile(default_file):
        print('%s not found in recording directory. Pulling parameters from defaults' % param_name)
        out = read_dict_from_json(default_file)
        if out.get('multi') is True and default_keyword is None:
            raise ValueError('Multple defaults in %s file, but no keyword provided' % param_name)

        elif out and default_keyword:
            out = out.get(default_keyword)
            if out is None:
                print('No %s found for keyword %s' % (param_name, default_keyword))

        elif out is None:
            print('%s default file is empty' % param_name)

    else:
        print('%s.json not found in recording directory or in defaults')
        out = None

    return out


def write_params_to_json(param_name, rec_dir, params):
    '''Writes params into a json file placed in the analysis_params folder in
    rec_dir with the name param_name.json

    Parameters
    ----------
    param_name : str, name of parameter file
    rec_dir : str, recording directory
    params : dict, paramters
    '''
    if not param_name.endswith('.json'):
        param_name += '.json'

    p_dir = os.path.join(rec_dir, 'analysis_params')
    save_file = os.path.join(p_dir, param_name)
    print('Writing %s to %s' % (param_name, save_file))
    if not os.path.isdir(p_dir):
        os.mkdir(p_dir)

    write_dict_to_json(params, save_file)

