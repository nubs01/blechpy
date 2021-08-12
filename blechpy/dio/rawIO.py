import numpy as np 
import os, re
from blechpy.dio import load_intan_rhd_format
from blechpy.utils import print_tools as pt, userIO

support_rec_types = {'one file per channel':'amp-\S-\d*\.dat',
                     'one file per signal type':'amplifier\.dat'}
voltage_scaling = 0.195

def get_sampling_rate(rec_dir):
    '''Returns sampling rate in Hz of intan recording data

    Parameters
    ----------
    rec_dir : str, full path to raw recording directory

    Returns
    -------
    float : sampling rate in Hz
    '''
    info_file = os.path.join(rec_dir, 'info.rhd')
    if not os.path.exists(info_file):
        raise FileNotFoundError('No info.rhd in recording directory')

    info = load_intan_rhd_format.read_data(info_file)
    return info['frequency_parameters']['amplifier_sample_rate']


def read_rec_info(file_dir, shell=True):
    '''Reads the info.rhd file to get relevant parameters.
    TODO: Check for zero size files and exclude from rec_info and suggest deletion to users
    Parameters
    ----------
    file_dir : str, path to recording directory

    Returns
    -------
    dict, necessary analysis info from info.rhd
        fields: amplifier_sampling_rate, dig_in_sampling_rate, notch_filter,
                ports (list, corresponds to channels), channels (list)

    Throws
    ------
    FileNotFoundError : if info.rhd is not in file_dir
    '''
    info_file = os.path.join(file_dir,'info.rhd')
    if not os.path.isfile(info_file):
        raise FileNotFoundError('info.rhd file not found in %s' % file_dir)
    out = {}
    print('Reading info.rhd file...')
    try:
        info = load_intan_rhd_format.read_data(info_file)
    except Exception as e:
        # TODO: Have a way to manually input settings
        info = None
        userIO.tell_user('%s was unable to be read. May be corrupted or '
                         'recording may have been interrupted' % info_file,
                         shell=True)
        raise e

    freq_params = info['frequency_parameters']
    notch_freq = freq_params['notch_filter_frequency']
    amp_fs = freq_params['amplifier_sample_rate']
    dig_in_fs = freq_params['board_dig_in_sample_rate']
    out = {'amplifier_sampling_rate':amp_fs,
            'dig_in_sampling_rate':dig_in_fs,
            'notch_filter':notch_freq}

    amp_ch = info['amplifier_channels']
    ports = [x['port_prefix'] for x in amp_ch]
    channels = [x['native_order'] for x in amp_ch]

    out['ports'] = ports
    out['channels'] = channels
    out['num_channels'] = len(channels)

    if info.get('board_dig_in_channels'):
        dig_in = info['board_dig_in_channels']
        din = [x['native_order'] for x in dig_in]
        out['dig_in'] = din

    if info.get('board_dig_out_channels'):
        dig_out = info['board_dig_out_channels']
        dout = [x['native_order'] for x in dig_out]
        out['dig_out'] = dout


    out['file_type'] = get_recording_filetype(file_dir)

    print('\nRecording Info\n--------------\n')
    print(pt.print_dict(out))
    return out


def read_time_dat(file_dir,sampling_rate=None):
    '''Reads the time vector out of time.dat and converts to seconds
    
    Parameters
    ----------
    file_dir : str, path to recording directory
    sampling_rate: int (optional), sampling rate of amplifier data in Hz

    Returns
    -------
    numpy.ndarray, time vector in seconds

    Throws
    ------
    FileNotFoundError : if time.dat is not in file_dir
    '''
    time_file = os.path.join(file_dir,'time.dat')
    if not os.path.isfile(time_file):
        raise FileNotFoundError('Time file not found at %s' % time_file)
    if sampling_rate is None:
        rec_info = read_rec_info(file_dir)
        sampling_rate = rec_info['amplifier_sampling_rate']

    time = np.fromfile(time_file,dtype=np.dtype('int32'))
    time = time.astype('float')/sampling_rate
    return time


def read_amplifier_dat(file_dir,num_channels=None):
    '''Reads intan amplifier.dat file to get recording channel data from
    recordings done with 'one file per signal type' setting
    WARNING: Memory intensive as all channel data will be held in memory
    TO_CHECK: Do we analyze with scaled or unscaled voltage, Narendra's code
    uses unscaled, scale factor to get microvolts is 0.195

    Parameters
    ----------
    file_dir: str, path to recording directory
    num_channels: int (optional), number of channels in recording, if not
                                  provided, info is taken from info.rhd

    Returns
    -------
    numpy.ndarray :  array with row for each channel corresponding to channels
                     in rec_info from read_rec_info, voltage is unscaled, to
                     get microvolts multiply by 0.195

    Throws
    ------
    FileNotFoundError : if amplifier.dat file is not found in file_dir
    '''
    if num_channels is None:
        rec_info = read_rec_info(file_dir)
        num_channels = len(rec_info['channels'])

    amp_file = os.path.join(file_dir,'amplifier.dat')
    if not os.path.isfile(amp_file):
        raise FileNotFoundError('Could not find amplfier file at %s' % amp_file)

    amp_dat = np.fromfile(amp_file,dtype=np.dtype('int16'))
    amp_dat = amp_dat.reshape(num_channels,-1,order='F')
    return amp_dat


def read_digital_dat(file_dir,dig_channels=None,dig_type='in'):
    '''Reads digitalin.dat from intan recording with file_type 'one file per signal type'

    Parameters
    ----------
    file_dir : str, file directory for recording data
    dig_channels : list (optional), digital channel numbers to get
    dig_type : {'in','out'}, type of digital signal to get (default 'in')

    Returns
    -------
    numpy.ndarray : one row per digital_input channel corresponding to dig_in
                    from rec_info

    Throws
    ------
    FileNotFoundError : if digitalin.dat is not in file_dir
    '''
    if dig_channels is None:
        rec_info = read_rec_info(file_dir)
        dig_channels = rec_info['dig_%s' % dig_type]
    dat_file = os.path.join(file_dir,'digital%s.dat' % dig_type)
    if not os.path.isfile(dat_file):
        raise FileNotFoundError('No file found at %s' % dig_in_file)
    file_dat = np.fromfile(dat_file,dtype=np.dtype('int16'))
    chan_dat = []
    for ch in dig_channels:
        tmp_dat = (file_dat&pow(2,ch)>0).astype(np.dtype('uint16'))
        chan_dat.append(tmp_dat)
    out = np.array(chan_dat)
    return out


def read_one_channel_file(file_name):
    '''Reads a single amp or din channel file created by an intan 'one file per
    channel' recording

    Parameters
    ----------
    file_name : str, absolute path to file to read data from 

    Returns
    -------
    numpy.ndarray : int16, 1D array of data from amp file

    Throws
    ------
    FileNotFoundError : if file_name is not found
    '''
    if not os.path.isfile(file_name):
        raise FileNotFoundError('Could not locate file %s' % file_name)

    chan_dat = np.fromfile(file_name,dtype=np.dtype('int16'))
    return chan_dat


def get_recording_filetype(file_dir):
    '''Check Intan recording directory to determine type of recording and thus
    extraction method to use. Asks user to confirm, and manually correct if
    incorrect

    Parameters
    ----------
    file_dir : str, recording directory to check

    Returns
    -------
    str : file_type of recording
    '''
    file_list = os.listdir(file_dir)
    file_type = None
    for k,v in support_rec_types.items():
        regex = re.compile(v)
        if any([True for x in file_list if regex.match(x) is not None]):
            file_type = k

    if file_type is None:
        msg = '\n   '.join(['unsupported recording type. Supported types are:',
                                    *list(support_rec_types.keys())])
    else:
        msg = '\"'+file_type+'\"'

    return file_type

    # Removing query since this is pretty accurate
    #query = 'Detected recording type is %s \nIs this correct?:  ' % msg
    #q = userIO.ask_user(query,,
    #                    shell=shell)

    #if q == 1:
    #    return file_type
    #else:
    #    choice = userIO.select_from_list('Select correct recording type',
    #                                     list(support_rec_types.keys()),
    #                                     'Select Recording Type',
    #                                     shell=shell)
    #    choice = list(support_rec_types.keys())[choice]
    #    return choice
