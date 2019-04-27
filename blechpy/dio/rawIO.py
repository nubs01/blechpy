import numpy as np
import os
from blechpy.dio import load_intan_rhd_format

def read_rec_info(file_dir):
    '''Reads the info.rhd file to get relevant parameters.
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
    info = load_intan_rhd_format.read_data(info_file)
    amp_ch = info['amplifier_channels']
    dig_in = info['board_dig_in_channels']
    ports,channels = zip(*[(x['port_prefix'],x['native_order']) for x in amp_ch])
    din_ports,din = zip(*[(x['port_number'],x['native_order']) for x in dig_in if x['port_prefix']=='DIN'])
    freq_params = info['frequency_parameters']
    notch_freq = freq_params['notch_filter_frequency']
    amp_fs = freq_params['amplifier_sample_rate']
    dig_in_fs = freq_params['board_dig_in_sample_rate']
    out = {'amplifier_sampling_rate':amp_fs,
            'dig_in_sampling_rate':dig_in_fs,
            'notch_filter':notch_freq,
            'dig_in':din,'dig_in_port_nums':din_ports,
            'ports':ports,
            'channels':channels}
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
                                  provided info is taken from info.rhd

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

def read_digitalin_dat(file_dir,rec_info=None):
    '''Reads digitalin.dat from intan recording with file_type 'one file per signal type'

    Parameters
    ----------
    file_dir : str, file directory for recording data
    rec_info : dict (optional), rec_info dict returned from read_rec_info

    Returns
    -------
    numpy.ndarray : one row per digital_input channel corresponding to dig_in
                    from rec_info

    Throws
    ------
    FileNotFoundError : if digitalin.dat is not in file_dir
    '''
    if rec_info is None:
        rec_info = read_rec_info(file_dir)

    dig_in = rec_info['dig_in']
    dig_in_file = os.path.join(file_dir,'digitalin.dat')
    if not os.path.isfile(dig_in_file):
        raise FileNotFoundError('digitalin.dat not found at %s' % dig_in_file)
    file_dat = np.fromfile(dig_in_file,dtype=np.dtype('int16'))
    chan_dat = []
    for din in dig_in:
        tmp_dat = (file_dat&pow(2,din)>0).astype(np.dtype('uint16'))
        chan_dat.append(tmp_dat)
    out = np.array(chan_dat)
    return out
    
