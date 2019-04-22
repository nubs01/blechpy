import os
import easygui as eg
import numpy as np

clustering_params = {'Max Number of Clusters':7,
                    'Max Number of Iterations':1000,
                    'Convergence Criterion':0.001,
                    'GMM random restarts':10}

data_params = {'clean':{'V_cutoff for disconnected headstage':1500,
                        'Max rate of cutoff breach per second':0.2,
                        'Max allowed seconds with a breach': 10,
                        'Max allowed breacher per second':20,
                        'Intra-cluster waveform amp SD cutoff':3},
                'noisy':{'V_cutoff for disconnected headstage':3000,
                        'Max rate of cutoff breach per second':2,
                        'Max allowed seconds with a breach': 20,
                        'Max allowed breacher per second':40,
                        'Intra-cluster waveform amp SD cutoff':3}}

bandpass_params = {'Lower freq cutoff':300,
                    'Upper freq cutoff':3000} # in Hz

spike_snapshot = {'Time before spike':.5,
                    'Time after spike':1} # in ms


def parse_amplifier_files(file_dir):
    '''
    parses the filenames of amp-*-*.dat files in file_dir and returns port and
    channel numbers
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
    '''
    ports,ch = parse_amplifier_files(file_dir)
    return ports

def get_channels_on_port(file_dir,port):
    '''
    reads files in file_dir to determine which amplifier channels are on port
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
    '''
    sampling_rate = np.fromfile(os.path.join(file_dir,'info.rhd'), dtype = np.dtype('float32'))
    sampling_rate = int(sampling_rate[2])   
    return sampling_rate

def get_din_channels(file_dir):
    '''
    returns a list of DIN channels read from filenames in file_dir
    '''
    DIN,DOUT = parse_board_files(file_dir)
    return DIN

def get_CAR_params(file_dir,num_groups,EMG_port=None,EMG_ch=None):
    '''
    Returns a dict containing standard params for common average referencing
    Each dict field with fields, num groups, ports, channels, emg port and emg
    channels
    Can set num_groups to an integer or as unilateral or bilateral
    Settings as unilateral or bilateral will automatically assign channels to
    groups, setting to a number will allow choice of channels for each group
    unilateral: 1 CAR group, all channels on port
    bilateral: 2 CAR groups, [0-7,24-31] & [8-23], assumes same port for both
    Can also provide EMG_port and EMG_ch(list or single channel) if used, to
    exclude from list
    '''
    if num_groups=='bilateral':
        num_groups = 2
        implant_type = 'bilateral'
    elif num_groups=='unilateral':
        num_groups = 1
        implant_type = 'unilateral'
    elif isinstance(num_groups,int) and num_groups>0:
        implant_type=None
    else:
        raise ValueError('Num groups must be an integer >0 or a string bilateral or unlateral')

    ports,channels = parse_amplifier_files(file_dir)
    if len(ports)>1:
        car_ports = []
        for i in range(num_groups):
            tmp = select_from_list('Choose port for CAR group %i: ' % i,'Multiple Ports Found',ports)
            if tmp is None:
                return None
            car_ports.append(tmp)
    else:
        car_ports = [ports[0] for p in range(num_groups)]

    if implant_type=='bilateral':
        car_channels = []
        port_idx = [ports.index(x) for x in car_ports]
        g1 = [*range(8),*[x+24 for x in range(8)]]
        g2 = [x+8 for x in range(16)]
        # check is files for any channels are missing
        g1_check = any([x not in channels[port_idx[0]] for x in g1])
        g2_check = any([x not in channels[port_idx[1]] for x in g2]) 
        if  g1_check or g2_check:
            raise LookupError('Missing some amp files for 32ch bilateral recording')
        car_channels = [g1,g2]
    elif implant_type=='unilateral':
        car_channels = channels
    else:
        car_channels = []
        for i,p in enumerate(car_ports,start=0):
            idx = ports.index(p)
            tmp = select_from_list('Choose CAR channels on port %s for CAR group %i: ' % (p,i),
                                    'Choose channels',channels[idx],multi_select=True)
            if tmp is None:
                return None
            car_channels.append(tmp)

    # Exclude emg channels from lists
    if not isinstance(EMG_ch,list):
        EMG_ch = [EMG_ch]
    if EMG_ch is not None and EMG_port in car_ports:
        idx = [i for i,x in enumerate(car_ports) if x==EMG_port]
        for i,p in enumerate(car_ports):
            if p==EMG_port:
                car_channels[i] = [x for x  in car_channels[i] if x not in EMG_ch]

    out = {'num groups':num_groups,'ports':car_ports,'channels':car_channels,
            'emg port':EMG_port,'emg channels':EMG_ch}
    return out
                

def select_from_list(prompt,title,items,multi_select=False):
    '''
    makes a popup for list selections, can be multichoice or single choice
    default is single selection
    '''
    if multi_select is False:
        choice = eg.choicebox(prompt,title,items)
    else:
        choice = eg.multchoicebox(prompt,title,items,None)

    return choice
