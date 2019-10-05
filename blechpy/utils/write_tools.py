from blechpy.utils import print_tools as pt
import json
import os


def write_clustering_params(file_name,params):
    '''Writes parameters into a file for use by blech_process.py

    Parameters
    ----------
    file_name : str, path to .params file to write params in
    params : dict, dictionary of parameters with keys:
                   clustering_params, data_params,
                   bandpass_params, spike_snapshot
    '''
    if not file_name.endswith('.params'):
        file_name += '.params'
    print('File: ' + file_name)
    pt.print_dict(params)
    with open(file_name,'w') as f:
        for c in clust_param_order:
            print(params['clustering_params'][c],file=f)
        for c in data_param_order:
            print(params['data_params'][c],file=f)
        for c in band_param_order:
            print(params['bandpass_params'][c],file=f)
        for c in spike_snap_order:
            print(params['spike_snapshot'][c],file=f)
        print(params['sampling_rate'],file=f)


def write_dict_to_json(dat, save_file):
    '''writes a dict to a json file

    Parameters
    ----------
    dat : dict
    save_file : str
    '''
    with open(save_file, 'w') as f:
        json.dump(dat, f, indent=True)


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
    if not os.path.isdir(p_dir):
        os.mkdir(p_dir)

    write_dict_to_json(params, save_file)


def read_dict_from_json(save_file):
    '''reads dict from json file

    Parameters
    ----------
    save_file : str
    '''
    with open(save_file, 'r') as f:
        out = json.load(f)

    return out
