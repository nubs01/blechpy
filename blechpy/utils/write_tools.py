from blechpy.utils import print_tools as pt
import json
import os


def write_dict_to_json(dat, save_file):
    '''writes a dict to a json file

    Parameters
    ----------
    dat : dict
    save_file : str
    '''
    with open(save_file, 'w') as f:
        json.dump(dat, f, indent=4)


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
