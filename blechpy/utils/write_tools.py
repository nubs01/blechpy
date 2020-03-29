from blechpy.utils import print_tools as pt, userIO
import pandas as pd
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


def write_pandas_to_table(df, save_file, overwrite=False, shell=True):
    if os.path.isfile(save_file) and not overwrite:
        q = userIO.ask_user('File already exists: %s\nDo you want to overwrite it?' % save_file, shell=shell)
        if q == 0:
            return

    df.to_csv(save_file, sep='\t')


def read_pandas_from_table(fn):
    df = pd.read_csv(fn, sep='\t', index_col=0)
    return df

