import os
import tables
import numpy as np
import pandas as pd
from blechpy.utils.particles import HMMInfoParticle


def setup_hmm_hdf5(h5_file):
    print('Initializing hdf5 store: %s' % h5_file)
    with tables.open_file(h5_file, 'a') as hf5:
        if 'data_overview' not in hf5.root:
            print('Initializing data_overview table in hdf5 store...')
            table = hf5.create_table('/', 'data_overview', HMMInfoParticle,
                                     'Basic info for each digital_input')
            table.flush()


def read_hmm_from_hdf5(h5_file, hmm_id):
    with tables.open_file(h5_file, 'r') as hf5:
        h_str = 'hmm_%s' % hmm_id
        if h_str not in hf5.root or len(hf5.list_nodes('/'+h_str)) == 0:
            return None

        print('Loading HMM %i for hdf5' % hmm_id)
        tmp = hf5.root[h_str]
        PI = tmp['initial_distribution'][:]
        A = tmp['transition'][:]
        B = tmp['emission'][:]
        time = tmp['time'][:]
        best_paths = tmp['state_sequences'][:]
        table = hf5.root.data_overview
        for row in table.where('hmm_id == id', condvars={'id':hmm_id}):
            params = {}
            for k in table.colnames:
                if table.coltypes[k] == 'string':
                    params[k] = row[k].decode('utf-8')
                else:
                    params[k] = row[k]

            return PI, A, B, time, best_paths, params
        else:
            raise ValueError('Parameters not found for hmm %i' % hmm_id)


def write_hmm_to_hdf5(h5_file, hmm, time, params):
    hmm_id = hmm.hmm_id
    if 'hmm_id' in params and hmm_id is None:
        hmm.hmm_id = hmm_id = params['hmm_id']
    elif 'hmm_id' in params and hmm_id != params['hmm_id']:
        raise ValueError('ID of HMM %i does not match ID in params %i'
                         % (hmm_id, params['hmm_id']))
    else:
        pass

    if not os.path.isfile(h5_file):
        setup_hmm_hdf5(h5_file)

    print('\n' + '='*80)
    print('Writing HMM %s to hdf5 file...' % hmm_id)
    print(params)
    print('PI: %s' % repr(hmm.initial_distribution.shape))
    print('A: %s' % repr(hmm.transition.shape))
    print('B: %s' % repr(hmm.emission.shape))
    with tables.open_file(h5_file, 'a') as hf5:
        if hmm_id is None:
            ids = hf5.root.data_overview.col('hmm_id')
            hmm_id = np.max(ids) + 1
            hmm.hmm_id = hmm_id
            params['hmm_id'] = hmm_id
            print('HMM assigned id #%i' % hmm_id)

        h_str = 'hmm_%s' % hmm_id
        if h_str in hf5.root:
            print('Deleting existing data for %s...' % h_str)
            hf5.remove_node('/', h_str, recursive=True)

        print('Writing new data for %s' % h_str)
        hf5.create_group('/', h_str, 'Data for HMM #%i' % hmm_id)
        hf5.create_array('/'+h_str, 'initial_distribution',
                         hmm.initial_distribution)
        hf5.create_array('/'+h_str, 'transition', hmm.transition)
        hf5.create_array('/'+h_str, 'emission', hmm.emission)
        hf5.create_array('/'+h_str, 'time', time)
        hf5.create_array('/'+h_str, 'state_sequences', hmm.best_sequences)

        table = hf5.root.data_overview
        for row in table.where('hmm_id == id', condvars={'id': hmm_id}):
            print('Editing existing row in data_overview with new values for HMM %s' % hmm_id)
            row['BIC'] = hmm.BIC
            row['cost'] = hmm.cost
            row['converged'] = hmm.converged
            row['fitted'] = hmm.fitted
            row['max_log_prob'] = hmm.max_log_prob
            row['n_iterations'] = hmm.iteration
            row.update()
            break
        else:
            print('Creating new row in data_overview for HMM %s' % hmm_id)
            row = table.row
            for k,v in params.items():
                row[k] = v

            row['BIC'] = hmm.BIC
            row['cost'] = hmm.cost
            row['converged'] = hmm.converged
            row['fitted'] = hmm.fitted
            row['max_log_prob'] = hmm.max_log_prob
            row['n_iterations'] = hmm.iteration
            row.append()

        table.flush()
        hf5.flush()

    print('='*80+'\n')
