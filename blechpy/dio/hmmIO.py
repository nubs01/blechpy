import os
import tables
import numpy as np
import pandas as pd
from blechpy.utils.particles import HMMInfoParticle
from copy import deepcopy
import glob
from blechpy import load_dataset

def fix_hmm_overview(h5_file):
    '''made to add the area column to the hmm overview
    now adds the hmm_class column
    '''
    fix = True
    if not os.path.isfile(h5_file):
        return

    #with tables.open_file(h5_file, 'r') as hf5:
    #    if 'notes' not in hf5.root.data_overview.colnames:
    #        fix = True
 
    #if not fix:
    #    return

    print('Fixing data overview table in %s' % h5_file)
    with tables.open_file(h5_file, 'a') as hf5:
        for row in hf5.root.data_overview.where('log_likelihood == 0.'):
            hmm_id = row['hmm_id']
            h_str = 'hmm_%s' % hmm_id
            fit_LL = hf5.root[h_str]['fit_LL'][-1]
            row['log_likelihood'] = fit_LL
            row.update()

        hf5.root.data_overview.flush()
        hf5.flush()


        #if 'tmp_overview' in hf5.root:
        #    hf5.remove_node('/tmp_overview')

        #new_table = hf5.create_table('/', 'tmp_overview', HMMInfoParticle,
        #                             'Parameters and goodness-of-fit info for HMMs in file')
        #table = hf5.root.data_overview
        #columns = table.colnames
        #new_row = new_table.row
        #for row in table.iterrows():
        #    for x in columns:
        #        new_row[x] = row[x]
 
        #    new_row['notes'] = 'PI & A constrained'
        #    new_row.append()
 
        #new_table.flush()
        #hf5.move_node('/tmp_overview', '/', 'data_overview', overwrite=True)

#       # Now change state_sequences to best_sequences
#       nodes = [x for x in hf5.walk_nodes('/') if 'log_likelihood_hist' in x._v_pathname]
#       for x in nodes:
#           hf5.move_node(x._v_pathname, x._v_parent._v_pathname, 'max_log_prob')

        #hf5.flush()


def setup_hmm_hdf5(h5_file, infoParticle=HMMInfoParticle):
    if os.path.isfile(h5_file):
        return

    print('Initializing hdf5 store: %s' % h5_file)
    with tables.open_file(h5_file, 'a') as hf5:
        if 'data_overview' not in hf5.root:
            print('Initializing data_overview table in hdf5 store...')
            table = hf5.create_table('/', 'data_overview', infoParticle,
                                     'Parameters and goodness-of-fit info for HMMs in file')
            table.flush()


def read_hmm_from_hdf5(h5_file, hmm_id):
    hmm_id = int(hmm_id)
    with tables.open_file(h5_file, 'r') as hf5:
        h_str = 'hmm_%s' % hmm_id
        if h_str not in hf5.root or len(hf5.list_nodes('/'+h_str)) == 0:
            return None

        # print('Loading HMM %i from hdf5' % hmm_id)
        nodes = [x._v_name for x in hf5.list_nodes('/'+h_str)]
        tmp = hf5.root[h_str]
        stat_arrays = {}
        for k in nodes:
            stat_arrays[k] = tmp[k][:]

        PI = stat_arrays.pop('initial_distribution')
        A = stat_arrays.pop('transition')
        B = stat_arrays.pop('emission')

        rs = stat_arrays['row_id'].shape
        tmp = np.array([x.decode('utf-8') for x in stat_arrays['row_id'].ravel()])
        stat_arrays['row_id'] = tmp.reshape(rs)


        table = hf5.root.data_overview
        for row in table.where('hmm_id == id', condvars={'id':hmm_id}):
            params = {}
            for k in table.colnames:
                if table.coltypes[k] == 'string':
                    params[k] = row[k].decode('utf-8')
                    if '..' in params[k]:
                        params[k] = params[k].split('..')

                else:
                    params[k] = row[k]

            if isinstance(params['taste'], list):
                params['channel'] = list_channel_hash(params['channel'])

            return PI, A, B, stat_arrays, params
        else:
            raise ValueError('Parameters not found for hmm %i' % hmm_id)


def write_hmm_to_hdf5(h5_file, hmm, params):
    params = deepcopy(params)
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
            tmp = np.where(np.diff(ids) > 1)[0]
            if len(ids) == 0:
                hmm_id = 0
            elif len(tmp) == 0:
                hmm_id = np.max(ids) + 1
            else:
                hmm_id = ids[tmp[0]] + 1

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
        for k, v in hmm.stat_arrays.items():
            if not isinstance(v, np.ndarray):
                tmp_v = np.array(v)
            else:
                tmp_v = v

            hf5.create_array('/'+h_str, k, tmp_v)

        table = hf5.root.data_overview
        edited_rows = 0
        for row in table.where('hmm_id == id', condvars={'id': hmm_id}):
            print('Editing existing row in data_overview with new values for HMM %s' % hmm_id)
            print('New iteration: %i, fit_LL: %.4E, BIC: %.3E' % (hmm.iteration, hmm.fit_LL, hmm.BIC))
            print('Old iteration: %i, fit_LL: %.4E, BIC: %.3E' % (row['n_iterations'], row['log_likelihood'], row['BIC']))
            row['BIC'] = hmm.BIC
            row['cost'] = hmm.cost
            row['converged'] = hmm.converged
            row['fitted'] = hmm.fitted
            row['max_log_prob'] = hmm.max_log_prob
            row['log_likelihood'] = hmm.fit_LL
            row['n_iterations'] = hmm.iteration
            row.update()
            edited_rows += 1

        if edited_rows == 0:
            print('Creating new row in data_overview for HMM %s' % hmm_id)
            row = table.row
            for k,v in params.items():
                if table.coltypes[k] == 'string' and isinstance(v, list):
                    row[k] = '..'.join(v)
                elif isinstance(v, list) and k == 'channel':
                    print('channels: %s' % str(v))
                    row[k] = hash_channel_list(v)
                    print('channel hash: %i' % row[k])
                elif not isinstance(v, list):
                    row[k] = v
                else:
                    raise ValueError('Something fucked up')


            row['BIC'] = hmm.BIC
            row['cost'] = hmm.cost
            row['converged'] = hmm.converged
            row['fitted'] = hmm.fitted
            row['max_log_prob'] = hmm.max_log_prob
            row['log_likelihood'] = hmm.fit_LL
            row['n_iterations'] = hmm.iteration
            row.append()

        table.flush()
        hf5.flush()

    print('='*80+'\n')


def delete_hmm_from_hdf5(h5_file, **kwargs):
    with tables.open_file(h5_file, 'a') as hf5:
        table = hf5.root.data_overview
        ids = []
        rmv = list(np.arange(len(table)))
        for k,v in kwargs.items():
            tmp = table[:][k]
            if isinstance(v, str):
                tmp = [x.decode('utf-8') for x in tmp]
                tmp = np.array(tmp)

            if v in tmp:
                idx = np.where(tmp == v)[0]
                ids.append(idx)

        for x in ids:
            rmv = [y for y in rmv if y in x]

        rmv.sort()
        for x in reversed(rmv):
            hmm_id = table[x]['hmm_id']
            h_str = 'hmm_%s' % hmm_id
            if h_str in hf5.root:
                print('Deleting existing data for %s...' % h_str)
                hf5.remove_node('/', h_str, recursive=True)
            else:
                print('HMM %s not found in hdf5.' % hmm_id)

            table.remove_rows(x, x+1)

        table.flush()
        hf5.flush()


def compare_hmm_params(p1, p2):
    compare_keys = ['taste', 'unit_type', 'dt', 'max_iter', 'time_start',
                    'time_end', 'n_states', 'n_trials', 'hmm_class', 'area', 'notes']
    for k in compare_keys:
        if p1[k] != p2[k]:
            return False

    return True


def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    return tmp[0]


def get_hmm_overview_from_hdf5(h5_file):
    with tables.open_file(h5_file, 'r') as hf5:
        table = hf5.root.data_overview
        ids = table[:]['hmm_id']

    params = []
    for i in ids:
        _, _, _, _, p = read_hmm_from_hdf5(h5_file, i)
        params.append(p)

    df = pd.DataFrame(params)

    return df


def hash_channel_list(channels):
    channels.insert(0, len(channels)) # gives elements and array and prevent leading 0 from dropping
    return ''.join([str(x) for x in channels])


def list_channel_hash(num):
    tmp = [int(x) for x in str(num)]
    return tmp[1:]


