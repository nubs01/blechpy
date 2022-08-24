import os
import tables
import numpy as np
from blechpy.dio import h5io
from numba import jit
import itertools
    
def make_spike_arrays(h5_file, params):
    '''Makes stimulus triggered spike array for all sorted units

    Parameters
    ----------
    h5_file : str, full path to hdf5 store
    params : dict
        Parameters for arrays with fields:
            dig_ins_to_use : list of int, which dig_in channels for arrays
            laser_channels : list of int or None if no lasers
            sampling_rate : float, sampling rate of data in Hz
            pre_stimulus: : int, ms before stimulus to include in array
            post_stimulus : int, ms after stimulus to include in array
    '''
    print('\n----------\nMaking Unit Spike Arrays\n----------\n')
    dig_in_ch = params['dig_ins_to_use']
    laser_ch = params['laser_channels']
    fs = params['sampling_rate']
    pre_stim = int(params['pre_stimulus'])
    post_stim = int(params['post_stimulus'])
    pre_idx = int(pre_stim * (fs/1000))
    post_idx = int(post_stim * (fs/1000))
    n_pts = pre_stim + post_stim

    if dig_in_ch is None or dig_in_ch == []:
        raise ValueError('Must provide dig_ins_to_use in params in '
                         'order to make spike arrays')
    # get digital input table
    dig_in_table = h5io.read_trial_data_table(h5_file, 'in',
                                              dig_in_ch)

    # Get laser input table
    if laser_ch is not None or laser_ch == []:
        laser_table = h5io.read_trial_data_table(h5_file, 'in',
                                                 laser_ch)
        lasers = True
        n_lasers = len(laser_ch)
    else:
        laser_table = None
        lasers = False
        n_lasers = 1

    with tables.open_file(h5_file, 'r+') as hf5:
        n_units = len(hf5.list_nodes('/sorted_units'))

        # Get experiment end time from last spike time in case headstage fell
        # off
        exp_end_idx = 0
        for unit in hf5.root.sorted_units:
            tmp = np.max(unit.times)
            if tmp > exp_end_idx:
                exp_end_idx = tmp

        exp_end_time = exp_end_idx/fs

        if '/spike_trains' in hf5:
            hf5.remove_node('/', 'spike_trains', recursive=True)

        hf5.create_group('/', 'spike_trains')

        for i in dig_in_ch:
            print('Creating spike arrays for dig_in_%i...' % i)

            # grab trials for dig_in_ch that end more than post_stim ms before
            # the end of the experiment
            tmp_trials = dig_in_table.query('channel == @i')
            trial_cutoff_idx = exp_end_idx - post_idx

            # get the end indices for those trials
            # off_idx = np.array(tmp_trials['off_index'])
            # off_idx.sort()
            # n_trials = len(off_idx)
            
            on_idx = np.array(tmp_trials['on_index'])
            on_idx.sort()
            n_trials = len(on_idx)

            # loop through trials and get spike train array for each
            spike_train = []
            cond_array = np.zeros(n_trials)
            laser_start = np.zeros(n_trials)
            laser_single = np.zeros((n_trials, n_lasers))

            for ti, trial_off in enumerate(on_idx):
                if trial_off >= trial_cutoff_idx:
                    cond_array[ti] = -1
                    continue

                window = (trial_off - pre_idx, trial_off + post_idx)
                spike_shift = pre_idx - trial_off
                spike_array = np.zeros((n_units, n_pts))

                # loop through units
                for unit in hf5.root.sorted_units:
                    unit_num = int(unit._v_name[-3:])
                    spike_idx = np.where((unit.times[:] >= window[0]) &
                                         (unit.times[:] <= window[1]))[0]
                    spike_times = unit.times[spike_idx]

                    # Shift to align to trial window and convert to ms
                    spike_times = (spike_times + spike_shift) / (fs/1000)
                    spike_times = spike_times.astype(int)

                    #Drop spikes that come too late after adjustment
                    spike_times = [x for x in spike_times if x < n_pts]
                    spike_array[unit_num, spike_times] = 1

                spike_train.append(spike_array)

                if lasers:
                    # figure out which laser trial matches with this dig_in
                    # trial and get the duration and onset lag
                    for li, l in enumerate(laser_ch):
                        tmp_trial = laser_table.query('abs(off_index - '
                                                      '@trial_off) <= '
                                                      '@post_idx')
                        if not tmp_trial.empty:
                            # Mark which laser was on
                            laser_single[ti, li] = 1.0

                            # Get duration of laser
                            duration = (tmp_trial['off_index'] -
                                        tmp_trial['on_index']) / (fs/1000)
                            # round duration down to nearest multiple of 10ms
                            cond_array[ti] = 10*int(duration.iloc[0]/10)

                            # Get onset lag of laser, time between laser start
                            # and end of the trial
                            lag = (tmp_trial['on_index'] -
                                   trial_off) / (fs/1000)
                            # Round lag down to nearest multiple of 10ms
                            laser_start[ti] = 10*int(lag.iloc[0]/10)

            array_time = np.arange(-pre_stim, post_stim, 1)  # time array in ms
            hf5.create_group('/spike_trains', 'dig_in_%i' % i)
            time = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                    'array_time', array_time)
            tmp = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                   'spike_array', np.array(spike_train))
            hf5.flush()

            if lasers:
                ld = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                      'laser_durations', cond_array)
                ls = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                      'laser_onset_lag', laser_start)
                ol = hf5.create_array('/spike_trains/dig_in_%i' % i,
                                      'on_laser', laser_single)
                hf5.flush()
    print('Done with spike array creation!\n----------\n')


@jit(nogil=True)
def count_similar_spikes(unit1_times, unit2_times):
    '''Compiled function to compute the number of spikes in unit1 that are
    within 1ms of a spike in unit2

    Parameters
    ----------
    unit1_times : numpy.array, 1D array of unit times in ms
    unit2_times : numpy.array, 1D array of unit times in ms

    Returns
    -------
    int : number of spikes in unit1 within 1ms of a spike in unit2
    '''
    unit_counter = 0
    for t1 in unit1_times:
        diff_arr = np.abs(unit2_times - t1)
        if np.where(diff_arr <= 1.0)[0].size > 0:
            unit_counter += 1

    return unit_counter


def calc_units_similarity(h5_file, fs, similarity_cutoff=50,
                          violation_file=None):
    '''Creates an ixj similarity matrix with values being the percentage of
    spike in unit i within 1ms of spikes in unit j, and add it to the HDF5
    store
    Additionally, if this percentage is greater than the similarity cutoff then
    units are output to a text file of unit similarity violations

    Parameters
    ----------
    h5_file : str, full path to HDF5 store
    fs : float, sampling rate of data in Hz
    similarity_cutoff : float (optional)
        similarity cutoff percentage in % (0-100)
        default is 50
    violation_file : str (optional)
        full path to text file to write violations in default is
        unit_similarity_violations.txt saved in the same directory as the hf5

    Returns
    -------
    similarity_matrix : numpy.array
    '''
    print('\n---------\nBeginning unit similarity calculation\n----------')
    violations = 0
    violation_pairs = []
    if violation_file is None:
        violation_file = os.path.join(os.path.dirname(h5_file),
                                      'unit_similarity_violations.txt')

    with tables.open_file(h5_file, 'r+') as hf5:
        units = hf5.list_nodes('/sorted_units')
        unit_distances = np.zeros((len(units),
                                  len(units)))

        for pair in itertools.product(units, repeat=2):
            u1 = pair[0]
            u2 = pair[1]
            u1_idx = h5io.parse_unit_number(u1._v_name)
            u2_idx = h5io.parse_unit_number(u2._v_name)
            print('Computing similarity between Unit %i and Unit %i' %
                  (u1_idx, u2_idx))
            n_spikes = len(u1.times[:])
            u1_times = u1.times[:] / (fs/1000.0)
            u2_times = u2.times[:] / (fs/1000.0)
            n_similar = count_similar_spikes(u1_times, u2_times)
            tmp_dist = 100.0 * float(n_similar/n_spikes)
            unit_distances[u1_idx, u2_idx] = tmp_dist

            if u1_idx != u2_idx and tmp_dist >= similarity_cutoff:
                violations += 1
                violation_pairs.append((u1._v_name, u2._v_name))

        print('\nSimilarity calculation done!')
        if '/unit_distances' in hf5:
            hf5.remove_node('/', 'unit_distances')

        hf5.create_array('/', 'unit_distances', unit_distances)
        hf5.flush()

    if violations > 0:
        out_str = '%i units similarity violations found:\n' % violations
        out_str += 'Unit_1    Unit_2    Similarity\n'
        for x,y in violation_pairs:
            u1 = h5io.parse_unit_number(x)
            u2 = h5io.parse_unit_number(y)
            out_str += '   {:<10}{:<10}{}\n'.format(x, y,
                                                    unit_distances[u1][u2])

    else:
        out_str = 'No unit similarity violations found!'

    print(out_str)
    with open(violation_file, 'w') as vf:
        print(out_str, file=vf)

    return violation_pairs, unit_distances
