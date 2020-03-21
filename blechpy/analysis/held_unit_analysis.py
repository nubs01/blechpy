import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from blechpy.analysis import spike_analysis as sas
from blechpy.datastructures.objects import load_dataset
from blechpy.dio import h5io
from blechpy.utils import print_tools as pt, userIO
import os


def calc_J1(wf_day1, wf_day2):
    # Get the mean PCA waveforms on days 1 and 2
    day1_mean = np.mean(wf_day1, axis=0)
    day2_mean = np.mean(wf_day2, axis=0)

    # Get the Euclidean distances of each day from its daily mean
    day1_dists = cdist(wf_day1, day1_mean.reshape((-1, 3)), metric='euclidean')
    day2_dists = cdist(wf_day2, day2_mean.reshape((-1, 3)), metric='euclidean')

    # Sum up the distances to get J1
    J1 = np.sum(day1_dists) + np.sum(day2_dists)
    return J1


def calc_J2(wf_day1, wf_day2):
    # Get the mean PCA waveforms on days 1 and 2
    day1_mean = np.mean(wf_day1, axis=0)
    day2_mean = np.mean(wf_day2, axis=0)

    # Get the overall inter-day mean
    overall_mean = np.mean(np.concatenate((wf_day1, wf_day2), axis=0), axis=0)

    # Get the distances of the daily means from the inter-day mean
    dist1 = cdist(day1_mean.reshape((-1, 3)), overall_mean.reshape((-1, 3)))
    dist2 = cdist(day2_mean.reshape((-1, 3)), overall_mean.reshape((-1, 3)))

    # Multiply the distances by the number of points on both days and sum to
    # get J2
    J2 = wf_day1.shape[0]*np.sum(dist1) + wf_day2.shape[0]*np.sum(dist2)
    return J2


def calc_J3(wf_day1, wf_day2):
    '''Calculate J3 value between 2 sets of PCA waveforms

    Parameters
    ----------
    wf_day1 : numpy.array
        PCA waveforms for a single unit from session 1
    wf_day2 : numpy.array
        PCA waveforms for a single unit from session 2

    Returns
    -------
    J3 : float
    '''
    J1 = calc_J1(wf_day1, wf_day2)
    J2 = calc_J2(wf_day1, wf_day2)
    J3 = J2 / J1
    return J3


def get_intra_J3(rec_dirs, raw_waves=False):
    print('\n----------\nComputing Intra J3s\n----------\n')
    # Go through each recording directory and compute intra_J3 array
    intra_J3 = []
    for rd in rec_dirs:
        print('Processing single units in %s...' % rd)
        unit_names = h5io.get_unit_names(rd)

        for un in unit_names:
            print('    Computing for %s...' % un)
            if raw_waves:
                waves, descrip, fs = h5io.get_raw_unit_waveforms(rd, un)
            else:
                waves, descrip, fs = h5io.get_unit_waveforms(rd, un)

            if descrip['single_unit'] == 1:
                pca = PCA(n_components=3)
                pca.fit(waves)
                pca_waves = pca.transform(waves)
                idx1 = int(waves.shape[0] * (1.0 / 3.0))
                idx2 = int(waves.shape[0] * (2.0 / 3.0))

                tmp_J3 = calc_J3(pca_waves[:idx1, :],
                                 pca_waves[idx2:, :])
                intra_J3.append(tmp_J3)

    print('Done!\n==========')
    return intra_J3


def find_held_units(rec_dirs, percent_criterion=95, rec_names=None, raw_waves=False):
    # TODO: if any rec is 'one file per signal type' create tmp_raw.hdf5 and
    # delete after detection is finished 

    userIO.tell_user('Computing intra recording J3 values...', shell=True)
    intra_J3 = get_intra_J3(rec_dirs)
    if rec_names is None:
        rec_names = [os.path.basename(x) for x in rec_dirs]

    rec_labels = {x: y for x, y in zip(rec_names, rec_dirs)}

    print('\n----------\nComputing Inter J3s\n----------\n')
    rec_pairs = [(rec_names[i], rec_names[i+1])
                 for i in range(len(rec_names)-1)]

    held_df = pd.DataFrame(columns=['unit', 'electrode', 'single_unit',
                                    'unit_type', *rec_names, 'J3'])

    # Go through each pair of directories and computer inter_J3 between
    # units. If the inter_J3 values is below the percentile_criterion of
    # the intra_j3 array then mark units as held. Only compare the same
    # type of single units on the same electrode
    inter_J3 = []
    for rec1, rec2 in rec_pairs:
        rd1 = rec_labels.get(rec1)
        rd2 = rec_labels.get(rec2)
        h5_file1 = h5io.get_h5_filename(rd1)
        h5_file2 = h5io.get_h5_filename(rd2)
        print('Comparing %s vs %s' % (rec1, rec2))
        found_cells = []

        unit_names1 = h5io.get_unit_names(rd1)
        unit_names2 = h5io.get_unit_names(rd2)

        for unit1 in unit_names1:
            if raw_waves:
                wf1, descrip1, fs1 = h5io.get_raw_unit_waveforms(rd1, unit1)
            else:
                wf1, descrip1, fs1 = h5io.get_unit_waveforms(rd1, unit1)

            electrode = descrip1['electrode_number']
            single_unit = bool(descrip1['single_unit'])
            unit_type = h5io.read_unit_description(descrip1)

            if descrip1['single_unit'] == 1:
                for unit2 in unit_names2:
                    if raw_waves:
                        wf2, descrip2, fs2 = \
                                h5io.get_raw_unit_waveforms(rd2, unit2,
                                                            required_descrip=descrip1)
                    else:
                        wf2, descrip2, fs2 = h5io.get_unit_waveforms(rd2, unit2,
                                                                     required_descrip=descrip1)

                    if descrip1 == descrip2 and wf2 is not None:

                        print('Comparing %s %s vs %s %s' %
                              (rec1, unit1, rec2, unit2))
                        userIO.tell_user('Comparing %s %s vs %s %s' %
                                         (rec1, unit1, rec2, unit2), shell=True)

                        if fs1 > fs2:
                            wf1 = sas.interpolate_waves(wf1, fs1,
                                                        fs2)
                        elif fs1 < fs2:
                            wf2 = sas.interpolate_waves(wf2, fs2,
                                                        fs1)

                        pca = PCA(n_components=3)
                        pca.fit(np.concatenate((wf1, wf2), axis=0))
                        pca_wf1 = pca.transform(wf1)
                        pca_wf2 = pca.transform(wf2)

                        J3 = calc_J3(pca_wf1, pca_wf2)
                        inter_J3.append(J3)

                        if J3 <= np.percentile(intra_J3,
                                               percent_criterion):
                            print('Detected held unit:\n    %s %s and %s %s'
                                  % (rec1, unit1, rec2, unit2))
                            userIO.tell_user('Detected held unit:\n    %s %s and %s %s'
                                             % (rec1, unit1, rec2, unit2), shell=True)
                            found_cells.append((h5io.parse_unit_number(unit1),
                                                h5io.parse_unit_number(unit2),
                                                J3, single_unit, unit_type))

        found_cells = np.array(found_cells)
        userIO.tell_user('\n-----\n%s vs %s\n-----' % (rec1, rec2), shell=True)
        userIO.tell_user(str(found_cells)+'\n', shell=True)
        userIO.tell_user('Resolving duplicates...', shell=True)
        found_cells = resolve_duplicate_matches(found_cells)
        userIO.tell_user('Results:\n%s\n' % str(found_cells), shell=True)
        for i, row in enumerate(found_cells):
            if held_df.empty:
                uL = 'A'
            else:
                uL = held_df['unit'].iloc[-1]
                uL = pt.get_next_letter(uL)

            unit1 = 'unit%03d' % int(row[0])
            unit2 = 'unit%03d' % int(row[1])
            j3 = row[2]
            idx1 = np.where(held_df[rec1] == unit1)[0]
            idx2 = np.where(held_df[rec2] == unit2)[0]
            if row[3] == 'True':
                single_unit = True
            else:
                single_unit = False

            if idx1.size == 0 and idx2.size == 0:
                tmp = {'unit': uL,
                       'single_unit': single_unit,
                       'unit_type': row[4],
                       rec1: unit1,
                       rec2: unit2,
                       'J3': [float(j3)]}
                held_df = held_df.append(tmp, ignore_index=True)
            elif idx1.size != 0 and idx2.size != 0:
                userIO.tell_user('WTF...', shell=True)
                continue
            elif idx1.size != 0:
                held_df[rec2].iloc[idx1[0]] = unit2
                held_df['J3'].iloc[idx1[0]].append(float(j3))
            else:
                held_df[rec1].iloc[idx2[0]] = unit1
                held_df['J3'].iloc[idx2[0]].append(float(j3))

    return held_df, intra_J3, inter_J3

def resolve_duplicate_matches(found_cells):
    if len(found_cells) == 0:
        return found_cells

    unique_units = np.unique(found_cells[:,0])
    new_found = []
    for unit in unique_units:
        idx = np.where(found_cells[:,0] == unit)[0]
        if len(idx) == 1:
            new_found.append(found_cells[idx,:])
            continue

        min_j3 = np.argmin(found_cells[idx,2])
        new_found.append(found_cells[idx[min_j3],:])

    found = np.vstack(new_found)
    go_back = []
    new_found = []
    for unit in np.unique(found[:,1]):
        idx = np.where(found[:,1] == unit)[0]
        if len(idx) == 1:
            new_found.append(found[idx,:])
            continue

        min_j3 = np.argmin(found[idx,2])
        i = idx[min_j3]
        idx = np.delete(idx, min_j3)
        new_found.append(found[i, :])
        go_back.append(found[idx, :])

    for row in go_back:
        idx = np.where((found_cells[:,0] == row[0][0]) & (found_cells[:,1] != row[0][1]))[0]
        if len(idx) == 1:
            new_found.append(found_cells[idx,:])
            continue
        elif len(idx) == 0:
            continue

        min_j3 = np.argmin(found_cells[idx, 2])
        new_found.append(found_cells[idx[min_j3],:])

    out = np.vstack(new_found)
    uni = True
    for unit in np.unique(out[:,0]):
        idx = np.where(out[:,0] == unit)[0]
        if len(idx) > 1:
            uni = False
            break

    for unit in np.unique(out[:,1]):
        idx = np.where(out[:,1] == unit)[0]
        if len(idx) > 1:
            uni = False
            break

    # Sort
    a = [int(x) for x in out[:,0]]
    idx = np.argsort(a)
    out = out[idx,:]

    if uni:
        return out
    else:
        print('Duplicates still found. Re-running')
        print(out)
        return resolve_duplicate_matches(out)

### Delete after here

def get_response_change(unit_name, rec1, unit1,
                        din1, rec2, unit2, din2,
                        bin_size=250, bin_step=25, norm_func=None):
    '''Uses the spike arrays to compute the change in
    firing rate of the response to the tastant.

    Parameters
    ----------
    unit_name : str, name of held unit
    rec1 : str, path to recording directory 1
    unit1: str, name of unit in rec1
    din1 : int, number of din to use from rec1
    rec2 : str, path to recording directory 2
    unit2: str, name of unit in rec2
    din2 : int, number of din to use from rec2
    bin_size : int, default=250
        width of bins in units of time vector saved in hf5 spike_trains
        usually ms
    bin_step : int, default=25
        step size to take from one bin to the next in same units (usually ms)
    norm_func: function (optional)
        function with which to normalize the firing rates before getting difference
        must take inputs (time_vector, firing_rate_array) where time_vector is
        1D numpy.array and firing_rate_array is a Trial x Time numpy.array
        Must return a numpy.array with same size as firing rate array

    Returns
    -------
    difference_of_means : numpy.array
    SEM : numpy.array, standard error of the mean difference
    '''
    # Get metadata
    dat1 = load_dataset(rec1)
    dat2 = load_dataset(rec2)

    # Get data from hf5 files
    time1, spikes1 = dio.h5io.get_spike_data(rec1, unit1, din1)
    time2, spikes2 = dio.h5io.get_spike_data(rec2, unit2, din2)

    # Get Firing Rates
    bin_time1, fr1 = sas.get_binned_firing_rate(time1, spikes1, bin_size, bin_step)
    bin_time2, fr2 = sas.get_binned_firing_rate(time2, spike2, bin_size, bin_step)

    if not np.array_equal(bin_time1, bin_time2):
        raise ValueError('Time of spike trains is not aligned')

    # Normalize firing rates
    if norm_func:
        fr1 = norm_func(bin_time1, fr1)
        fr2 = norm_fun(bin_time2, fr2)

    difference_of_mean, SEM = sas.get_mean_difference(fr1, fr2, axis=0)

    return difference_of_mean, SEM, bin_time1




