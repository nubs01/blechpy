import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from blechpy.analysis import spike_analysis as sas
from blechpy.datastructures.objects import load_dataset
from blechpy.dio import h5io
from blechpy.utils import print_tools as pt
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


def get_intra_J3(rec_dirs):
    print('\n----------\nComputing Intra J3s\n----------\n')
    # Go through each recording directory and compute intra_J3 array
    intra_J3 = []
    for rd in rec_dirs:
        print('Processing single units in %s...' % rd)
        unit_names = h5io.get_unit_names(rd)

        for un in unit_names:
            print('    Computing for %s...' % un)
            # waves, descrip, fs = get_unit_waveforms(rd, unit)
            waves, descrip, fs = h5io.get_raw_unit_waveforms(rd, un)
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


def find_held_units(rec_dirs, percent_criterion):
    # TODO: if any rec is 'one file per signal type' create tmp_raw.hdf5 and
    # delete after detection is finished 

    intra_J3 = get_intra_J3(rec_dirs)

    print('\n----------\nComputing Inter J3s\n----------\n')
    rec_pairs = [(rec_dirs[i], rec_dirs[i+1])
                 for i in range(len(rec_dirs)-1)]

    held_df = pd.DataFrame(columns=['unit',
                                    *[os.path.basename(x) for x in rec_dirs]])
    J3_df = held_df.copy()

    # Go through each pair of directories and computer inter_J3 between
    # units. If the inter_J3 values is below the percentile_criterion of
    # the intra_j3 array then mark units as held. Only compare the same
    # type of single units on the same electrode
    inter_J3 = []
    for rd1, rd2 in rec_pairs:
        h5_file1 = h5io.get_h5_filename(rd1)
        h5_file2 = h5io.get_h5_filename(rd2)
        rec1 = os.path.basename(rd1)
        rec2 = os.path.basename(rd2)
        print('Comparing %s vs %s' % (rec1, rec2))

        unit_names1 = h5io.get_unit_names(rd1)
        unit_names2 = h5io.get_unit_names(rd2)

        for unit1 in unit_names1:
            # wf1, descrip1, fs1 = get_unit_waveforms(rd1, unit1)
            wf1, descrip1, fs1 = h5io.get_raw_unit_waveforms(rd1, unit1)

            if descrip1['single_unit'] == 1:
                for unit2 in unit_names2:
                    # wf2, descrip2, fs2 = get_unit_waveforms(rd2, unit2)
                    wf2, descrip2, fs2 = \
                            h5io.get_raw_unit_waveforms(rd2, unit2,
                                                        required_descrip=descrip1)
                    if descrip1 == descrip2 and wf2 is not None:

                        print('Comparing %s %s vs %s %s' %
                              (rec1, unit1, rec2, unit2))

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
                            # Add unit to proper spot in Dataframe
                            if held_df.empty:
                                held_df = \
                                    held_df.append({'unit': 'A',
                                                    rec1: unit1,
                                                    rec2: unit2},
                                                   ignore_index=True)
                                J3_df = J3_df.append({'unit': 'A',
                                                      rec1: [J3],
                                                      rec2: [J3]},
                                                     ignore_index=True)

                                continue

                            idx1 = np.where(held_df[rec1] == unit1)[0]
                            idx2 = np.where(held_df[rec2] == unit2)[0]

                            if idx1.size == 0 and idx2.size == 0:
                                uL = held_df['unit'].iloc[-1]
                                uL = pt.get_next_letter(uL)
                                tmp = {'unit': uL,
                                       rec1: unit1,
                                       rec2: unit2}
                                held_df = held_df.append(
                                    tmp,
                                    ignore_index=True)
                                J3_df = J3_df.append({'unit': uL,
                                                      rec1: [J3],
                                                      rec2: [J3]},
                                                     ignore_index=True)
                            elif idx1.size != 0 and idx2.size != 0:
                                continue
                            elif idx1.size != 0:
                                held_df[rec2].iloc[idx1[0]] = unit2
                                J3_df[rec1].iloc[idx1[0]].append(J3)
                                J3_df[rec2].iloc[idx1[0]] = [J3]
                            else:
                                held_df[rec1].iloc[idx2[0]] = unit1
                                J3_df[rec2].iloc[idx2[0]].append(J3)
                                J3_df[rec1].iloc[idx2[0]] = [J3]

    return held_df, intra_J3, inter_J3, J3_df

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
    bin_time1, fr1 = get_binned_firing_rate(time1, spikes1, bin_size, bin_step)
    bin_time2, fr2 = get_binned_firing_rate(time2, spike2, bin_size, bin_step)

    if not np.array_equal(bin_time1, bin_time2):
        raise ValueError('Time of spike trains is not aligned')

    # Normalize firing rates
    if norm_func:
        fr1 = norm_func(bin_time1, fr1)
        fr2 = norm_fun(bin_time2, fr2)

    difference_of_mean, SEM = get_mean_difference(fr1, fr2, axis=0)

    return difference_of_mean, SEM, bin_time1





# For spike_train_analysis.py


def get_binned_firing_rate(time, spikes, bin_size=250, bin_step=25):
    '''Take a spike array and returns a firing rate array (row-wise)

    Parameters
    ----------
    time :  numpy.array, time vector in ms
    spikes : numpy.array, Trial x Time array with 1s at spike times
    bin_size: int (optional), bin width in ms, default=250
    bin_step : int (optional), step size in ms, default=25

    Returns
    -------
    bin_time : numpy.array
        time vector for binned firing rate array, times correspond to center of
        bins in ms
    firing_rate : numpy.array
        Trial x Time firing rate array in Hz
    '''
    bin_start = np.arange(time[0], time[-1] - bin_size + bin_step, bin_step)
    bin_time = bin_start + int(bin_size/2)
    n_trials = spikes.shape[0]
    n_bins = len(bin_start)

    firing_rate = np.zeros((n_trials, n_bins))
    for i, start in enumerate(bin_start):
        idx = np.where((time >= start) & (time <= start+bin_size))[0]
        firing_rate[:, i] = np.sum(spikes[:, idx], axis=1) / (bin_size/1000)

    return bin_time, firing_rate


def get_mean_difference(A, B, axis=0):
    '''Returns the difference of the means of arrays A and B along an axis and
    propogates the uncertainty of the means

    Parameters
    ----------
    A,B : numpy.array
    arrays to get difference between. arrays must be the same size along
    the axis being compared. For example, if A is MxN and B is LxN and
    axis=0 then they can be compared since axis 0 will be meaned and axis 1
    will be subtracted.
    axis : int, axis to be meaned

    Returns
    -------
    difference_of_means : numpy.array, 1D array
    SEM : numpy.array, standard error of the mean differences, 1D array
    '''
    shape_ax = int(not axis)

    m1 = np.mean(A, axis=axis)
    sd1 = np.std(A, axis=axis)
    n1 = A.shape[shape_ax]
    m2 = np.mean(B, axis=axis)
    sd2 = np.std(B, axis=axis)
    n2 = B.shape[shape_ax]
    C = m2 - m1
    SEM = np.sqrt((np.power(sd1, 2)/n1) + (np.power(sd2,2)/n2)) / \
           np.sqrt(n1+n2)

    return C, SEM



def zscore_to_baseline(time, fr):
    '''takes a firing rate array and zscores each row using the mean and st.
    dev over all trials during times < 0

    Parameters
    ----------
    time : numpy.array, 1D time vector
    fr : numpy.array, Trial x Time array of firing rates

    Returns
    -------
    norm_fr : numpy.array, array of firing rate traces
    '''
    idx = np.where(time < 0)[0]
    baselines = np.mean(fr[:, idx], axis=1)
    m = np.mean(baselines)
    sd = np.std(baselines)

    norm_fr = (fr - m) / sd

    return norm_fr


def remove_baseline(time, fr):
    '''takes a firing rate and substracts the group baseline mean from the each
    trials' firing rates

    Parameters
    ----------
    time : numpy.array, 1D time vector
    fr : numpy.array, Trial x Time array of firing rates

    Returns
    -------
    norm_fr : numpy.array, array of firing rate traces
    '''
    idx = np.where(time < 0)[0]
    baseline = np.mean(fr[:, idx])
    norm_fr = fr - baseline
    return norm_fr


