from blechpy.analysis import dataset
import os

def compute_MDS(exp):
    rec_dirs = exp.recording_dirs
    held_df = exp.held_units['units'].copy()
    rec_names = [os.path.basename(x) for x in rec_dirs]
    held_df = held_df.select_dtypes(include=['object'])

    # Only use units held over all sessions
    held_df = held_df.dropna(axis=0)

    early_window = [0, 750]
    late_window = [750, 1500]
    baseline_window = [-750, 0]

    # Make matrix with row for each taste/session and column for each neuron
    # with values being the average firing rate (minus baseline) of the neuron in the given time
    # window (early or late, matrix for each)

    # make symmetric distance matrix of the distances between rows
    # sklearn.metrics.pariwise.euclidean_distances
    # confirm its symmetric np.allclose(a, a.T)
    #




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
    dat1 = dataset.load_dataset(rec1)
    dat2 = dataset.load_dataset(rec2)

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


