import numpy as np
import tables
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d


def interpolate_waves(waves, fs, fs_new, axis=1):
    end_time = waves.shape[axis] / (fs/1000)
    x = np.arange(0, end_time, 1/(fs/1000))
    x_new = np.arange(0, end_time, 1/(fs_new/1000))
    f = interp1d(x, waves, axis=axis)
    return f(x_new)


def make_single_trial_psth(spike_train, win_size, win_step, time=None):
    '''Takes a spike train and returns firing rate trace in Hz

    Parameters
    ----------
    spike_train : 1D numpy.array
        spike train with 1s in bins with spikes and 0s elsewhere
    win_size : float, window size of psth in ms
    win_step : float, step size of psth in ms
    time : numpy.array (optional)
        time array with times corresponding to bins in spike_train
        if not provided then on is created starting at 0 and assuming 1ms bins

    Returns
    -------
    psth : numpy.array, firing rate vector with units of Hz
    psth_time: numpy.array, time vector corresponding to the psth
    '''
    if time is None:
        time = np.arange(0, len(spike_train), 1)  # assume 1ms bins

    psth_time = np.arange(np.min(time) + (win_size/2),
                          np.max(time) - (win_size/2),
                          win_step)
    psth = np.zeros(psth_time.shape)
    window = np.array([-win_size/2, win_size/2])

    for i, t in enumerate(psth_time):
        t_win = t + window
        idx = np.where((time >= t_win[0]) & (time <= t_win[1]))[0]
        psth[i] = np.sum(spike_train[idx]) / (win_size/1000.0)  # in Hz

    return psth, psth_time


def make_mean_PSTHs(h5_file, win_size, win_step, dig_in_ch):

    with tables.open_file(h5_file, 'r') as hf5:
        spike_data = hf5.root.spike_trains['dig_in_%i' % dig_in_ch]
        spike_array = spike_data.spike_array[:]
        time = spike_data.array_time[:]

        psth_time = np.arange(np.min(time) - (win_size/2),
                              np.max(time) + (win_size/2),
                              win_step)
        PSTHs = np.zeros((len(psth_time), spike_array.shape[1]))

        for trial in spike_array:
            for i, unit in enumerate(trial):
                tmp, tmp_time = make_single_trial_psth(unit, win_size,
                                                       win_step, time)
                PSTHs[:, i] += tmp
        PSTHs /= spike_array.shape[0]

    return PSTHs, psth_time


def make_psths_for_tastant(h5_file, win_size, win_step, dig_in_ch, smoothing_width=3):
    dig_str = 'dig_in_%i' % dig_in_ch
    with tables.open_file(h5_file, 'r+') as hf5:
        spike_data = hf5.root.spike_trains[dig_str]
        spike_array = spike_data.spike_array[:]
        time = spike_data.array_time[:]

        psth_time = None
        PSTHs = None

        for ti, trial in enumerate(spike_array):
            for ui, unit in enumerate(trial):
                tmp, tmp_time = make_single_trial_psth(unit, win_size,
                                                       win_step, time)
                if psth_time is None:
                    psth_time = tmp_time
                    PSTHs = np.zeros((spike_array.shape[1],
                                     spike_array.shape[0],
                                     len(psth_time)))

                # Smooth firing rate trace
                tmp = gaussian_filter1d(tmp, sigma=smoothing_width)

                PSTHs[ui, ti, :] = tmp

        if '/PSTHs' not in hf5:
            hf5.create_group('/', 'PSTHs')

        if '/PSTHs/%s' % dig_str in hf5:
            hf5.remove_node('/PSTHs', dig_str, recursive=True)

        hf5.create_group('/PSTHs', dig_str)
        hf5.create_array('/PSTHs/%s' % dig_str, 'time', psth_time)
        hf5.create_array('/PSTHs/%s' % dig_str, 'psth_array', PSTHs)
        hf5.create_array('/PSTHs/%s' % dig_str, 'mean_psths',
                         np.mean(PSTHs, axis=1))
        hf5.flush()

    return PSTHs, psth_time


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



