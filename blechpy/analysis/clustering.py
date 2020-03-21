import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.interpolate import interp1d


def get_filtered_electrode(data, freq = [300.0, 3000.0], sampling_rate = 30000.0):
    el = data
    m, n = butter(2, [2.0*freq[0]/sampling_rate, 2.0*freq[1]/sampling_rate], btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el


def dejitter(slices, spike_times, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
    '''Upsamples (by 10) and aligns spike waveforms to minima. Returns the
    upsampled waveforms are correct spike_times
    '''
    x = np.arange(0,len(slices[0]),1)
    xnew = np.arange(0,len(slices[0])-1,0.1)

    # Calculate the number of samples to be sliced out around each spike's minimum
    before = int((sampling_rate/1000.0)*(spike_snapshot[0]))
    after = int((sampling_rate/1000.0)*(spike_snapshot[1]))

    slices_dejittered = []
    spike_times_dejittered = []
    for i in range(len(slices)):
        f = interp1d(x, slices[i])
        # 10-fold interpolated spike
        ynew = f(xnew)
        orig_min = np.where(slices[i] == np.min(slices[i]))[0][0]
        orig_min_time = x[orig_min] / (sampling_rate/1000)
        minimum = np.where(ynew == np.min(ynew))[0][0]
        min_time = xnew[minimum] / (sampling_rate/1000)
        # Only accept spikes if the interpolated minimum has shifted by
        # less than 1/10th of a ms (3 samples for a 30kHz recording, 30
        # samples after interpolation)
        if np.abs(min_time - orig_min_time) <= 0.1:
            # If minimum is too close to the end for a full snapshot then toss out spike
            if minimum + after*10 < len(ynew) and minimum - before*10 >= 0:
                slices_dejittered.append(ynew[minimum - before*10 : minimum + after*10])
                spike_times_dejittered.append(spike_times[i])

    return np.array(slices_dejittered), np.array(spike_times_dejittered)


def get_waveforms(el_trace, spike_times, snapshot = [0.5, 1.0],
                  sampling_rate = 30000.0, bandpass=[300, 3000]):
    '''Returns waveform slices based on the given spike_times (in samples)
    '''
    # Filter and extract waveforms
    filt_el = get_filtered_electrode(el_trace, freq=bandpass,
                                     sampling_rate=sampling_rate)
    del el_trace
    pre_pts = int((snapshot[0]+0.1) * (sampling_rate/1000))
    post_pts = int((snapshot[1]+0.2) * (sampling_rate/1000))
    slices = np.zeros((spike_times.shape[0], pre_pts+post_pts))
    for i, st in enumerate(spike_times):
        slices[i, :] = filt_el[st - pre_pts: st + post_pts]

    slices_dj, times_dj = dejitter(slices, spike_times, snapshot, sampling_rate)

    return slices_dj, sampling_rate*10
