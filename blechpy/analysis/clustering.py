import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import pylab as plt
from sklearn.decomposition import PCA

def get_filtered_electrode(data, freq = [300.0, 3000.0], sampling_rate = 30000.0):
    el = data
    m, n = butter(2, [2.0*freq[0]/sampling_rate, 2.0*freq[1]/sampling_rate], btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

def extract_waveforms(filt_el, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
    m = np.mean(filt_el)
    th = 5.0*np.median(np.abs(filt_el)/0.6745)
    pos = np.where(filt_el <= m-th)[0]
    changes = []
    for i in range(len(pos)-1):
        if pos[i+1] - pos[i] > 1:
            changes.append(i+1)

    # slices = np.zeros((len(changes)-1,150))
    slices = []
    spike_times = []
    for i in range(len(changes) - 1):
        minimum = np.where(filt_el[pos[changes[i]:changes[i+1]]] \
                == np.min(filt_el[pos[changes[i]:changes[i+1]]]))[0]
        #print minimum, len(slices), len(changes), len(filt_el)
        # try slicing out the putative waveform, only do this if there are 10ms
        # of data points (waveform is not too close to the start or end of the
        # recording)
        if pos[minimum[0]+changes[i]] - int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0)) \
                > 0 and pos[minimum[0]+changes[i]] + \
                int((spike_snapshot[1] + 0.1)*(sampling_rate/1000.0)) < len(filt_el):
            slices.append(filt_el[pos[minimum[0]+changes[i]] - int((spike_snapshot[0] + 0.1) \
                    *(sampling_rate/1000.0)) : pos[minimum[0]+changes[i]] + \
                    int((spike_snapshot[1] + 0.1)*(sampling_rate/1000.0))])
            spike_times.append(pos[minimum[0]+changes[i]])

    return np.array(slices), spike_times

def dejitter(slices, spike_times, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
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

def scale_waveforms(slices_dejittered):
    energy = np.sqrt(np.sum(slices_dejittered**2, axis = 1))/len(slices_dejittered[0])
    scaled_slices = np.zeros((len(slices_dejittered),len(slices_dejittered[0])))
    for i in range(len(slices_dejittered)):
        scaled_slices[i] = slices_dejittered[i]/energy[i]

    return scaled_slices, energy

def implement_pca(scaled_slices):
    pca = PCA()
    pca_slices = pca.fit_transform(scaled_slices)    
    return pca_slices, pca.explained_variance_ratio_

def clusterGMM(data, n_clusters, n_iter, restarts, threshold):
    g = []
    bayesian = []

    for i in range(restarts):
        g.append(GaussianMixture(n_components = n_clusters, covariance_type = 'full', 
            tol = threshold, random_state = i, max_iter = n_iter))
        g[-1].fit(data)
        if g[-1].converged_:
            bayesian.append(g[-1].bic(data))
        else:
            del g[-1]

    #print len(akaike)
    bayesian = np.array(bayesian)
    best_fit = np.where(bayesian == np.min(bayesian))[0][0]
    
    predictions = g[best_fit].predict(data)

    return g[best_fit], predictions, np.min(bayesian)
