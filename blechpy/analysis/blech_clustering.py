import os
import shutil
import numpy as np
import pandas as pd
import itertools as it
import umap
import pywt
from statsmodels.stats.diagnostic import lilliefors
from copy import deepcopy
from scipy.spatial.distance import mahalanobis
from scipy import linalg
from scipy.signal import find_peaks
from scipy.stats import sem
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from blechpy.utils import write_tools as wt, print_tools as pt, math_tools as mt, userIO
from blechpy.dio import h5io
from blechpy.analysis import clustering, spike_analysis as sas
from blechpy.plotting import data_plot as dplt
import datetime as dt


def detect_spikes(filt_el, spike_snapshot = [0.5, 1.0], fs = 30000.0):
    '''Detects spikes in the filtered electrode trace and return the waveforms
    and spike_times

    Parameters
    ----------
    filt_el : np.array, 1-D
        filtered electrode trace
    spike_snapshot : list
        2-elements, [ms before spike minimum, ms after spike minimum] 
        time around spike to snap as waveform
    fs : float, sampling rate in Hz

    Returns
    -------
    waves : np.array
        matrix of de-jittered, spike waveforms, upsampled by 10x, row for each spike
    times : np.array
        array of spike times in samples
    threshold: float
        spike detection threshold
    '''
    # get indices of spike snapshot, expand by .1 ms in each direction
    snapshot = np.arange(-(spike_snapshot[0]+0.1)*fs/1000,
                         1+(spike_snapshot[1]+0.1)*fs/1000).astype('int64')
    m = np.mean(filt_el)
    th = 5.0*np.median(np.abs(filt_el)/0.6745)
    pos = np.where(filt_el <= m-th)[0]
    consecutive = mt.group_consecutives(pos)

    waves = []
    times = []
    for idx in consecutive:
        minimum = idx[np.argmin(filt_el[idx])]
        spike_idx = minimum + snapshot
        if spike_idx[0] >= 0 and spike_idx[-1] < len(filt_el):
            waves.append(filt_el[spike_idx])
            times.append(minimum)

    if len(waves) == 0:
        return None, None

    waves_dj, times_dj = clustering.dejitter(np.array(waves), np.array(times), spike_snapshot, fs)
    return waves_dj, times_dj, m-th


def implement_pca(scaled_slices):
    pca = PCA()
    pca_slices = pca.fit_transform(scaled_slices)
    return pca_slices, pca.explained_variance_ratio_


def implement_umap(waves, n_pc=3, n_neighbors=30, min_dist=0.0):
    reducer = umap.UMAP(n_components=n_pc,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist)
    return reducer.fit_transform(waves)


def implement_wavelet_transform(waves, n_pc=10):
    coeffs = pywt.wavedec(waves, 'haar', axis=1)
    all_coeffs = np.column_stack(coeffs)
    k_stats = np.zeros((all_coeffs.shape[1],))
    p_vals = np.ones((all_coeffs.shape[1],))
    for i, c in enumerate(all_coeffs.T):
        k_stats[i], p_vals[i] = lilliefors(c, dist='norm')

    idx = np.argsort(p_vals)
    return all_coeffs[:, idx[:n_pc]]


def compute_waveform_metrics(waves, n_pc=3, umap=False):
    '''Make clustering data array with columns:
         - amplitudes, energy, slope, pc1, pc2, pc3, etc
    Parameters
    ----------
    waves : np.array
        waveforms with a row for each spike waveform
    n_pc : int (optional)
        number of principal components to include in data array

    Returns
    -------
    np.array
    '''
    data = np.zeros((waves.shape[0], 3))
    for i, wave in enumerate(waves):
        data[i,0] = np.min(wave)
        data[i,1] = np.sqrt(np.sum(wave**2))/len(wave)
        peaks = find_peaks(wave)[0]
        minima = np.argmin(wave)
        if not any(peaks < minima):
            maxima = np.argmax(wave[:minima])
        else:
            maxima = max(peaks[np.where(peaks < minima)[0]])

        data[i,2] = (wave[minima]-wave[maxima])/(minima-maxima)

    # Scale waveforms to energy before running PCA
    if umap:
        pc_waves = implement_umap(waves, n_pc=n_pc)
    else:
        scaled_waves = scale_waveforms(waves, energy=data[:,1])
        pc_waves, _ = implement_pca(scaled_waves)

    data = np.hstack((data, pc_waves[:,:n_pc]))
    data_columns = ['amplitude', 'energy', 'spike_slope']
    data_columns.extend(['PC%i' % i for i in range(n_pc)])
    return data, data_columns


def get_waveform_amplitudes(waves):
    '''Returns array of waveform amplitudes

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    '''
    return np.min(waves,axis = 1)


def get_waveform_energy(waves):
    '''Returns array of waveform energies

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    '''
    energy = np.sqrt(np.sum(waves**2, axis=1))/waves.shape[1]
    return energy


def get_spike_slopes(waves):
    '''Returns array of spike slopes (initial downward slope of spike)

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    '''
    slopes = np.zeros((waves.shape[0],))
    for i, wave in enumerate(waves):
        peaks = find_peaks(wave)[0]
        minima = np.argmin(wave)
        if not any(peaks < minima):
            maxima = np.argmax(wave[:minima])
        else:
            maxima = max(peaks[np.where(peaks < minima)[0]])

        slopes[i] = (wave[minima]-wave[maxima])/(minima-maxima)

    return slopes


def get_ISI_and_violations(spike_times, fs, rec_map=None):
    '''returns array of ISIs in ms and # of 1ms and 2ms violations

    Parameters
    ----------
    spike_time  numpy.array
    fs : float, sampling rate in Hz
    rec_map : np.array (optional)
        if not passed, it is assumed all spike times are from same recording
        if passed, spike times are split into recordings and ISIs are computed
        per recording.
        If fs is different for each recording, fs should be a dict with keys as
        rec ids in rec_map

    Returns
    -------
    np.array : ISIs
    int : 1ms violations
    int : 2ms violations
    '''
    if rec_map is not None:
        if not isinstance(fs, dict):
            fs = dict.fromkeys(np.unique(rec_map), fs)

        ISIs = np.array([])
        violations1 = 0
        violations2 = 0
        for i in np.unique(rec_map):
            idx = np.where(rec_map == i)[0]
            tmp_isi, v1, v2 = get_ISI_and_violations(spike_times[idx], fs[i])
            violations1 += v1
            violations2 += v2
            ISIs = np.concatenate((ISIs, tmp_isi))

    else:
        fs = float(fs/1000.0)
        ISIs = np.ediff1d(np.sort(spike_times))/fs
        violations1 = np.sum(ISIs < 1.0)
        violations2 = np.sum(ISIs < 2.0)

    return ISIs, violations1, violations2


def scale_waveforms(waves, energy=None):
    '''Scales each waveform to its own energy

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike
    energy : np.array (optional)
        array of waveform energies, saves computation time

    Returns
    -------
    np.array
    '''
    if energy is None:
        energy = get_waveform_energy(waves)
    elif len(energy) != waves.shape[0]:
        raise ValueError(('Energies must correspond to each waveforms.'
                          'Different lengths are not allowed'))

    scaled_slices = np.zeros(waves.shape)
    for i, w in enumerate(zip(waves, energy)):
        scaled_slices[i] = w[0]/w[1]

    return scaled_slices


def get_mahalanobis_distances_to_cluster(data, model, clusters, target_cluster):
    '''computes mahalanobis distance from spikes in target_cluster to all clusters
    in GMM model

    Parameters
    ----------
    data : np.array, data used to train GMM
    model : fitted GMM model
    clusters : np.array, maps data points to clusters
    target_cluster : int, cluster for which to compute distances

    Returns
    -------
    np.array
    '''
    unique_clusters = np.unique(abs(clusters))
    out_distances = dict.fromkeys(unique_clusters)
    cluster_idx = np.where(clusters == target_cluster)[0]
    for other_cluster in unique_clusters:
        mahalanobis_dist = np.zeros((len(cluster_idx),))
        other_cluster_mean = model.means_[other_cluster, :]
        other_cluster_covar_I = linalg.inv(model.covariances_[other_cluster, :, :])
        for i, idx in enumerate(cluster_idx):
            mahalanobis_dist[i] = mahalanobis(data[idx, :],
                                              other_cluster_mean,
                                              other_cluster_covar_I)

        out_distances[other_cluster] = mahalanobis_dist

    return out_distances


def get_recording_cutoff(filt_el, sampling_rate, voltage_cutoff,
                         max_breach_rate, max_secs_above_cutoff,
                         max_mean_breach_rate_persec, **kwargs):
    breach_idx = np.where(filt_el > voltage_cutoff)[0]
    breach_rate = float(len(breach_idx)*int(sampling_rate))/len(filt_el)
    # truncate to nearest second and make 1 sec bins
    filt_el = filt_el[:int(sampling_rate)*int(len(filt_el)/sampling_rate)]
    test_el = np.reshape(filt_el, (-1, int(sampling_rate)))
    breaches_per_sec = [len(np.where(test_el[i] > voltage_cutoff)[0])
                        for i in range(len(test_el))]
    breaches_per_sec = np.array(breaches_per_sec)
    secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
    if secs_above_cutoff == 0:
        mean_breach_rate_persec = 0
    else:
        mean_breach_rate_persec = np.mean(breaches_per_sec[np.where(breaches_per_sec > 0)[0]])

    # And if they all exceed the cutoffs, assume that the headstage fell off mid-experiment
    recording_cutoff = int(len(filt_el)/sampling_rate) # cutoff in seconds
    if (breach_rate >= max_breach_rate and
        secs_above_cutoff >= max_secs_above_cutoff and
        mean_breach_rate_persec >= max_mean_breach_rate_persec):
        # Find the first 1 second epoch where the number of cutoff breaches is
        # higher than the maximum allowed mean breach rate 
        recording_cutoff = np.where(breaches_per_sec > max_mean_breach_rate_persec)[0][0]
        # cutoff is still in seconds since 1 sec bins

    return recording_cutoff


def UMAP_METRICS(waves, n_pc):
    return compute_waveform_metrics(waves, n_pc, umap=True)


class SpikeDetection(object):
    '''Interface to manage spike detection and data extraction in preparation
    for GMM clustering. Intended to help create and access the neccessary
    files. If object will detect is file already exist to avoid re-creation
    unless overwrite is specified as True.
    '''

    def __init__(self, file_dir, electrode, params=None, overwrite=False):
        # Setup paths to files and directories needed
        self._file_dir = file_dir
        self._electrode = electrode
        self._out_dir = os.path.join(file_dir, 'spike_detection',
                                     'electrode_%i' % electrode)
        self._data_dir = os.path.join(self._out_dir, 'data')
        self._plot_dir = os.path.join(self._out_dir, 'plots')
        self._files = {'params': os.path.join(file_dir,'analysis_params', 'spike_detection_params.json'),
                       'spike_waveforms': os.path.join(self._data_dir, 'spike_waveforms.npy'),
                       'spike_times' : os.path.join(self._data_dir, 'spike_times.npy'),
                       'energy' : os.path.join(self._data_dir, 'energy.npy'),
                       'spike_amplitudes' : os.path.join(self._data_dir, 'spike_amplitudes.npy'),
                       'pca_waveforms' : os.path.join(self._data_dir, 'pca_waveforms.npy'),
                       'slopes' : os.path.join(self._data_dir, 'spike_slopes.npy'),
                       'recording_cutoff' : os.path.join(self._data_dir, 'cutoff_time.txt'),
                       'detection_threshold' : os.path.join(self._data_dir, 'detection_threshold.txt')}

        self._status = dict.fromkeys(self._files.keys(), False)
        self._referenced = True

        # Delete existing data if overwrite is True
        if overwrite and os.path.isdir(self._out_dir):
            shutil.rmtree(self._out_dir)

        # See what data already exists
        self._check_existing_files()

        # Make directories if needed
        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)

        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)

        if not os.path.isdir(self._plot_dir):
            os.makedirs(self._plot_dir)

        if not os.path.isdir(os.path.join(file_dir, 'analysis_params')):
            os.makedirs(os.path.join(file_dir, 'analysis_params'))

        # grab recording cutoff time if it already exists
        # cutoff should be in seconds
        self.recording_cutoff = None
        if os.path.isfile(self._files['recording_cutoff']):
            self._status['recording_cutoff'] = True
            with open(self._files['recording_cutoff'], 'r') as f:
                self.recording_cutoff = float(f.read())

        self.detection_threshold = None
        if os.path.isfile(self._files['detection_threshold']):
            self._status['detection_threshold'] = True
            with open(self._files['detection_threshold'], 'r') as f:
                self.detection_threshold = float(f.read())

        # Read in parameters
        # Parameters passed as an argument will overshadow parameters saved in file
        # Input parameters should be formatted as dataset.clustering_parameters
        if params is None and os.path.isfile(self._files['params']):
            self.params = wt.read_dict_from_json(self._files['params'])
        elif params is None:
            raise FileNotFoundError('params must be provided if spike_detection_params.json does not exist.')
        else:
            self.params = {}
            self.params['voltage_cutoff'] = params['data_params']['V_cutoff for disconnected headstage']
            self.params['max_breach_rate'] = params['data_params']['Max rate of cutoff breach per second']
            self.params['max_secs_above_cutoff'] = params['data_params']['Max allowed seconds with a breach']
            self.params['max_mean_breach_rate_persec'] = params['data_params']['Max allowed breaches per second']
            band_lower = params['bandpass_params']['Lower freq cutoff']
            band_upper = params['bandpass_params']['Upper freq cutoff']
            self.params['bandpass'] = [band_lower, band_upper]
            snapshot_pre = params['spike_snapshot']['Time before spike (ms)']
            snapshot_post = params['spike_snapshot']['Time after spike (ms)']
            self.params['spike_snapshot'] = [snapshot_pre, snapshot_post]
            self.params['sampling_rate'] = params['sampling_rate']
            # Write params to json file
            wt.write_dict_to_json(self.params, self._files['params'])
            self._status['params'] = True

    def _check_existing_files(self):
        '''Checks which files already exist and updates _status so as to avoid
        re-creation later
        '''
        for k, v in self._files.items():
            if os.path.isfile(v):
                self._status[k] = True
            else:
                self._status[k] = False

    def run(self):
        status = self._status
        file_dir = self._file_dir
        electrode = self._electrode
        params = self.params
        fs = params['sampling_rate']

        # Check if this even needs to be run
        if all(status.values()):
            return electrode, 1, self.recording_cutoff

        # Grab referenced electrode or raw if ref is not available
        ref_el = h5io.get_referenced_trace(file_dir, electrode)
        if ref_el is None:
            print('Could not find referenced data for electrode %i. Using raw.' % electrode)
            self._referenced = False
            ref_el = h5io.get_raw_trace(file_dir, electrode)
            if ref_el is None:
                raise KeyError('Neither referenced nor raw data found for electrode %i in %s' % (electrode, file_dir))

        # Filter electrode trace
        filt_el = clustering.get_filtered_electrode(ref_el, freq=params['bandpass'],
                                               sampling_rate = fs)
        del ref_el
        # Get recording cutoff
        if not status['recording_cutoff']:
            self.recording_cutoff = get_recording_cutoff(filt_el, **params)
            with open(self._files['recording_cutoff'], 'w') as f:
                f.write(str(self.recording_cutoff))

            status['recording_cutoff'] = True
            fn = os.path.join(self._plot_dir, 'cutoff_time.png')
            dplt.plot_recording_cutoff(filt_el, fs, self.recording_cutoff,
                                       out_file=fn)

        # Truncate electrode trace, deal with early cutoff (<60s)
        if self.recording_cutoff < 60:
            print('Immediate Cutoff for electrode %i...exiting' % electrode)
            return electrode, 0, self.recording_cutoff

        filt_el = filt_el[:int(self.recording_cutoff*fs)]

        if status['spike_waveforms'] and status['spike_times']:
            waves = np.load(self._files['spike_waveforms'])
            times = np.load(self._files['spike_times'])
        else:
            # Detect spikes and get dejittered times and waveforms
            # detect_spikes returns waveforms upsampled by 10x and times in units
            # of samples
            waves, times, threshold = detect_spikes(filt_el, params['spike_snapshot'], fs)
            self.detection_threshold = threshold
            if waves is None:
                print('No waveforms detected on electrode %i' % electrode)
                return electrode, 0, self.recording_cutoff

            # Save waveforms and times
            np.save(self._files['spike_waveforms'], waves)
            np.save(self._files['spike_times'], times)
            with open(self._files['detection_threshold'], 'w') as f:
                f.write(str(threshold))

            status['detection_threshold'] = True
            status['spike_waveforms'] = True
            status['spike_times'] = True

        # Get various metrics and scale waveforms
        if not status['spike_amplitudes']:
            amplitudes = get_waveform_amplitudes(waves)
            np.save(self._files['spike_amplitudes'], amplitudes)
            status['spike_amplitudes'] = True

        if not status['slopes']:
            slopes = get_spike_slopes(waves)
            np.save(self._files['slopes'], slopes)
            status['slopes'] = True

        if not status['energy']:
            energy = get_waveform_energy(waves)
            np.save(self._files['energy'], energy)
            status['energy'] = True
        else:
            energy=None

        # get pca of scaled waveforms
        if not status['pca_waveforms']:
            scaled_waves = scale_waveforms(waves, energy=energy)
            pca_waves, explained_variance_ratio = implement_pca(scaled_waves)

            # Plot explained variance
            fn = os.path.join(self._plot_dir, 'pca_variance.png')
            dplt.plot_explained_pca_variance(explained_variance_ratio,
                                             out_file = fn)

        return electrode, 1, self.recording_cutoff

    def get_spike_waveforms(self):
        '''Returns spike waveforms if they have been extracted, None otherwise
        Dejittered waveforms upsampled to 10 x sampling_rate

        Returns
        -------
        numpy.array
        '''
        if os.path.isfile(self._files['spike_waveforms']):
            return np.load(self._files['spike_waveforms'])
        else:
            return None

    def get_spike_times(self):
        '''Returns spike times if they have been extracted, None otherwise
        In units of samples.

        Returns
        -------
        numpy.array
        '''
        if os.path.isfile(self._files['spike_times']):
            return np.load(self._files['spike_times'])
        else:
            return None

    def get_energy(self):
        '''Returns spike energies if they have been extracted, None otherwise

        Returns
        -------
        numpy.array
        '''
        if os.path.isfile(self._files['energy']):
            return np.load(self._files['energy'])
        else:
            return None

    def get_spike_amplitudes(self):
        '''Returns spike amplitudes if they have been extracted, None otherwise

        Returns
        -------
        numpy.array
        '''
        if os.path.isfile(self._files['spike_amplitudes']):
            return np.load(self._files['spike_amplitudes'])
        else:
            return None

    def get_spike_slopes(self):
        '''Returns spike slopes if they have been extracted, None otherwise

        Returns
        -------
        numpy.array
        '''
        if os.path.isfile(self._files['slopes']):
            return np.load(self._files['slopes'])
        else:
            return None

    def get_pca_waveforms(self):
        '''Returns pca of sclaed spike waveforms if they have been extracted,
        None otherwise
        Dejittered waveforms upsampled to 10 x sampling_rate, scaled to energy
        and transformed via PCA

        Returns
        -------
        numpy.array
        '''
        if os.path.isfile(self._files['spike_waveforms']):
            return np.load(self._files['spike_waveforms'])
        else:
            return None

    def get_clustering_metrics(self, n_pc=3):
        '''Returns array of metrics to use for feature based clustering
        Row for each waveform with columns:
            - amplitude, energy, spike slope, PC1, PC2, etc
        '''
        amplitude = self.get_spike_amplitudes()
        energy = self.get_energy()
        slopes = self.get_spike_slopes()
        pca_waves = self.get_pca_waveforms()
        out = np.vstack((amplitude, energy, slopes)).T
        out = np.hstack((out, pca_waves[:,:n_pc]))
        return out

    def __str__(self):
        out = []
        out.append('SpikeDetection\n--------------')
        out.append('Recording Directory: %s' % self._file_dir)
        out.append('Electrode: %i' % self._electrode)
        out.append('Output Directory: %s' % self._out_dir)
        out.append('###################################\n')
        out.append('Status:')
        out.append(pt.print_dict(self._status))
        out.append('-------------------\n')
        out.append('Parameters:')
        out.append(pt.print_dict(self.params))
        out.append('-------------------\n')
        out.append('Data files:')
        out.append(pt.print_dict(self._files))
        return '\n'.join(out)


class BlechClust(object):
    def __init__(self, rec_dirs, electrode, out_dir=None, params=None,
                 overwrite=False, no_write=False, n_pc=3,
                 data_transform=compute_waveform_metrics):
        '''Recording directories should be ordered to make spike sorting easier later on
        '''
        if isinstance(rec_dirs, str):
            rec_dirs = [rec_dirs]

        rec_dirs = [x[:-1] if x.endswith(os.sep) else x for x in rec_dirs]
        self.rec_dirs = rec_dirs
        self.electrode = electrode
        self._data_transform = data_transform
        self._n_pc = n_pc
        if out_dir is None:
            if len(rec_dirs) > 1:
                top = os.path.dirname(rec_dirs[0])
                out_dir = os.path.join(top, 'BlechClust', 'electrode_%i' % electrode)
            else:
                out_dir = os.path.join(rec_dirs[0], 'BlechClust', 'electrode_%i' % electrode)

        if overwrite:
            shutil.rmtree(out_dir)

        # Make directories
        self.out_dir = out_dir
        self._plot_dir = os.path.join(out_dir, 'plots')
        self._data_dir = os.path.join(out_dir, 'clustering_results')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)

        if not os.path.isdir(self._plot_dir):
            os.mkdir(self._plot_dir)

        # Check files
        params_file = os.path.join(out_dir, 'BlechClust_params.json')
        map_file = os.path.join(self._data_dir, 'spike_id.npy')
        key_file = os.path.join(self._data_dir, 'rec_key.json')
        results_file = os.path.join(self._data_dir, 'clustering_results.json')
        self._files = {'params': params_file, 'spike_map': map_file,
                       'rec_key': key_file, 'clustering_results': results_file}
        self.params = params
        self._load_existsing_data()

        if self._rec_key is None and not no_write:
            # Create new rec key
            rec_key = {x:y for x,y in enumerate(self.rec_dirs)}
            self._rec_key = rec_key
            wt.write_dict_to_json(rec_key, self._files['rec_key'])
        elif self._rec_key is None:
            ValueError('Existing rec_key not found and no_write is enabled')

        # Check to see if spike detection is already completed on all recording directories
        spike_check = self._check_spike_detection()
        if not all(spike_check):
            invalid = [rec_dirs[i] for i, x in enumerate(spike_check) if x==False]
            error_str = '\n\t'.join(invalid)
            raise ValueError('Spike detection has not been run on:\n\t%s' % error_str)

    def _load_existsing_data(self):
        params = self.params
        file_check = self._check_existing_files()

        # Check params files and create if new params are passed
        if file_check['params']:
            self.params = wt.read_dict_from_json(self._files['params'])

        # Make new params or overwrite existing with passed params
        if params is None and not file_check['params']:
            raise ValueError(('Params file does not exists at %s. Must provide'
                              ' clustering parameters.') % self._files['params'])
        elif params is not None:
            self.params['max_clusters'] = params['clustering_params']['Max Number of Clusters']
            self.params['max_iterations'] = params['clustering_params']['Max Number of Iterations']
            self.params['threshold'] = params['clustering_params']['Convergence Criterion']
            self.params['num_restarts'] = params['clustering_params']['GMM random restarts']
            self.params['wf_amplitude_sd_cutoff'] = params['data_params']['Intra-cluster waveform amp SD cutoff']
            wt.write_dict_to_json(self.params, self._files['params'])

        # Deal with existing rec key
        if file_check['rec_key']:
            rec_dirs = self.rec_dirs
            rec_key = wt.read_dict_from_json(self._files['rec_key'])
            rec_key = {int(x): y for x,y in rec_key.items()}
            if len(rec_key) != len(rec_dirs):
                raise ValueError('Rec key does not match rec dirs')

            # Correct rec key in case rec_dir roots have changed
            for rd in rec_dirs:
                rn = os.path.basename(rd)
                dn = os.path.dirname(rd)
                kd = [(x, y) for x,y in rec_key.items() if rn in y]
                if len(kd) == 0:
                    raise ValueError('%s not found in rec_key' % rn)

                kd = kd[0]
                if kd[1] != rd:
                    rec_key[kd[0]] = rd

            inverted = {v:k for k,v in rec_key.items()}
            self.rec_dirs = sorted(self.rec_dirs, key=lambda i: inverted[i])
            self._rec_key = rec_key
        else:
            self._rec_key = None

        # Check is clustering has already been done, load results
        if file_check['clustering_results']:
            self.results = wt.read_pandas_from_table(self._files['clustering_results'])
            expected_results = np.arange(2, self.params['max_clusters'] + 1)
            if not all([x in self.results['clusters'] for x in expected_results]):
                self.clustered = False
            else:
                self.clustered = True

        else:
            self.results = None
            self.clustered = False

    def _check_existing_files(self):
        out = dict.fromkeys(self._files.keys(), False)
        for k,v in self._files.items():
            if os.path.isfile(v):
                out[k] = True

        return out

    def _check_spike_detection(self):
        '''Check to see if spike detection has been run on all recording directories
        '''
        out = []
        for rec in  self.rec_dirs:
            try:
                spike_detect = SpikeDetection(rec, self.electrode)
                if all(spike_detect._status):
                    out.append(True)
                else:
                    out.append(False)

            except FileNotFoundError:
                out.append(False)

        return out

    def run(self, n_pc=None, overwrite=False):
        if self.clustered and not overwrite:
            return True

        if n_pc is None:
            n_pc = self._n_pc

        GMM = ClusterGMM(self.params['max_iterations'],
                         self.params['num_restarts'], self.params['threshold'])

        # Collect data from all recordings
        waveforms, spike_times, spike_map, fs, offsets = self.get_spike_data()

        # Save array to map spikes and predictions back to original recordings
        np.save(self._files['spike_map'], spike_map)

        data, data_columns = self._data_transform(waveforms, n_pc)
        amplitudes = get_waveform_amplitudes(waveforms)

        # Run GMM for each number of clusters from 2 to max_clusters
        tested_clusters = np.arange(2, self.params['max_clusters']+1)
        clust_results = pd.DataFrame(columns=['clusters','converged',
                                              'BIC','spikes_per_cluster'],
                                     index=tested_clusters)
        for n_clust in tested_clusters:
            data_dir = os.path.join(self._data_dir, '%i_clusters' % n_clust)
            plot_dir = os.path.join(self._plot_dir, '%i_clusters' % n_clust)
            wave_plot_dir = os.path.join(self._plot_dir, '%i_clusters_waveforms_ISIs' % n_clust)
            bic_file = os.path.join(data_dir, 'bic.npy')
            pred_file = os.path.join(data_dir, 'predictions.npy')

            if os.path.isfile(bic_file) and os.path.isfile(pred_file) and not overwrite:
                bic = np.load(bic_file)
                predictions = np.load(pred_file)
                spikes_per_clust = [len(np.where(predictions == c)[0])
                                    for c in np.unique(predictions)]
                clust_results.loc[n_clust] = [n_clust, True, bic, spikes_per_clust]
                continue

            if not os.path.isdir(wave_plot_dir):
                os.makedirs(wave_plot_dir)

            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)

            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            model, predictions, bic = GMM.fit(data, n_clust)
            if model is None:
                clust_results.loc[n_clust] = [n_clust, bic, False, [0]]
                # Nothing converged
                continue

            # Go through each cluster and throw out any spikes too far from the
            # mean
            spikes_per_clust = []
            for c in range(n_clust):
                idx = np.where(predictions == c)[0]
                mean_amp = np.mean(amplitudes[idx])
                sd_amp = np.std(amplitudes[idx])
                cutoff_amp = mean_amp - (sd_amp * self.params['wf_amplitude_sd_cutoff'])
                rejected_idx = np.array([i for i in idx if amplitudes[i] <= cutoff_amp])
                if len(rejected_idx) > 0:
                    predictions[rejected_idx] = -1

                idx = np.where(predictions == c)[0]
                spikes_per_clust.append(len(idx))

                if len(idx) == 0:
                    continue

                # Plot waveforms and ISIs of cluster
                ISIs, violations_1ms, violations_2ms = get_ISI_and_violations(spike_times[idx], fs, spike_map[idx])
                cluster_waves = waveforms[idx]
                cluster_times = spike_times[idx]
                isi_fn = os.path.join(wave_plot_dir, 'Cluster%i_ISI.png' % c)
                wave_fn = os.path.join(wave_plot_dir, 'Cluster%i_waveforms.png' % c)
                title_str = ('Cluster%i\nviolations_1ms = %i, '
                             'violations_2ms = %i\n'
                             'Number of waveforms = %i' %
                             (c, violations_1ms, violations_2ms, len(idx)))
                dplt.plot_waveforms(cluster_waves, title=title_str, save_file=wave_fn)
                if len(ISIs) > 0:
                    dplt.plot_ISIs(ISIs, total_spikes=len(idx), save_file=isi_fn)


            clust_results.loc[n_clust] = [n_clust, True, bic, spikes_per_clust]

            # Plot feature pairs
            feature_pairs = it.combinations(list(range(data.shape[1])), 2)
            for f1, f2 in feature_pairs:
                fn = '%sVS%s.png' % (data_columns[f1], data_columns[f2])
                fn = os.path.join(plot_dir, fn)
                dplt.plot_cluster_features(data[:, [f1,f2]], predictions,
                                           x_label = data_columns[f1],
                                           y_label = data_columns[f2],
                                           save_file = fn)

            # For each cluster plot mahanalobis distances to all other clusters
            for c in range(n_clust):
                distances = get_mahalanobis_distances_to_cluster(data,  model,
                                                                 predictions, c)
                fn = os.path.join(plot_dir, 'Mahalanobis_cluster%i.png' % c)
                title = ('Mahalanobis distance of Cluster %i from all other clusters' % c)
                dplt.plot_mahalanobis_to_cluster(distances, title=title, save_file=fn)

            # Save data
            np.save(bic_file, bic)
            np.save(pred_file, predictions)

        # Save results table
        self.results = clust_results
        wt.write_pandas_to_table(clust_results,
                                 self._files['clustering_results'],
                                 overwrite=True)
        self.clustered = True
        return True

    def get_spike_data(self):
        # Collect data from all recordings
        tmp_waves = []
        tmp_times = []
        tmp_id = []
        fs = dict.fromkeys(self._rec_key.keys())
        offsets = dict.fromkeys(self._rec_key.keys())
        offset = 0
        for i in sorted(self._rec_key.keys()):
            rec = self._rec_key[i]
            spike_detect = SpikeDetection(rec, self.electrode)
            t = spike_detect.get_spike_times()
            fs[i] = spike_detect.params['sampling_rate']
            if t is None:
                offsets[i] = int(offset)
                offset = offset + 3*fs[i]
                continue

            tmp_waves.append(spike_detect.get_spike_waveforms())
            tmp_times.append(t)
            tmp_id.append(np.ones((t.shape[0],))*i)
            offsets[i] = int(offset)
            offset = offset + max(t) + 3*fs[i]

        waveforms = np.vstack(tmp_waves)
        spike_times = np.hstack(tmp_times)
        spike_map = np.hstack(tmp_id)

        # Double check that spike_map matches up with existing spike_map
        if os.path.isfile(self._files['spike_map']):
            orig_map = np.load(self._files['spike_map'])
            if len(orig_map) != len(spike_map):
                raise ValueError('Spike detection has changed, please re-cluster with overwrite=True')

        return waveforms, spike_times, spike_map, fs, offsets

    def get_clusters(self, solution_num, cluster_nums):
        if not isinstance(cluster_nums, list):
            cluster_nums = [cluster_nums]

        waveforms, times, spike_map, fs, offsets = self.get_spike_data()
        predictions = self.get_predictions(solution_num)
        out = []
        for c in cluster_nums:
            idx = np.where(predictions == c)[0]
            if len(idx)==0:
                continue

            tmp_clust = SpikeCluster('Cluster_%i' % c,
                                     self.electrode,
                                     solution_num,
                                     c,
                                     1,
                                     waveforms[idx],
                                     times[idx],
                                     spike_map[idx],
                                     self._rec_key.copy(),
                                     fs.copy(),
                                     offsets.copy(),
                                     manipulations='')
            out.append(tmp_clust)

        return out

    def get_predictions(self, n_clusters):
        fn = os.path.join(self._data_dir, '%i_clusters' % n_clusters,
                          'predictions.npy')
        if os.path.isfile(fn):
            return np.load(fn)
        else:
            return None


class ClusterGMM(object):
    def __init__(self, n_iters, n_restarts, thresh):
        self.params = {'iterations': n_iters,
                       'restarts': n_restarts,
                       'thresh': thresh}

    def fit(self, data, n_clusters):
        min_bic = None
        best_model = None
        if n_clusters is not None:
            self.params['clusters'] = n_clusters

        for i in range(self.params['restarts']):
            model = GaussianMixture(n_components = self.params['clusters'],
                                    covariance_type = 'full',
                                    tol = self.params['thresh'],
                                    random_state = i,
                                    max_iter = self.params['iterations'])
            model.fit(data)
            if model.converged_:
                new_bic = model.bic(data)
                if min_bic is None:
                    min_bic = model.bic(data)
                    best_model = model
                elif new_bic < min_bic:
                    best_model = model
                    min_bic = new_bic

        predictions = best_model.predict(data)
        self._model = best_model
        self._predictions = predictions
        self._bic = min_bic
        return best_model, predictions, min_bic


class SpikeSorter(object):
    def __init__(self, rec_dirs, electrode, clustering_dir=None, shell=False):
        if isinstance(rec_dirs, str):
            rec_dirs = [rec_dirs]

        rec_dirs = [x[:-1] if x.endswith(os.sep) else x for x in rec_dirs]
        self.rec_dirs = rec_dirs
        self.electrode = electrode
        if clustering_dir is None:
            if len(rec_dirs) > 1:
                top = os.path.dirname(rec_dirs[0])
                clustering_dir = os.path.join(top, 'BlechClust', 'electrode_%i' % electrode)
            else:
                clustering_dir = os.path.join(rec_dirs[0], 'BlechClust', 'electrode_%i' % electrode)

        self.clustering_dir = clustering_dir
        try:
            clust = BlechClust(rec_dirs, electrode, out_dir = clustering_dir, no_write=True)
        except FileNotFoundError:
            clust = None

        if clust is None or not clust.clustered:
            raise ValueError('Recordings have not been clustered yet.')

        # Match recording directory ordering to clustering
        self.rec_dirs = clust.rec_dirs
        self.clustering = clust
        self._current_solution = None
        self._active = None
        self._last_saved = None
        self._previous = None
        self._shell = shell
        self._split_results = None
        self._split_starter = None
        self._split_index = None
        self._last_umap_embedding = None
        self._last_action = None
        self._last_popped = None  # Dict of indices to clusters
        self._last_added = None  # List of indices

        thresh = []
        for rd in rec_dirs:
            sd = SpikeDetection(rd, electrode)
            thresh.append(sd.detection_threshold)

        self._detection_thresholds = thresh

    def undo(self):
        if self._last_action is None:
            return

        if self._last_action == 'save':
            self.undo_last_save()
            return

        # Remove last added
        for k in reversed(sorted(self._last_added)):
            self._active.pop(k)

        # Insert previous clusters
        for k in sorted(self._last_popped.keys()):
            self._active.insert(k, self._last_popped[k])

        # reset
        self._last_action = None
        self._last_popped = None
        self._last_added = None

    def set_active_clusters(self, solution_num):
        self._current_solution = solution_num
        cluster_nums = list(range(solution_num))
        clusters = self.clustering.get_clusters(solution_num, cluster_nums)
        if len(clusters) == 0:
            raise ValueError('Solution or clusters not found')

        self._active = clusters
        self._last_action = None
        self._last_popped = None
        self._last_added = None

    def save_clusters(self, target_clusters, single_unit, pyramidal, interneuron):
        '''Saves active clusters as cells, write them to the h5_files in the
        appropriate recording directories

        Parameters
        ----------
        target_clusters: list of int
            indicies of active clusters to save
        single_unit : list of bool
            elements in list must correspond to elements in active clusters
        pyramidal : list of bool
        interneuron : list of bool
        '''
        if self._active is None:
            return

        if any([i >= len(self._active) for i in target_clusters]):
            raise ValueError('Target cluster is out of range.')

        n_clusters = len(target_clusters)
        if (len(single_unit) != n_clusters or len(pyramidal) != n_clusters or
            len(interneuron) != n_clusters):
            raise ValueError('Length of input lists must match number of '
                             'active clusters. Expected %i' % n_clusters)

        self._last_action = 'save'
        self._last_popped = {i: self._active[i] for i in target_clusters}
        self._last_added = []
        clusters = [self._active[i] for i in target_clusters]
        rec_key = self.clustering._rec_key
        self._last_saved = dict.fromkeys(rec_key.keys(), None)

        for clust, single, pyr, intr in zip(clusters, single_unit,
                                            pyramidal, interneuron):
            for i, rec in rec_key.items():
                idx = np.where(clust['spike_map'] == i)[0]
                if len(idx) == 0:
                    continue

                waves = clust['spike_waveforms'][idx]
                times = clust['spike_times'][idx]
                unit_name = h5io.add_new_unit(rec, self.electrode, waves,
                                              times, single, pyr, intr)
                if self._last_saved[i] is None:
                    self._last_saved[i] = [unit_name]
                else:
                    self._last_saved[i].append(unit_name)

                metrics_dir = os.path.join(rec,'sorted_unit_metrics', unit_name)
                if not os.path.isdir(metrics_dir):
                    os.makedirs(metrics_dir)

                # Write cluster info to file
                print_clust = clust.copy()
                for k,v in clust.items():
                    if isinstance(v, np.ndarray):
                        print_clust.pop(k)

                print_clust.pop('rec_key')
                print_clust.pop('fs')
                clust_info_file = os.path.join(metrics_dir, 'cluster.info')
                with open(clust_info_file, 'a+') as log:
                    print('%s sorted on %s'
                          % (unit_name,
                             dt.datetime.today().strftime('%m/%d/%y %H:%M')),
                          file=log)
                    print('Cluster info: \n----------', file=log)
                    print(pt.print_dict(print_clust), file=log)
                    print('Saved metrics to %s' % metrics_dir, file=log)
                    print('--------------\n', file=log)

        userIO.tell_user('Target clusters successfully saved to recording '
                         'directories.', shell=True)
        self._active = [self._active[i] for i in range(len(self._active))
                        if i not in target_clusters]

    def undo_last_save(self):
        if self._last_saved is None:
            return

        rec_key = self.clustering._rec_key
        last_saved = self._last_saved
        for i, rec in rec_key.items():
            for unit in reversed(np.sort(last_saved[i])):
                h5io.delete_unit(rec, unit)

        for k in sorted(self._last_popped.keys()):
            self._active.insert(k, self._last_popped[k])

        self._last_saved = None
        self._last_popped = None
        self._last_added = None
        self._last_action = None

    def split_cluster(self, target_clust, n_iter, n_restart, thresh, n_clust,
                      store_split=False, umap=False):
        '''splits the target active cluster using a GMM
        '''
        if target_clust >= len(self._active):
            raise ValueError('Invalid target. Only %i active clusters' % len(self._active))

        cluster = self._active.pop(target_clust)
        self._split_starter = cluster
        self._split_index = target_clust

        try:
            GMM = ClusterGMM(n_iter, n_restart, thresh)
            waves = cluster['spike_waveforms']
            data, data_columns = compute_waveform_metrics(waves, umap=umap)
            model, predictions, bic = GMM.fit(data, n_clust)
            new_clusts = []
            for i in np.unique(predictions):
                idx = np.where(predictions == i)[0]
                edit_str = (cluster['manipulations'] + '\nSplit %s into %i '
                            'clusters. This is sub-cluster %i'
                            % (cluster['Cluster_Name'], n_clust, i))
                tmp_clust = SpikeCluster(cluster['Cluster_Name'] + '-%i' % i,
                                         cluster['electrode_num'],
                                         cluster['solution_num'],
                                         cluster['cluster_num'],
                                         cluster['cluster_id']*10+i,
                                         waves[idx],
                                         cluster['spike_times'][idx],
                                         cluster['spike_map'][idx],
                                         cluster['rec_key'].copy(),
                                         cluster['fs'].copy(),
                                         cluster['offsets'].copy(),
                                         manipulations=edit_str)
                new_clusts.append(tmp_clust)

            # Plot cluster and ask to choose which to keep
            figs = []
            for i, c in enumerate(new_clusts):
                _, viol_1ms, viol_2ms = get_ISI_and_violations(c['spike_times'], c['fs'], c['spike_map'])
                plot_title = ('Index: %i\n1ms violations: %i, 2ms violations: %i\n'
                              'Total Waveforms: %i'
                              % (i, viol_1ms, viol_2ms, len(c['spike_times'])))
                tmp_fig, _ = dplt.plot_waveforms(c['spike_waveforms'], title=plot_title)
                figs.append(tmp_fig)
                tmp_fig.show()

            f2 = dplt.plot_waveforms_pca([c['spike_waveforms'] for c in new_clusts])
            figs.append(f2)
            f2.show()
        except:
            # So cluster isn't lost with error
            self._active.insert(target_clust, cluster)
            self._split_starter = None
            self._split_index = None
            raise

        if store_split:
            self._split_results = new_clusts
            return new_clusts
        else:
            self._split_starter = None
            self._split_index = None
            selection_list = ['all'] + ['%i' % i for i in range(len(new_clusts))]
            prompt = 'Select split clusters to keep\nCancel to reset.'
            ans = userIO.select_from_list(prompt, selection_list,
                                          multi_select=True, shell=self._shell)
            if ans is None or 'all' in ans:
                print('Reset to before split')
                self._active.insert(target_clust, cluster)
            else:
                keepers = [new_clusts[int(i)] for i in ans]
                start_idx = len(self._active)
                self._last_added = list(range(start_idx, start_idx+len(keepers)))
                self._last_popped = {target_clust: cluster}
                self._last_action = 'split'
                self._active.extend(keepers)

            return True

    def set_split(self, choices):
        if self._split_starter is None:
            raise ValueError('Not split stored.')

        if len(choices) == 0:
            self._active.insert(self._split_index, self._split_starter)
        else:
            keepers = [self._split_results[i] for i in choices]
            start_idx = len(self._active)
            self._last_added = list(range(start_idx, start_idx+len(keepers)))
            self._last_popped = {self._split_index: self._split_starter}
            self._last_action = 'split'
            self._active.extend(keepers)

        self._split_index = None
        self._split_results = None
        self._split_starter = None

    def merge_clusters(self, target_clusters):
        if any([i >= len(self._active) for i in target_clusters]):
            raise ValueError('Target cluster is out of range.')

        new_clust = []
        self._last_popped = {}
        self._last_action = 'merge'
        self._last_added = []
        for c in target_clusters:
            self._last_popped[c] = self._active[c]

            if len(new_clust) == 0:
                new_clust = deepcopy(self._active[c])
                continue

            clust = self._active[c]
            sm1 = new_clust['spike_map']
            sm2 = clust['spike_map']
            st1 = new_clust['spike_times']
            st2 = clust['spike_times']
            sw1 = new_clust['spike_waveforms']
            sw2 = clust['spike_waveforms']

            spike_map = np.hstack((sm1, sm2))
            spike_times = np.hstack((st1, st2))
            spike_waveforms = np.vstack((sw1, sw2))

            # Re-order to spike_map
            idx = np.argsort(spike_map)
            spike_map = spike_map[idx]
            spike_times = spike_times[idx]
            spike_waveforms = spike_waveforms[idx]

            # Re-order so spike_times within a reocrding are in order
            times = []
            waves = []
            new_map = []
            for i in np.unique(spike_map):
                idx = np.where(spike_map == i)[0]
                st = spike_times[idx]
                sw = spike_waveforms[idx]
                sm = spike_map[idx]
                idx2 = np.argsort(st)
                st = st[idx2]
                sw = sw[idx2]
                sm = sm[idx2]
                times.append(st)
                waves.append(sw)
                new_map.append(sm)

            times = np.hstack(times)
            waves = np.vstack(waves)
            spike_map = np.hstack(new_map)
            del new_map, spike_times, spike_waveforms

            new_clust['spike_map'] = spike_map
            new_clust['spike_times'] = times
            new_clust['spike_waveforms'] = waves
            new_clust['manipulations'] += '\nMerged with %s.' % clust['Cluster_Name']
            new_clust['Cluster_Name'] += '+' + clust['Cluster_Name'].replace('Cluster_','')

        self._active = [self._active[i] for i in range(len(self._active))
                        if i not in target_clusters]

        self._last_added = [len(self._active)]
        self._active.append(new_clust)

    def discard_clusters(self, target_clusters):
        if isinstance(target_clusters, int):
            target_clusters = [target_clusters]

        if len(target_clusters) == 0:
            return

        self._last_action = 'discard'
        self._last_popped = {i: self._active[i] for i in target_clusters}
        self._last_added = []
        self._active = [self._active[i] for i in range(len(self._active))
                        if i not in target_clusters]

    def plot_clusters_waveforms(self, target_clusters):
        if len(target_clusters) == 0:
            return

        for i in target_clusters:
            c = self._active[i]
            isi, v1, v2 = get_ISI_and_violations(c['spike_times'], c['fs'], c['spike_map'])
            title = ('Index : %i\n1ms violations: %0.1f, 2ms violations: %0.1f'
                     '\ntotal waveforms: %i'
                     % (i, v1, v2, len(c['spike_waveforms'])))
            fig, ax = dplt.plot_waveforms(c['spike_waveforms'], title=title,
                                          threshold=self._detection_thresholds[0])
            fig.show()

    def split_by_rec(self, target_cluster):
        if isinstance(target_cluster, list) and len(target_cluster) != 1:
            return
        elif isinstance(target_cluster, list):
            target_cluster = target_clsuter[0]

        clust = self._active[target_cluster]
        sm = clust['spike_map']
        recs = np.unique(sm)
        if len(sm) == 1:
            return
        else:
            clust = self._active.pop(target_cluster)
            st = clust['spike_times']
            sw = clust['spike_waveforms']
            keepers = []
            for i in recs:
                idx = np.where(sm == i)[0]
                new_clust = deepcopy(clust)
                new_clust['spike_times'] = st[idx]
                new_clust['spike_waveforms'] = sw[idx, :]
                new_clust['cluster_id'] = clust['cluster_id']*10 + i
                new_clust['spike_map'] = sm[idx]
                new_clust['manipulations'] = '\nSplit by recording'
                keepers.append(new_clust)

        start_idx = len(self._active)
        self._last_added = list(range(start_idx, start_idx+len(keepers)))
        self._last_popped = {target_cluster: clust}
        self._last_action = 'split'
        self._active.extend(keepers)

    def plot_cluster_waveforms_by_rec(self, target_cluster):
        if isinstance(target_cluster, list) and len(target_cluster) != 1:
            return
        elif isinstance(target_cluster, list):
            target_cluster = target_cluster[0]

        c = self._active[target_cluster]
        sm = c['spike_map']
        for i in np.unique(sm):
            idx = np.where(sm==i)[0]
            waves = c['spike_waveforms'][idx, :]
            isi, v1, v2 = get_ISI_and_violations(c['spike_times'][idx], c['fs'][i])
            title = ('Index : %i, Rec: %i\n1ms violations: %0.1f, 2ms violations: %0.1f'
                     '\ntotal waveforms: %i'
                     % (target_cluster, i, v1, v2, len(waves)))
            fig, ax = dplt.plot_waveforms(waves, title=title)
            fig.show()

    def plot_cluster_waveforms_over_time(self, target_cluster, interval):
        if isinstance(target_cluster, list) and len(target_cluster) != 1:
            return
        elif isinstance(target_cluster, list):
            target_cluster = target_cluster[0]

        c = self._active[target_cluster]
        spike_times = c.get_spike_time_vector('s')
        start_times = np.arange(spike_times[0], spike_times[-1]+1, interval)
        if len(start_times) > 10:
            userIO.tell_user('This would open more than 10 figures, choose a larger interval')
            return

        if len(start_times) == 0:
            return

        for i, start_time in enumerate(start_times):
            idx = np.where((spike_times >= start_time) & (spike_times < start_time+interval))[0]
            waves = c['spike_waveforms'][idx,:]
            isi, v1, v2 = get_ISI_and_violations(c['spike_times'][idx], c['fs'][0])
            title = ('Index : %i, Rec: %i\n1ms violations: %0.1f, 2ms violations: %0.1f'
                     '\ntotal waveforms: %i'
                     % (target_cluster, i, v1, v2, len(waves)))
            fig, ax = dplt.plot_waveforms(waves, title=title,
                                          threshold=self._detection_thresholds[0])
            fig.show()

    def plot_clusters_pca(self, target_clusters):
        if len(target_clusters) == 0:
            return

        waves = [self._active[i]['spike_waveforms'] for i in target_clusters]
        fig = dplt.plot_waveforms_pca(waves, cluster_ids=target_clusters)
        fig.show()

    def plot_clusters_umap(self, target_clusters):
        if len(target_clusters) == 0:
            return

        waves = [self._active[i]['spike_waveforms'] for i in target_clusters]
        fig = dplt.plot_waveforms_umap(waves, cluster_ids=target_clusters)
        fig.show()

    def plot_clusters_wavelets(self, target_clusters):
        if len(target_clusters) == 0:
            return

        waves = [self._active[i]['spike_waveforms'] for i in target_clusters]
        fig, ax = dplt.plot_waveforms_wavelet_tranform(waves,
                                                       cluster_ids=target_clusters,
                                                       n_pc=4)
        fig.show()

    def plot_clusters_raster(self, target_clusters):
        if len(target_clusters) == 0:
            return

        clusters = [self._active[i] for i in target_clusters]
        spike_times = []
        spike_waves = []
        vlines = {}
        for c in clusters:
            # Adjust spike times by offset so recordings are not overlapping
            st = c.get_spike_time_vector(units='s')
            vlines = {i: c['offsets'][i] / c['fs'][i] for i in c['fs'].keys()}
            spike_times.append(st)
            spike_waves.append(c['spike_waveforms'])

        fig, ax = dplt.plot_spike_raster(spike_times, spike_waves, target_clusters)
        ax.set_xlabel('Time (s)')
        for x in vlines.values():
            ax.axvline(x, color='black', linewidth=2)

        fig.show()

    def plot_clusters_ISI(self, target_clusters):
        if len(target_clusters) == 0:
            return

        for i in target_clusters:
            cluster = self._active[i]
            isi, v1, v2 = get_ISI_and_violations(cluster['spike_times'],
                                                 cluster['fs'],
                                                 cluster['spike_map'])
            fig, ax = dplt.plot_ISIs(isi, total_spikes=len(cluster['spike_times']))
            title= ax.get_title()
            title = 'Index: %i\n%s' % (i, title)
            ax.set_title(title)
            fig.show()

    def plot_clusters_acorr(self, target_clusters):
        if len(target_clusters) == 0 or not all([x < len(self._active) for x in target_clusters]):
            return

        for i in target_clusters:
            cluster = self._active[i]
            acf, bin_centers, edges = sas.spike_time_acorr(cluster.get_spike_time_vector(units='ms'))
            fig, ax = dplt.plot_correlogram(acf, bin_centers, edges)
            title = 'Index: %i\nAutocorrelogram' % (i)
            ax.set_title(title)
            fig.show()

    def plot_clusters_xcorr(self, target_clusters):
        if len(target_clusters) == 0 or not all([x < len(self._active) for x in target_clusters]):
            return

        pairs = it.combinations(target_clusters, 2)
        for x, y in pairs:
            clust1 = self._active[x]
            clust2 = self._active[y]
            xcf, bin_centers, edges = sas.spike_time_xcorr(clust1.get_spike_time_vector(units='ms'),
                                                    clust2.get_spike_time_vector(units='ms'))
            fig, ax = dplt.plot_correlogram(xcf, bin_centers, edges)
            title = 'Cross-correlogram\n%i vs %i' % (x, y)
            ax.set_title(title)
            fig.show()

    def get_mean_waveform(self, target_cluster):
        '''Returns mean waveform of target_cluster in active clusters. Also
        returns St. Dev. of waveforms
        '''
        cluster = self._active[target_cluster]
        return cluster.get_mean_waveform()

    def get_possible_solutions(self):
        results = self.clustering.results.dropna()
        converged = list(results[results['converged']].index)
        return converged


class SpikeCluster(dict):
    def __init__(self, name, electrode, solution, cluster, cluster_id, waves, times,
                 spike_map, rec_key, fs={0: 30000}, offsets={0:0}, manipulations=''):
        # Confirm spike_map, rec_key, fs and offsets are all in sync
        rec_nums = np.unique(spike_map)
        if (not all([x in rec_key.keys() for x in rec_nums]) or
            not all([x in fs.keys() for x in rec_nums]) or
            not all([x in offsets.keys() for x in rec_nums])):
            raise ValueError('rec_key, fs and offsets must have entries for '
                             'each unique element of spike_map')

        # Confirm same number of waves, times and map entries
        if waves.shape[0] != len(times) or len(times) != len(spike_map):
            raise ValueError('Must have same number of waves, times and map entries')

        super(SpikeCluster, self).__init__(Cluster_Name=name,
                                           electrode_num=electrode,
                                           solution_num=solution,
                                           cluster_num = cluster,
                                           cluster_id=cluster_id,
                                           spike_waveforms=waves,
                                           spike_times=times,
                                           spike_map=spike_map,
                                           rec_key=rec_key,
                                           fs=fs,
                                           offsets=offsets,
                                           manipulations=manipulations)

    def delete_spikes(self, idx, msg=None):
        self['spike_waveforms'] = np.delete(self['spike_waveforms'],
                                               idx, axis=0)
        self['spike_times'] = np.delete(self['spike_times'], idx)
        self['spike_map'] = np.delete(self['spike_map'], idx)
        print('deleted %i spikes.' % len(idx))
        if msg is not None:
            self['manipulations'] += '/n' + msg + '\n-Removed %i spikes' % len(idx)

    def get_spike_time_vector(self, units='samples'):
        '''Return vector of all spike times with offsets added if multiple
        recordings are present

        Parameters
        ----------
        units : {'samples' (default), 'ms', 's'}, units for spike times returned

        Returns
        -------
        numpy.ndarray
        '''
        if units.lower() == 'ms':
            times = np.array([(a + self['offsets'][b]) / (self['fs'][b] / 1000)
                              for a, b in
                              zip(self['spike_times'].astype('float64'), self['spike_map'])])
        elif units.lower() == 's':
            times = np.array([(a + self['offsets'][b]) / self['fs'][b]
                              for a, b in
                              zip(self['spike_times'].astype('float64'), self['spike_map'])])
        elif units.lower() == 'samples':
            times = np.array([(a + self['offsets'][b]) for a, b in
                              zip(self['spike_times'], self['spike_map'])])
        else:
            raise ValueError('units must be either samples or ms')

        return times

    def __eq__(self, other):
        times1 = self.get_spike_time_vector(units='samples')
        times2 = other.get_spike_time_vector(units='samples')
        times1 = np.sort(times1)
        times2= np.sort(times2)
        return np.array_equal(times1, times2)

    def get_mean_waveform(self):
        '''Returns mean waveform of cluster. Also
        returns St. Dev. of waveforms and number of waveforms

        Returns
        -------
        np.ndarray, np.ndarray, int
        mean waveform, st. dev of waveforms, number of waveforms 
        '''
        mean_wave = np.mean(self['spike_waveforms'], axis=0)
        std_wave = np.std(self['spike_waveforms'], axis=0)
        n_waves = self['spike_waveforms'].shape[0]
        return mean_wave, std_wave, n_waves

    # TODO: Finish this section
    def _dist(self, other=None):
        if other is None:
            other = self

        pass

    def _add(self, other):
        pass

    def _subtract(self, other):
        pass

    def _divide(self, N, method='pca'):
        pass

    def _is_subcluster(self, other):
        '''less than?'''
        pass

    def _is_supercluster(self, other):
        ''' greater than?'''
        pass

