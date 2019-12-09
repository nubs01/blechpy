import os
import numpy as np
from blechpy.utils import write_tools as wt, print_tools as pt, math_tools as mt
import pylab as plt


def detect_spikes(filt_el, spike_snapshot = [0.5, 1.0], fs = 30000.0):
    '''Detects spikes in the filtered electrode trace and returns the waveforms
    and spike_times
    '''
    # get indices of spike snapshot, expand by .1 ms in each direction
    snapshot = np.arange(-(spike_snapshot[0]+0.1)*fs/1000,
                         1+(spike_snapshot[1]+0.1)*fs/1000)
    m = np.mean(filt_el)
    th = 5.0*np.median(np.abs(filt_el)/0.6745)
    pos = np.where(filt_el <= m-th)[0]
    consecutive = mt.group_consecutives(pos)

    for idx in consecutive:
        minimum = idx[np.argmin(filt_el[idx])]
        spike_idx = minimum + snapshot
        if spike_idx[0] >= 0 and spike_idx[-1] < len(filt_el):
            waves.append(filt_el[spike_idx])
            times.append(minimum)

    if len(waves) == 0:
        return None, None

    waves_dj, times_dj = clust.dejitter(np.array(waves), times, spike_snapshot, fs)
    return waves_dj, times_dj


def compute_waveform_metrics(waves, n_pc=3):
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
    amplitudes =  get_waveform_amplitudes(waves)
    energy = get_waveform_energy(waves)
    slopes = get_spike_slopes(waves)
    # Scale waveforms to energy before running PCA
    scaled_waves = scale_waveforms(waves, energy=energy)
    pc_waves = clust.implement_pca(scaled_waves)
    out = np.vstack((amplitudes, energy, slopes)).T
    out = np.hstack((out, pc_waves[:,:n_pc]))
    return out


def get_waveform_amplitudes(waves):
    return np.min(waves,axis = 1)


def get_waveform_energy(waves):
    energy = np.sqrt(np.sum(waves**2, axis=1))/waves.shape[1]
    return energy


def get_spike_slopes(waves):
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


def scale_waveforms(waves, energy=None):
    if energy is None:
        energy = get_waveform_energy(waves)
    elif len(energy) != waves.shape[0]:
        raise ValueError(('Energies must correspond to each waveforms.
                         Different lengths are not allowed'))

    sclaed_slices = np.zeros(waves.shape)
    for i, w in enumerate(zip(waves, energy)):
        scaled_slices[i] = w[0]/w[1]

    return scaled_slices


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


class SpikeDetection(object):
    '''Interface to manage spike detection and data extraction in preparation
    for GMM clustering. Intended to help create and access the neccessary
    files. If object will detect is file already exist to avoid re-creation
    unless overwrite is specified as True.
    '''

    def __init__(self, file_dir, electode, params=None, overwrite=False):
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
                       'recording_cutoff' : os.path.join(self._data_dir, 'cutoff_time.txt')}

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
            self.params['wf_amplitude_sd_cutoff'] = params['data_params']['Intra-cluster waveform amp SD cutoff']
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
        for k, v in self._file.items():
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
        filt_el = clust.get_filtered_electrode(ref_el, freq=params['bandpass'],
                                               sampling_rate = fs)
        del ref_el
        # Get recording cutoff
        if not status['recording_cutoff']:
            self.recording_cutoff = get_recording_cutoff(filt_el, fs, **params)
            with open(self._files['recording_cutoff'], 'w') as f:
                f.write(str(self.recording_cutoff))

            status['recording_cutoff'] = True
            fn = os.path.join(self._plot_dir, 'cutoff_time.png')
            dplt.plot_recording_cutoff(filt_el, fs, self.recording_cutoff,
                                       out_file=fn)

        # Truncate electrode trace, deal with early cutoff
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
            waves, times = detect_spikes(filt_el, params['spike_snapshot'], fs)
            if waves is None:
                print('No waveforms detected on electrode %i' % electrode)
                return electrode, 0, self.recording_cutoff

            # Save waveforms and times
            np.save(self._files['spike_waveforms'], waves)
            np.save(self._files['spike_times'], times)
            status['spike_waveforms'] = True
            status['spike_times'] = True

        # Get various metrics and scale waveforms
        if not status['spike_amplitudes']:
            amplitudes = get_spike_amplitudes(waves)
            np.save(self._files['spike_amplitudes'], ampltitudes)
            status['spike_amplitudes'] = True

        if not status['slopes']:
            slopes = get_spike_slopes(waves)
            np.save(self._files['slopes'], ampltitudes)
            status['slopes'] = True

        if not status['energy']:
            energy = get_spike_energy(waves)
            np.save(self._files['energy'], ampltitudes)
            status['energy'] = True

        # get pca of scaled waveforms
        if not status['pca_waveforms']:
            scaled_waves = scale_waveforms(waves, energy=energy)
            pca_waves, explained_variance_ratio = clust.implement_pca(scaled_waves)

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
        out = np.vstack((ampltiude, energy, slopes)).T
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
        out.append(pt.print_dict(self.parms))
        out.append('-------------------\n')
        out.append('Data files:')
        out.append(pt.print_dict(self._files))
        return '\n'.join(out)

