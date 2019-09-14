import pylab as plt
import numpy as np
import tables
import os
from blechpy import dio
from blechpy.analysis import spike_analysis as sas
from scipy.stats import sem
from scipy.ndimage.filters import gaussian_filter1d


def plot_traces_and_outliers(h5_file, window=60):
    '''plot first 30 sec of raw data traces as well as a subplot with a metric
    to help identify dead channels (max(abs(trace)) * std(trace))

    Parameters
    ----------
    h5_file : str, full path to h5_file with raw data
    '''
    if not os.path.isfile(h5_file):
        raise FileNotFoundError('%s not found.' % h5_file)

    with tables.open_file(h5_file, 'r') as hf5:
        if '/raw' not in hf5:
            raise ValueError('No raw data in %s' % h5_file)

        electrodes = hf5.list_nodes('/raw')
        t_idx = np.where(lambda x: x.v_name == 'amplifier_time')[0]
        time = electrodes.pop(t_idx[0])[:]
        n_electrodes = len(electrodes)
        max_amp = np.zeros(n_electrodes)
        max_amp_idx = np.zeros(n_electrodes)
        std_amp = np.zeros(n_electrodes)
        range_amp = np.zeros(n_electrodes)

        for node in electrodes:
            i = int(node._v_name.replace('electrode',''))
            trace = node[:] * dio.rawIO.voltage_scaling
            max_amp[i] = np.max(np.abs(trace))
            max_amp_idx[i] = int(np.argmax(np.abs(trace)))
            std_amp[i] = np.std(trace)
            range_amp[i] = np.max(trace) - np.min(trace)

        max_v = np.max(max_amp)
        max_idx = int(max_amp_idx[np.argmax(max_amp)])
        metric = max_amp * std_amp
        idx = np.where((time >= time[max_idx] - window/2) &
                       (time <= time[max_idx] + window/2))[0]

        fig, ax = plt.subplots(nrows=2, figsize=(30,30))
        for node in electrodes:
            i = int(node._v_name.replace('electrode',''))
            trace = node[:] * dio.rawIO.voltage_scaling / max_v
            ax[0].plot(time[idx], trace[idx] + i, linewidth=0.5)
            ax[1].plot([i, i], [0, metric[i]], color='black', linewidth=0.5)

        ax[1].scatter(np.arange(n_electrodes), metric)
        med = np.median(metric)
        sd = np.std(metric)
        ax[1].plot([0, n_electrodes-1], [med, med], color='blue',
                   linewidth=0.5, alpha=0.5)
        ax[1].plot([0, n_electrodes-1], [med + 1.5*sd, med + 1.5*sd],
                   color='red', linewidth=0.5, alpha=0.5)

        ax[0].set_ylabel('Electrode')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title('Raw Traces')

        ax[1].set_ylabel('max * st. dev.')
        ax[1].set_xlabel('Electrode')
        over = np.where(metric > med+1.5*sd)[0]
        ax[1].set_title('Electrodes over line: %s' % over)

    return fig, ax


def plot_overlay_psth(rec_dir, unit, din_map, plot_window=[-1500, 2500],
                      bin_size=250, bin_step=25, dig_ins=None, smoothing_width=3,
                      save_file=None):
    '''
    Plots overlayed PSTHs for all tastants or a specified subset

    Parameters
    ----------
    rec_dir: str
    unit: int
    plot_window: list of int, time window for plotting in ms
    bin_size: int, window size for binning spikes in ms
    bin_step: int, step size for binning spikes in ms
    dig_ins: list of int (optional)
        which digital inputs to plot PSTHs for, None (default) plots all
    save_file: str (optional), full path to save file, if None, saves in Overlay_PSTHs subfolder
    '''
    if isinstance(unit, str):
        unit = dio.h5io.parse_unit_number(unit)

    if dig_ins is None:
        dig_ins = din_map.query('spike_array==True').channel.values

    if save_file is None:
        save_dir = os.path.join(rec_dir, 'Overlay_PSTHs')
        save_file = os.path.join(save_dir, 'Overlay_PSTH_unit%03d' % unit)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    fig, ax = plt.subplots(figsize=(20,15))
    for din in dig_ins:
        name = din_map.query('channel==@din').name.values[0]
        time, spike_train = dio.h5io.get_spike_data(rec_dir, unit, din)
        psth_time, fr = sas.get_binned_firing_rate(time, spike_train, bin_size, bin_step)

        mean_fr = np.mean(fr, axis=0)
        sem_fr = sem(fr, axis=0)

        t_idx = np.where((psth_time >= plot_window[0]) & (psth_time <= plot_window[1]))[0]
        psth_time = psth_time[t_idx]
        mean_fr = mean_fr[t_idx]
        sem_fr = sem_fr[t_idx]
        mean_fr = gaussian_filter1d(mean_fr, smoothing_width)

        ax.fill_between(psth_time, mean_fr - sem_fr, mean_fr + sem_fr, alpha=0.3)
        ax.plot(psth_time, mean_fr, linewidth=3, label=name)

    ax.set_title('Peri-stimulus Firing Rate Plot\nUnit %i' % unit, fontsize=34)
    ax.set_xlabel('Time (ms)', fontsize=28)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=28)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(loc='best')
    ax.axvline(0, color='red', linestyle='--')
    fig.savefig(save_file)
    plt.close('all')
