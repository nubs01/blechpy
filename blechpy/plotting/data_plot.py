import pylab as plt
import numpy as np
import tables
import os
from blechpy import dio
from blechpy.analysis import spike_analysis as sas
from scipy.stats import sem
from scipy.ndimage.filters import gaussian_filter1d


def plot_traces_and_outliers(h5_file, window=60, save_file=None):
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

    if save_file is not None:
        fig.savefig(save_file)
        plt.close('all')

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


def plot_J3s(intra_J3, inter_J3, save_dir, percent_criterion):
    print('\n----------\nPlotting J3 distribution\n----------\n')
    fig = plt.figure(figsize=(10,5))
    plt.hist([inter_J3, intra_J3], bins=20, alpha=0.7,
             label=['Across-session J3', 'Within-session J3'])
    plt.legend(prop={'size':12}, loc='upper right')
    plt.axvline(np.percentile(intra_J3, percent_criterion), linewidth=2,
                color='black', linestyle='dashed')
    plt.xlabel('J3', fontsize=18)
    plt.ylabel('Number of single unit pairs', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    fig.savefig(os.path.join(save_dir, 'J3_distribution.png'),
                bbox_inches='tight')
    plt.close('all')


def plot_held_units(rec_dirs, held_df, J3_df, save_dir):
    '''Plot waveforms of held units side-by-side

    Parameters
    ----------
    rec_dirs : list of str
        full paths to recording directories
    held_df : pandas.DataFrame
        dataframe listing held units with columns matching the names of the
        recording directories and a unit column with the unit names
    J3_df : pandas.DataFrame
        dataframe with same rows and columns as held df except the values are
        lists fo inter_J3 values for units that were found to be held
    save_dir : str, directory to save plots in
    '''

    print('\n----------\nPlotting held units\n----------\n')
    for idx, row in held_df.iterrows():
        unit_name = row.pop('unit')
        electrode = row.pop('electrode')
        area = row.pop('area')
        n_subplots = row.notnull().sum()
        idx = np.where(row.notnull())[0]
        cols = row.keys()[idx]
        units = row.values[idx]

        fig = plt.figure(figsize=(18, 6))
        ylim = [0, 0]
        row_ax = []

        for i, x in enumerate(zip(cols, units)):
            J3_vals = J3_df[x[0]][J3_df['unit'] == unit_name].values[0]
            J3_str = np.array2string(np.array(J3_vals), precision=3)
            ax = plt.subplot(1, n_subplots, i+1)
            row_ax.append(ax)
            rd = [y for y in rec_dirs if x[0] in y][0]
            params = get_clustering_parameters(rd)
            if params is None:
                raise FileNotFoundError('No dataset pickle file for %s' % rd)

            #waves, descriptor, fs = get_unit_waveforms(rd, x[1])
            waves, descriptor, fs = get_raw_unit_waveforms(rd, x[1])
            waves = waves[:, ::10]
            fs = fs/10
            time = np.arange(0, waves.shape[1], 1) / (fs/1000)
            snapshot = params['spike_snapshot']
            t_shift = snapshot['Time before spike (ms)']
            time = time - t_shift
            mean_wave = np.mean(waves, axis=0)
            std_wave = np.std(waves, axis=0)
            plt.plot(time, mean_wave,
                     linewidth=5.0, color='black')
            plt.plot(time, mean_wave - std_wave,
                     linewidth=2.0, color='black',
                     alpha=0.5)
            plt.plot(time, mean_wave + std_wave,
                     linewidth=2.0, color='black',
                     alpha=0.5)
            plt.xlabel('Time (ms)',
                       fontsize=35)
            if i==0:
                plt.ylabel('Voltage (microvolts)', fontsize=35)

            plt.title('%s %s\ntotal waveforms = %i, Electrode: %i\n'
                      'J3: %s, Single Unit: %i, RSU: %i, FS: %i'
                      % (x[0], x[1], waves.shape[0],
                         descriptor['electrode_number'],
                         J3_str,
                         descriptor['single_unit'],
                         descriptor['regular_spiking'],
                         descriptor['fast_spiking']),
                      fontsize = 20)
            plt.tick_params(axis='both', which='major', labelsize=32)

            if np.min(mean_wave - std_wave) - 20 < ylim[0]:
                ylim[0] = np.min(mean_wave - std_wave) - 20

            if np.max(mean_wave + std_wave) + 20 > ylim[1]:
                ylim[1] = np.max(mean_wave + std_wave) + 20

        for ax in row_ax:
            ax.set_ylim(ylim)

        plt.subplots_adjust(top=.7)
        plt.suptitle('Unit %s' % unit_name)
        fig.savefig(os.path.join(save_dir,
                                 'Unit%s_waveforms.png' % unit_name),
                    bbox_inches='tight')
        plt.close('all')
