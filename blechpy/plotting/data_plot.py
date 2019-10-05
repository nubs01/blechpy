import pylab as plt
import pandas as pd
import numpy as np
import tables
import os
from blechpy import dio
from blechpy.analysis import spike_analysis as sas
from blechpy.dio import h5io
from scipy.stats import sem
from scipy.ndimage.filters import gaussian_filter1d
from blechpy.plotting import blech_waveforms_datashader
import matplotlib

plot_params = {'xtick.labelsize': 14, 'ytick.labelsize': 14,
               'axes.titlesize': 26, 'figure.titlesize': 28,
               'axes.labelsize': 24}

def make_unit_plots(file_dir, unit_name, save_dir=None):
    '''Makes waveform plots for sorted unit in unit_waveforms_plots

    Parameters
    ----------
    file_dir : str, full path to recording directory
    fs : float, smapling rate in Hz
    '''
    matplotlib.use('Qt5Agg')
    matplotlib.rcParams.update(plot_params)

    if isinstance(unit_name, int):
        unit_num = unit_name
        unit_name = 'unit%03i' % unit_num
    else:
        unit_num = dio.h5io.parse_unit_number(unit_name)

    waveforms, descriptor, fs = dio.h5io.get_unit_waveforms(file_dir, unit_name)
    fs_str = '%g samples per ms' % (fs/10/1000.0)  # since both theses plots
                                                   # downsample by 10 and then to convert to samples/ms

    fig, ax = blech_waveforms_datashader.waveforms_datashader(waveforms)
    ax.set_xlabel('Samples (%s)' % fs_str)
    ax.set_ylabel('Voltage (microvolts)')
    unit_title = (('Unit %i, total waveforms = %i\nElectrode: %i, '
                   'Single Unit: %i, RSU: %i, FSU: %i') %
                  (unit_num, waveforms.shape[0],
                   descriptor['electrode_number'],
                   descriptor['single_unit'],
                   descriptor['regular_spiking'],
                   descriptor['fast_spiking']))
    ax.set_title(unit_title)
    fig.savefig(os.path.join(save_dir, 'Unit%i.png' % unit_num))
    plt.close('all')

    # Plot mean and SEM of waveforms
    # Downsample by 10 to remove upsampling from de-jittering
    fig, ax = plt.subplots(figsize=(12,8))
    mean_wave = np.mean(waveforms[:, ::10], axis=0)
    std_wave = np.std(waveforms[:, ::10], axis=0)
    mean_x = np.arange(mean_wave.shape[0]) + 1
    ax.plot(mean_x, mean_wave, linewidth=4.0)
    ax.fill_between(mean_x, mean_wave - std_wave,
                     mean_wave + std_wave, alpha=0.4)
    ax.set_xlabel('Samples (%s)' % fs_str)
    ax.set_ylabel('Voltage (microvolts)')
    ax.set_title(unit_title)
    fig.savefig(os.path.join(save_dir, 'Unit%i_mean_sd.png' % unit_num))
    plt.close('all')


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


def plot_held_units(rec_dirs, held_df, save_dir, rec_names=None):
    '''Plot waveforms of held units side-by-side

    Parameters
    ----------
    rec_dirs : list of str
        full paths to recording directories
    held_df : pandas.DataFrame
        dataframe listing held units with columns matching the names of the
        recording directories or the given rec_names. Also colulmns:
            - unit : str, unit name
            - single_unit : bool
            - unit_type : str, unit_type
            - electrode : int
            - J3 : list of float, J3 values for the held unit
    save_dir : str, directory to save plots in
    rec_names : list of str (optional)
        abbreviated rec_names if any were used for held_df creation
        if not given, rec_names are assumed to be the basenames of rec_dirs
    '''
    matplotlib.use('Qt5Agg')
    if rec_names is None:
        rec_names = [os.path.basename(x) for x in rec_dirs]

    rec_labels = {x: y for x, y in zip(rec_names, rec_dirs)}

    print('\n----------\nPlotting held units\n----------\n')
    for idx, row in held_df.iterrows():
        n_subplots = 0
        units = {}
        for rn in rec_names:
            if not pd.isna(row.get(rn)):
                n_subplots += 1
                units[rn] = row.get(rn)

        if n_subplots == 0:
            continue

        single_unit = row['single_unit']
        if single_unit:
            single_str = 'single-unit'
        else:
            single_str = 'multi-unit'

        unit_type = row['unit_type']
        unit_name = row['unit']
        electrode = row['electrode']
        area = row['area']
        J3_vals = row['J3']
        J3_str = np.array2string(np.array(J3_vals), precision=3)
        print('Plotting Unit %s...' % unit_name)

        title_str = 'Unit %s\nElectrode %i: %s %s\nJ3: %s' % (unit_name, electrode,
                                                              unit_type,
                                                              single_str, J3_str)


        fig, fig_ax = plt.subplots(ncols=n_subplots, figsize=(20, 10))
        ylim = [0, 0]
        row_ax = []

        for ax, unit_info in zip(fig_ax, units.items()):
            rl = unit_info[0]
            u = unit_info[1]
            rd = rec_labels.get(rl)
            params = dio.params.load_params('clustering_params', rd)
            if params is None:
                raise FileNotFoundError('No dataset pickle file for %s' % rd)

            #waves, descriptor, fs = get_unit_waveforms(rd, x[1])
            waves, descriptor, fs = dio.h5io.get_raw_unit_waveforms(rd, u)
            waves = waves[:, ::10]
            fs = fs/10
            time = np.arange(0, waves.shape[1], 1) / (fs/1000)
            snapshot = params['spike_snapshot']
            t_shift = snapshot['Time before spike (ms)']
            time = time - t_shift
            mean_wave = np.mean(waves, axis=0)
            std_wave = np.std(waves, axis=0)
            ax.plot(time, mean_wave,
                    linewidth=5.0, color='black')
            ax.plot(time, mean_wave - std_wave,
                    linewidth=2.0, color='black',
                    alpha=0.5)
            ax.plot(time, mean_wave + std_wave,
                    linewidth=2.0, color='black',
                    alpha=0.5)
            ax.set_xlabel('Time (ms)',
                          fontsize=35)

            ax.set_title('%s %s\ntotal waveforms = %i'
                         % (rl, u, waves.shape[0]),
                         fontsize = 20)
            ax.autoscale(axis='x', tight=True)
            plt.tick_params(axis='both', which='major', labelsize=32)

            if np.min(mean_wave - std_wave) - 20 < ylim[0]:
                ylim[0] = np.min(mean_wave - std_wave) - 20

            if np.max(mean_wave + std_wave) + 20 > ylim[1]:
                ylim[1] = np.max(mean_wave + std_wave) + 20

        for ax in row_ax:
            ax.set_ylim(ylim)

        fig_ax[0].set_ylabel('Voltage (microvolts)', fontsize=35)
        plt.subplots_adjust(top=.75)
        plt.suptitle(title_str)
        fig.savefig(os.path.join(save_dir,
                                 'Unit%s_waveforms.png' % unit_name),
                    bbox_inches='tight')
        plt.close('all')
