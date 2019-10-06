from blechpy import dio
from blechpy.widgets import userIO
from blechpy.analysis import spike_sorting as ss, dataset, clustering as clust
import pandas as pd
import numpy as np
import pylab as plt
import easygui as eg
import os
import tables
import pickle
import shutil
import json
from copy import deepcopy
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
from scipy.stats import mannwhitneyu, spearmanr, sem
import matplotlib.gridspec as gridspec
from collections.abc import Iterable
from scipy.ndimage.filters import gaussian_filter1d



class multi_dataset(object):

    def __init__(self, exp_dir=None, shell=False):
        '''Setup for analysis across recording sessions

        Parameters
        ----------
        exp_dir : str (optional)
            path to directory containing all recording directories
            if None (default) is passed then a popup to choose file
            will come up
        shell : bool (optional)
            True to use command-line interface for user input
            False (default) for GUI
        '''
        if exp_dir is None:
            exp_dir = eg.diropenbox('Select Experiment Directory',
                                    'Experiment Directory')
            if exp_dir is None or exp_dir == '':
                return

        fd = [os.path.join(exp_dir, x) for x in os.listdir(exp_dir)]
        file_dirs = [x for x in fd if os.path.isdir(x)]
        order_dict = dict.fromkeys(file_dirs, 0)
        tmp = userIO.dictIO(order_dict, shell=shell)
        order_dict = tmp.fill_dict(prompt=('Set order of recordings (1-%i)\n'
                                           'Leave blank to delete directory'
                                           ' from list') % len(file_dirs))
        if order_dict is None:
            return

        file_dirs = [k for k, v in order_dict.items()
                     if v is not None and v != 0]
        file_dirs = sorted(file_dirs, key=order_dict.get)
        file_dirs = [os.path.join(exp_dir, x) for x in file_dirs]
        file_dirs = [x[:-1] if x.endswith('/') else x
                     for x in file_dirs]
        self.recording_dirs = file_dirs
        self.experiment_dir = exp_dir
        self.shell = shell

        dat = dataset.load_dataset(file_dirs[0])
        em = dat.electrode_mapping.copy()
        if 'area' not in em.columns:
            ingc = userIO.select_from_list('Select all eletrodes confirmed in GC',
                                           em['Electrode'],
                                           multi_select=True, shell=shell)
            ingc = list(map(int, ingc))
            em['area'] = np.where(em['Electrode'].isin(ingc), 'GC', 'Other')

        self.electrode_mapping = em
        self.save_file = os.path.join(exp_dir, '%s_experiment.p'
                                      % os.path.basename(exp_dir))

    def save(self):
        '''Saves multi_dataset object to experiment.p file in recording directory
        '''
        with open(self.save_file, 'wb') as f:
            pickle.dump(self, f)
            print('Saved experiment dataset to %s' % self.save_file)

    def _order_dirs(self, shell=None):
        '''set order of redcording directories
        '''
        if shell is None:
            shell = self.shell
        self.recording_dirs = [x[:-1] if x.endswith('/') else x
                               for x in self.recording_dirs]
        top_dirs = {os.path.basename(x): os.path.dirname(x)
                    for x in self.recording_dirs}
        file_dirs = list(top_dirs.keys())
        order_dict = dict.fromkeys(file_dirs, 0)
        tmp = userIO.dictIO(order_dict, shell=shell)
        order_dict = tmp.fill_dict(prompt=('Set order of recordings (1-%i)\n'
                                           'Leave blank to delete directory'
                                           ' from list'))
        if order_dict is None:
            return

        file_dirs = [k for k, v in order_dict.items()
                     if v is not None and v != 0]
        file_dirs = sorted(file_dirs, key=order_dict.get)
        file_dirs = [os.path.join(top_dirs.get(x), x) for x in file_dirs]
        self.recording_dirs = file_dirs

    def _add_dir(self, new_dir=None, shell=None):
        '''Add recording directory to experiment

        Parameters
        ----------
        new_dir : str (optional)
            full path to new directory to add to recording dirs
        shell : bool (optional)
            True for command-line interface for user input
            False (default) for GUI
            If not passed then the preference set upon object creation is used
        '''
        if shell is None:
            shell = self.shell

        if new_dir is None:
            if shell:
                new_dir = input('Full path to new directory:  ')
            else:
                new_dir = eg.diropenbox('Select new recording directory',
                                        'Add Recording Directory')

        if os.path.isdir(new_dir):
            self.recording_dirs.append(new_dir)
        else:
            raise NotADirectoryError('new directory must be a valid full'
                                     ' path to a directory')

        self._order_dirs(shell=shell)

    def _assign_area(self, row):
        data_dir = self.experiment_dir
        em = self.electrode_mapping.copy()
        tmp_row = row.dropna()
        tmp = tmp_row.keys().to_list()
        idx = tmp.index('unit')
        if idx == 0:
            idx = 1
        else:
            idx = 0
        rec = tmp[idx]
        unit = tmp_row[rec]
        unit_num = dio.h5io.parse_unit_number(unit)
        rec_dir = os.path.join(data_dir, rec)
        descrip = dio.h5io.get_unit_descriptor(rec_dir, unit_num)
        electrode = descrip['electrode_number']
        area = em.query('Electrode == @electrode')['area'].values[0]

        row['electrode'] = electrode
        row['area'] = area
        return row


    def get_unit_stats(self, significance=0.01):
        '''Go through all recordings and units and get number of units in the
        correct area
        also count number of held units in an area and the number of taste
        responsive units in each recording session
        '''
        em = self.electrode_mapping
        recs = self.recording_dirs
        taste_map, tastants = get_taste_mapping(recs)

        out = []

        for t in tastants:
            for rd in recs:
                units = dio.h5io.get_unit_names(rd)
                rec_name = os.path.basename(rd)
                din = taste_map[t].get(rec_name)
                if din is None:
                    continue

                for u in units:
                    descrip = dio.h5io.get_unit_descriptor(rd, u)
                    el = descrip['electrode_number']
                    area = em.query('Electrode == @el')['area'].values[0]

                    taste_stat, taste_p = check_taste_response(rd, u, din,
                                                               window=1500)
                    if taste_p <= significance:
                        taste_resp = True
                    else:
                        taste_resp = False

                    single = bool(descrip['single_unit'])

                    unit_type = '-'
                    if descrip['regular_spiking'] == 1:
                        unit_type = 'pyramidal'
                    elif descrip['fast_spiking'] == 1:
                        unit_type = 'interneuron'

                    tmp = {'rec': rec_name, 'electrode': el, 'area': area,
                           'unit': u, 'single_unit': single,
                           'unit_type': unit_type,
                           'taste_responsive': taste_resp, 'tastant': t,
                           'test_stat': taste_stat, 'p-Value': taste_p}
                    out.append(tmp)

        tmp_stats = pd.DataFrame(out)
        gc_units = tmp_stats.query('single_unit & area=="GC"')
        single_unit_stat_str = gc_units.groupby(['rec', 'tastant',
                                                 'unit_type',
                                                 'taste_responsive']
                                               ).count()['unit'].to_string()
        print('\nSingle Unit Counts\n====================')
        print(single_unit_stat_str)
        stat_file = os.path.join(self.experiment_dir, 'Single_Unit_Counts.txt')
        if os.path.isfile(stat_file):
            os.remove(stat_file)

        with open(stat_file, 'w') as f:
            print(single_unit_stat_str, file=f)

        self.unit_stats = tmp_stats
        return tmp_stats.copy()


    def held_units_compare(self, significance=0.05):
        rec_dirs = self.recording_dirs
        rec_names = [os.path.basename(x) for x in rec_dirs]
        exp_name = os.path.basename(self.experiment_dir)

        # Figure out which recording sessions had the same tastants
        taste_map, tastants = get_taste_mapping(rec_dirs)
        save_dir = os.path.join(self.experiment_dir, 'held_units_comparison')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        os.mkdir(save_dir)

        out = {}

        # Iterate through held units and compare
        # held_df & counts only reflect responses to Saccharin
        # TODO: Fix hardcoding of Saccharin as only taste to keep summary stats
        held_df = self.held_units['units'].copy()
        held_dict = self.held_units['units'].to_dict(orient='records')
        sig_units = {}
        held_df['baseline_shift'] = False
        held_df['taste_response1'] = False
        held_df['taste_response2'] = False
        held_df['response_change'] = False
        held_df['norm_response_change'] = False
        held_df['z_response_change'] = False
        held_df['divergence'] = np.nan
        held_df['norm_divergence'] = np.nan
        held_df['z_divergence'] = np.nan
        perc_changed = {t: [] for t in tastants}
        perc_changed_time = None
        N_held = {t: 0 for t in tastants}

        for t in tastants:
            # Summary changes
            avg_mag_change = None
            avg_mag_change_SEM = None
            mag_time = None
            avg_norm_mag_change = None
            avg_norm_mag_change_SEM = None
            avg_z_mag_change = None
            avg_z_mag_change_SEM = None
            for i, unit in enumerate(held_dict):
                u = unit.copy()
                unit_name = u.pop('unit')
                elecrtrode = u.pop('electrode')
                area = u.pop('area')
                df_idx = held_df['unit'] == unit_name

                # Restrict to GC units
                if area != 'GC':
                    continue

                recs = list(u.keys())

                t_recs = [x for x in recs if taste_map[t].get(x) is not None]
                t_recs = sorted(t_recs, key=lambda x: rec_names.index(x))

                # For now only compare pairs
                if len(t_recs) != 2:
                    continue

                if not all([isinstance(u.get(x), str) for x in t_recs]):
                    continue

                N_held[t] += 1
                sd = os.path.join(save_dir, t)
                if not os.path.exists(sd):
                    os.mkdir(sd)

                dins = [taste_map[t][x] for x in t_recs]
                rds = [os.path.join(self.experiment_dir, x)
                       for x in t_recs]
                uns = [u.get(x) for x in t_recs]
                unit_descrip = dio.h5io.get_unit_descriptor(rds[0], uns[0])
                stats = compare_held_unit(unit_name, rds[0],
                                          uns[0], dins[0],
                                          rds[1], uns[1], dins[1],
                                          significance=significance,
                                          tastant=t,
                                          save_dir=sd)

                # Get magnitude of change
                mc, mc_SEM, mc_time = get_response_change(unit_name, rds[0],
                                                          uns[0], dins[0],
                                                          rds[1], uns[1],
                                                          dins[1])
                if avg_mag_change is not None:
                    avg_mag_change += np.abs(mc)
                    avg_mag_change_SEM += np.power(mc_SEM, 2)
                else:
                    avg_mag_change = np.abs(mc)
                    avg_mag_change_SEM = mc_SEM
                    mag_time = mc_time

                # Again but with baseline removed
                mc, mc_SEM, mc_time = get_response_change(unit_name, rds[0],
                                                          uns[0], dins[0],
                                                          rds[1], uns[1],
                                                          dins[1],
                                                          norm_func=remove_baseline)
                if avg_norm_mag_change is not None:
                    avg_norm_mag_change += np.abs(mc)
                    avg_norm_mag_change_SEM += np.power(mc_SEM, 2)
                else:
                    avg_norm_mag_change = np.abs(mc)
                    avg_norm_mag_change_SEM = mc_SEM

                # Again but Z-scored to baseline
                mc, mc_SEM, mc_time = get_response_change(unit_name, rds[0],
                                                          uns[0], dins[0],
                                                          rds[1], uns[1],
                                                          dins[1],
                                                          norm_func=zscore_to_baseline)
                if avg_z_mag_change is not None:
                    avg_z_mag_change += np.abs(mc)
                    avg_z_mag_change_SEM += np.power(mc_SEM, 2)
                else:
                    avg_z_mag_change = np.abs(mc)
                    avg_z_mag_change_SEM = mc_SEM

                # Save signififance stats
                if sig_units.get(t) is None:
                    sig_units[t] = {}

                if sig_units[t].get(unit_name) is None:
                    sig_units[t][unit_name] = {'rec1': rds[0], 'unit1': uns[0],
                                               'din1': dins[0], 'rec2': rds[1],
                                               'unit2': uns[1], 'din2': dins[1],
                                               'baseline_shift': [],
                                               'taste_response1': [],
                                               'taste_response2': [],
                                               'response_change': [],
                                               'norm_response_change': [],
                                               'z_response_change': [],
                                               'statistics': stats}

                if perc_changed[t] == []:
                    perc_changed[t] = np.where(stats['norm_response_p'] <= significance, 1, 0)
                else:
                    perc_changed[t] += np.where(stats['norm_response_p'] <= significance, 1, 0)

                if perc_changed_time is None:
                    perc_changed_time = stats['window_starts']

                if stats['baseline_shift']:
                    sig_units[t][unit_name]['baseline_shift'].append(unit_name)
                    if t == 'Saccharin' or t == 'saccharin':
                        held_df.loc[df_idx, 'baseline_shift'] = True

                if stats['taste_response1']:
                    sig_units[t][unit_name]['taste_response1'].append(unit_name)
                    if t == 'Saccharin' or t == 'saccharin':
                        held_df.loc[df_idx, 'taste_response1'] = True

                if stats['taste_response2']:
                    sig_units[t][unit_name]['taste_response2'].append(unit_name)
                    if t == 'Saccharin' or t == 'saccharin':
                        held_df.loc[df_idx, 'taste_response2'] = True

                if stats['response_change']:
                    sig_units[t][unit_name]['response_change'].append(unit_name)
                    if t == 'Saccharin' or t == 'saccharin':
                        held_df.loc[df_idx, 'response_change'] = True
                        held_df.loc[df_idx, 'divergence'] = stats['divergence']

                if stats['norm_change']:
                    sig_units[t][unit_name]['norm_response_change'].append(unit_name)
                    if t == 'Saccharin' or t == 'saccharin':
                        held_df.loc[df_idx, 'norm_response_change'] = True
                        held_df.loc[df_idx, 'norm_divergence'] = stats['norm_divergence']

                if stats['z_change']:
                    sig_units[t][unit_name]['z_response_change'].append(unit_name)
                    if t == 'Saccharin' or t == 'saccharin':
                        held_df.loc[df_idx, 'z_response_change'] = True
                        held_df.loc[df_idx, 'z_divergence'] = stats['z_divergence']

            # Plot average magnitude of changes
            if sig_units.get(t) is not None:
                avg_mag_change = avg_mag_change / N_held[t]
                avg_norm_mag_change = avg_norm_mag_change / N_held[t]
                avg_z_mag_change = avg_z_mag_change / N_held[t]
                avg_mag_change_SEM = np.sqrt(avg_mag_change_SEM)
                avg_norm_mag_change_SEM = np.sqrt(avg_norm_mag_change_SEM)
                avg_z_mag_change_SEM = np.sqrt(avg_z_mag_change_SEM)

                np_dir = os.path.join(sd, 'metrics')
                if not os.path.isdir(np_dir):
                    os.mkdir(np_dir)

                mag_change_file = os.path.join(np_dir, 'avg_norm_mag_change.npz')
                np.savez(mag_change_file, avg_norm_mag_change=avg_norm_mag_change,
                         avg_norm_mag_change_SEM=avg_norm_mag_change_SEM,
                         mag_time=mag_time)
                cut_win = np.array([-500, 2000])
                t_idx = np.where((mag_time >= cut_win[0]) & (mag_time <= cut_win[1]))[0]
                fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(30,10))
                ax[0].fill_between(mag_time, avg_mag_change - avg_mag_change_SEM,
                                   avg_mag_change + avg_mag_change_SEM, alpha=0.4)
                ax[0].plot(mag_time, avg_mag_change, linewidth=2)
                ax[0].axvline(x=0, linestyle='--', color='red', alpha=0.6)
                ax[0].set_title('Average change in magnitude of response',fontsize=14)
                ax[0].set_xlabel('Time (ms)',fontsize=14)
                ax[0].set_ylabel('Firing Rate (Hz)',fontsize=14)

                ax[1].fill_between(mag_time[t_idx], avg_norm_mag_change[t_idx] - avg_norm_mag_change_SEM[t_idx],
                                   avg_norm_mag_change[t_idx] + avg_norm_mag_change_SEM[t_idx], alpha=0.4)
                ax[1].plot(mag_time[t_idx], avg_norm_mag_change[t_idx], linewidth=2)
                ax[1].axvline(x=0, linestyle='--', color='red', alpha=0.6)
                ax[1].set_title('Average change in magnitude of response\nBaseline removed',fontsize=14)
                ax[1].set_xlabel('Time (ms)',fontsize=14)

                save_file = os.path.join(sd, 'Summary_Magnitude_Change.png')
                fig.savefig(save_file)
                plt.close('all')

                # Plot % units changed at each time point
                perc_changed[t] = 100 * perc_changed[t] / N_held[t]
                fig, ax = plt.subplots(figsize=(14.5,7))
                ax.step(perc_changed_time, perc_changed[t], color='black')
                ax.set_ylabel('% GC Units Changed', fontsize=20)
                ax.set_xlabel('Time (ms)', fontsize=20)
                ax.set_title('%s: %% units changed\n%s, N=%i' % (exp_name, t, N_held[t]), fontsize=24)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                save_file = os.path.join(sd, '%_units_changed.png')
                fig.savefig(save_file)
                plt.close('all')

        self.significant_units = sig_units
        self.perc_changed = perc_changed
        self.perc_changed_time = perc_changed_time
        self.held_units['units'] = held_df.copy()
        sig_file = os.path.join(sd, 'significant_units.json')
        held_df.to_json(sig_file, orient='records', lines=True)

        counts = {}
        counts['held'] = N_held
        sums = held_df.groupby(['area']).sum().drop(['electrode',
                                                     'divergence',
                                                     'norm_divergence',
                                                     'z_divergence'],axis=1)
        counts.update(sums.to_dict(orient='records')[0])
        n_taste_gain = len(held_df.query('not taste_response1 & taste_response2'))
        n_taste_loss = len(held_df.query('taste_response1 & not taste_response2'))
        n_resp_overlap = len(held_df.query('response_change & norm_response_change'))
        z_resp_overlap = len(held_df.query('response_change & z_response_change'))
        counts['taste_response_gain'] = n_taste_gain
        counts['taste_response_loss'] = n_taste_loss
        counts['response_change_overlap'] = n_resp_overlap
        counts['z_response_change_overlap'] = z_resp_overlap
        counts['mean_divergence_time'] = (held_df['divergence'].mean(),
                                          held_df['divergence'].std())
        counts['mean_norm_divergence_time'] = (held_df['norm_divergence'].mean(),
                                               held_df['norm_divergence'].std())
        counts['mean_z_divergence_time'] = (held_df['z_divergence'].mean(),
                                               held_df['z_divergence'].std())
        self.held_units['counts'] = counts
        counts_file = os.path.join(sd, 'held_unit_counts.json')
        dio.params.write_dict_to_json(counts, counts_file)


        # return sig_units

    def load_held_units(self):
        save_dir = os.path.join(self.experiment_dir, 'held_units')
        if not os.path.exists(save_dir):
            raise FileNotFoundError('%s not found' % save_dir)

        intra_J3 = np.load(os.path.join(save_dir, 'intra_J3.npy'))
        inter_J3 = np.load(os.path.join(save_dir, 'inter_J3.npy'))
        with open(os.path.join(save_dir, 'held_units.json'), 'r') as f:
            held_dict = json.load(f)

        held_df =  pd.DataFrame.from_dict(held_dict)
        J3_df = pd.read_csv(os.path.join(save_dir, 'held_units_J3.txt'),
                           sep='\t')

        self.held_units = {'units': held_df,
                           'J3_df': J3_df,
                           'intra_J3': intra_J3,
                           'inter_J3': inter_J3}


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


def get_unit_waveforms(rec_dir, unit_name, shell=True):
    '''returns the waveforms of a single unit read from the h5 file in rec_dir

    Parameters
    ----------
    rec_dir : str, full path to recording directory
    unit_num : int, number of unit to grab

    Returns
    -------
    numpy.array with row for each spike waveform
    '''
    print('Getting waveforms for %s %s'
          % (os.path.basename(rec_dir), unit_name))
    dat = dataset.load_dataset(rec_dir)
    unit_num = ss.parse_unit_number(unit_name)

    with tables.open_file(dat.h5_file, 'r') as hf5:
        unit = hf5.root.sorted_units[unit_name]
        waves = unit.waveforms[:]
        descriptor = hf5.root.unit_descriptor[unit_num]

    return waves, descriptor, dat.sampling_rate*10

def get_raw_unit_waveforms(rec_dir, unit_name, shell=True, required_descrip=None):
    '''Returns the waveforms of a single unit extracting them directly from the
    raw data files.
    '''
    unit_num = ss.parse_unit_number(unit_name)
    dat = dataset.load_dataset(rec_dir)
    e_map = dat.electrode_mapping
    snapshot = dat.clustering_params['spike_snapshot']
    snapshot = [snapshot['Time before spike (ms)'],
                snapshot['Time after spike (ms)']]
    bandpass = dat.clustering_params['bandpass_params']
    bandpass = [bandpass['Lower freq cutoff'],
                bandpass['Upper freq cutoff']]
    fs = dat.sampling_rate


    # Get spike times for unit
    h5_file = dio.h5io.get_h5_filename(rec_dir, shell=shell)
    with tables.open_file(h5_file, 'r') as hf5:
        spike_times = hf5.root.sorted_units[unit_name]['times'][:]
        descriptor = hf5.root.unit_descriptor[unit_num]

    if required_descrip is not None:
        if descriptor != required_descrip:
            return None, descriptor, fs

    print('Getting raw waveforms for %s %s'
          % (os.path.basename(rec_dir), unit_name))
    tmp_h5 = os.path.join(rec_dir, 'tmp_raw.hdf5')
    # Create temporary hdf5 store for raw traces 
    if not os.path.exists(tmp_h5):
        dio.h5io.create_empty_data_h5(tmp_h5, shell=shell)
        dio.h5io.create_hdf_arrays(tmp_h5, dat.rec_info,
                                            dat.electrode_mapping,
                                            dat.emg_mapping)
        dio.h5io.read_files_into_arrays(tmp_h5, dat.rec_info,
                                                 dat.electrode_mapping,
                                                 dat.emg_mapping)

    # Get raw trace for unit
    electrode_num = descriptor['electrode_number']
    with tables.open_file(tmp_h5, 'r') as hf5:
        raw_el = hf5.root.raw['electrode%i' % electrode_num][:]

    # Filter and extract waveforms
    filt_el = clust.get_filtered_electrode(raw_el, freq=bandpass,
                                           sampling_rate=fs)
    del raw_el
    pre_pts = int((snapshot[0]+0.1) * (fs/1000))
    post_pts = int((snapshot[1]+0.2) * (fs/1000))
    slices = np.zeros((spike_times.shape[0], pre_pts+post_pts))
    for i, st in enumerate(spike_times):
        slices[i, :] = filt_el[st - pre_pts: st + post_pts]

    slices_dj, times_dj = clust.dejitter(slices, spike_times, snapshot, fs)

    return slices_dj, descriptor, fs*10


def get_clustering_parameters(rec_dir):
    file_list = os.listdir(rec_dir)
    dat_file = [x for x in file_list if x.endswith('dataset.p')][0]
    if dat_file == []:
        return None

    dat_file = os.path.join(rec_dir, dat_file)
    with open(dat_file, 'rb') as f:
        dat = pickle.load(f)

    return dat.clustering_params

def check_changes_in_taste_responses(pre_h5, post_h5, held_df, pre_dig_in,
                                     post_dig_in):
    pass


def apply_test_along_axis(data1, data2, test_func=mannwhitneyu, axis=1):
    '''Takes two data matrices and runs the test function pairwise using
    columns (axis=1) ro rows (axis=0). data matrices must be the same length
    along the axis to iterate comparisons over. i.e. if both data1 and data2
    are MxN matrices and axis=1 then this will return two 1xN matrices
    containing the test statistic and p-values when comparing each column of
    data1 and data2
    test_func must return stat and pvalue and take 2 arrays are arguments.
    i.e. stat, pval = test_func(slice1, slice2)

    Applies Bonferroni correction for multiple comparisons
    '''
    n_pts = data1.shape[axis]
    if data2.shape[axis] != n_pts:
        raise ValueError('data1 and data2 must be the same size along axis %i' % axis)

    stats = np.zeros(n_pts)
    pvals = np.zeros(n_pts)

    for i in range(n_pts):
        try:
            if axis == 1:
                s, p = test_func(data1[:, i], data2[:, i])
            elif axis == 0:
                s, p = test_func(data1[i, :], data2[i, :])

            stats[i] = s
            pvals[i] = p*n_pts  # apply bonferroni correction to pvalue
        except ValueError:
            stats[i] = 0
            pvals[i] = 1


    return stats, pvals


def check_taste_response(rec_dir, unit_name, din, window=1500):
    h5_file = dio.h5io.get_h5_filename(rec_dir)
    unit_num = dio.h5io.parse_unit_number(unit_name)
    din_str = 'dig_in_%i' % din

    with tables.open_file(h5_file, 'r') as hf5:
        spikes = hf5.root.spike_trains[din_str]['spike_array'][:, unit_num, :]
        time = hf5.root.spike_trains[din_str]['array_time'][:]

    pre_idx = np.where((time >= -window) & (time <= 0))[0]
    post_idx = np.where((time >= 0) & (time <= window))[0]
    pre = 1000 * np.sum(spikes[:, pre_idx], axis=1) / window
    post = 1000 * np.sum(spikes[:, post_idx], axis=1) / window
    stat, pval = mannwhitneyu(pre, post, alternative='two-sided')

    return stat, pval


def compare_held_unit(unit_name, rec1, unit1, dig_in1, rec2, unit2, dig_in2,
                      significance=0.05, n_sig_win=5, normalized=False,
                      tastant=None, save_dir=None, time_window=[-1500, 2500],
                      comp_win=250):
    out = {}
    u1 = ss.parse_unit_number(unit1)
    u2 = ss.parse_unit_number(unit2)

    dat1 = dataset.load_dataset(rec1)
    dat2 = dataset.load_dataset(rec2)

    if tastant is None:
        din_map = dat1.dig_in_mapping
        tastant = din_map['name'][din_map['channel'] == dig_in1].values[0]

    di1 = 'dig_in_%i' % dig_in1
    di2 = 'dig_in_%i' % dig_in2

    rn1 = os.path.basename(rec1)
    rn2 = os.path.basename(rec2)

    with tables.open_file(dat1.h5_file, 'r') as hf5:
        tmp = hf5.root.PSTHs[di1]
        fr1 = tmp.psth_array[u1, :, :]
        time1 = tmp.time[:]
        i1 = np.where((time1 >= time_window[0]) & (time1 <= time_window[1]))[0]
        fr1 = fr1[:, i1]
        time1 = time1[i1]

        spikes1 = hf5.root.spike_trains[di1]['spike_array'][:, u1, :]
        s_time1 = hf5.root.spike_trains[di1]['array_time'][:]
        i1 = np.where((s_time1 >= time_window[0]) &
                      (s_time1 <= time_window[1]))[0]
        spikes1 = spikes1[:, i1]
        s_time1 = s_time1[i1]

    with tables.open_file(dat2.h5_file, 'r') as hf5:
        tmp = hf5.root.PSTHs[di2]
        fr2 = tmp.psth_array[u2, :, :]
        time2 = tmp.time[:]
        i2 = np.where((time2 >= time_window[0]) & (time2 <= time_window[1]))[0]
        fr2 = fr2[:, i2]
        time2 = time2[i2]

        spikes2 = hf5.root.spike_trains[di2]['spike_array'][:, u2, :]
        s_time2 = hf5.root.spike_trains[di2]['array_time'][:]
        i2 = np.where((s_time2 >= time_window[0]) &
                      (s_time2 <= time_window[1]))[0]
        spikes2 = spikes2[:, i2]
        s_time2 = s_time2[i2]

    if not np.array_equal(time1, time2):
        raise ValueError('Units have different time arrays. %s %s vs %s %s'
                        % (rn1, unit1, rn2, unit2))

    if not np.array_equal(s_time1, s_time2):
        raise ValueError('Units have different spike time arrays. %s %s vs %s %s'
                        % (rn1, unit1, rn2, unit2))

    # Compare baseline firing from spike arrays
    pre_stim_idx = np.where(s_time1 < 0)[0]
    baseline1 = np.sum(spikes1[:, pre_stim_idx], axis=1) / abs(time_window[0])
    baseline2 = np.sum(spikes2[:, pre_stim_idx], axis=1) / abs(time_window[0])
    baseline1 = baseline1 * 1000 # convert to Hz
    baseline2 = baseline2 * 1000 # convert to Hz

    base_u, base_p = mannwhitneyu(baseline1, baseline2,
                                  alternative='two-sided')

    if base_p <= significance:
        out['baseline_shift'] = True
    else:
        out['baseline_shift'] = False

    out['baseline_stat'] = base_u
    out['baseline_p'] = base_p
    out['baseline1'] = (np.mean(baseline1), np.std(baseline1))
    out['baseline2'] = (np.mean(baseline2), np.std(baseline2))
    baseline_txt = ('Baseline\n   - Test Stat: %g\n   - p-Value: %g\n'
                 '   - Baseline 1 Mean (Hz): %0.2f \u00b1 %0.2f\n'
                 '   - Baseline 2 Mean (Hz): %0.2f \u00b1 %0.2f\n') % \
            (base_u, base_p, out['baseline1'][0], out['baseline1'][1],
             out['baseline2'][0], out['baseline2'][1])

    # Check if each is taste responsive
    # Compare 1.5 second before stim to 1.5 sec after stim
    post_stim_idx = np.where((s_time1 > 0) &
                             (s_time1 < abs(time_window[0])))[0]
    post1 = np.sum(spikes1[:, post_stim_idx], axis=1) / abs(time_window[0])
    post2 = np.sum(spikes2[:, post_stim_idx], axis=1) / abs(time_window[0])
    post1 = post1 * 1000
    post2 = post2 * 1000
    taste_u1, taste_p1 = mannwhitneyu(baseline1, post1,
                                      alternative='two-sided')
    taste_u2, taste_p2 = mannwhitneyu(baseline2, post2,
                                      alternative='two-sided')

    if taste_p1 <= significance:
        out['taste_response1'] = True
    else:
        out['taste_response1'] = False

    out['taste1_stat'] = taste_u1
    out['taste1_p'] = taste_p1
    out['taste1'] = (np.mean(post1), np.std(post1))

    taste_txt1 = ('Taste Responsive 1\n   - Test Stat: %g\n   - p-Value: %g\n'
                  '   - Mean Evoked (Hz): %0.2f \u00b1 %0.2f\n') % \
            (taste_u1, taste_p1, out['taste1'][0], out['taste1'][1])

    if taste_p2 <= significance:
        out['taste_response2'] = True
    else:
        out['taste_response2'] = False

    out['taste2_stat'] = taste_u2
    out['taste2_p'] = taste_p2
    out['taste2'] = (np.mean(post2), np.std(post2))

    taste_txt2 = ('Taste Responsive 2\n   - Test Stat: %g\n   - p-Value: %g\n'
                  '   - Mean Evoked (Hz): %0.2f \u00b2 %0.2f\n') % \
            (taste_u2, taste_p2, out['taste2'][0], out['taste2'][1])

    # Check for difference in posts-stimulus firing 
    # Compare consecutive 250ms post-stimulus bins
    win_starts = np.arange(0, time_window[1], comp_win)
    resp_u = np.zeros(win_starts.shape)
    resp_p = np.ones(win_starts.shape)
    norm_resp_u = np.zeros(win_starts.shape)
    norm_resp_p = np.ones(win_starts.shape)
    z_resp_u = np.zeros(win_starts.shape)
    z_resp_p = np.ones(win_starts.shape)
    for i, ws in enumerate(win_starts):
        idx = np.where((s_time1 >= ws) & (s_time1 <= ws+comp_win))[0]
        rate1 = 1000 * np.sum(spikes1[:, idx], axis=1) / comp_win
        nrate1 = rate1 - baseline1
        zrate1 = (rate1 - np.mean(baseline1)) / np.std(baseline1)
        rate2 = 1000 * np.sum(spikes2[:, idx], axis=1) / comp_win
        nrate2 = rate2 - baseline2
        zrate2 = (rate2 - np.mean(baseline2)) / np.std(baseline2)

        try:
            resp_u[i], resp_p[i] = mannwhitneyu(rate1, rate2,
                                                alternative='two-sided')
        except ValueError:
            resp_u[i] = 0
            resp_p[i] = 1

        try:
            norm_resp_u[i], norm_resp_p[i] = mannwhitneyu(nrate1, nrate2,
                                                          alternative='two-sided')
        except ValueError:
            norm_resp_u[i] = 0
            norm_resp_p[i] = 1

        try:
            z_resp_u[i], z_resp_p[i] = mannwhitneyu(zrate1, zrate2,
                                                          alternative='two-sided')
        except ValueError:
            z_resp_u[i] = 0
            z_resp_p[i] = 1

    # Bonferroni correction
    resp_p = resp_p * len(win_starts)
    norm_resp_p = norm_resp_p * len(win_starts)
    z_resp_p = z_resp_p * len(win_starts)

    sig_pts = np.where(resp_p <= significance, 1, 0)
    n_sig_pts = np.where(norm_resp_p <= significance, 1, 0)
    z_sig_pts = np.where(z_resp_p <= significance, 1, 0)

    out['comparison_window'] = comp_win
    out['window_starts'] = win_starts

    if any(sig_pts == 1):
        tmp_idx  = np.where(resp_p <= significance)[0][0]
        init_diff = win_starts[tmp_idx]
        out['response_change'] = True
    else:
        init_diff = np.nan
        out['response_change'] = False

    out['response_p'] = resp_p
    out['response_stat'] = resp_u
    out['divergence'] = init_diff
    out['sig_windows'] = np.sum(sig_pts)

    diff_txt = ('Response Change\n   - Significant Change: %s\n') % \
            out['response_change']
    if not np.isnan(init_diff):
        diff_txt += ('   - Divergence time (ms): %g\n'
                     '   - Number of different windows: %i\n'
                     '   - Minimum p-Value: %g\n') % \
                (init_diff, np.sum(sig_pts), np.min(resp_p))

    if any(n_sig_pts == 1):
        tmp_idx = np.where(norm_resp_p <= significance)[0][0]
        n_init_diff = win_starts[tmp_idx]
        n_init_p = norm_resp_p[tmp_idx]
        out['norm_change'] = True
    else:
        n_init_diff = np.nan
        n_init_p = np.nan
        out['norm_change'] = False

    out['norm_response_p'] = norm_resp_p
    out['norm_response_stat'] = norm_resp_u
    out['norm_divergence'] = n_init_diff
    out['norm_sig_windows'] = np.sum(n_sig_pts)

    norm_diff_txt = ('Relative Response Change\n'
                     '   - Significant Change: %s\n') %  out['norm_change']
    if not np.isnan(n_init_diff):
        norm_diff_txt += ('   - Divergence time (ms): %g\n'
                          '   - Number of different windows: %i\n'
                          '   - Minimum p-Value: %g\n'
                          '   - Inital p-Value: %g\n') % \
                (n_init_diff, np.sum(n_sig_pts), np.min(norm_resp_p), n_init_p)

    if any(z_sig_pts == 1):
        tmp_idx = np.where(z_resp_p <= significance)[0][0]
        z_init_diff = win_starts[tmp_idx]
        z_init_p = z_resp_p[tmp_idx]
        out['z_change'] = True
    else:
        z_init_diff = np.nan
        z_init_p = np.nan
        out['z_change'] = False

    out['z_response_p'] = z_resp_p
    out['z_response_stat'] = z_resp_u
    out['z_divergence'] = z_init_diff
    out['z_sig_windows'] = np.sum(z_sig_pts)

    z_diff_txt = ('Z-scored Response Change\n'
                  '   - Significant Change: %s\n') %  out['z_change']
    if not np.isnan(z_init_diff):
        z_diff_txt += ('   - Divergence time (ms): %g\n'
                       '   - Number of different windows: %i\n'
                       '   - Minimum p-Value: %g\n'
                       '   - Inital p-Value: %g\n') % \
                (z_init_diff, np.sum(z_sig_pts), np.min(z_resp_p), z_init_p)

    # Plot 1x3 with left being raw and center being with baseline removed &
    # right being z-scored to baseline& sig stars
    fig = plt.figure(figsize=(45, 20))
    gs = gridspec.GridSpec(4, 15)
    ax1 = plt.subplot(gs[:3, :5]) # raw psth
    ax2 = plt.subplot(gs[:3, 5:10]) # relative psth
    axz = plt.subplot(gs[:3, 10:]) # z psth
    ax3 = plt.subplot(gs[3, 0]) # Taste Responsive 1
    ax4 = plt.subplot(gs[3, 2]) # Taste Responsive 2
    ax5 = plt.subplot(gs[3, 5]) # Baseline change
    ax6 = plt.subplot(gs[3, 8]) # Response Change
    ax7 = plt.subplot(gs[3, 11]) # Relative response change
    ax8 = plt.subplot(gs[3, 13]) # Z response change
    nfr1 = (fr1.transpose() - baseline1).transpose()
    nfr2 = (fr2.transpose() - baseline2).transpose()
    zfr1 = (fr1 - np.mean(baseline1)) / np.std(baseline1)
    zfr2 = (fr2 - np.mean(baseline2)) / np.std(baseline2)
    plot_psth(fr1, time1, ax=ax1, label=rn1)
    plot_psth(fr2, time2, ax=ax1, label=rn2)

    sig_ints = find_contiguous(sig_pts).get(1)
    new_sig_ints = []
    if sig_ints is not None:
        for iv in sig_ints:
            i1 = np.where((s_time1 >= win_starts[iv[0]]) &
                          (s_time1 <= win_starts[iv[1]]+comp_win))[0]
            new_sig_ints.append((i1[0], i1[-1]))

    if out['baseline_shift']:
        i1 = np.where(s_time1 < 0)[0]
        new_sig_ints.append((i1[0], i1[-1]))


    if new_sig_ints != []:
        plot_significance(s_time1, new_sig_ints, ax=ax1)


    plot_psth(nfr1, time1, ax=ax2, label=rn1)
    plot_psth(nfr2, time2, ax=ax2, label=rn2)
    n_sig_ints = find_contiguous(n_sig_pts).get(1)
    new_sig_ints = []
    if n_sig_ints is not None:
        for iv in n_sig_ints:
            i1 = np.where((s_time1 >= win_starts[iv[0]]) &
                          (s_time1 <= win_starts[iv[1]]+comp_win))[0]
            new_sig_ints.append((i1[0], i1[-1]))

    if new_sig_ints != []:
        plot_significance(s_time1, new_sig_ints, ax=ax2)

    plot_psth(zfr1, time1, ax=axz)
    plot_psth(zfr2, time2, ax=axz)
    z_sig_ints = find_contiguous(z_sig_pts).get(1)
    new_sig_ints = []
    if z_sig_ints is not None:
        for iv in z_sig_ints:
            i1 = np.where((s_time1 >= win_starts[iv[0]]) &
                          (s_time1 <= win_starts[iv[1]]+comp_win))[0]
            new_sig_ints.append((i1[0], i1[-1]))

    if new_sig_ints != []:
        plot_significance(s_time1, new_sig_ints, ax=axz)

    ax1.legend(loc='lower left')

    ax1.set_title('Smoothed Firing Rate', fontsize=24)
    ax1.set_xlabel('Time (ms)', fontsize=20)
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=20)
    ax2.set_title('Smoothed Firing Rate\nBaseline removed', fontsize=24)
    ax2.set_xlabel('Time (ms)', fontsize=20)
    axz.set_title('Smoothed Firing Rate\nZ-scored to Baseline', fontsize=24)
    axz.set_xlabel('Time (ms)', fontsize=20)
    plt.subplots_adjust(top=0.87)
    plt.suptitle('Comparison of held unit %s\n%s %s vs %s %s'
                 % (unit_name, rn1, unit1, rn2, unit2), fontsize=34)

    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax7.axis('off')
    ax8.axis('off')
    ax3.text(0, 0.8, taste_txt1, fontsize=18, verticalalignment='top')
    ax4.text(0, 0.8, taste_txt2, fontsize=18, verticalalignment='top')
    ax5.text(0, 0.8, baseline_txt, fontsize=18, verticalalignment='top')
    ax6.text(0, 0.8, diff_txt, fontsize=18, verticalalignment='top')
    ax7.text(0, 0.8, norm_diff_txt, fontsize=18, verticalalignment='top')
    ax8.text(0, 0.8, z_diff_txt, fontsize=18, verticalalignment='top')
    plt.savefig(os.path.join(save_dir, 'Unit_%s_Comparison.png' % unit_name))
    plt.close('all')

    # Return if baseline if diff, if sig post_stim diff & when, same with baseline removed

    # Plot spike raster and psth for pre and post
    fig = plt.figure(figsize=(30,30))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222, sharex=ax1)
    ax3 = plt.subplot(223, sharex=ax1)
    ax4 = plt.subplot(224, sharey=ax3, sharex=ax2)
    plot_spike_raster(spikes1, s_time1, ax=ax1)
    plot_spike_raster(spikes2, s_time2, ax=ax2)
    plot_psth(fr1, time1, ax=ax3)
    plot_psth(fr2, time2, ax=ax4)
    ax1.set_ylabel('Trial #', fontsize=36)
    if normalized:
        ax3.set_ylabel('Mean Firing Rate\nbaseline removed', fontsize=32)
    else:
        ax3.set_ylabel('Mean Firing Rate (Hz)', fontsize=36)

    ax3.set_xlabel('Time (ms)', fontsize=36)
    ax4.set_xlabel('Time (ms)', fontsize=36)
    ax1.set_title('%s\n%s\n\nSpike Raster' % (rn1, unit1), fontsize=32)
    ax2.set_title('%s\n%s\n\nSpike Raster' % (rn2, unit2), fontsize=32)
    ax3.set_title('Firing Rate', fontsize=32)
    ax4.set_title('Firing Rate', fontsize=32)
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Unit %s: %s Response' % (unit_name, tastant))
    plt.savefig(os.path.join(save_dir, 'Unit_%s_Raster_%s_vs_%s.png' %
                             (unit_name, rn1, rn2)))
    plt.close('all')
    return out


def plot_spike_raster(spikes, time, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    sa = spikes.copy()
    for i, row in enumerate(spikes):
        idx = np.where(row != 0)[0]
        ax.scatter(time[idx], (i+1)*row[idx], marker='|', color='black')

    ax.axvline(x=0, linestyle='--', color='red', alpha=0.6)


def plot_psth(psth_array, time, **kwargs):
    mean_psth = np.mean(psth_array, axis=0)
    sem_psth = sem(psth_array, axis=0)
    ax = kwargs.pop('ax', plt.gca())
    baseline, = ax.plot(time, mean_psth, linewidth=3, **kwargs)
    ax.fill_between(time, mean_psth - sem_psth, mean_psth + sem_psth,
                    facecolor=baseline.get_color(), alpha=0.3,
                    edgecolor=baseline.get_color())
    ax.axvline(x=0, linestyle='--', color='red', alpha=0.6)
    ax.margins(.1)
    ax.autoscale(axis='y', tight=False)
    ax.autoscale(axis='x', tight=True)


def plot_significance(x, sig_ints, **kwargs):
    if sig_ints is None or sig_ints == []:
        return

    ax = kwargs.pop('ax', plt.gca())
    bb = ax.viewLim
    ymax = bb.ymax
    ylim = ax.get_ylim()

    for a,b in sig_ints:
        if b-a == 0:
            x1 = x[a]
            if a+1 < len(x):
                x2 = x[a+1]
            else:
                x2 = x[a]
            x3 = (x1 + x2) / 2
            ax.text(x3, ymax+.1, '*')
        else:
            x1 = x[a]
            if len(x) == b+1:
                x2 = x[b]
            else:
                x2 = x[b+1]

            ax.plot([x1, x2], [ymax, ymax], color='black', linewidth=3)
            ax.text((x1+x2)/2, ymax+.1, '*')

    ax.margins(.2)
    ax.autoscale(axis='y', tight=False)
    ax.autoscale(axis='x', tight=True)


def find_contiguous(arr):
    cont = dict.fromkeys(np.unique(arr))
    start_idx = 0
    end_idx = 0
    num = arr[0]
    for i, x in enumerate(arr):
        if x == num:
            end_idx = i

        if x != num or i == len(arr)-1:
            tmp = (start_idx, end_idx)
            if cont.get(num) is None:
                cont[num] = []

            cont[num].append(tmp)
            num = x
            start_idx = i
            end_idx = i

    return cont


def get_taste_mapping(rec_dirs):
    rec_names = [os.path.basename(x) for x in rec_dirs]
    tastants = []
    for rd in rec_dirs:
        dat = dataset.load_dataset(rd)
        tmp = dat.dig_in_mapping
        tastants.extend(tmp['name'].to_list())

    tastants = np.unique(tastants)
    taste_map = {}
    for rd in rec_dirs:
        rn = os.path.basename(rd)
        dat = dataset.load_dataset(rd)
        din = dat.dig_in_mapping
        for t in tastants:
            if taste_map.get(t) is None:
                taste_map[t] = {}

            tmp = din['channel'][din['name'] == t]
            if not tmp.empty:
                taste_map[t][rn] = tmp.values[0]

    return taste_map, tastants


def get_mean_difference(A, B, axis=0):
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
    bin_time2, fr2 = get_binned_firing_rate(time2, spikes2, bin_size, bin_step)

    if not np.array_equal(bin_time1, bin_time2):
        raise ValueError('Time of spike trains is not aligned')

    # Normalize firing rates
    if norm_func:
        fr1 = norm_func(bin_time1, fr1)
        fr2 = norm_func(bin_time2, fr2)

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


def plot_held_unit_firing(exp, unit_name, norm_func=None,
                          bin_size=250, bin_step=25, tastants=None,
                          smoothing_width=5, save_dir=None, significance=0.05):
    if save_dir is None:
        save_dir = os.path.join(exp.experiment_dir, 'held_unit_comparison_plots')

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    held_df = exp.held_units['units']
    held_df = held_df.query('unit == @unit_name')
    held_df = held_df.dropna(axis=1)

    rec_dirs = exp.recording_dirs
    taste_map, found_tastants = get_taste_mapping(rec_dirs)
    rec_names = [os.path.basename(x) for x in rec_dirs]

    if tastants is None:
        tastants = found_tastants
    elif not isinstance(tastants, Iterable):
        tastants = [tastants]

    for taste in tastants:
        t_recs = list(taste_map.get(taste).keys())
        valid_recs = []
        units = []
        dins = []

        for tr in t_recs:
            tmp = held_df.get(tr)
            if tmp is not None:
                units.append(held_df[tr].values[0])
                valid_recs.append(tr)
                dins.append(taste_map.get(taste).get(tr))

        if len(valid_recs) == 0:
            continue

        save_file = os.path.join(save_dir, 'Unit_%s_%s.svg' % (unit_name, taste))
        fig = plt.figure(figsize=(20,15))
        ax = plt.subplot(111)
        for rec, unit, din in zip(valid_recs, units, dins):
            rec_dir = os.path.join(exp.experiment_dir, rec)
            unit_num = dio.h5io.parse_unit_number(unit)
            if 'Train' in rec or 'pre' in rec:
                rec_label = 'Pre-CTA'
            else:
                rec_label = 'Post-CTA'

            spike_time, spike_trains = dio.h5io.get_spike_data(rec_dir,
                                                               unit_num, din)
            fr_time, fr = get_binned_firing_rate(spike_time, spike_trains,
                                                 bin_size=bin_size,
                                                 bin_step=bin_step)
            if norm_func is not None:
                fr = norm_func(fr_time, fr)

            smooth_fr = gaussian_filter1d(fr, smoothing_width)
            mean_fr = np.mean(smooth_fr, axis=0)
            sd_fr = sem(smooth_fr, axis=0)
            t_idx = np.where((fr_time>=-1500) & (fr_time<=2500))[0]
            mean_fr = mean_fr[t_idx]
            fr_time = fr_time[t_idx]
            sd_fr = sd_fr[t_idx]
            ax.fill_between(fr_time, mean_fr - sd_fr, mean_fr + sd_fr, alpha=0.4)
            ax.plot(fr_time, mean_fr, linewidth=2, label=rec_label)

        # Significance
        if exp.significant_units.get(taste) is not None:
            stats = exp.significant_units[taste][unit_name]['statistics']
            resp_p = stats['norm_response_p']
            win_starts = stats['window_starts']
            sig_pts = np.where(resp_p <= significance, 1, 0)
            sig_ints = find_contiguous(sig_pts).get(1)
            plot_significance(win_starts, sig_ints, ax=ax)
            ax.margins(.2)
            ax.autoscale(axis='y', tight=False)
            ax.autoscale(axis='x', tight=True)
        else:
            ax.margins(.2)
            ax.autoscale(axis='y', tight=False)
            ax.autoscale(axis='x', tight=True)


        ax.axvline(0, linestyle='--', color='red')
        exp_name = os.path.basename(exp.experiment_dir)
        ax.set_title('%s: Unit %s\nResponse to %s'
                     % (exp_name, unit_name, taste), fontsize=32)
        ax.set_xlabel('Time (ms)', fontsize=28)
        ax.set_ylabel('Smoothed Relative Firing Rate (Hz)', fontsize=28)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=len(valid_recs))
        ax.set_xlim(-1500, 2500)
        fig.savefig(save_file)
        plt.close('all')



def compute_MDS(exp):
    rec_dirs = exp.recording_dirs
    held_df = exp.held_units['units'].copy()
    rec_names = [os.path.basename(x) for x in rec_dirs]
    held_df = held_df.select_dtypes(include=['object'])

    # Only use units held over all sessions
    held_df = held_df.dropna(axis=0)
    N_units = len(held_df)

    early_window = [0, 750]
    late_window = [750, 1500]
    baseline_window = [-750, 0]

    taste_map, tastants = get_taste_mapping(rec_dirs)
    N_rows = 0
    for t, v in taste_map.items():
        N_rows += len(v)

    # Make matrix with row for each taste/session and column for each neuron
    # with values being the average firing rate (minus baseline) of the neuron in the given time
    # window (early or late, matrix for each)
    early_fr = np.zeros((N_rows, N_units))
    late_fr = np.zeros((N_rows, N_units))
    labels = []
    row_idx = 0
    for rd in rec_dirs:
        row = np.zeros((N_units,))
        rec_name = os.path.basename(rd)
        for t in tastants:
            if rec_name not in taste_map[t]:
                continue

            if 'pre' in rd or 'Train' in rd:
                labels.append((t,'pre'))
            else:
                labels.append((t,'post'))

            for u_idx, unit in enumerate(held_df[rec_name]):
                time, spike_train = dio.h5io.get_spike_data(rd, unit, taste_map[t][rec_name])
                b_idx = np.where((time >= baseline_window[0]) & (time <= baseline_window[1]))[0]
                e_idx = np.where((time >= early_window[0]) & (time <= early_window[1]))[0]
                l_idx = np.where((time >= late_window[0]) & (time <= late_window[1]))[0]

                baseline = np.mean(np.sum(spike_train[:, b_idx], axis=1) / np.diff(baseline_window)[0] / 1000)
                early = np.mean(np.sum(spike_train[:, e_idx], axis=1) / np.diff(early_window)[0] / 1000)
                early -= baseline
                late = np.mean(np.sum(spike_train[:, l_idx], axis=1) / np.diff(late_window)[0] / 1000)
                late -= baseline

                early_fr[row_idx, u_idx] = float(early)
                late_fr[row_idx, u_idx] = float(late)

            row_idx+=1


    # make symmetric distance matrix of the distances between rows
    # sklearn.metrics.pariwise.euclidean_distances
    # confirm its symmetric np.allclose(a, a.T)
    early_sim = euclidean_distances(early_fr)
    late_sim = euclidean_distances(late_fr)
    early_sim -= np.mean(early_sim)
    late_sim -= np.mean(late_sim)
    early_sim = early_sim / np.max(early_sim)
    late_sim = late_sim / np.max(late_sim)

    early_mds = MDS(n_components=2, dissimilarity='precomputed')
    early_pos = early_mds.fit_transform(1-early_sim)

    late_mds = MDS(n_components=2, dissimilarity='precomputed')
    late_pos = late_mds.fit_transform(1-late_sim)

    fig, ax = plt.subplots(nrows=2)
    cols = {'Saccharin':'green','Water':'blue','NaCl':'yellow','Citric Acid': 'orange', 'Quinine':'red'}
    markers = {'pre':'o', 'post':'s'}
    points = []
    leg_entry = []
    for l, ep, lp in zip(labels, early_pos, late_pos):
        if l[1] == 'pre':
            ax[0].scatter(ep[0], ep[1], label=l[0], marker='o', s=20, color=cols[l[0]])
            p = ax[1].scatter(lp[0], lp[1], label=l[0], marker='o', s=20, color=cols[l[0]])
            points.append(p)
            leg_entry.append(l[0])
        else:
            ax[0].scatter(ep[0], ep[1], marker='s', s=20, color=cols[l[0]])
            ax[1].scatter(lp[0], lp[1], marker='s', s=20, color=cols[l[0]])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax[0].set_xlabel('Dim 1', fontsize=20)
    ax[0].set_ylabel('Dim 2', fontsize=20)
    ax[0].set_title('Early response', fontsize=24)

    ax[1].set_xlabel('Dim 1', fontsize=20)
    ax[1].set_ylabel('Dim 2', fontsize=20)
    ax[1].set_title('Late response', fontsize=24)
    ax[1].legend(points, leg_entry, fontsize=18, ncol=2, bbox_to_anchor=(-0.5, -0.5))

    return fig, ax, early_mds, late_mds





def plot_magnitude_change(exp, win=[-500, 2000], exp_labels=None, save_dir=None, shell=False):
    if not isinstance(exp, list):
        exp = [exp]

    if save_dir is None:
        save_dir = userIO.get_file('Select save directory', shell=shell)

    tastants = set()
    exp_names = []
    for e in exp:
        _, t = get_taste_mapping(e.recording_dirs)
        tastants.update(t)
        exp_names.append(os.path.basename(e.experiment_dir))

    if exp_labels is None:
        exp_labels = exp_names

    for t in tastants:
        fig, ax = plt.subplots(figsize=(15, 7))
        labels = []
        for e, exp_name in zip(exp, exp_labels):
            if e.significant_units.get(t) is None:
                continue

            fn = os.path.join(e.experiment_dir, 'held_units_comparison',
                              t, 'metrics', 'avg_norm_mag_change.npz')
            mag = np.load(fn)
            y = mag['avg_norm_mag_change']
            time = mag['mag_time']
            y_err = mag['avg_norm_mag_change_SEM']

            t_idx = np.where((time >= win[0]) & (time <= win[1]))[0]
            time = time[t_idx]
            y = y[t_idx]
            y_err = y_err[t_idx]

            y = gaussian_filter1d(y, sigma=3)
            ax.fill_between(time, y - y_err, y + y_err, alpha=0.5)
            ax.plot(time, y, linewidth=3, label=exp_name)

        ax.axvline(0, linestyle='--', color='red')
        ax.set_xlabel('Time (ms)', fontsize=24)
        ax.set_ylabel('Magnitude of response change (Hz)', fontsize=24)
        ax.legend(fontsize=18)
        ax.set_title('Response Change\n%s' % t, fontsize=28)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.autoscale(axis='x', tight=True)
        save_file = os.path.join(save_dir, 'Response_Change-%s.svg' % t)
        fig.savefig(save_file)
        plt.close('all')
