from blechpy.datastructures import dataset, experiment


def compare_held_units(exp, response_change_significicance=0.05,
                       taste_responsive_significance=0.01):
    rec_names = list(exp.rec_labels.keys())
    rec_dirs = list(exp.rec_labels.values())
    exp_name = exp.data_name
    taste_map = exp.taste_map
    tastants = list(exp.taste_map.keys())
    save_dir = os.path.join(self.analysis_dir, 'held_units_comparison')
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

            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(30,10))
            ax[0].fill_between(mag_time, avg_mag_change - avg_mag_change_SEM,
                               avg_mag_change + avg_mag_change_SEM, alpha=0.4)
            ax[0].plot(mag_time, avg_mag_change, linewidth=2)
            ax[0].axvline(x=0, linestyle='--', color='red', alpha=0.6)
            ax[0].set_title('Average change in magnitude of response',fontsize=14)
            ax[0].set_xlabel('Time (ms)',fontsize=14)
            ax[0].set_ylabel('Firing Rate (Hz)',fontsize=14)

            ax[1].fill_between(mag_time, avg_norm_mag_change - avg_norm_mag_change_SEM,
                               avg_norm_mag_change + avg_norm_mag_change_SEM, alpha=0.4)
            ax[1].plot(mag_time, avg_norm_mag_change, linewidth=2)
            ax[1].axvline(x=0, linestyle='--', color='red', alpha=0.6)
            ax[1].set_title('Average change in magnitude of response\nBaseline removed',fontsize=14)
            ax[1].set_xlabel('Time (ms)',fontsize=14)

            ax[2].fill_between(mag_time, avg_z_mag_change - avg_z_mag_change_SEM,
                               avg_z_mag_change + avg_z_mag_change_SEM, alpha=0.4)
            ax[2].plot(mag_time, avg_z_mag_change, linewidth=2)
            ax[2].axvline(x=0, linestyle='--', color='red', alpha=0.6)
            ax[2].set_title('Average change in magnitude of response\nZ-scored to baseline',fontsize=14)
            ax[2].set_xlabel('Time (ms)',fontsize=14)

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
