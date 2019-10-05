from blechpy.datastructures import dataset, experiment
from blechpy.dio import h5io
from blechpy.analysis import spike_analysis as sas, stat_tests as stats


def compare_held_units(exp, response_change_significicance=0.05,
                       taste_responsive_significance=0.01):
    rec_names = list(exp.rec_labels.keys())
    rec_labels = exp.rec_labels.copy()
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
    out_df = None
    out_data = {}

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
        out_data[t] = {}
        for i, row in held_df.iterrows():
            unit_name = row['unit']
            elecrtrode = row['electrode']
            area = row['area']
            out_row = row.copy()
            out_row['tastant'] = t
            out_row['baseline_shift'] = False
            out_row['baseline_shift_p'] = 1.0
            out_row['taste_responsive_1'] = False
            out_row['taste_responsive_1_p'] = 1.0
            out_row['taste_responsive_2'] = False
            out_row['taste_resposive_2_p'] = 1.0
            out_row['response_change'] = False
            out_row['divergence_time'] = 0
            out_row['response_min_p'] = 1.0
            out_row['respons_max_p'] = 1.0
            out_row['norm_response_change'] = False
            out_row['norm_divergence_time'] = 0
            out_row['norm_response_min_p'] = 1.0
            out_row['norm_response_max_p'] = 1.0

            # Restrict to GC units
            #if area != 'GC':
            #    continue

            t_recs = [x for x in rec_names if taste_map[t].get(x) is not None]
            t_recs = sorted(t_recs, key=lambda x: rec_names.index(x))

            # TODO: For now only compare pairs
            if len(t_recs) != 2:
                continue

            if any([pd.isna(row[x]) for x in t_recs]):
                continue

            N_held[t] += 1
            sd = os.path.join(save_dir, t)
            if not os.path.exists(sd):
                os.mkdir(sd)

            rec_info = [(rec_labels.get(x), row.loc[x], taste_map[t][x]) for x in t_recs]
            unit_descrip = dio.h5io.get_unit_descriptor(rec_info[0][0], rec_info[0][1])
            out_dict = {'rec_info': rec_info, 'unit_descrip': unit_descrip}

            # Check taste responsiveness pre and post
            stat_pre, p_pre = stats.check_taste_response(*rec_info[0])
            stat_post, p_post = stats.check_taste_response(*rec_info[1])
            out_row['taste_responsive_1_p'] = p_pre
            out_row['taste_responsive_2_p'] = p_post
            if stat_pre <= taste_responsive_significance:
                out_row['taste_responsive_1'] = True

            if stat_post <= taste_responsive_significance:
                out_row['taste_responsive_2'] = True

            out_dict['taste_responsive_1_stat'] : stat_pre
            out_dict['taste_responsive_1_p'] : p_pre
            out_dict['taste_responsive_2_stat'] : stat_post
            out_dict['taste_responsive_2_p'] : p_post

            # compare baseline firing rate
            base_stats, base_p = compare_baseline(*rec_info[0], *rec_info[1])
            out_row['baseline_shift_p'] = base_p
            if base_p <= taste_responsive_significance:
                out_row['baseline_shift'] = True

            out_dict['baseline_stat'] = base_stats
            out_dict['baseline_p'] = base_p


            # Check if the response changed
            win_starts, resp_u, resp_p = compare_taste_response(*rec_info[0], *rec_info[1])
            if any(resp_p <= response_change_significicance):
                out_row['response_change'] = True
                idx = np.where(resp_p <= response_change_significicance)[0]
                out_row['divergence_time'] = win_starts[idx[0]]
                out_row['response_min_p'] = np.min(resp_p)
                out_row['response_max_p'] = np.max(resp_p)

            out_dict['response_change_time'] = win_starts
            out_dict['response_change_stats'] = resp_u
            out_dict['response_change_p'] = resp_p



            # Check is the normalized response changed
            norm_win_starts, norm_resp_u, norm_resp_p = compare_taste_response(*rec_info[0], *rec_info[0],
                                                                               norm_func=stats.remove_baseline)
            if any(norm_resp_p <= response_change_significicance):
                out_row['norm_response_change'] = True
                idx = np.where(norm_resp_p <= response_change_significicance)[0]
                out_row['norm_divergence_time'] = norm_win_starts[idx[0]]
                out_row['norm_response_min_p'] = np.min(norm_resp_p)
                out_row['norm_response_max_p'] = np.max(norm_resp_p)

            # Get magnitude of change
            mc, mc_SEM, mc_time = get_response_change(unit_name, *rec_info[0], *rec_info[1])

            if avg_mag_change is not None:
                avg_mag_change += np.abs(mc)
                avg_mag_change_SEM += np.power(mc_SEM, 2)
            else:
                avg_mag_change = np.abs(mc)
                avg_mag_change_SEM = np.power(mc_SEM, 2)
                mag_time = mc_time

            # Now with the baseline removed
            mc, mc_SEM, mc_time = get_response_change(unit_name, *rec_info[0], *rec_info[1],
                                                      norm_func=stats.remove_baseline)
            if avg_norm_mag_change is not None:
                avg_norm_mag_change += np.abs(mc)
                avg_norm_mag_change_SEM += np.power(mc_SEM, 2)
            else:
                avg_norm_mag_change = np.abs(mc)
                avg_norm_mag_change_SEM = np.power(mc_SEM,2)


            # Save metrics
            # Update out Row
            # Add row to output_df
            if out_df is None:
                out_df = pd.DataFrame([out_row])
            else:
                out_df = out_df.append(out_row)

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

def compare_baseline(rd1, u1, din1, rd2, u2, din2, win_size=1500):
    time1, spikes1 = h5io.get_spike_data(rd1, u1, din1)
    time2, spikes2 = h5io.get_spike_data(rd2, u2, din2)

    p_idx_1 = np.where((time1 >= -win_size) & (time1 < 0))[0]
    p_idx_2 = np.where((time2 >= -win_size) & (time2 < 0))[0]
    baseline1 = np.sum(spikes1[:, p_idx_1], axis=1) / abs(win_size)
    baseline2 = np.sum(spikes2[:, p_idx_2], axis=1) / abs(win_size)
    baseline1 = baseline1 * 1000 # convert to Hz
    baseline2 = baseline2 * 1000 # convert to Hz

    base_u, base_p = mannwhitneyu(baseline1, baseline2,
                                  alternative='two-sided')
    stats = {'u-stat': base_u, 'p-val': base_p,
             'baseline1': (np.mean(baseline1), sem(baseline1)),
             'baseline2': (np.mean(baseline2), sem(baseline2))}

    return stats, base_p


def compare_taste_response(rd1, u1, din1, rd2, u2, din2,
                           time_window=[0, 2000], bin_size=250,
                           norm_func=None):
    s_time1, spikes1 = h5io.get_spike_data(rd1, u1, din1)
    s_time2, spikes2 = h5io.get_spike_data(rd2, u2, din2)

    t_idx1 = np.where((s_time1 >= time_window[0]) & (s_time1 <= time_window[1]))[0]
    t_idx2 = np.where((s_time2 >= time_window[0]) & (s_time2 <= time_window[1]))[0]
    s_time1 = s_time1[t_idx1]
    spikes1 = spikes1[:, t_idx1]
    s_time2 = s_time2[t_idx2]
    spikes2 = spikes2[:, t_idx2]

    time1, fr1 = sas.get_binned_firing_rate(time1, spikes1, bin_size, bin_size)
    time2, fr2 = sas.get_binned_firing_rate(time2, spikes2, bin_size, bin_size)
    if norm_func is not None:
        fr1 = norm_func(time, fr1)
        fr2 = norm_func(time, fr2)

    win_starts = np.arange(time_window[0], time_window[1], bin_size)
    resp_u = np.zeros(win_starts.shape)
    resp_p = np.ones(win_starts.shape)
    for i, ws in enumerate(win_starts):
        rate1 = fr1[:, i]
        rate2 = fr2[:, i]
        try:
            resp_u[i], resp_p[i] = mannwhitneyu(rate1, rate2,
                                                alternative='two-sided')
        except ValueError:
            resp_u[i] = 0
            resp_p[i] = 1


    # Bonferroni correction
    resp_p = resp_p * len(win_starts)

    return win_starts, resp_u, resp_p

