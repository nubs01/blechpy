from blechpy.datastructures.objects import load_dataset
from blechpy.dio import h5io
from blechpy.utils import print_tools as pt, userIO
import numpy as np
import pandas as pd
import tables
from scipy.stats import f_oneway, ttest_ind, spearmanr, pearsonr, rankdata
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeavePOut, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import os
import itertools
import warnings



default_pal_id_params ={'window_size': 250, 'window_step': 25,
                        'num_comparison_bins': 5, 'comparison_bin_size': 250,
                        'discrim_p': 0.01, 'pal_deduce_start_time': 700,
                        'pal_deduce_end_time': 1200, 'unit_type': 'Single'}

def palatability_identity_calculations(rec_dir, pal_ranks=None,
                                       params=None, shell=False):
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    dat = load_dataset(rec_dir)
    dim = dat.dig_in_mapping
    if 'palatability_rank' in dim.columns:
        pass
    elif pal_ranks is None:
        dim = get_palatability_ranks(dim, shell=shell)
    else:
        dim['palatability_rank'] = dim['name'].map(pal_ranks)

    dim = dim.dropna(subset=['palatability_rank'])
    dim = dim[dim['palatability_rank'] > 0]
    dim = dim.reset_index(drop=True)
    num_tastes = len(dim)
    taste_names = dim.name.to_list()

    trial_list = dat.dig_in_trials.copy()
    trial_list = trial_list[[True if x in taste_names else False
                             for x in trial_list.name]]
    num_trials = trial_list.groupby('channel').count()['name'].unique()
    if len(num_trials) > 1:
        raise ValueError('Unequal number of trials for tastes to used')
    else:
        num_trials = num_trials[0]

    dim['num_trials'] = num_trials

    # Get which units to use
    unit_table = h5io.get_unit_table(rec_dir)
    unit_types = ['Single', 'Multi', 'All', 'Custom']
    unit_type = params.get('unit_type')
    if unit_type is None:
        q = userIO.ask_user('Which units do you want to use for taste '
                            'discrimination and  palatability analysis?',
                            choices=unit_types,
                            shell=shell)
        unit_type = unit_types[q]

    if unit_type == 'Single':
        chosen_units = unit_table.loc[unit_table['single_unit'],
                                      'unit_num'].to_list()
    elif unit_type == 'Multi':
        chosen_units = unit_table.loc[unit_table['single_unit'] == False,
                                      'unit_num'].to_list()
    elif unit_type == 'All':
        chosen_units = unit_table['unit_num'].to_list()
    else:
        selection = userIO.select_from_list('Select units to use:',
                                            unit_table['unit_num'],
                                            'Select Units',
                                            multi_select=True)
        chosen_units = list(map(int, selection))

    num_units = len(chosen_units)
    unit_table = unit_table.loc[chosen_units]

    # Enter Parameters
    if params is None or params.keys() != default_pal_id_params.keys():
        params = default_pal_id_params.copy()
        params = userIO.confirm_parameter_dict(params,
                                               ('Palatability/Identity '
                                                'Calculation Parameters'
                                                '\nTimes in ms'), shell=shell)

    win_size = params['window_size']
    win_step = params['window_step']
    print('Running palatability/identity calculations with parameters:\n%s' %
          pt.print_dict(params))

    with tables.open_file(dat.h5_file, 'r+') as hf5:
        trains_dig_in = hf5.list_nodes('/spike_trains')
        time = trains_dig_in[0].array_time[:]
        bin_times = np.arange(time[0], time[-1] - win_size + win_step,
                             win_step)
        num_bins = len(bin_times)

        palatability = np.empty((num_bins, num_units, num_tastes*num_trials),
                                dtype=int)
        identity = np.empty((num_bins, num_units, num_tastes*num_trials),
                            dtype=int)
        unscaled_response = np.empty((num_bins, num_units, num_tastes*num_trials),
                                     dtype=np.dtype('float64'))
        response  = np.empty((num_bins, num_units, num_tastes*num_trials),
                             dtype=np.dtype('float64'))
        laser = np.empty((num_bins, num_units, num_tastes*num_trials, 2),
                         dtype=float)

        # Fill arrays with data
        print('Filling data arrays...')
        onesies = np.ones((num_bins, num_units, num_trials))
        for i, row in dim.iterrows():
            idx = range(num_trials*i, num_trials*(i+1))
            palatability[:, :, idx] = row.palatability_rank * onesies
            identity[:, :, idx] = row.channel * onesies
            for j, u in enumerate(chosen_units):
                for k,t in enumerate(bin_times):
                    t_idx = np.where((time >= t) & (time <= t+win_size))[0]
                    unscaled_response[k, j, idx] = \
                            np.mean(trains_dig_in[i].spike_array[:, u, t_idx],
                                    axis=1)
                    try:
                        lasers[k, j, idx] = \
                            np.vstack((trains_dig_in[i].laser_durations[:],
                                       trains_dig_in[i].laser_onset_lag[:]))
                    except:
                        laser[k, j, idx] = np.zeros((num_trials, 2))

        # Scaling was not done, so:
        response = unscaled_response.copy()

        # Make ancillary_analysis node and put in arrays
        if '/ancillary_analysis' in hf5:
            hf5.remove_node('/ancillary_analysis', recursive=True)

        hf5.create_group('/', 'ancillary_analysis')
        hf5.create_array('/ancillary_analysis', 'palatability', palatability)
        hf5.create_array('/ancillary_analysis', 'identity', identity)
        hf5.create_array('/ancillary_analysis', 'laser', laser)
        hf5.create_array('/ancillary_analysis', 'scaled_neural_response',
                         response)
        hf5.create_array('/ancillary_analysis', 'window_params',
                         np.array([win_size, win_step]))
        hf5.create_array('/ancillary_analysis', 'bin_times', bin_times)
        hf5.create_array('/ancillary_analysis', 'unscaled_neural_response',
                         unscaled_response)

        # for backwards compatibility
        hf5.create_array('/ancillary_analysis', 'params',
                        np.array([win_size, win_step]))
        hf5.create_array('/ancillary_analysis', 'pre_stim', np.array(time[0]))
        hf5.flush()

        # Get unique laser (duration, lag) combinations
        print('Organizing trial data...')
        unique_lasers = np.vstack(list({tuple(row) for row in laser[0, 0, :, :]}))
        unique_lasers = unique_lasers[unique_lasers[:, 1].argsort(), :]
        num_conditions = unique_lasers.shape[0]
        trials = []
        for row in unique_lasers:
            tmp_trials = [j for j in range(num_trials * num_tastes)
                          if np.array_equal(laser[0, 0, j, :], row)]
            trials.append(tmp_trials)

        trials_per_condition = [len(x) for x in trials]
        if not all(x == trials_per_condition[0] for x in trials_per_condition):
            raise ValueError('Different number of trials for each laser condition')

        trials_per_condition = int(trials_per_condition[0] / num_tastes)  #assumes same number of trials per taste per condition
        print('Detected:\n    %i tastes\n    %i laser conditions\n'
              '    %i trials per condition per taste' %
              (num_tastes, num_conditions, trials_per_condition))
        trials = np.array(trials)

        # Store laser conditions and indices of trials per condition in trial x
        # taste space
        hf5.create_array('/ancillary_analysis', 'trials', trials)
        hf5.create_array('/ancillary_analysis', 'laser_combination_d_l',
                         unique_lasers)
        hf5.flush()

        # Taste Similarity Calculation
        neural_response_laser = np.empty((num_conditions, num_bins,
                                          num_tastes, num_units,
                                          trials_per_condition),
                                         dtype=np.dtype('float64'))
        taste_cosine_similarity = np.empty((num_conditions, num_bins,
                                            num_tastes, num_tastes),
                                           dtype=np.dtype('float64'))
        taste_euclidean_distance = np.empty((num_conditions, num_bins,
                                             num_tastes, num_tastes),
                                            dtype=np.dtype('float64'))

        # Re-format neural responses from bin x unit x (trial*taste) to
        # laser_condition x bin x taste x unit x trial
        print('Reformatting data arrays...')
        for i, trial in enumerate(trials):
            for j, _ in enumerate(bin_times):
                for k, _ in dim.iterrows():
                    idx = np.where((trial >= num_trials*k) &
                                   (trial < num_trials*(k+1)))[0]
                    neural_response_laser[i, j, k, :, :] = \
                            response[j, :, trial[idx]].T

        # Compute taste cosine similarity and euclidean distances
        print('Computing taste cosine similarity and euclidean distances...')
        for i, _ in enumerate(trials):
            for j, _ in enumerate(bin_times):
                for k, _ in dim.iterrows():
                    for l, _ in dim.iterrows():
                        taste_cosine_similarity[i, j, k, l] = \
                                np.mean(cosine_similarity(
                                    neural_response_laser[i, j, k, :, :].T,
                                    neural_response_laser[i, j, l, :, :].T))
                        taste_euclidean_distance[i, j, k, l] = \
                                np.mean(cdist(
                                    neural_response_laser[i, j, k, :, :].T,
                                    neural_response_laser[i, j, l, :, :].T,
                                    metric='euclidean'))

        hf5.create_array('/ancillary_analysis', 'taste_cosine_similarity',
                         taste_cosine_similarity)
        hf5.create_array('/ancillary_analysis', 'taste_euclidean_distance',
                         taste_euclidean_distance)
        hf5.flush()

        # Taste Responsiveness calculations
        bin_params = [params['num_comparison_bins'],
                      params['comparison_bin_size']]
        discrim_p = params['discrim_p']

        responsive_neurons = []
        discriminating_neurons = []
        taste_responsiveness = np.zeros((bin_params[0], num_units, 2))
        new_bin_times = np.arange(0, np.prod(bin_params), bin_params[1])
        baseline = np.where(bin_times < 0)[0]
        print('Computing taste responsiveness and taste discrimination...')
        for i, t in enumerate(new_bin_times):
            places = np.where((bin_times >= t) &
                              (bin_times <= t+bin_params[1]))[0]
            for j, u in enumerate(chosen_units):
                # Check taste responsiveness
                f, p = f_oneway(np.mean(response[places, j, :], axis=0),
                                np.mean(response[baseline, j, :], axis=0))
                if np.isnan(f):
                    f = 0.0
                    p = 1.0

                if p <= discrim_p and u not in responsive_neurons:
                    responsive_neurons.append(u)
                    taste_responsiveness[i, j, 0] = 1

                # Check taste discrimination
                taste_idx = [np.arange(num_trials*k, num_trials*(k+1))
                             for k in range(num_tastes)]
                taste_responses = [np.mean(response[places, j, :][:, k], axis=0)
                                   for k in taste_idx]
                f, p = f_oneway(*taste_responses)
                if np.isnan(f):
                    f = 0.0
                    p = 1.0

                if p <= discrim_p and u not in discriminating_neurons:
                    discriminating_neurons.append(u)

        responsive_neurons = np.sort(responsive_neurons)
        discriminating_neurons = np.sort(discriminating_neurons)

        # Write taste responsive and taste discriminating units to text file
        save_file = os.path.join(rec_dir, 'discriminative_responsive_neurons.txt')
        with open(save_file, 'w') as f:
            print('Taste discriminative neurons', file=f)
            for u in discriminating_neurons:
                print(u, file=f)

            print('Taste responsive neurons', file=f)
            for u in responsive_neurons:
                print(u, file=f)

        hf5.create_array('/ancillary_analysis', 'taste_disciminating_neurons',
                         discriminating_neurons)
        hf5.create_array('/ancillary_analysis', 'taste_responsive_neurons',
                         responsive_neurons)
        hf5.create_array('/ancillary_analysis', 'taste_responsiveness',
                         taste_responsiveness)
        hf5.flush()

        # Get time course of taste discrimibility
        print('Getting taste discrimination time course...')
        p_discrim = np.empty((num_conditions, num_bins, num_tastes, num_tastes,
                              num_units), dtype=np.dtype('float64'))
        for i in range(num_conditions):
            for j, t in enumerate(bin_times):
                for k in range(num_tastes):
                    for l in range(num_tastes):
                        for m in range(num_units):
                            _, p = ttest_ind(neural_response_laser[i, j, k, m, :],
                                             neural_response_laser[i, j, l, m, :],
                                             equal_var = False)
                            if np.isnan(p):
                                p = 1.0

                            p_discrim[i, j, k, l, m] = p

        hf5.create_array('/ancillary_analysis', 'p_discriminability',
                          p_discrim)
        hf5.flush()

        # Palatability Rank Order calculation (if > 2 tastes)
        t_start = params['pal_deduce_start_time']
        t_end = params['pal_deduce_end_time']
        if num_tastes > 2:
            print('Deducing palatability rank order...')
            palatability_rank_order_deduction(rec_dir, neural_response_laser,
                                              unique_lasers,
                                              bin_times, [t_start, t_end])

        # Palatability calculation
        r_spearman = np.zeros((num_conditions, num_bins, num_units))
        p_spearman = np.ones((num_conditions, num_bins, num_units))
        r_pearson = np.zeros((num_conditions, num_bins, num_units))
        p_pearson = np.ones((num_conditions, num_bins, num_units))
        f_identity = np.ones((num_conditions, num_bins, num_units))
        p_identity = np.ones((num_conditions, num_bins, num_units))
        lda_palatability = np.zeros((num_conditions, num_bins))
        lda_identity = np.zeros((num_conditions, num_bins))
        r_isotonic = np.zeros((num_conditions, num_bins, num_units))
        id_pal_regress = np.zeros((num_conditions, num_bins, num_units, 2))
        pairwise_identity = np.zeros((num_conditions, num_bins, num_tastes, num_tastes))
        print('Computing palatability metrics...')

        for i, t in enumerate(trials):
            for j in range(num_bins):
                for k in range(num_units):
                    ranks = rankdata(response[j, k, t])
                    r_spearman[i, j, k], p_spearman[i, j, k] = \
                            spearmanr(ranks, palatability[j, k, t])
                    r_pearson[i, j, k], p_pearson[i, j, k] = \
                            pearsonr(response[j, k, t], palatability[j, k, t])
                    if np.isnan(r_spearman[i, j, k]):
                        r_spearman[i, j, k] = 0.0
                        p_spearman[i, j, k] = 1.0

                    if np.isnan(r_pearson[i, j, k]):
                        r_pearson[i, j, k] = 0.0
                        p_pearson[i, j, k] = 1.0

                    # Isotonic regression of firing against palatability
                    model = IsotonicRegression(increasing = 'auto')
                    model.fit(palatability[j, k, t], response[j, k, t])
                    r_isotonic[i, j, k] = model.score(palatability[j, k, t],
                                                      response[j, k, t])

                    # Multiple Regression of firing rate against palatability and identity
                    # Regress palatability on identity
                    tmp_id = identity[j, k, t].reshape(-1, 1)
                    tmp_pal = palatability[j, k, t].reshape(-1, 1)
                    tmp_resp = response[j, k, t].reshape(-1, 1)
                    model_pi = LinearRegression()
                    model_pi.fit(tmp_id, tmp_pal)
                    pi_residuals = tmp_pal - model_pi.predict(tmp_id)

                    # Regress identity on palatability
                    model_ip = LinearRegression()
                    model_ip.fit(tmp_pal, tmp_id)
                    ip_residuals = tmp_id - model_ip.predict(tmp_pal)

                    # Regress firing on identity
                    model_fi = LinearRegression()
                    model_fi.fit(tmp_id, tmp_resp)
                    fi_residuals = tmp_resp - model_fi.predict(tmp_id)

                    # Regress firing on palatability
                    model_fp = LinearRegression()
                    model_fp.fit(tmp_pal, tmp_resp)
                    fp_residuals = tmp_resp - model_fp.predict(tmp_pal)

                    # Get partial correlation coefficient of response with identity
                    idp_reg0, p = pearsonr(fp_residuals, ip_residuals)
                    if np.isnan(idp_reg0):
                        idp_reg0 = 0.0

                    idp_reg1, p = pearsonr(fi_residuals, pi_residuals)
                    if np.isnan(idp_reg1):
                        idp_reg1 = 0.0

                    id_pal_regress[i, j, k, 0] = idp_reg0
                    id_pal_regress[i, j, k, 1] = idp_reg1

                    # Identity Calculation
                    samples = []
                    for _, row in dim.iterrows():
                        taste = row.channel
                        samples.append([trial for trial in t
                                        if identity[j, k, trial] == taste])

                    tmp_resp = [response[j, k, sample] for sample in samples]
                    f_identity[i, j, k], p_identity[i, j, k] = f_oneway(*tmp_resp)
                    if np.isnan(f_identity[i, j, k]):
                        f_identity[i, j, k] = 0.0
                        p_identity[i, j, k] = 1.0


                # Linear Discriminant analysis for palatability
                X = response[j, :, t]
                Y = palatability[j, 0, t]
                test_results = []
                c_validator = LeavePOut(1)
                for train, test in c_validator.split(X, Y):
                    model = LDA()
                    model.fit(X[train, :], Y[train])
                    tmp = np.mean(model.predict(X[test]) == Y[test])
                    test_results.append(tmp)

                lda_palatability[i, j] = np.mean(test_results)

                # Linear Discriminant analysis for identity
                Y = identity[j, 0, t]
                test_results = []
                c_validator = LeavePOut(1)
                for train, test in c_validator.split(X, Y):
                    model = LDA()
                    model.fit(X[train, :], Y[train])
                    tmp = np.mean(model.predict(X[test]) == Y[test])
                    test_results.append(tmp)

                lda_identity[i, j] = np.mean(test_results)

                # Pairwise Identity Calculation
                for ti1, r1 in dim.iterrows():
                    for ti2, r2 in dim.iterrows():
                        t1 = r1.channel
                        t2 = r2.channel
                        tmp_trials = np.where((identity[j, 0, :] == t1) |
                                              (identity[j, 0, :] == t2))[0]
                        idx = [trial for trial in t if trial in tmp_trials]
                        X = response[j, :, idx]
                        Y = identity[j, 0, idx]
                        test_results = []
                        c_validator = StratifiedShuffleSplit(n_splits=10,
                                                             test_size=0.25,
                                                             random_state=0)
                        for train, test in c_validator.split(X, Y):
                            model = GaussianNB()
                            model.fit(X[train, :], Y[train])
                            tmp_score = model.score(X[test, :], Y[test])
                            test_results.append(tmp_score)

                        pairwise_identity[i, j, ti1, ti2] = np.mean(test_results)

        hf5.create_array('/ancillary_analysis', 'r_pearson', r_pearson)
        hf5.create_array('/ancillary_analysis', 'r_spearman', r_spearman)
        hf5.create_array('/ancillary_analysis', 'p_pearson', p_pearson)
        hf5.create_array('/ancillary_analysis', 'p_spearman', p_spearman)
        hf5.create_array('/ancillary_analysis', 'lda_palatability', lda_palatability)
        hf5.create_array('/ancillary_analysis', 'lda_identity', lda_identity)
        hf5.create_array('/ancillary_analysis', 'r_isotonic', r_isotonic)
        hf5.create_array('/ancillary_analysis', 'id_pal_regress', id_pal_regress)
        hf5.create_array('/ancillary_analysis', 'f_identity', f_identity)
        hf5.create_array('/ancillary_analysis', 'p_identity', p_identity)
        hf5.create_array('/ancillary_analysis', 'pairwise_NB_identity', pairwise_identity)
        hf5.flush()

    warnings.filterwarnings('default', category=UserWarning)
    warnings.filterwarnings('default', category=RuntimeWarning)


def palatability_rank_order_deduction(rec_dir, response, lasers, time, window):
    num_conditions = response.shape[0]
    num_tastes = response.shape[2]
    num_units = response.shape[3]
    num_trials = response.shape[4]
    if num_tastes == 3:
        base_p_patterns = [[1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 2, 3]]
    elif num_tastes == 4:
        base_p_patterns = [[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2],
                           [1, 1, 2, 3], [1, 2, 2, 3], [1, 2, 3, 4]]

    save_file = os.path.join(rec_dir, 'palatability_deduction.txt')
    with open(save_file, 'w') as f:
        idx = np.where((time >= window[0]) & (time <= window[1]))[0]
        for i, ul in enumerate(lasers):
            print('Laser Condition: %s' % ul, file=f)
            for pattern in base_p_patterns:
                order = []
                corrs = []
                for per in itertools.permutations(pattern):
                    order.append(per)
                    this_corr = []
                    for j in range(num_units):
                        resp = np.mean(response[i, idx, :, j, :], axis=0)
                        resp = resp.T.reshape(-1)
                        comp = np.tile(per, num_trials)
                        tmp = pearsonr(resp, comp)[0]**2
                        this_corr.append(tmp)

                    corrs.append(np.mean(this_corr))

                max_order = order[np.argmax(corrs)]
                print('Base pattern: %s, Max pattern: %s, Max avg corr: %g'
                      % (pattern, max_order, np.max(corrs)), file=f)

            print("", file=f)


def get_palatability_ranks(dig_in_mapping, shell=True):
    '''Queries user for palatability rankings for digital inputs (tastants) and
    adds a column to dig_in_mapping DataFrame

    Parameters
    ----------
    dig_in_mapping: pandas.DataFrame,
        DataFrame with at least columns 'channel' and 'name', for mapping
        digital input channel number to a str name
    '''
    dim = dig_in_mapping.copy()
    tmp = dict.fromkeys(dim['name'], 0)
    filler = userIO.dictIO(tmp, shell=shell)
    tmp = filler.fill_dict('Rank Palatability\n1 for the lowest\n'
                           'Leave blank to exclude from palatability analysis')
    dim['palatability_rank'] = dim['name'].map(tmp)
    return dim
