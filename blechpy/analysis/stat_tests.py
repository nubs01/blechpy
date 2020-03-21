import numpy as np
from blechpy import dio
from scipy.stats import mannwhitneyu, sem


def check_taste_response(rec_dir, unit_name, din, win_size=1500):
    h5_file = dio.h5io.get_h5_filename(rec_dir)
    if isinstance(unit_name, str):
        unit_num = dio.h5io.parse_unit_number(unit_name)
    elif isinstance(unit_name, int):
        unit_num = unit_name
        unit_name = 'unit%03i' % unit_num
    else:
        raise ValueError('%s is not a valid input type for unit_name. str or int only')

    din_str = 'dig_in_%i' % din

    time, spikes = dio.h5io.get_spike_data(rec_dir, unit_num, din)

    pre_idx = np.where((time >= -win_size) & (time < 0))[0]
    post_idx = np.where((time >= 0) & (time < win_size))[0]
    pre = 1000 * np.sum(spikes[:, pre_idx], axis=1) / win_size
    post = 1000 * np.sum(spikes[:, post_idx], axis=1) / win_size
    try:
        stat, pval = mannwhitneyu(pre, post, alternative='two-sided')
    except ValueError:
        pval = 1.0
        stat = 0.0


    mean_delta = get_mean_difference(pre, post)

    stats = {'u-stat': stat, 'p-val': pval, 'baseline': (np.mean(pre), sem(pre)),
             'response': (np.mean(post), sem(post)), 'delta': mean_delta}

    return pval, stats
