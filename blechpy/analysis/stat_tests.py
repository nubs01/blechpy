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
    if len(A.shape) == 1 and len(B.shape) == 1:
        shape_ax = 0
    elif len(A.shape) != len(B.shape):
        raise ValueError('A and B must have same number of dimensions')

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
