import numpy as np

def group_consecutives(arr):
    '''Searches array for runs of consecutive numbers. Returns a list of lists
    of consecutive numbers found.

    Parameters
    ----------
    arr : list or np.array

    Returns
    -------
    list of lists
    '''
    diff_arr = np.diff(arr)
    change = np.where(diff_arr>1)[0] + 1
    out = []
    prev = 0
    for i in change:
        out.append(arr[prev:i])
        prev = i

    out.append(arr[prev:])
    return out
