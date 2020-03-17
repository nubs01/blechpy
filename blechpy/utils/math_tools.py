import numpy as np
from numba import njit

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


@njit
def levenshtein(seq1, seq2):
    ''' Computes edit distance between 2 sequences
    '''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x

    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1],
                                   matrix[x, y-1] + 1)
            else:
                matrix [x,y] = min(matrix[x-1,y] + 1, matrix[x-1,y-1] + 1,
                                   matrix[x,y-1] + 1)

    return (matrix[size_x - 1, size_y - 1])


@njit
def euclidean(a, b):
    c = np.power(a-b,2)
    return np.sqrt(np.sum(c))
