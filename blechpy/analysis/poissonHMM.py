import os
import math
import numpy as np
import itertools as it
import pandas as pd
import tables
import time as sys_time
from numba import njit
from scipy.ndimage.filters import gaussian_filter1d
from blechpy.utils.particles import HMMInfoParticle
from blechpy import load_dataset
from blechpy.dio import h5io, hmmIO
from blechpy.plotting import hmm_plot as hmmplt
from blechpy.utils import math_tools as mt
from joblib import Parallel, delayed, Memory, cpu_count
from appdirs import user_cache_dir
cachedir = user_cache_dir('blechpy')
memory = Memory(cachedir, verbose=0)



TEST_PARAMS = {'n_cells': 10, 'n_states': 4, 'state_seq_length': 5,
               'trial_time': 3.5, 'dt': 0.001, 'max_rate': 50, 'n_trials': 15,
               'min_state_dur': 0.05, 'noise': 0.01, 'baseline_dur': 1}


HMM_PARAMS = {'hmm_id': None, 'taste': None, 'channel': None,
              'unit_type': 'single', 'dt': 0.001, 'threshold': 1e-7,
              'max_iter': 200, 'n_cells': None, 'n_trials': None,
              'time_start': -250, 'time_end': 2000, 'n_repeats': 25,
              'n_states': 3, 'fitted': False, 'area': 'GC',
              'hmm_class': 'PoissonHMM', 'notes': ''}


FACTORIAL_LOOKUP = np.array([math.factorial(x) for x in range(20)])
MIN_PROB = 1e-100


@njit
def fast_factorial(x):
    if x < len(FACTORIAL_LOOKUP):
        return FACTORIAL_LOOKUP[x]
    else:
        y = 1
        for i in range(1,x+1):
            y = y*i

        return y


@njit
def poisson(rate, n, dt):
    '''Gives probability of each neurons spike count assuming poisson spiking
    '''
    #tmp = np.power(rate*dt, n) / np.array([fast_factorial(x) for x in n])
    #tmp = tmp * np.exp(-rate*dt)
    tmp = n*np.log(rate*dt) - np.array([np.log(fast_factorial(x)) for x in n])
    tmp = tmp - rate*dt
    return np.exp(tmp)

@njit
def log_emission(rate, n , dt):
    return np.sum(np.log(poisson(rate, n, dt)))


@njit
def fix_arrays(PI,A,B):
    '''copy and remove zero values so that log probabilities can be computed
    '''
    PI = PI.copy()
    A = A.copy()
    B = B.copy()
    nx, ny = A.shape
    for i in range(nx):
        for j in range(ny):
            if A[i,j] == 0.:
                A[i,j] = MIN_PROB

    nx, ny = B.shape
    for i in range(nx):
        for j in range(ny):
            if B[i,j] == 0.:
                B[i,j] = MIN_PROB

    for i in range(len(PI)):
        if PI[i] == 0.:
            PI[i] = MIN_PROB

    return PI, A, B


@njit
def forward(spikes, dt, PI, A, B):
    '''Run forward algorithm to compute alpha = P(Xt = i| o1...ot, pi)
    Gives the probabilities of being in a specific state at each time point
    given the past observations and initial probabilities

    Parameters
    ----------
    spikes : np.array
        N x T matrix of spike counts with each entry ((i,j)) holding the # of
        spikes from neuron i in timebine j
    nStates : int, # of hidden states predicted to have generate the spikes
    dt : float, timebin in seconds (i.e. 0.001)
    PI : np.array
        nStates x 1 vector of initial state probabilities
    A : np.array
        nStates x nStates state transmission matrix with each entry ((i,j))
        giving the probability of transitioning from state i to state j
    B : np.array
        N x nSates rate matrix. Each entry ((i,j)) gives this predicited rate
        of neuron i in state j

    Returns
    -------
    alpha : np.array
        nStates x T matrix of forward probabilites. Each entry (i,j) gives
        P(Xt = i | o1,...,oj, pi)
    norms : np.array
        1 x T vector of norm used to normalize alpha to be a probability
        distribution and also to scale the outputs of the backward algorithm.
        norms(t) = sum(alpha(:,t))
    '''
    nTimeSteps = spikes.shape[1]
    nStates = A.shape[0]
    PI, A, B = fix_arrays(PI, A, B)

    # For each state, use the the initial state distribution and spike counts
    # to initialize alpha(:,1)
    #row = np.array([PI[i] * np.prod(poisson(B[:,i], spikes[:,0], dt))
    #row = np.array([np.log(PI[i]) + np.sum(np.log(poisson(B[:,i], spikes[:,0], dt)))
    #                for i in range(nStates)])
    a0 = [np.exp(np.log(PI[i]) + log_emission(B[:,i], spikes[:,0], dt))
          for i in range(nStates)]
    a0 = np.array(a0)
    alpha = np.zeros((nStates, nTimeSteps))
    norms = [np.sum(a0)]
    alpha[:, 0] = a0/norms[0]
    for t in range(1, nTimeSteps):
        for s in range(nStates):
            tmp_em = log_emission(B[:,s], spikes[:, t], dt)
            tmp_a = np.sum(np.exp(np.log(alpha[:, t-1]) + np.log(A[:,s])))
            tmp = np.exp(tmp_em + np.log(tmp_a))
            alpha[s,t] = tmp

        tmp_norm = np.sum(alpha[:,t])
        norms.append(tmp_norm)
        alpha[:, t] = alpha[:,t] / tmp_norm

    return alpha, norms


@njit
def backward(spikes, dt, A, B, norms):
    ''' Runs the backward algorithm to compute beta = P(ot+1...oT | Xt=s)
    Computes the probability of observing all future observations given the
    current state at each time point

    Paramters
    ---------
    spike : np.array, N x T matrix of spike counts
    nStates : int, # of hidden states predicted
    dt : float, timebin size in seconds
    A : np.array, nStates x nStates matrix of transition probabilities
    B : np.array, N x nStates matrix of estimated spike rates for each neuron

    Returns
    -------
    beta : np.array, nStates x T matrix of backward probabilities
    '''
    _, A, B = fix_arrays(np.array([0]), A, B)
    nTimeSteps = spikes.shape[1]
    nStates = A.shape[0]
    beta = np.zeros((nStates, nTimeSteps))
    beta[:, -1] = 1  # Initialize final beta to 1 for all states
    tStep = list(range(nTimeSteps-1))
    tStep.reverse()
    for t in tStep:
        for s in range(nStates):
            tmp_em = log_emission(B[:,s], spikes[:,t+1], dt)
            tmp_b = np.log(beta[:,t+1]) + np.log(A[s,:]) + tmp_em
            beta[s,t] = np.sum(np.exp(tmp_b))

        beta[:, t] = beta[:, t] / norms[t+1]

    return beta


@njit
def compute_baum_welch(spikes, dt, A, B, alpha, beta):
    _, A, B = fix_arrays(np.array([0]), A, B)
    nTimeSteps = spikes.shape[1]
    nStates = A.shape[0]
    gamma = np.zeros((nStates, nTimeSteps))
    epsilons = np.zeros((nStates, nStates, nTimeSteps-1))
    for t in range(nTimeSteps):
        tmp_g = np.exp(np.log(alpha[:, t]) + np.log(beta[:, t]))
        gamma[:, t] = tmp_g / np.sum(tmp_g)
        if t < nTimeSteps-1:
            epsilonNumerator = np.zeros((nStates, nStates))
            for si in range(nStates):
                for sj in range(nStates):
                    probs = log_emission(B[:, sj], spikes[:, t+1], dt)
                    tmp_en = (np.log(alpha[si, t]) + np.log(A[si, sj]) +
                              np.log(beta[sj, t+1]) + probs)
                    epsilonNumerator[si, sj] = np.exp(tmp_en)

            epsilons[:, :, t] = epsilonNumerator / np.sum(epsilonNumerator)

    return gamma, epsilons


@njit
def baum_welch(trial_dat, dt, PI, A, B):
    alpha, norms = forward(trial_dat, dt, PI, A, B)
    beta = backward(trial_dat, dt, A, B, norms)
    tmp_gamma, tmp_epsilons = compute_baum_welch(trial_dat, dt, A, B, alpha, beta)
    return tmp_gamma, tmp_epsilons, norms


def compute_new_matrices(spikes, dt, gammas, epsilons):
    nTrials, nCells, nTimeSteps = spikes.shape
    n_states = gammas.shape[1]
    minFR = 1/(nTimeSteps*dt)

    PI = np.mean(gammas[:, :, 0], axis=0)
    A = np.zeros((n_states, n_states))
    B = np.zeros((nCells, n_states))
    for si in range(n_states):
        for sj in range(n_states):
            Anumer = np.sum(epsilons[:, si, sj, :])
            Adenom = np.sum(gammas[:, si, -1])
            if np.isfinite(Adenom) and Adenom != 0.:
                A[si, sj] = Anumer / Adenom
            else:
                A[si, sj] = 0 # incase of floating point errors resulting in zeros

        #A[si, A[si,:] < 1e-50] = 0
        row = A[si,:]
        if np.sum(row) == 0.0:
            A[si, sj] = 1.0
        else:
            A[si, :] = A[si,:] / np.sum(row)


    for si in range(n_states):
        for tri in range(nTrials):
            for t in range(nTimeSteps-1):
                for u in range(nCells):
                    B[u,si] = B[u,si] + gammas[tri, si, t]*spikes[tri, u, t]

    # Convert and really small transition values into zeros
    #A[A < 1e-50] = 0
    #sums = np.sum(A, axis=1)
    #A = A/np.sum(A, axis=1) # This divides columns not rows
    #Bnumer = np.sum(np.array([np.matmul(tmp_y, tmp_g.T)
    #                          for tmp_y, tmp_g in zip(spikes, gammas)]),
    #                axis=0)
    Bdenom =  np.sum(np.sum(gammas, axis=2), axis=0)
    B = (B / Bdenom)/dt
    B[B < minFR] = minFR
    A[A <= MIN_PROB] = 0.0

    return PI, A, B


def poisson_viterbi_deprecated(spikes, dt, PI, A, B):
    '''
    Parameters
    ----------
    spikes : np.array, Neuron X Time matrix of spike counts
    PI : np.array, nStates x 1 vector of initial state probabilities
    A : np.array, nStates X nStates matric of state transition probabilities
    B : np.array, Neuron X States matrix of estimated firing rates
    dt : float, time step size in seconds

    Returns
    -------
    bestPath : np.array
        1 x Time vector of states representing the most likely hidden state
        sequence
    maxPathLogProb : float
        Log probability of the most likely state sequence
    T1 : np.array
        State X Time matrix where each entry (i,j) gives the log probability of
        the the most likely path so far ending in state i that generates
        observations o1,..., oj
    T2: np.array
        State X Time matrix of back pointers where each entry (i,j) gives the
        state x(j-1) on the most likely path so far ending in state i
    '''
    if A.shape[0] != A.shape[1]:
        raise ValueError('Transition matrix is not square')

    nStates = A.shape[0]
    nCells, nTimeSteps = spikes.shape
    # get rid of zeros for computation 
    A[np.where(A==0)] = 1e-300
    T1 = np.zeros((nStates, nTimeSteps))
    T2 = np.zeros((nStates, nTimeSteps))
    T1[:,0] = np.array([np.log(PI[i]) +
                        np.log(np.prod(poisson(B[:,i], spikes[:, 1], dt)))
                        for i in range(nStates)])
    for t, s in it.product(range(1,nTimeSteps), range(nStates)):
        probs = np.log(np.prod(poisson(B[:, s], spikes[:, t], dt)))
        vec2 = T1[:, t-1] + np.log(A[:,s])
        vec1 = vec2 + probs
        T1[s, t] = np.max(vec1)
        idx = np.argmax(vec1)
        T2[s, t] = idx

    bestPathEndState = np.argmax(T1[:, -1])
    maxPathLogProb = T1[bestPathEndState, -1]
    bestPath = np.zeros((nTimeSteps,))
    bestPath[-1] = bestPathEndState
    tStep = list(range(nTimeSteps-1))
    tStep.reverse()
    for t in tStep:
        bestPath[t] = T2[int(bestPath[t+1]), t+1]

    return bestPath, maxPathLogProb, T1, T2


def poisson_viterbi(spikes, dt, PI, A, B):
    n_states = A.shape[0]
    PI, A, B = fix_arrays(PI, A, B)
    n_cells, n_steps = spikes.shape
    T1 = np.ones((n_states, n_steps))*1e-300
    T2 = np.zeros((n_states, n_steps))
    T1[:, 0] = [np.log(PI[i])+np.sum(np.log(poisson(B[:,i], spikes[:,0], dt)))
                for i in range(n_states)]
    #for t,s in it.product(range(1,n_steps), range(n_states)):
    for t in range(1,n_steps):
        for s in range(n_states):
            probs = np.sum(np.log(poisson(B[:,s], spikes[:,t], dt)))
            vec1 = T1[:,t-1]+np.log(A[:,s])+probs
            T1[s,t] = np.max(vec1)
            T2[s,t] = np.argmax(vec1)

    best_end_state = np.argmax(T1[:,-1])
    max_log_prob = T1[best_end_state, -1]
    bestPath = np.zeros((n_steps,))
    bestPath[-1] = best_end_state
    tStep = list(range(n_steps-1))
    tStep.reverse()
    for t in tStep:
        bestPath[t] = T2[int(bestPath[t+1]), t+1]

    return bestPath, max_log_prob, T1, T2


def compute_BIC(PI, A, B, spikes=None, dt=None, maxLogProb=None, n_time_steps=None):
    if (maxLogProb is None or n_time_steps is None) and (spikes is None or dt is None):
        raise ValueError('Must provide max log prob and n_time_steps or spikes and dt')

    nParams = (A.shape[0]*(A.shape[1]-1) +
               (PI.shape[0]-1) +
               B.shape[0]*(B.shape[1]-1))
    if maxLogProb and n_time_steps:
        pass
    else:
        bestPaths, path_probs = compute_best_paths(spikes, dt, PI, A, B)
        maxLogProb = np.sum(path_probs)
        n_time_steps = spikes.shape[-1]

    BIC = -2 * maxLogProb + nParams * np.log(n_time_steps)
    return BIC, bestPaths, maxLogProb


def compute_hmm_cost(spikes, dt, PI, A, B, win_size=0.25, true_rates=None):
    if true_rates is None:
        true_rates = convert_spikes_to_rates(spikes, dt, win_size,
                                             step_size=win_size)

    BIC, bestPaths, maxLogProb = compute_BIC(PI, A, B, spikes=spikes, dt=dt)
    hmm_rates = generate_rate_array_from_state_seq(bestPaths, B, dt, win_size,
                                                   step_size=win_size)
    RMSE = compute_rate_rmse(true_rates, hmm_rates)
    return RMSE, BIC, bestPaths, maxLogProb


def compute_best_paths(spikes, dt, PI, A, B):
    if len(spikes.shape) == 2:
        spikes = np.array([spikes])

    nTrials, nCells, nTimeSteps = spikes.shape
    bestPaths = np.zeros((nTrials, nTimeSteps))-1
    pathProbs = np.zeros((nTrials,))

    for i, trial in enumerate(spikes):
        bestPaths[i,:], pathProbs[i], _, _ = poisson_viterbi(trial, dt, PI,
                                                             A, B)
    return bestPaths, pathProbs


@njit
def compute_rate_rmse(rates1, rates2):
    # Compute RMSE per trial
    # Mean over trials
    n_trials, n_cells, n_steps = rates1.shape
    RMSE = np.zeros((n_trials,))
    for i in range(n_trials):
        t1 = rates1[i, :, :]
        t2 = rates2[i, :, :]
        # Compute RMSE from euclidean distances at each time point
        distances = np.zeros((n_steps,))
        for j in range(n_steps):
            distances[j] =  mt.euclidean(t1[:,j], t2[:,j])

        RMSE[i] = np.sqrt(np.mean(np.power(distances,2)))

    return np.mean(RMSE)


def convert_path_state_numbers(paths, state_map):
    newPaths = np.zeros(paths.shape)
    for k,v in state_map.items():
        idx = np.where(paths == k)
        newPaths[idx] = v

    return newPaths


def match_states(emission1, emission2):
    '''Takes 2 Cell X State firing rate matrices and determines which states
    are most similar. Returns dict mapping emission2 states to emission1 states
    '''
    distances = np.zeros((emission1.shape[1], emission2.shape[1]))
    for x, y in it.product(range(emission1.shape[1]), range(emission2.shape[1])):
        tmp = mt.euclidean(emission1[:, x], emission2[:, y])
        distances[x, y] = tmp

    states = list(range(emission2.shape[1]))
    out = {}
    for i in range(emission2.shape[1]):
        s = np.argmin(distances[:,i])
        r = np.argmin(distances[s, :])
        if r == i and s in states:
            out[i] = s
            idx = np.where(states == s)[0]
            states.pop(int(idx))

    for i in range(emission2.shape[1]):
        if i not in out:
            s = np.argmin(distances[states, i])
            out[i] = states[s]

    return out


@memory.cache
@njit
def convert_spikes_to_rates(spikes, dt, win_size, step_size=None):
    if step_size is None:
        step_size = win_size

    n_trials, n_cells, n_steps = spikes.shape
    n_pts = int(win_size/dt)
    n_step_pts = int(step_size/dt)
    win_starts = np.arange(0, n_steps, n_step_pts)
    out = np.zeros((n_trials, n_cells, len(win_starts)))
    for i, w in enumerate(win_starts):
        out[:, :, i] = np.sum(spikes[:, :, w:w+n_pts], axis=2) / win_size

    return out


@memory.cache
@njit
def generate_rate_array_from_state_seq(bestPaths, B, dt, win_size,
                                       step_size=None):
    if not step_size:
        step_size = win_size

    n_trials, n_steps = bestPaths.shape
    n_cells, n_states = B.shape
    rates = np.zeros((n_trials, n_cells, n_steps))
    for j in range(n_trials):
        seq = bestPaths[j, :].astype(np.int64)
        rates[j, :, :] = B[:, seq]

    n_pts = int(win_size / dt)
    n_step_pts = int(step_size/dt)
    win_starts = np.arange(0, n_steps, n_step_pts)
    mean_rates = np.zeros((n_trials, n_cells, len(win_starts)))
    for i, w in enumerate(win_starts):
        mean_rates[:, :, i] = np.sum(rates[:, : , w:w+n_pts], axis=2) / n_pts

    return mean_rates


@memory.cache
@njit
def rebin_spike_array(spikes, dt, time, new_dt):
    if dt == new_dt:
        return spikes, time

    n_trials, n_cells, n_steps = spikes.shape
    n_bins = int(new_dt/dt)
    new_time = np.arange(time[0], time[-1], n_bins)
    new_spikes = np.zeros((n_trials, n_cells, len(new_time)))
    for i, w in enumerate(new_time):
        idx = np.where((time >= w) & (time < w+new_dt))[0]
        new_spikes[:,:,i] = np.sum(spikes[:,:,idx], axis=-1)

    return new_spikes.astype(np.int32), new_time


@memory.cache
def get_hmm_spike_data(rec_dir, unit_type, channel, time_start=None,
                       time_end=None, dt=None, trials=None, area=None):
    # unit type can be 'single', 'pyramidal', or 'interneuron', or a list of unit names
    if isinstance(unit_type, str):
        units = query_units(rec_dir, unit_type, area=area)
    elif isinstance(unit_type, list):
        units = unit_type

    time, spike_array = h5io.get_spike_data(rec_dir, units, channel, trials=trials)
    spike_array = spike_array.astype(np.int32)
    if len(units) == 1:
        spike_array = np.expand_dims(spike_array, 1)

    time = time.astype(np.float64)
    curr_dt = np.unique(np.diff(time))[0] / 1000
    if dt is not None and curr_dt < dt:
        print('%s: Rebinning Spike Array' % os.getpid())
        spike_array, time = rebin_spike_array(spike_array, curr_dt, time, dt)
    elif dt is not None and curr_dt > dt:
        raise ValueError('Cannot upsample spike array from %f sec '
                         'bins to %f sec bins' % (dt, curr_dt))
    else:
        dt = curr_dt

    if time_start is not None and time_end is not None:
        print('%s: Trimming spike array' % os.getpid())
        idx = np.where((time >= time_start) & (time < time_end))[0]
        time = time[idx]
        spike_array = spike_array[:, :, idx]

    return spike_array, dt, time


@memory.cache
def query_units(dat, unit_type, area=None):
    '''Returns the units names of all units in the dataset that match unit_type

    Parameters
    ----------
    dat : blechpy.dataset or str
        Can either be a dataset object or the str path to the recording
        directory containing that data .h5 object
    unit_type : str, {'single', 'pyramidal', 'interneuron', 'all'}
        determines whether to return 'single' units, 'pyramidal' (regular
        spiking single) units, 'interneuron' (fast spiking single) units, or
        'all' units
    area : str
        brain area of cells to return, must match area in
        dataset.electrode_mapping

    Returns
    -------
        list of str : unit_names
    '''
    if isinstance(dat, str):
        units = h5io.get_unit_table(dat)
        el_map = h5io.get_electrode_mapping(dat)
    else:
        units = dat.get_unit_table()
        el_map = dat.electrode_mapping.copy()

    u_str = unit_type.lower()
    q_str = ''
    if u_str == 'single':
        q_str = 'single_unit == True'
    elif u_str == 'pyramidal':
        q_str = 'single_unit == True and regular_spiking == True'
    elif u_str == 'interneuron':
        q_str = 'single_unit == True and fast_spiking == True'
    elif u_str == 'all':
        return units['unit_name'].tolist()
    else:
        raise ValueError('Invalid unit_type %s. Must be '
                         'single, pyramidal, interneuron or all' % u_str)

    units = units.query(q_str)
    if area is None or area == '' or area == 'None':
        return units['unit_name'].to_list()

    out = []
    el_map = el_map.set_index('Electrode')
    for i, row in units.iterrows():
        if el_map.loc[row['electrode'], 'area'] == area:
            out.append(row['unit_name'])

    return out


def fit_hmm_mp(rec_dir, params, h5_file=None, constraint_func=None):
    hmm_id = params['hmm_id']
    n_states = params['n_states']
    dt = params['dt']
    time_start = params['time_start']
    time_end = params['time_end']
    max_iter = params['max_iter']
    threshold = params['threshold']
    unit_type = params['unit_type']
    channels = params['channel']
    tastes = params['taste']
    n_trials = params['n_trials']
    if 'area' in params.keys():
        area = params['area']
    else:
        area = None

    if not isinstance(channels, list):
        channels = [channels]

    if not isinstance(tastes, list):
        tastes = [tastes]

    spikes = []
    row_id = []
    time = None
    for ch, tst in zip(channels, tastes):
        tmp_s, _, time = get_hmm_spike_data(rec_dir, unit_type, ch,
                                             time_start=time_start,
                                             time_end=time_end, dt=dt,
                                             trials=n_trials, area=area)
        tmp_id = np.vstack([(hmm_id, ch, tst, x) for x in range(tmp_s.shape[0])])
        spikes.append(tmp_s)
        row_id.append(tmp_id)

    spikes = np.vstack(spikes)
    row_id = np.vstack(row_id)

    if params['hmm_class'] == 'PoissonHMM':
        hmm = PoissonHMM(n_states, hmm_id=hmm_id)
    elif params['hmm_class'] == 'ConstrainedHMM':
        hmm = ConstrainedHMM(len(channels), hmm_id=hmm_id)

    hmm.randomize(spikes, dt, time, row_id=row_id, constraint_func=constraint_func)
    success = hmm.fit(spikes, dt, time, max_iter=max_iter, threshold=threshold)
    if not success:
        print('%s: Fitting Aborted for hmm %s' % (os.getpid(), hmm_id))
        if h5_file:
            return hmm_id, False
        else:
            return hmm_id, hmm

    # hmm = roll_back_hmm_to_best(hmm, spikes, dt, threshold)
    print('%s: Done Fitting for hmm %s' % (os.getpid(), hmm_id))
    written = False
    if h5_file:
        pid = os.getpid()
        lock_file = h5_file + '.lock'
        while os.path.exists(lock_file):
            print('%s: Waiting for file lock' % pid)
            sys_time.sleep(20)

        locked = True
        while locked:
            try:
                os.mknod(lock_file)
                locked=False
            except:
                sys_time.sleep(10)

        try:
            old_hmm, _, old_params = load_hmm_from_hdf5(h5_file, hmm_id)

            if old_hmm is None:
                print('%s: No existing HMM %s. Writing ...' % (pid, hmm_id))
                hmmIO.write_hmm_to_hdf5(h5_file, hmm, params)
                written = True
            else:
                print('%s: Existing HMM %s found. Comparing log likelihood ...' % (pid, hmm_id))
                print('New %.3E vs Old %.3E' % (hmm.fit_LL, old_hmm.fit_LL))
                if hmm.fit_LL > old_hmm.fit_LL:
                    print('%s: Replacing HMM %s due to higher log likelihood' % (pid, hmm_id))
                    hmmIO.write_hmm_to_hdf5(h5_file, hmm, params)
                    written = True

        except Exception as e:
            os.remove(lock_file)
            raise Exception(e)

        os.remove(lock_file)
        del old_hmm, hmm, spikes, dt, time
        return hmm_id, written
    else:
        return hmm_id, hmm


def load_hmm_from_hdf5(h5_file, hmm_id):
    hmm_id = int(hmm_id)
    try:
        existing_hmm = hmmIO.read_hmm_from_hdf5(h5_file, hmm_id)
    except:
        existing_hmm = None

    if existing_hmm is None:
        return None, None, None

    PI, A, B, stat_arrays, params = existing_hmm
    hmm = PoissonHMM(params['n_states'], hmm_id=hmm_id)
    hmm._init_history()
    hmm.initial_distribution = PI
    hmm.transition = A
    hmm.emission = B
    hmm.iteration = params['n_iterations']
    for k,v in stat_arrays.items():
        if k in hmm.stat_arrays.keys() and isinstance(hmm.stat_arrays[k], list):
            hmm.stat_arrays[k] = list(v)
        else:
            hmm.stat_arrays[k] = v

    hmm.BIC = params.pop('BIC')
    hmm.converged = params.pop('converged')
    hmm.fitted = params.pop('fitted')
    hmm.cost = params.pop('cost')
    hmm.fit_LL = params.pop('log_likelihood')
    hmm.max_log_prob = params.pop('max_log_prob')

    return hmm, stat_arrays['time'], params


def isConverged(hmm, thresh):
    '''Check HMM convergence based on the log-likelihood
    NOT WORKING YET
    '''
    pass


def check_ll_trend(hmm, thresh, n_iter=None):
    '''Check the trend of the log-likelihood to see if it has plateaued, is
    decreasing or is increasing
    '''
    if n_iter is None:
        n_iter = hmm.iteration

    ll_hist = np.array(hmm.stat_arrays['max_log_prob'])
    iterations = np.array(hmm.stat_arrays['iterations'])
    if n_iter not in iterations:
        raise ValueError('Iteration %i is not in history' % n_iter)

    idx = np.where(iterations <= n_iter)[0]
    ll_hist = ll_hist[idx]
    filt_ll = gaussian_filter1d(ll_hist, 4)
    diff_ll = np.diff(filt_ll)

    # Linear fit, if overall trend is decreasing, it fails
    z = np.polyfit(range(len(ll_hist)), filt_ll, 1)
    if z[0] <= 0:
        return 'decreasing'

    # Check if it has plateaued
    if all(np.abs(diff_ll[-5:]) <= thresh):
        return 'plateau'

    # if its a maxima and hasn't plateaued it needs to continue fitting
    if np.max(filt_ll) == filt_ll[-1]:
        return 'increasing'

    return 'flux'


def roll_back_hmm_to_best(hmm, spikes, dt, thresh):
    '''Looks at the log likelihood over fitting and determines the best
    iteration to have stopped at by choosing a local maxima during a period
    where the smoothed LL trace has plateaued
    '''
    ll_hist = np.array(hmm.stat_arrays['max_log_prob'])
    idx = np.where(np.isfinite(ll_hist))[0]
    if len(idx) == 0:
        return hmm

    iterations = np.array(hmm.stat_arrays['iterations'])
    ll_hist = ll_hist[idx]
    iterations = iterations[idx]
    filt_ll = gaussian_filter1d(ll_hist, 4)
    diff_ll = np.diff(filt_ll)
    below = np.where(np.abs(diff_ll) < thresh)[0] + 1 # since diff_ll is 1 smaller than ll_hist
    # Exclude maxima less than 50 iterations since its pretty spikey early on
    below = [x for x in below if (iterations[x] > 50)]
    # If there are none that fit criteria, just pick best past 50
    if len(below) == 0:
        below = np.where(iterations > 50)[0]

    if len(below) == 0:
        below = np.arange(len(iterations))

    below = below[below>2]

    tmp = [x for x in below if check_ll_trend(hmm, thresh, n_iter=iterations[x]) == 'plateau']
    if len(tmp) != 0:
        below = tmp

    maxima = np.argmax(ll_hist[below]) # this gives the index in below
    maxima = iterations[below[maxima]] # this is the iteration at which the maxima occurred
    hmm.roll_back(maxima, spikes=spikes, dt=dt)
    return hmm


def get_new_id(ids=None):
    if ids is None or len(ids) == 0:
        return 0

    nums = np.arange(0, np.max(ids) + 2)
    diff_nums = [x for x in nums if x not in ids]
    return np.min(diff_nums)


class PoissonHMM(object):
    def __init__(self, n_states, hmm_id=None):
        self.stat_arrays = {} # dict of cumulative stats to keep while fitting
                              # iterations, max_log_likelihood, fit log
                              # likelihood, cost, best_sequences, gamma
                              # probabilities, time, row_id
        self.n_states = n_states
        self.hmm_id = hmm_id

        self.transition = None
        self.emission = None
        self.initial_distribution = None

        self.fitted = False
        self.converged = False

        self.cost = None
        self.BIC = None
        self.max_log_prob = None
        self.fit_LL = None

    def randomize(self, spikes, dt, time, row_id=None, constraint_func=None):
        '''Initialize and randomize HMM matrices: initial_distribution (PI),
        transition (A) and emission/rates (B)
        Parameters
        ----------
        spikes : np.ndarray, dtype=int
            matrix of spike counts with dimensions trials x cells x time with binsize dt
        dt : float
            time step of spikes matrix in seconds
        time : np.ndarray
            1-D time vector corresponding to final dimension of spikes matrix,
            in milliseconds
        row_id : np.ndarray
            array to uniquely identify each row of the spikes array. This will
            thus identify each row of the best_sequences and gamma_probability
            matrices that are computed and stored
            useful when fitting a single HMM to trials with differing stimuli
        constrain_func : function
            user can provide a function that is used after randomization to
            constrain the PI, A and B matrices. The function must take PI, A, B
            as arguments and return PI, A, B.
        '''
        # setup parameters 
        # make transition matrix
        # all baseline states have equal probability of staying or changing
        # into each other and the early states
        # each early state has high stay probability and low chance to transition into 
        np.random.seed(None)
        n_trials, n_cells, n_steps = spikes.shape
        n_states = self.n_states

        # Initialize transition matrix with high stay probability
        # A is prob from going from state row to state column
        print('%s: Randomizing' % os.getpid())
        # Design transition matrix with large diagnonal and small everything else
        diag = np.abs(np.random.normal(.99, .01, n_states))
        A = np.abs(np.random.normal(0.01/(n_states-1), 0.01, (n_states, n_states)))
        for i in range(n_states):
            A[i, i] = diag[i]
            A[i,:] = A[i,:] / np.sum(A[i,:]) # normalize row to sum to 1

        # Initialize rate matrix ("Emission" matrix)
        spike_counts = np.sum(spikes, axis=2) / (len(time)*dt)
        mean_rates = np.mean(spike_counts, axis=0)
        std_rates = np.std(spike_counts, axis=0)
        B = np.vstack([np.abs(np.random.normal(x, y, n_states))
                       for x,y in zip(mean_rates, std_rates)])
        PI = np.ones((n_states,)) / n_states

        # RN10 preCTA fit better without constraining initial firing rate
        # mr = np.mean(np.sum(spikes[:, :, :int(500/dt)], axis=2), axis=0)
        # sr = np.std(np.sum(spikes[:, :, :int(500/dt)], axis=2), axis=0)
        # B[:, 0] = [np.abs(np.random.normal(x, y, 1))[0] for x,y in zip(mr, sr)]

        if constraint_func is not None:
            PI, A, B = constraint_func(PI, A, B)

        self.transition = A
        self.emission = B
        self.initial_distribution = PI
        self.fitted = False
        self.converged = False
        self.iteration = 0
        self.stat_arrays['row_id'] = row_id
        self._init_history()
        self.stat_arrays['gamma_probabilities'] = self.get_gamma_probabilities(spikes, dt)
        self.stat_arrays['time'] = time
        self._update_cost(spikes, dt)
        self.fit_LL = self.max_log_prob
        self._update_history()

    def _init_history(self):
        self.stat_arrays['cost'] = []
        self.stat_arrays['BIC'] = []
        self.stat_arrays['max_log_prob'] = []
        self.stat_arrays['fit_LL'] = []
        self.stat_arrays['iterations'] = []
        self.history = {'A': [], 'B': [], 'PI': [], 'iterations':[]}

    def _update_history(self):
        itr = self.iteration
        self.history['A'].append(self.transition)
        self.history['B'].append(self.emission)
        self.history['PI'].append(self.initial_distribution)
        self.history['iterations'].append(itr)

        self.stat_arrays['cost'].append(self.cost)
        self.stat_arrays['BIC'].append(self.BIC)
        self.stat_arrays['max_log_prob'].append(self.max_log_prob)
        self.stat_arrays['fit_LL'].append(self.fit_LL)
        self.stat_arrays['iterations'].append(itr)

    def fit(self, spikes, dt, time, max_iter = 500, threshold=1e-5, parallel=False):
        '''using parallels for processing trials actually seems to slow down
        processing (with 15 trials). Might still be useful if there is a very
        large nubmer of trials
        '''
        spikes = spikes.astype('int32')
        if (self.initial_distribution is None or
            self.transition is None or
            self.emission is None):
            raise ValueError('Must first initialize fit matrices either manually or via randomize')

        converged = False
        last_logl = None
        self.stat_arrays['time'] = time
        while (not converged and (self.iteration < max_iter)):
            self.fit_LL = self._step(spikes, dt, parallel=parallel)
            self._update_history()
            # if self.iteration >= 100:
            #     trend = check_ll_trend(self, threshold)
            #     if trend == 'decreasing':
            #         return False
            #     elif trend == 'plateau':
            #         converged = True

            if last_logl is None:
                delta_ll = np.abs(self.fit_LL)
            else:
                delta_ll = np.abs((last_logl - self.fit_LL)/self.fit_LL)

            if (last_logl is not None and
                np.isfinite(delta_ll) and
                delta_ll < threshold and
                np.isfinite(self.fit_LL) and
                self.iteration>2):
                # This log likelihood measure doesn't look right, the change
                # seems to always be 0
                # 8/24/20: Fixed, this is now a good measure
                converged = True
                print('%s: %s: Change in log likelihood converged' % (os.getpid(), self.hmm_id))

            last_logl = self.fit_LL

            # Convergence check is replaced by checking LL trend for plateau
            # converged = self.isConverged(convergence_thresh)
            print('%s: %s: Iter #%i complete. Log-likelihood is %.2E. Delta is %.2E'
                  % (os.getpid(), self.hmm_id, self.iteration, self.fit_LL, delta_ll))

        self.fitted = True
        self.converged = converged
        return True

    def _step(self, spikes, dt, parallel=False):
        if len(spikes.shape) == 2:
            spikes = np.expand_dims(spikes, 0)

        nTrials, nCells, nTimeSteps = spikes.shape

        A = self.transition
        B = self.emission
        PI = self.initial_distribution
        nStates = self.n_states

        # For multiple trials need to cmpute gamma and epsilon for every trial
        # and then update
        if parallel:
            n_cores = cpu_count() - 1
        else:
            n_cores = 1

        results = Parallel(n_jobs=n_cores)(delayed(baum_welch)(trial, dt, PI, A, B)
                                           for trial in spikes)
        gammas, epsilons, norms = zip(*results)
        gammas = np.array(gammas)
        epsilons = np.array(epsilons)
        norms = np.array(norms)
        #logl = np.sum(norms)
        logl = np.sum(np.log(norms))

        PI, A, B = compute_new_matrices(spikes, dt, gammas, epsilons)
        # Make sure rates are non-zeros for computations
        # B[np.where(B==0)] = 1e-300
        A[A < 1e-50] = 0.0
        for i in range(self.n_states):
            A[i,:] = A[i,:] / np.sum(A[i,:])

        self.transition = A
        self.emission = B
        self.initial_distribution = PI
        self.stat_arrays['gamma_probabilities'] = gammas
        self.iteration = self.iteration + 1
        self._update_cost(spikes, dt)
        return logl

    def get_best_paths(self, spikes, dt):
        if 'best_sequences' is self.stat_arrays.keys():
            return self.stat_arrays['best_sequences'], self.max_log_prob

        PI = self.initial_distribution
        A = self.transition
        B = self.emission

        bestPaths, pathProbs = compute_best_paths(spikes, dt, PI, A, B)
        return bestPaths, np.sum(pathProbs)

    def get_forward_probabilities(self, spikes, dt, parallel=False):
        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        if parallel:
            n_cpu = cpu_count() -1
        else:
            n_cpu = 1

        a_results = Parallel(n_jobs=n_cpu)(delayed(forward)
                                           (trial, dt, PI, A, B)
                                           for trial in spikes)
        alphas, norms = zip(*a_results)
        return np.array(alphas), np.array(norms)

    def get_backward_probabilities(self, spikes, dt, parallel=False):
        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        betas = []
        if parallel:
            n_cpu = cpu_count() -1
        else:
            n_cpu = 1

        a_results = Parallel(n_jobs=n_cpu)(delayed(forward)(trial, dt, PI, A, B)
                                         for trial in spikes)
        _, norms = zip(*a_results)
        b_results = Parallel(n_jobs=n_cpu)(delayed(backward)(trial, dt, A, B, n)
                                           for trial, n in zip(spikes, norms))
        betas = np.array(b_results)

        return betas

    def get_gamma_probabilities(self, spikes, dt, parallel=False):
        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        if parallel:
            n_cpu = cpu_count()-1
        else:
            n_cpu = 1

        results = Parallel(n_jobs=n_cpu)(delayed(baum_welch)(trial, dt, PI, A, B)
                                         for trial in spikes)
        gammas, _, _ = zip(*results)
        return np.array(gammas)

    def _update_cost(self, spikes, dt):
        spikes = spikes.astype('int')
        PI = self.initial_distribution
        A  = self.transition
        B  = self.emission
        cost, BIC, bestPaths, maxLogProb = compute_hmm_cost(spikes, dt, PI, A, B)
        self.cost = cost
        self.BIC = BIC
        self.max_log_prob = maxLogProb
        self.stat_arrays['best_sequences'] = bestPaths

    def roll_back(self, iteration, spikes=None, dt=None):
        itrs = self.history['iterations']
        idx = np.where(itrs == iteration)[0]
        if len(idx) == 0:
            raise ValueError('Iteration %i not found in history' % iteration)

        idx = idx[0]
        self.emission = self.history['B'][idx]
        self.transition = self.history['A'][idx]
        self.initial_distribution = self.history['PI'][idx]
        self.iteration = iteration

        itrs = self.stat_arrays['iterations']
        idx = np.where(itrs == iteration)[0][0]
        self.fit_LL = self.stat_arrays['fit_LL'][idx]
        self.max_log_prob = self.stat_arrays['max_log_prob'][idx]
        self.BIC = self.stat_arrays['BIC'][idx]
        self.cost = self.stat_arrays['cost'][idx]
        if spikes is not None and dt is not None:
            self.stat_arrays['gamma_probabilities'] = self.get_gamma_probabilities(spikes, dt)
            self._update_cost(spikes, dt)

        self._update_history()


class HmmHandler(object):
    def __init__(self, dat, save_dir=None):
        '''Takes a blechpy dataset object and fits HMMs for each tastant

        Parameters
        ----------
        dat: blechpy.dataset
        params: dict or list of dicts
            each dict must have fields:
                time_window: list of int, time window to cut around stimuli in ms
                convergence_thresh: float
                max_iter: int
                n_repeats: int
                unit_type: str, {'single', 'pyramidal', 'interneuron', 'all'}
                bin_size: time bin for spike array when fitting in seconds
                n_states: predicted number of states to fit
        '''
        if isinstance(dat, str):
            fd = dat
            dat = load_dataset(dat)
            if os.path.realpath(fd) != os.path.realpath(dat.root_dir):
                print('Changing dataset root_dir to match local directory')
                dat._change_root(fd)

            if dat is None:
                raise FileNotFoundError('No dataset.p file found given directory')

        if save_dir is None:
            save_dir = os.path.join(dat.root_dir,
                                    '%s_analysis' % dat.data_name)

        self._dataset = dat
        self.root_dir = dat.root_dir
        self.save_dir = save_dir
        self.h5_file = os.path.join(save_dir, '%s_HMM_Analysis.hdf5' % dat.data_name)
        self.load_params()

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.plot_dir = os.path.join(save_dir, 'HMM_Plots')
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        hmmIO.setup_hmm_hdf5(self.h5_file)
        # this function can be edited to account for parameters added in the
        # future
        # hmmIO.fix_hmm_overview(self.h5_file)

    def load_params(self):
        self._data_params = []
        self._fit_params = []
        h5_file = self.h5_file
        if not os.path.isfile(h5_file):
            return

        overview = self.get_data_overview()
        if overview.empty:
            return

        for i in overview.hmm_id:
            _, _, _, _, p = hmmIO.read_hmm_from_hdf5(h5_file, i)
            for k in list(p.keys()):
                if k not in HMM_PARAMS.keys():
                    _ = p.pop(k)

            self.add_params(p)

    def get_parameter_overview(self):
        df = pd.DataFrame(self._data_params)
        return df

    def get_data_overview(self):
        return hmmIO.get_hmm_overview_from_hdf5(self.h5_file)

    def run(self, parallel=True, overwrite=False, constraint_func=None):
        h5_file = self.h5_file
        rec_dir = self.root_dir
        if overwrite:
            fit_params = self._fit_params
        else:
            fit_params = [x for x in self._fit_params if not x['fitted']]

        if len(fit_params) == 0:
            return

        print('Running fittings')
        if parallel:
            n_cpu = np.min((cpu_count()-1, len(fit_params)))
        else:
            n_cpu = 1

        results = Parallel(n_jobs=n_cpu, verbose=100)(delayed(fit_hmm_mp)
                                                     (rec_dir, p, h5_file,
                                                      constraint_func)
                                                     for p in fit_params)


        memory.clear(warn=False)
        print('='*80)
        print('Fitting Complete')
        print('='*80)
        print('HMMs written to hdf5:')
        for hmm_id, written in results:
            print('%s : %s' % (hmm_id, written))

        #self.plot_saved_models()
        self.load_params()

    def plot_saved_models(self):
        print('Plotting saved models')
        data = self.get_data_overview().set_index('hmm_id')
        rec_dir = self.root_dir
        for i, row in data.iterrows():
            hmm, _, params = load_hmm_from_hdf5(self.h5_file, i)
            spikes, dt, time = get_hmm_spike_data(rec_dir, params['unit_type'],
                                                  params['channel'],
                                                  time_start=params['time_start'],
                                                  time_end=params['time_end'],
                                                  dt=params['dt'],
                                                  trials=params['n_trials'],
                                                  area=params['area'])
            plot_dir = os.path.join(self.plot_dir, 'hmm_%s' % i)
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            print('Plotting HMM %s...' % i)
            hmmplt.plot_hmm_figures(hmm, spikes, dt, time, save_dir=plot_dir)

    def add_params(self, params):
        if isinstance(params, list):
            for p in params:
                self.add_params(p)

            return
        elif not isinstance(params, dict):
            raise ValueError('Input must  be a dict or list of dicts')

        # Fill in blanks with defaults
        for k, v in HMM_PARAMS.items():
            if k not in params.keys():
                params[k] = v
                print('Parameter %s not provided. Using default value: %s'
                      % (k, repr(v)))

        # Grab existing parameters
        data_params = self._data_params
        fit_params = self._fit_params

        # Get taste and trial info from dataset
        dat = self._dataset
        dim = dat.dig_in_mapping.query('exclude == False and spike_array == True')
        if params['taste'] is None:
            tastes = dim['name'].tolist()
            single_taste = True
        elif isinstance(params['taste'], list):
            tastes = [t for t in params['taste'] if any(dim['name'] == t)]
            single_taste = False
        elif params['taste'] == 'all':
            tastes = dim['name'].tolist()
            single_taste = False
        else:
            tastes = [params['taste']]
            single_taste = True

        dim = dim.set_index('name')
        if not hasattr(dat, 'dig_in_trials'):
            dat.create_trial_list()

        trials = dat.dig_in_trials
        hmm_ids = [x['hmm_id'] for x in data_params]
        if single_taste:
            for t in tastes:
                p = params.copy()

                p['taste'] = t
                # Skip if parameter is already in parameter set
                if any([hmmIO.compare_hmm_params(p, dp) for dp in data_params]):
                    print('Parameter set already in data_params, '
                          'to re-fit run with overwrite=True')
                    continue

                if t not in dim.index:
                    print('Taste %s not found in dig_in_mapping or marked to exclude. Skipping...' % t)
                    continue

                if p['hmm_id'] is None:
                    hid = get_new_id(hmm_ids)
                    p['hmm_id'] = hid
                    hmm_ids.append(hid)

                p['channel'] = dim.loc[t, 'channel']
                unit_names = query_units(dat, p['unit_type'], area=p['area'])
                p['n_cells'] = len(unit_names)
                if p['n_trials'] is None:
                    p['n_trials'] = len(trials.query('name == @t'))

                data_params.append(p)
                for i in range(p['n_repeats']):
                    fit_params.append(p.copy())

        else:
            if any([hmmIO.compare_hmm_params(p, dp) for dp in data_params]):
                print('Parameter set already in data_params, '
                      'to re-fit run with overwrite=True')
                return

            channels = [dim.loc[x,'channel'] for x in tastes]
            params['taste'] = tastes
            params['channel'] = channels

            # this is basically meaningless right now, since this if clause
            # should only be used with ConstrainedHMM which will fit 5
            # baseline states and 2 states per taste
            params['n_states'] = params['n_states']*len(tastes)

            if params['hmm_id'] is None:
                hid = get_new_id(hmm_ids)
                params['hmm_id'] = hid
                hmm_ids.append(hid)

            unit_names = query_units(dat, params['unit_type'],
                                     area=params['area'])
            params['n_cells'] = len(unit_names)
            if params['n_trials'] is None:
                params['n_trials'] = len(trials.query('name == @t'))

            data_params.append(params)
            for i in range(params['n_repeats']):
                fit_params.append(params.copy())

        self._data_params = data_params
        self._fit_params = fit_params

    def get_hmm(self, hmm_id):
        return load_hmm_from_hdf5(self.h5_file, hmm_id)

    def delete_hmm(self, **kwargs):
        '''Deletes any HMMs whose parameters match the kwargs. i.e. n_states=2,
        taste="Saccharin" would delete all 2-state HMMs for Saccharin trials
        also reload parameters from hdf5, so any added but un-fit params will
        be lost
        '''
        hmmIO.delete_hmm_from_hdf5(self.h5_file, **kwargs)
        self.load_params()


def sequential_constraint(PI, A, B):
    '''Forces all states to occur sequentially
    Can be passed to HmmHandler.run() or fit_hmm_mp as the constraint_func
    argument

    Parameters
    ----------
    PI: np.ndarray, initial state probability vector
    A: np.ndarray, transition matrix
    B: np.ndarray, emission or rate matrix

    Returns
    -------
    np, ndarray, np.ndarray, np.ndarray : PI, A, B
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    for i in np.arange(n_states):
        if i > 0:
            A[i, :i] = 0.0

        if i < n_states-2:
            A[i, i+2:] = 0.0

        A[i, :] = A[i,:]/np.sum(A[i,:])

    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    return PI, A, B


class ConstrainedHMM(PoissonHMM):
    def __init__(self, n_tastes, n_baseline=3, hmm_id=None):
        self.stat_arrays = {} # dict of cumulative stats to keep while fitting
                              # iterations, max_log_likelihood, fit log
                              # likelihood, cost, best_sequences, gamma
                              # probabilities, time, row_id
        self.n_tastes = n_tastes
        self.n_baseline = n_baseline
        n_states = n_baseline + 2*n_tastes
        super().__init__(n_states, hmm_id=hmm_id)

    def randomize(self, spikes, dt, time, row_id=None, constraint_func=None):
        # setup parameters 
        # make transition matrix
        # all baseline states have equal probability of staying or changing
        # into each other and the early states
        # each early state has high stay probability and low chance to transition into 
        n_trials, n_cells, n_steps = spikes.shape
        n_tastes = self.n_tastes
        n_baseline = self.n_baseline
        n_states = n_baseline + n_tastes*2

        # Transition Matrix: state X state, A[i,j] is prob to go from state i to state j
        unit = 1/(n_baseline + n_tastes)
        A0 = np.random.normal(unit, 0.01, (n_baseline, n_baseline)).astype('float64')
        A1 = np.vstack([[unit, 0]*n_tastes]*n_baseline).astype('float64')
        A2 = np.zeros((n_tastes*2, n_baseline)).astype('float64')
        A3 = np.zeros((n_tastes*2, n_tastes*2)).astype('float64')
        for i in range(n_tastes):
            j = 2*i
            A3[j, j] = np.min((0.999, np.random.normal(0.98, 0.01, 1)))
            A3[j, j+1] = 1-A3[j,j]
            A3[j+1, j] = 0
            A3[j+1, j+1] = 1

        A = np.hstack((np.vstack((A0, A2)), np.vstack((A1, A3))))

        # Rate Matrix: cells X states, Bij is firing rate of cell i in state j
        b_idx = np.where(time < 0)[0]
        e_idx = np.where((time >= 0) & (time < np.max(time)/2))[0]
        l_idx = np.where(time >= np.max(time)/2)[0]
        if len(b_idx) == 0:
            b_idx = np.arange(n_steps)

        baseline = np.mean(np.sum(spikes[:, :, b_idx], axis=2), axis=0) / (len(b_idx)*dt)
        b_sd = np.std(np.sum(spikes[:, :, b_idx], axis=2), axis=0) / (len(b_idx)*dt)
        early = np.mean(np.sum(spikes[:, :, e_idx], axis=2), axis=0) / (len(e_idx)*dt)
        e_sd = np.std(np.sum(spikes[:, :, e_idx], axis=2), axis=0) / (len(e_idx)*dt)
        late = np.mean(np.sum(spikes[:, :, l_idx], axis=2), axis=0) / (len(l_idx)*dt)
        l_sd = np.std(np.sum(spikes[:, :, l_idx], axis=2), axis=0) / (len(l_idx)*dt)

        rates = np.zeros((n_cells, n_states))
        minFR = 1/n_steps
        for i in range(n_cells):
            row = [np.random.normal(baseline[i], b_sd[i], n_baseline)]
            for j in range(n_tastes):
                row.append(np.random.normal(early[i], e_sd[i], 1))
                row.append(np.random.normal(late[i], l_sd[i], 1))

            row = np.hstack(row)
            rates[i, :] = np.array([np.max((x, minFR)) for x in row])

        # Initial probabilities
        # Equal prob of all baseline states
        unit = 1/n_baseline
        PI = np.hstack([[np.random.normal(unit, 0.02, 1)[0]
                         for x in range(n_baseline)],
                        np.zeros((n_tastes*2,))])
        PI = PI/np.sum(PI)

        self.transition = A
        self.emission = rates
        self.initial_distribution = PI
        self.fitted = False
        self.converged = False
        self.iteration = 0
        self.stat_arrays['row_id'] = row_id
        self._init_history()
        self.stat_arrays['gamma_probabilities'] = self.get_gamma_probabilities(spikes, dt)
        self.stat_arrays['time'] = time
        self._update_cost(spikes, dt)
        self.fit_LL = self.max_log_prob
        self._update_history()
        self._update_history()

    def fit(self, spikes, dt, time, max_iter = 500, threshold=1e-5, parallel=False):
        '''using parallels for processing trials actually seems to slow down
        processing (with 15 trials). Might still be useful if there is a very
        large nubmer of trials
        '''
        spikes = spikes.astype('int32')
        if (self.initial_distribution is None or
            self.transition is None or
            self.emission is None):
            raise ValueError('Must first initialize fit matrices either manually or via randomize')

        converged = False
        last_logl = None
        self.stat_arrays['time'] = time
        while (not converged and (self.iteration < max_iter)):
            self.fit_LL = self._step(spikes, dt, parallel=parallel)
            self._update_history()
            # if self.iteration >= 100:
            #     trend = check_ll_trend(self, threshold)
            #     if trend == 'decreasing':
            #         return False
            #     elif trend == 'plateau':
            #         converged = True

            if last_logl is None:
                delta_ll = np.abs(self.fit_LL)
            else:
                delta_ll = np.abs((last_logl - self.fit_LL)/self.fit_LL)

            if (last_logl is not None and
                np.isfinite(delta_ll) and
                delta_ll < threshold and
                np.isfinite(self.fit_LL) and
                self.iteration>2):
                converged = True
                print('%s: %s: Change in log likelihood converged' % (os.getpid(), self.hmm_id))

            last_logl = self.fit_LL

            # Convergence check is replaced by checking LL trend for plateau
            # converged = self.isConverged(convergence_thresh)
            print('%s: %s: Iter #%i complete. Log-likelihood is %.2E. Delta is %.2E'
                  % (os.getpid(), self.hmm_id, self.iteration, self.fit_LL, delta_ll))

        self.fitted = True
        self.converged = converged
        return True

    def get_baseline_states(self):
        return np.arange(self.n_baseline)

    def get_early_states(self):
        return np.arange(self.n_baseline, self.n_states, 2)

    def get_late_states(self):
        return np.arange(self.n_baseline+1, self.n_states, 2)
