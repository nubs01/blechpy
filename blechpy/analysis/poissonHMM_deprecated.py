import os
import math
import numpy as np
import itertools as it
import pylab as plt
import seaborn as sns
import pandas as pd
import multiprocessing as mp
import tables
#from scipy.spatial.distance import euclidean
from numba import njit
from blechpy.utils.particles import HMMInfoParticle
from blechpy import load_dataset
from blechpy.dio import h5io
from blechpy.plotting import hmm_plot as hmmplt
from joblib import Parallel, delayed, Memory
from appdirs import user_cache_dir
cachedir = user_cache_dir('blechpy')
memory = Memory(cachedir, verbose=0)



TEST_PARAMS = {'n_cells': 10, 'n_states': 4, 'state_seq_length': 5,
               'trial_time': 3.5, 'dt': 0.001, 'max_rate': 50, 'n_trials': 15,
               'min_state_dur': 0.05, 'noise': 0.01, 'baseline_dur': 1}

FACTORIAL_LOOKUP = np.array([math.factorial(x) for x in range(20)])


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
    tmp = np.power(rate*dt, n) / np.array([fast_factorial(x) for x in n])
    tmp = tmp * np.exp(-rate*dt)
    return tmp


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

    # For each state, use the the initial state distribution and spike counts
    # to initialize alpha(:,1)
    row = np.array([PI[i] * np.prod(poisson(B[:,i], spikes[:,0], dt))
                    for i in range(nStates)])
    alpha = np.zeros((nStates, nTimeSteps))
    norms = [np.sum(row)]
    alpha[:, 0] = row/norms[0]
    for t in range(1, nTimeSteps):
        tmp = np.array([np.prod(poisson(B[:, s], spikes[:, t], dt)) *
                        np.sum(alpha[:, t-1] * A[:,s])
                        for s in range(nStates)])
        tmp_norm = np.sum(tmp)
        norms.append(tmp_norm)
        tmp = tmp / tmp_norm
        alpha[:, t] = tmp

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
    nTimeSteps = spikes.shape[1]
    nStates = A.shape[0]
    beta = np.zeros((nStates, nTimeSteps))
    beta[:, -1] = 1  # Initialize final beta to 1 for all states
    tStep = list(range(nTimeSteps-1))
    tStep.reverse()
    for t in tStep:
        for s in range(nStates):
            beta[s,t] = np.sum((beta[:, t+1] * A[s,:]) *
                               np.prod(poisson(B[:, s], spikes[:, t+1], dt)))

        beta[:, t] = beta[:, t] / norms[t+1]

    return beta


@njit
def baum_welch(spikes, dt, A, B, alpha, beta):
    nTimeSteps = spikes.shape[1]
    nStates = A.shape[0]
    gamma = np.zeros((nStates, nTimeSteps))
    epsilons = np.zeros((nStates, nStates, nTimeSteps-1))
    for t in range(nTimeSteps):
        if t < nTimeSteps-1:
            gamma[:, t] = (alpha[:, t] * beta[:, t]) / np.sum(alpha[:,t] * beta[:,t])
            epsilonNumerator = np.zeros((nStates, nStates))
            for si in range(nStates):
                for sj in range(nStates):
                    probs = np.prod(poisson(B[:,sj], spikes[:, t+1], dt))
                    epsilonNumerator[si, sj] = (alpha[si, t]*A[si, sj]*
                                                beta[sj, t]*probs)

            epsilons[:, :, t] = epsilonNumerator / np.sum(epsilonNumerator)

    return gamma, epsilons


def isNotConverged(oldPI, oldA, oldB, PI, A, B, thresh=1e-4):
    dPI = np.sqrt(np.sum(np.power(oldPI - PI, 2)))
    dA = np.sqrt(np.sum(np.power(oldA - A, 2)))
    dB = np.sqrt(np.sum(np.power(oldB - B, 2)))
    print('dPI = %f,  dA = %f,  dB = %f' % (dPI, dA, dB))
    if all([x < thresh for x in [dPI, dA, dB]]):
        return False
    else:
        return True

def poisson_viterbi(spikes, dt, PI, A, B):
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
    T1 = np.zeros((nStates, nTimeSteps))
    T2 = np.zeros((nStates, nTimeSteps))
    T1[:,1] = np.array([np.log(PI[i]) +
                        np.log(np.prod(poisson(B[:,i], spikes[:, 1], dt)))
                        for i in range(nStates)])
    for t, s in it.product(range(1,nTimeSteps), range(nStates)):
        probs = np.log(np.prod(poisson(B[:, s], spikes[:, t], dt)))
        vec2 = T1[:, t-1] + np.log(A[:,s])
        vec1 = vec2 + probs
        T1[s, t] = np.max(vec1)
        idx = np.argmax(vec2)
        T2[s, t] = idx

    bestPathEndState = np.argmax(T1[:, -1])
    maxPathLogProb = T1[idx, -1]
    bestPath = np.zeros((nTimeSteps,))
    bestPath[-1] = bestPathEndState
    tStep = list(range(nTimeSteps-1))
    tStep.reverse()
    for t in tStep:
        bestPath[t] = T2[int(bestPath[t+1]), t+1]

    return bestPath, maxPathLogProb, T1, T2


class TestData(object):
    def __init__(self, params=None):
        if params is None:
            params = TEST_PARAMS.copy()
            param_str = '\t'+'\n\t'.join(repr(params)[1:-1].split(', '))
            print('Using default parameters:\n%s' % param_str)

        self.params = params.copy()
        self.generate()

    def generate(self, params=None):
        print('-'*80)
        print('Simulating Data')
        print('-'*80)
        if params is not None:
            self.params.update(params)

        params = self.params
        param_str = '\t'+'\n\t'.join(repr(params)[1:-1].split(', '))
        print('Parameters:\n%s' % param_str)

        self._generate_ground_truth()
        self._generate_spike_trains()

    def _generate_ground_truth(self):
        print('Generating ground truth state sequence...')
        params = self.params
        nStates = params['n_states']
        seqLen = params['state_seq_length']
        minSeqDur = params['min_state_dur']
        baseline_dur = params['baseline_dur']
        maxFR = params['max_rate']
        nCells = params['n_cells']
        trialTime = params['trial_time']
        nTrials = params['n_trials']
        dt = params['dt']
        nTimeSteps = int(trialTime/dt)

        T = trialTime
        # Figure out a random state sequence and state durations
        stateSeq = np.random.randint(0, nStates, seqLen)
        stateSeq = np.array([0, *np.random.randint(0,nStates, seqLen-1)])
        stateDurs = np.zeros((nTrials, seqLen))
        for i in range(nTrials):
            tmp = np.abs(np.random.rand(seqLen-1))
            tmp = tmp * ((trialTime - baseline_dur) / np.sum(tmp))
            stateDurs[i, :] = np.array([baseline_dur, *tmp])

        # Make vector of state at each time point
        stateVec = np.zeros((nTrials, nTimeSteps))
        for trial in range(nTrials):
            t0 = 0
            for state, dur in zip(stateSeq, stateDurs[trial]):
                tn = int(dur/dt)
                stateVec[trial, t0:t0+tn] = state
                t0 += tn

        # Determine firing rates per neuron per state
        # For each neuron generate a mean firing rate and then draw state
        # firing rates from a normal distribution around that with 10Hz
        # variance
        mean_rates = np.random.rand(nCells, 1) * maxFR
        stateRates = np.zeros((nCells, nStates))
        for i, r in enumerate(mean_rates):
            stateRates[i, :] = np.array([r, *np.abs(np.random.normal(r, .5*r, nStates-1))])

        self.ground_truth = {'state_sequence': stateSeq,
                             'state_durations': stateDurs,
                             'firing_rates': stateRates,
                             'state_vectors': stateVec}

    def _generate_spike_trains(self):
        print('Generating new spike trains...')
        params = self.params
        nCells = params['n_cells']
        trialTime = params['trial_time']
        dt = params['dt']
        nTrials = params['n_trials']
        noise = params['noise']
        nTimeSteps = int(trialTime/dt)

        stateRates = self.ground_truth['firing_rates']
        stateVec = self.ground_truth['state_vectors']


        # Make spike arrays
        # Trial x Neuron x Time
        random_nums = np.abs(np.random.rand(nTrials, nCells, nTimeSteps))
        rate_arr = np.zeros((nTrials, nCells, nTimeSteps))
        for trial, cell, t in  it.product(range(nTrials), range(nCells), range(nTimeSteps)):
            state = int(stateVec[trial, t])
            mean_rate = stateRates[cell, state]
            # draw noisy rates from normal distrib with mean rate from ground
            # truth and width as noise*mean_rate
            r = np.random.normal(mean_rate, mean_rate*noise)
            rate_arr[trial, cell, t] = r

        spikes = (random_nums <= rate_arr *dt).astype('int')

        self.spike_trains = spikes

    def get_spike_trains(self):
        if not hasattr(self, 'spike_trains'):
            self._generate_spike_trains()

        return self.spike_trains

    def get_ground_truth(self):
        if not hasattr(self, 'ground_truth'):
            self._generate_ground_truth()

        return self.ground_truth

    def plot_state_rates(self, ax=None):
        fig, ax = plot_state_rates(self.ground_truth['firing_rates'], ax=ax)
        return fig, ax

    def plot_state_raster(self, ax=None):
        fig, ax = plot_state_raster(self.spike_trains,
                                    self.ground_truth['state_vectors'],
                                    self.params['dt'], ax=ax)
        return fig, ax


class PoissonHMM(object):
    '''Poisson implementation of Hidden Markov Model for fitting spike data
    from a neuronal population
    Author: Roshan Nanu
    Adpated from code by Ben Ballintyn
    '''
    def __init__(self, n_predicted_states, spikes, dt,
                 max_history=500, cost_window=0.25, set_data=None):
        if len(spikes.shape) == 2:
            spikes = np.array([spikes])

        self.data = spikes.astype('int32')
        self.dt = dt
        self._rate_data = None
        self.n_states = n_predicted_states
        self._cost_window = cost_window
        self._max_history = max_history
        self.cost = None
        self.BIC = None
        self.best_sequences = None
        self.max_log_prob = None
        self._rate_data = None
        self.history = None
        self._compute_data_rate_array()
        if set_data is None:
            self.randomize()
        else:
            self.fitted = set_data['fitted']
            self.initial_distribution = set_data['initial_distribution']
            self.transition = set_data['transition']
            self.emission = set_data['emission']
            self.iteration = 0
            self._update_cost()


    def randomize(self):
        nStates = self.n_states
        spikes = self.data
        dt = self.dt
        n_trials, n_cells, n_steps = spikes.shape
        total_time = n_steps * dt

        # Initialize transition matrix with high stay probability
        print('Randomizing')
        diag = np.abs(np.random.normal(.99, .01, nStates))
        A = np.abs(np.random.normal(0.01/(nStates-1), 0.01, (nStates, nStates)))
        for i in range(nStates):
            A[i, i] = diag[i]
            A[i,:] = A[i,:] / np.sum(A[i,:])

        # Initialize rate matrix ("Emission" matrix)
        spike_counts = np.sum(spikes, axis=2) / total_time
        mean_rates = np.mean(spike_counts, axis=0)
        std_rates = np.std(spike_counts, axis=0)
        B = np.vstack([np.abs(np.random.normal(x, y, nStates))
                       for x,y in zip(mean_rates, std_rates)])
        # B = np.random.rand(nCells, nStates)

        self.transition = A
        self.emission = B
        self.initial_distribution = np.ones((nStates,)) / nStates
        self.iteration = 0
        self.fitted = False
        self.history = None
        self._update_cost()

    def fit(self, spikes=None, dt=None, max_iter = 1000, convergence_thresh = 1e-4,
            parallel=False):
        '''using parallels for processing trials actually seems to slow down
        processing (with 15 trials). Might still be useful if there is a very
        large nubmer of trials
        '''
        if self.fitted:
            return

        if spikes is not None:
            spikes = spikes.astype('int32')
            self.data = spikes
            self.dt = dt
        else:
            spikes = self.data
            dt = self.dt

        while (not self.isConverged(convergence_thresh) and
               (self.iteration < max_iter)):
            self._step(spikes, dt, parallel=parallel)
            print('Iter #%i complete.' % self.iteration)

        self.fitted = True

    def _step(self, spikes, dt, parallel=False):
        if len(spikes.shape) == 2:
            spikes = np.array([spikes])

        nTrials, nCells, nTimeSteps = spikes.shape

        A = self.transition
        B = self.emission
        PI = self.initial_distribution
        nStates = self.n_states

        # For multiple trials need to cmpute gamma and epsilon for every trial
        # and then update
        gammas = np.zeros((nTrials, nStates, nTimeSteps))
        epsilons = np.zeros((nTrials, nStates, nStates, nTimeSteps-1))
        if parallel:
            def update(ans):
                idx = ans[0]
                gammas[idx, :, :] = ans[1]
                epsilons[idx, :, :, :] = ans[2]

            def error(ans):
                raise RuntimeError(ans)

            n_cores = mp.cpu_count() - 1
            pool = mp.get_context('spawn').Pool(n_cores)
            for i, trial in enumerate(spikes):
                pool.apply_async(wrap_baum_welch,
                                 (i, trial, dt, PI, A, B),
                                 callback=update, error_callback=error)

            pool.close()
            pool.join()
        else:
            for i, trial in enumerate(spikes):
                _, tmp_gamma, tmp_epsilons = wrap_baum_welch(i, trial, dt, PI, A, B)
                gammas[i, :, :] = tmp_gamma
                epsilons[i, :, :, :] = tmp_epsilons

        # Store old parameters for convergence check
        self.update_history()

        PI, A, B = compute_new_matrices(spikes, dt, gammas, epsilons)
        self.transition = A
        self.emission = B
        self.initial_distribution = PI
        self.iteration += 1
        self._update_cost()

    def update_history(self):
        A = self.transition
        B = self.emission
        PI = self.initial_distribution
        BIC = self.BIC
        cost = self.cost
        iteration = self.iteration

        if self.history is None:
            self.history = {}
            self.history['A'] = [A]
            self.history['B'] = [B]
            self.history['PI'] = [PI]
            self.history['iterations'] = [iteration]
            self.history['cost'] = [cost]
            self.history['BIC'] = [BIC]
        else:
            if iteration in self.history['iterations']:
                return self.history

            self.history['A'].append(A)
            self.history['B'].append(B)
            self.history['PI'].append(PI)
            self.history['iterations'].append(iteration)
            self.history['cost'].append(cost)
            self.history['BIC'].append(BIC)

        if len(self.history['iterations']) > self._max_history:
            nmax = self._max_history
            for k, v in self.history.items():
                self.history[k] = v[-nmax:]

        return self.history

    def isConverged(self, thresh):
        if self.history is None:
            return False

        oldPI = self.history['PI'][-1]
        oldA = self.history['A'][-1]
        oldB = self.history['B'][-1]
        oldCost = self.history['cost'][-1]

        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        cost = self.cost

        dPI = np.sqrt(np.sum(np.power(oldPI - PI, 2)))
        dA = np.sqrt(np.sum(np.power(oldA - A, 2)))
        dB = np.sqrt(np.sum(np.power(oldB - B, 2)))
        dCost = cost-oldCost
        print('dPI = %f,  dA = %f,  dB = %f, dCost = %f, cost = %f'
              % (dPI, dA, dB, dCost, cost))

        # TODO: determine if this is reasonable
        # dB takes waaaaay longer to converge than the rest, i'm going to
        # double the thresh just for that
        dB = dB/2

        if not all([x < thresh for x in [dPI, dA, dB]]):
            return False
        else:
            return True

    def get_best_paths(self):
        if self.best_sequences is not None:
            return self.best_sequences, self.max_log_prob

        spikes = self.data
        dt = self.dt
        PI = self.initial_distribution
        A = self.transition
        B = self.emission

        bestPaths, pathProbs = compute_best_paths(spikes, dt, PI, A, B)
        self.best_sequences = bestPaths
        self.max_log_prob = np.max(pathProbs)
        return bestPaths, self.max_log_prob

    def get_forward_probabilities(self):
        alphas = []
        for trial in self.data:
            tmp, _ = forward(trial, self.dt, self.initial_distribution,
                             self.transition, self.emission)
            alphas.append(tmp)

        return np.array(alphas)

    def get_backward_probabilities(self):
        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        betas = []
        for trial in self.data:
            alpha, norms = forward(trial, self.dt, PI, A, B)
            tmp = backward(trial, self.dt, A, B, norms)
            betas.append(tmp)

        return np.array(betas)

    def get_gamma_probabilities(self):
        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        gammas = []
        for i, trial in enumerate(self.data):
            _, tmp, _ = wrap_baum_welch(i, trial, self.dt, PI, A, B)
            gammas.append(tmp)

        return np.array(gammas)

    def get_BIC(self):
        if self.BIC is not None:
            return self.BIC

        PI = self.initial_distribution
        A = self.transition
        B = self.emission
        BIC, bestPaths, max_log_prob = compute_BIC(self.data, self.dt, PI, A, B)
        self.BIC = BIC
        self.best_sequences = bestPaths
        self.max_log_prob = max_log_prob
        return BIC, bestPaths, max_log_prob

    def _compute_data_rate_array(self):
        if self._rate_data is not None:
            return self._rate_data

        win_size = self._cost_window
        rate_array = convert_spikes_to_rates(self.data, self.dt,
                                             win_size, step_size=win_size)
        self._rate_data = rate_array

    def _compute_predicted_rate_array(self):
        B = self.emission
        bestPaths, _ = self.get_best_paths()
        bestPaths = bestPaths.astype('int32')
        win_size = self._cost_window
        dt = self.dt
        mean_rates = generate_rate_array_from_state_seq(bestPaths, B,
                                                        dt, win_size,
                                                        step_size=win_size)
        return mean_rates

    def set_to_lowest_cost(self):
        hist = self.update_history()
        idx = np.argmin(hist['cost'])
        iteration = hist['iterations'][idx]
        self.roll_back(iteration)

    def set_to_lowest_BIC(self):
        hist = self.update_history()
        idx = np.argmin(hist['BIC'])
        iteration = hist['iterations'][idx]
        self.roll_back(iteration)

    def find_best_in_history(self):
        hist = self.update_history()
        PIs = hist['PI']
        As = hist['A']
        Bs = hist['B']
        iters = hist['iterations']
        BICs = hist['BIC']
        idx = np.argmin(BICs)
        out = {'PI': PIs[idx], 'A': As[idx], 'B': Bs[idx]}
        return out, iters[idx], BICs

    def roll_back(self, iteration):
        hist = self.history
        try:
            idx = hist['iterations'].index(iteration)
        except ValueError:
            raise ValueError('Iteration %i not found in history' % iteration)

        self.initial_distribution = hist['PI'][idx]
        self.transition = hist['A'][idx]
        self.emission = hist['B'][idx]
        self.iteration = iteration
        self._update_cost()

    def set_matrices(self, new_mats):
        self.initial_distribution = new_mats['PI']
        self.transition = new_mats['A']
        self.emission = new_mats['B']
        if 'iteration' in new_mats:
            self.iteration = new_mats['iteration']
        self._update_cost()

    def set_data(self, new_data, dt):
        self.data = new_data
        self.dt = dt
        self._compute_data_rate_array()
        self._update_cost()

    def plot_state_raster(self, ax=None, state_map=None):
        bestPaths, _ = self.get_best_paths()
        if state_map is not None:
            bestPaths = convert_path_state_numbers(bestPaths, state_map)

        data = self.data
        fig, ax = plot_state_raster(data, bestPaths, self.dt, ax=ax)
        return fig, ax

    def plot_state_rates(self, ax=None, state_map=None):
        rates = self.emission
        if state_map:
            idx = [state_map[k] for k in sorted(state_map.keys())]
            maxState = np.max(list(state_map.values()))
            newRates = np.zeros((rates.shape[0], maxState+1))
            for k, v in state_map.items():
                newRates[:, v] = rates[:, k]

            rates = newRates

        fig, ax = plot_state_rates(rates, ax=ax)
        return fig, ax

    def reorder_states(self, state_map):
        idx = [state_map[k] for k in sorted(state_map.keys())]
        PI = self.initial_distribution
        A = self.transition
        B = self.emission

        newPI = PI[idx]
        newB = B[:, idx]
        newA = np.zeros(A.shape)
        for x in range(A.shape[0]):
            for y in range(A.shape[1]):
                i = state_map[x]
                j = state_map[y]
                newA[i,j] = A[x,y]

        self.initial_distribution = newPI
        self.transition = newA
        self.emission = newB
        self._update_cost()

    def _update_cost(self):
        spikes = self.data
        win_size = self._cost_window
        dt = self.dt
        PI = self.initial_distribution
        A  = self.transition
        B  = self.emission
        true_rates = self._rate_data
        cost, BIC, bestPaths, maxLogProb = compute_hmm_cost(spikes, dt, PI, A, B,
                                                            win_size=win_size,
                                                            true_rates=true_rates)
        self.cost = cost
        self.BIC = BIC
        self.best_sequences = bestPaths
        self.max_log_prob = maxLogProb

    def get_cost(self):
        if self.cost is None:
            self._update_cost()

        return self.cost


def compute_BIC(spikes, dt, PI, A, B):
    bestPaths, maxLogProb = compute_best_paths(spikes, dt, PI, A, B)
    maxLogProb = np.max(maxLogProb)

    nParams = (A.shape[0]*(A.shape[1]-1) +
               (PI.shape[0]-1) +
               B.shape[0]*(B.shape[1]-1))

    nPts = spikes.shape[-1]
    BIC = -2 * maxLogProb + nParams * np.log(nPts)
    return BIC, bestPaths, maxLogProb


def compute_hmm_cost(spikes, dt, PI, A, B, win_size=0.25, true_rates=None):
    if true_rates is None:
        true_rates = convert_spikes_to_rates(spikes, dt, win_size)

    BIC, bestPaths, maxLogProb = compute_BIC(spikes, dt, PI, A, B)
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


def convert_path_state_numbers(paths, state_map):
    newPaths = np.zeros(paths.shape)
    for k,v in state_map.items():
        idx = np.where(paths == k)
        newPaths[idx] = v

    return newPaths


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
            distances[j] =  euclidean(t1[:,j], t2[:,j])

        RMSE[i] = np.sqrt(np.mean(np.power(distances,2)))

    return np.mean(RMSE)


def plot_state_raster(data, stateVec, dt, ax=None):
    if len(data.shape) == 2:
        data = np.array([data])

    nTrials, nCells, nTimeSteps = data.shape
    nStates = np.max(stateVec) +1

    gradient = np.array([0 + i/(nCells+1) for i in range(nCells)])
    time = np.arange(0, nTimeSteps * dt * 1000, dt * 1000)
    colors = [plt.cm.jet(i) for i in np.linspace(0,1,nStates)]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for trial, spikes in enumerate(data):
        path = stateVec[trial]
        for i, row in enumerate(spikes):
            idx = np.where(row == 1)[0]
            ax.scatter(time[idx], row[idx]*trial + gradient[i],
                       c=[colors[int(x)] for x in path[idx]], marker='|')

    return fig, ax

def plot_state_rates(rates, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    nCells, nStates = rates.shape
    df = pd.DataFrame(rates, columns=['state %i' % i for i in range(nStates)])
    df['cell'] = ['cell %i' % i for i in df.index]
    df = pd.melt(df, 'cell', ['state %i' % i for i in range(nStates)], 'state', 'rate')
    sns.barplot(x='state', y='rate', hue='cell', data=df,
                palette='muted', ax=ax)

    return fig, ax

def compare_hmm_to_truth(truth_dat, hmm, state_map=None):
    if state_map is None:
        state_map = match_states(truth_dat.ground_truth['firing_rates'], hmm.emission)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    truth_dat.plot_state_raster(ax=ax[0,0])
    truth_dat.plot_state_rates(ax=ax[1,0])
    hmm.plot_state_raster(ax=ax[0,1], state_map=state_map)
    hmm.plot_state_rates(ax=ax[1,1], state_map=state_map)
    ax[0,0].set_title('Ground Truth States')
    ax[0,1].set_title('HMM Best Decoded States')
    ax[1,0].get_legend().remove()
    ax[1,1].legend(loc='upper center', bbox_to_anchor=[-0.4, -0.6, 0.5, 0.5], ncol=5)

    # Compute edit distances, histogram, return mean and median % correct
    truePaths = truth_dat.ground_truth['state_vectors']
    bestPaths, _ = hmm.get_best_paths()
    if state_map is not None:
        bestPaths = convert_path_state_numbers(bestPaths, state_map)

    edit_distances = np.zeros((truePaths.shape[0],))
    pool = mp.Pool(mp.cpu_count())
    def update(ans):
        edit_distances[ans[0]] = ans[1]

    print('Computing edit distances...')
    for i, x in enumerate(zip(truePaths, bestPaths)):
        pool.apply_async(levenshtein_mp, (i, *x), callback=update)

    pool.close()
    pool.join()
    print('Done!')

    nPts = truePaths.shape[1]
    mean_correct = 100*(nPts - np.mean(edit_distances)) / nPts
    median_correct = 100*(nPts - np.median(edit_distances)) / nPts

    # Plot:
    #   - edit distance histogram
    #   - side-by-side trial comparison
    h = 0.25
    dt = hmm.dt
    time = np.arange(0, nPts * (dt*1000), dt*1000)  # time in ms
    fig2, ax2 = plt.subplots(ncols=2, figsize=(15,10))
    ax2[0].hist(100*(nPts-edit_distances)/nPts)
    ax2[0].set_xlabel('Percent Correct')
    ax2[0].set_ylabel('Trial Count')
    ax2[0].set_title('Percent Correct based on edit distance\n'
                     'Mean Correct: %0.1f%%, Median: %0.1f%%'
                     % (mean_correct, median_correct))

    maxState = int(np.max((bestPaths, truePaths)))
    colors = [plt.cm.Paired(x) for x in np.linspace(0, 1, (maxState+1)*2)]
    trueCol = [colors[x] for x in np.arange(0, (maxState+1)*2, 2)]
    hmmCol = [colors[x] for x in np.arange(1, (maxState+1)*2, 2)]
    leg = {}
    leg['hmm'] = {k: None for k in np.unique((bestPaths, truePaths))}
    leg['truth'] = {k: None for k in np.unique((bestPaths, truePaths))}
    for i, x in enumerate(zip(truePaths, bestPaths)):
        y = x[0]
        z = x[1]
        t = 0
        while(t  < nPts):
            s = int(y[t])
            next_t = np.where(y[t:] != s)[0]
            if len(next_t) == 0:
                next_t = nPts - t
            else:
                next_t = next_t[0]

            t_start = time[t]
            t_end = time[t+next_t-1]
            tmp = ax2[1].fill_between([t_start, t_end], [i, i], [i+h, i+h], color=trueCol[s])
            if leg['truth'][s] is None:
                leg['truth'][s] = tmp

            t += next_t

        t = 0
        while(t  < nPts):
            s = int(z[t])
            next_t = np.where(z[t:] != s)[0]
            if len(next_t) == 0:
                next_t = nPts - t
            else:
                next_t = next_t[0]

            t_start = time[t]
            t_end = time[t+next_t-1]
            tmp = ax2[1].fill_between([t_start, t_end], [i, i], [i-h, i-h], color=hmmCol[s])
            if leg['hmm'][s] is None:
                leg['hmm'][s] = tmp

            t += next_t

        # Write % correct next to line
        t_str = '%0.1f%%' % (100 * (nPts - edit_distances[i])/nPts)
        ax2[1].text(nPts+5, i-h, t_str)

    ax2[1].set_xlim((0, nPts+int(nPts/3)))
    ax2[1].set_xlabel('Time (ms)')
    ax2[1].set_title('State Sequences')
    handles = list(leg['truth'].values()) + list(leg['hmm'].values())
    labels = (['True State %i' % i for i in leg['truth'].keys()] +
              ['HMM State %i' % i for i in leg['hmm'].keys()])
    ax2[1].legend(handles, labels, shadow=True,
                  bbox_to_anchor=(0.78, 0.5, 0.5, 0.5))

    fig.show()
    fig2.show()
    return fig, ax, fig2, ax2


def wrap_baum_welch(trial_id, trial_dat, dt, PI, A, B):
    alpha, norms = forward(trial_dat, dt, PI, A, B)
    beta = backward(trial_dat, dt, A, B, norms)
    tmp_gamma, tmp_epsilons = baum_welch(trial_dat, dt, A, B, alpha, beta)
    return trial_id, tmp_gamma, tmp_epsilons


def compute_new_matrices(spikes, dt, gammas, epsilons):
    nTrials, nCells, nTimeSteps = spikes.shape
    minFR = 1/(nTimeSteps*dt)

    PI = np.sum(gammas, axis=0)[:,1] / nTrials
    Anumer = np.sum(np.sum(epsilons, axis=3), axis=0)
    Adenom = np.sum(np.sum(gammas[:,:,:-1], axis=2), axis=0)
    A = Anumer/Adenom
    A = A/np.sum(A, axis=1)
    Bnumer = np.sum(np.array([np.matmul(tmp_y, tmp_g.T)
                              for tmp_y, tmp_g in zip(spikes, gammas)]),
                    axis=0)
    Bdenom =  np.sum(np.sum(gammas, axis=2), axis=0)
    B = (Bnumer / Bdenom)/dt
    idx = np.where(B < minFR)[0]
    B[idx] = minFR

    return PI, A, B


def match_states(rates1, rates2):
    '''Takes 2 Cell X State firing rate matrices and determines which states
    are most similar. Returns dict mapping rates2 states to rates1 states
    '''
    distances = np.zeros((rates1.shape[1], rates2.shape[1]))
    for x, y in it.product(range(rates1.shape[1]), range(rates2.shape[1])):
        tmp = euclidean(rates1[:, x], rates2[:, y])
        distances[x, y] = tmp

    states = list(range(rates2.shape[1]))
    out = {}
    for i in range(rates2.shape[1]):
        s = np.argmin(distances[:,i])
        r = np.argmin(distances[s, :])
        if r == i and s in states:
            out[i] = s
            idx = np.where(states == s)[0]
            states.pop(int(idx))

    for i in range(rates2.shape[1]):
        if i not in out:
            s = np.argmin(distances[states, i])
            out[i] = states[s]

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
def levenshtein_mp(i, seq1, seq2):
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

    return i, matrix[size_x - 1, size_y - 1]


def fit_hmm_mp(nStates, spikes, dt, max_iter=1000, thresh=1e-4):
    hmm = PoissonHMM(nStates, spikes, dt)
    hmm.fit(max_iter=max_iter, convergence_thresh=thresh, parallel=False)
    return hmm


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


@njit
def euclidean(a, b):
    c = np.power(a-b,2)
    return np.sqrt(np.sum(c))


def rebin_spike_array(spikes, dt, time, new_dt):
    if spikes.ndim == 2:
        spikes = np.expand_dims(spikes,0)

    n_trials, n_cells, n_steps = spikes.shape
    n_bins = int(new_dt/dt)
    new_time = np.arange(time[0], time[-1], new_dt)
    new_spikes = np.zeros((n_trials, n_cells, len(new_time)))
    for i, w in enumerate(new_time):
        idx = np.where((time >= w) & (time < w+new_dt))[0]
        new_spikes[:,:,i] = np.sum(spikes[:,:,idx], axis=-1)

    return new_spikes, new_time


HMM_PARAMS = {'unit_type': 'single', 'dt': 0.001, 'threshold': 1e-4, 'max_iter': 1000,
              'time_start': 0, 'time_end': 2000, 'n_repeats': 3, 'n_states': 3}


class HmmHandler(object):
    def __init__(self, dat, params=None, save_dir=None):
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
        if isinstance(params, dict):
            params = [params]

        if isinstance(dat, str):
            dat = load_dataset(dat)
            if dat is None:
                raise FileNotFoundError('No dataset.p file found given directory')

        if save_dir is None:
            save_dir = os.path.join(dat.root_dir,
                                    '%s_analysis' % dat.data_name)

        self._dataset = dat
        self.root_dir = dat.root_dir
        self.save_dir = save_dir
        self.h5_file = os.path.join(save_dir, '%s_HMM_Analysis.hdf5' % dat.data_name)
        dim = dat.dig_in_mapping.query('exclude==False')
        tastes = dim['name'].tolist()
        if params is None:
            # Load params and fitted models
            self.load_data()
        else:
            self.init_params(params)

        self.params = params

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.plot_dir = os.path.join(save_dir, 'HMM_Plots')
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self._setup_hdf5()

    def init_params(self, params):
        dat = self._dataset
        dim = dat.dig_in_mapping.query('exclude == False')
        tastes = dim['name'].tolist()
        dim = dim.set_index('name')
        if not hasattr(dat, 'dig_in_trials'):
            dat.create_trial_list()

        trials = dat.dig_in_trials
        data_params = []
        fit_objs = []
        fit_params = []
        for i, X in enumerate(it.product(params,tastes)):
            p = X[0].copy()
            t = X[1]
            p['hmm_id'] = i
            p['taste'] = t
            p['channel'] = dim.loc[t, 'channel']
            unit_names = query_units(dat, p['unit_type'])
            p['n_cells'] = len(unit_names)
            p['n_trials'] = len(trials.query('name == @t'))

            data_params.append(p)
            # Make fit object for each repeat
            # During fitting compare HMM as ones with the same ID are returned
            for i in range(p['n_repeats']):
                hmmFit = HMMFit(dat.root_dir, p)
                fit_objs.append(hmmFit)
                fit_params.append(p)

        self._fit_objects = fit_objs
        self._data_params = data_params
        self._fit_params = fit_params
        self._fitted_models = dict.fromkeys([x['hmm_id'] for x in data_params])
        self.write_overview_to_hdf5()

    def load_data(self):
        h5_file = self.h5_file
        if not os.path.isfile(h5_file):
            raise ValueError('No params to load')

        rec_dir = self._dataset.root_dir
        params = []
        fit_objs = []
        fit_params = []
        fitted_models = {}
        with tables.open_file(h5_file, 'r') as hf5:
            table = hf5.root.data_overview
            col_names = table.colnames
            for row in table[:]:
                p = {}
                for k in col_names:
                    if table.coltypes[k] == 'string':
                        p[k] = row[k].decode('utf-8')
                    else:
                        p[k] = row[k]

                params.append(p)
                for i in range(p['n_repeats']):
                    hmmFit = HMMFit(rec_dir, p)
                    fit_objs.append(hmmFit)
                    fit_params.append(p)

        for p in params:
            hmm_id = p['hmm_id']
            fitted_models[hmm_id] = read_hmm_from_hdf5(h5_file, hmm_id, rec_dir)

        self._data_params = params
        self._fit_objects = fit_objs
        self._fitted_models = fitted_models
        self._fit_params = fit_params


    def write_overview_to_hdf5(self):
        params = self._data_params
        h5_file = self.h5_file
        if hasattr(self, '_fitted_models'):
            models = self._fitted_models
        else:
            models = dict.fromkeys([x['hmm_id']
                                    for x in data_params])
            self._fitted_models = models


        if not os.path.isfile(h5_file):
            self._setup_hdf5()

        print('Writing data overview table to hdf5...')
        with tables.open_file(h5_file, 'a') as hf5:
            table = hf5.root.data_overview
            # Clear old table
            table.remove_rows(start=0)

            # Add new rows
            for p in params:
                row = table.row
                for k, v in p.items():
                    row[k] = v

                if models[p['hmm_id']] is not None:
                   hmm = models[p['hmm_id']]
                   row['n_iterations'] =  hmm.iterations
                   row['BIC'] = hmm.BIC
                   row['cost'] = hmm.cost
                   row['converged'] = hmm.isConverged(p['threshold'])
                   row['fitted'] = hmm.fitted

                row.append()

            table.flush()
            hf5.flush()

        print('Done!')

    def _setup_hdf5(self):
        h5_file = self.h5_file

        with tables.open_file(h5_file, 'a') as hf5:
            # Taste -> PI, A, B, BIC, state_sequences, nStates, nCells, dt
            if not 'data_overview' in hf5.root:
                # Contains taste, channel, n_cells, n_trials, n_states, dt, BIC
                table = hf5.create_table('/', 'data_overview', HMMInfoParticle,
                                         'Basic info for each digital_input')
                table.flush()


            if hasattr(self, '_data_params') and self._data_params is not None:
                for p in self._data_params:
                    hmm_str = 'hmm_%i' % p['hmm_id']
                    if hmm_str not in hf5.root:
                        hf5.create_group('/', hmm_str, 'Data for HMM #%i' % p['hmm_id'])

            hf5.flush()

    def run(self, parallel=True):
        self.write_overview_to_hdf5()
        h5_file = self.h5_file
        rec_dir = self._dataset.root_dir
        fit_objs = self._fit_objects
        fit_params = self._fit_params
        self._fitted_models = dict.fromkeys([x['hmm_id'] for x in self._data_params])
        errors = []

        # def update(ans):
        #     hmm_id = ans[0]
        #     hmm = ans[1]
        #     if self._fitted_models[hmm_id] is not None:
        #         best_hmm = pick_best_hmm([HMMs[hmm_id], hmm])
        #         self._fitted_models[hmm_id] = best_hmm
        #         write_hmm_to_hdf5(h5_file, hmm_id, best_hmm)
        #         del hmm, best_hmm
        #     else:
        #         # Check history for lowest BIC
        #         self._fitted_models[hmm_id] = hmm.set_to_lowest_BIC()
        #         write_hmm_to_hdf5(h5_file, hmm_id, hmm)
        #         del hmm

        # def error_call(e):
        #     errors.append(e)

        # if parallel:
        #     n_cpu = np.min((mp.cpu_count()-1, len(fit_objs)))
        #     if n_cpu > 10:
        #         pool = mp.get_context('spawn').Pool(n_cpu)
        #     else:
        #         pool = mp.Pool(n_cpu)

        #     for f in fit_objs:
        #         pool.apply_async(f.run, callback=update, error_callback=error_call)

        #     pool.close()
        #     pool.join()
        # else:
        #     for f in fit_objs:
        #         try:
        #             ans = f.run()
        #             update(ans)
        #         except Exception as e:
        #             raise Exception(e)
        #             error_call(e)
        print('Running fittings')
        if parallel:
            n_cpu = np.min((mp.cpu_count()-1, len(fit_params)))
        else:
            n_cpu = 1

        results = Parallel(n_jobs=n_cpu, verbose=20)(delayed(hmm_fit_mp)(rec_dir, p) for p in fit_params)
        for hmm_id, hmm in zip(*results):
            if self._fitted_models[hmm_id] is None:
                self._fitted_models[hmm_id] = hmm
            else:
                new_hmm = pick_best_hmm([hmm, self._fitted_models[hmm_id]])
                self._fitted_models[hmm_id] = new_hmm

        self.write_overview_to_hdf5()
        self.save_fitted_models()
        # if len(errors) > 0:
        #     print('Encountered errors: ')
        #     for e in errors:
        #         print(e)

    def save_fitted_models(self):
        models = self._fitted_models
        for k, v in models.items():
            write_hmm_to_hdf5(self.h5_file, k, v)
            plot_dir = os.path.join(self.plot_dir, 'HMM_%i' % k)
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            ids = [x['hmm_id'] for x in self._data_params]
            idx = ids.index(k)
            params = self._data_params[idx]
            time_window = [params['time_start'], params['time_end']]
            hmmplt.plot_hmm_figures(v, time_window, save_dir=plot_dir)


@memory.cache
def get_hmm_spike_data(rec_dir, unit_type, channel, time_start=None, time_end=None, dt=None):
    units = query_units(rec_dir, unit_type)
    time, spike_array = h5io.get_spike_data(rec_dir, units, channel)
    curr_dt = np.unique(np.diff(time))[0] / 1000
    if dt is not None and curr_dt < dt:
        spike_array, time = rebin_spike_array(spike_array, curr_dt, time, dt)
    elif dt is not None and curr_dt > dt:
        raise ValueError('Cannot upsample spike array from %f ms '
                         'bins to %f ms bins' % (dt, curr_dt))
    else:
        dt = curr_dt

    if time_start and time_end:
        idx = np.where((time >= time_start) & (time < time_end))[0]
        time = time[idx]
        spike_array = spike_array[:, :, idx]

    return spike_array.astype('int32'), dt, time


def read_hmm_from_hdf5(h5_file, hmm_id, rec_dir):
    print('Loading HMM %i for hdf5' % hmm_id)
    with tables.open_file(h5_file, 'r') as hf5:
        h_str = 'hmm_%i' % hmm_id
        if h_str not in hf5.root or len(hf5.list_nodes('/'+h_str)) == 0:
            return None

        table = hf5.root.data_overview
        row = list(table.where('hmm_id == id', condvars={'id':hmm_id}))
        if len(row) == 0:
            raise ValueError('Parameters not found for hmm %i' % hmm_id)
        elif len(row) > 1:
            raise ValueError('Multiple parameters found for hmm %i' % hmm_id)

        row = row[0]
        units = query_units(rec_dir, row['unit_type'].decode('utf-8'))
        spikes, dt, time = get_spike_data(rec_dir, units, row['channel'],
                                          dt=row['dt'],
                                          time_start=row['time_start'],
                                          time_end=row['time_end'])
        tmp = hf5.root[h_str]
        mats = {'initial_distribution': tmp['initial_distribution'][:],
                'transition': tmp['transition'][:],
                'emission': tmp['emission'][:],
                'fitted': row['fitted']}
        hmm = PoissonHMM(row['n_states'], spikes, dt, set_data=mats)

    return hmm


def write_hmm_to_hdf5(h5_file, hmm_id, hmm):
    h_str = 'hmm_%i' % hmm_id
    print('Writing HMM %i to hdf5 file...' % hmm_id)
    with tables.open_file(h5_file, 'a') as hf5:
        if h_str in hf5.root:
            hf5.remove_node('/', h_str, recursive=True)

        hf5.create_group('/', h_str, 'Data for HMM #%i' % hmm_id)
        hf5.create_array('/'+h_str, 'initial_distribution',
                         hmm.initial_distribution)
        hf5.create_array('/'+h_str, 'transition', hmm.transition)
        hf5.create_array('/'+h_str, 'emission', hmm.emission)

        best_paths, _ = hmm.get_best_paths()
        hf5.create_array('/'+h_str, 'state_sequences', best_paths)


def query_units(dat, unit_type):
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

    Returns
    -------
        list of str : unit_names
    '''
    if isinstance(dat, str):
        units = h5io.get_unit_table(dat)
    else:
        units = dat.get_unit_table()

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

    return units.query(q_str)['unit_name'].tolist()


    # Parameters
    # hmm_id
    # taste
    # channel
    # n_cells
    # unit_type
    # n_trials
    # dt
    # threshold
    # time_start
    # time_end
    # n_repeats
    # n_states
    # n_iterations
    # BIC
    # cost
    # converged
    # fitted
    #
    # Extras: unit_names, rec_dir


class HMMFit(object):
    def __init__(self, rec_dir, params):
        self._rec_dir = rec_dir
        self._params = params

    def run(self, parallel=False):
        params = self._params
        spikes, dt, time = self.get_spike_data()
        hmm = PoissonHMM(params['n_states'], spikes, dt)
        hmm.fit(max_iter=params['max_iter'],
                convergence_thresh=params['threshold'],
                parallel=parallel)
        del spikes, dt, time
        return params['hmm_id'], hmm

    def get_spike_data(self):
        p = self._params
        units = query_units(self._rec_dir, p['unit_type'])
        # Get stored spike array, time is in ms, dt is usually 1 ms
        spike_array, dt, time = get_spike_data(self._rec_dir, units,
                                               p['channel'], dt=p['dt'],
                                               time_start=p['time_start'],
                                               time_end=p['time_end'])
        return spike_array, dt, time

def hmm_fit_mp(rec_dir, params):
    hmm_id = params['hmm_id']
    n_states = params['n_states']
    dt = params['dt']
    time_start = params['time_start']
    time_end = params['time_end']
    max_iter = params['max_iter']
    threshold = params['threshold']
    unit_type = params['unit_type']
    channel = params['channel']
    spikes, dt, time = get_hmm_spike_data(rec_dir, unit_type, channel,
                                          time_start=time_start,
                                          time_end=time_end, dt = dt)
    hmm = PoissonHMM(params['n_states'], spikes, dt)
    hmm.fit(max_iter=max_iter, convergence_thresh=threshold)
    return hmm_id, hmm


def get_spike_data(rec_dir, units, channel, dt=None, time_start=None, time_end=None):
    time, spike_array = h5io.get_spike_data(rec_dir, units, channel)
    curr_dt = np.unique(np.diff(time))[0] / 1000
    if dt is not None and curr_dt < dt:
        spike_array, time = rebin_spike_array(spike_array, curr_dt, time, dt)
    elif dt is not None and curr_dt > dt:
        raise ValueError('Cannot upsample spike array from %f ms '
                         'bins to %f ms bins' % (dt, curr_dt))
    else:
        dt = curr_dt

    if time_start and time_end:
        idx = np.where((time >= time_start) & (time < time_end))[0]
        time = time[idx]
        spike_array = spike_array[:, :, idx]

    return spike_array.astype('int32'), dt, time


def pick_best_hmm(HMMs):
    '''For each HMM it searches the history for the HMM with lowest BIC Then it
    compares HMMs. Those with same # of free parameters are compared by BIC
    Those with different # of free parameters (namely # of states) are compared
    by cost Best HMM is returned

    Parameters
    ----------
    HMMs : list of PoissonHmm objects

    Returns
    -------
    PoissonHmm
    '''
    # First optimize each HMMs and sort into groups based on # of states
    groups = {}
    for hmm in HMMs:
        hmm.set_to_lowest_BIC()
        if hmm.n_states in groups:
            groups[hmm.n_states].append(hmm)
        else:
            groups[hmm.n_states] = [hmm]

    best_per_state = {}
    for k, v in groups:
        BICs = np.array([x.get_BIC()[0] for x in v])
        idx = np.argmin(BICs)
        best_per_state[k] = v[idx]

    hmm_list = best_per_state.values()
    costs = np.array([x.get_cost() for x in hmm_list])
    idx = np.argmin(costs)
    return hmm_list[idx]

    # Compare HMMs with same number of states by BIC

