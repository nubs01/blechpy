import math
import numpy as np
import itertools as it
import pylab as plt
import seaborn as sns
import pandas as pd
import multiprocessing as mp


TEST_PARAMS = {'n_cells': 10, 'n_states': 4, 'state_seq_length': 5,
               'trial_time': 3.5, 'dt': 0.001, 'max_rate': 30, 'n_trials': 15,
               'min_state_dur': 0.05, 'noise': 0.01, 'baseline_dur': 1}


def poisson(rate, n, dt):
    '''Gives probability of each neurons spike count assuming poisson spiking
    '''
    tmp = np.power(rate*dt, n) / np.array([math.factorial(x) for x in n])
    tmp = tmp * np.exp(-rate*dt)
    return tmp

def forward(spikes, nStates, dt, PI, A, B):
    '''Run forward algorithm to compute alpha = P(Xt = i| o1...ot, pi)
    Gives the probabilities of being in a specific state at each time point
    given the past observations and inital probabilities

    Parameters
    ----------
    spikes : np.array
        N x T matrix of spike counts with each entry ((i,j)) holding the # of
        spikes from neuron i in timebine j
    nStates : int, # of hidden states predicted to have generate the spikes
    dt : float, timebin in seconds (i.e. 0.001)
    PI : np.array
        nStates x 1 vector of inital state probabilities
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

    # For each state, use the the initial state distribution and spike counts
    # to initalize alpha(:,1)
    alpha = np.array([[PI[i] * np.prod(poisson(B[:,i], spikes[:,1], dt))
                      for i in range(nStates)]])
    norms = [np.sum(alpha)]
    alpha = alpha/norms[0]
    for t in range(1, nTimeSteps):
        tmp = np.array([np.prod(poisson(B[:, s], spikes[:, t], dt)) *
                        np.sum(alpha[t-1, :] * A[:,s])
                        for s in range(nStates)])
        norms.append(np.sum(tmp))
        tmp = tmp / np.sum(tmp)
        alpha = np.vstack((alpha, tmp))

    alpha = alpha.T
    return alpha, norms


def backward(spikes, nStates, dt, A, B, norms):
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
    beta = np.zeros((nStates, nTimeSteps))
    beta[:, -1] = 1  # Initialize final beta to 1 for all states
    for t in reversed(range(nTimeSteps-1)):
        for s in range(nStates):
            beta[s,t] = np.sum((beta[:, t+1] * A[s,:]) *
                               np.prod(poisson(B[:, s], spikes[:, t+1], dt)))

        beta[:, t] = beta[:, t] / norms[t+1]

    return beta

def baum_welch(spikes, nStates, dt, A, B, alpha, beta):
    nTimeSteps = spikes.shape[1]
    gamma = np.zeros((nStates, nTimeSteps))
    epsilons = np.zeros((nStates, nStates, nTimeSteps-1))
    for t in range(nTimeSteps):
        if t < nTimeSteps-1:
            gamma[:, t] = (alpha[:, t] * beta[:, t]) / np.sum(alpha[:,t] * beta[:,t])
            epsilonNumerator = np.zeros((nStates, nStates))
            for si, sj in it.product(range(nStates), range(nStates)):
                probs = np.prod(poisson(B[:,sj], spikes[:, t+1], dt))
                epsilonNumerator[si, sj] = (alpha[si, t]*A[si, sj]*
                                            beta[sj, t]*probs)

            epsilons[:, :, t] = epsilonNumerator / np.sum(epsilonNumerator)

    return gamma, epsilons


def poisson_baum_welch(spikes, nStates, dt, maxIter, convergence_thresh=1e-4):
    # Deprecated
    # TODO: Ask Ben about how to best change this to train using more than one
    # trial
    if len(spikes.shape) == 3:
        # spikes is Trial x Neuron x Time
        nTrials, nCells, nTimeSteps = spikes.shape
    elif len(spikes.shape) == 2:
        nTrials = 1
        nCells, nTimeSteps = spikes.shape

    minFR = 1/(nTimeSteps*dt)

    # Initial state distribution
    PI = np.ones((nStates, 1)) / nStates

    # Initialize transition matrix with high prob for transition to same
    # state and uniform elsewhere
    # For Baum-Welch, inital parameters cannot be flat distribution
    # TODO: Change this initialization
    A = np.zeros((nStates, nStates)) * (0.01 / (nStates-1))
    for i in range(nStates):
        A[i, i] = 0.99

    # Initialize emission (rate) matrix which is fed to poisson function to get
    # emission probabilities
    B = np.random.rand(nNeurons, nStates)
    notConverged = False
    iterNum = 0
    while (notConverged and (iterNum < maxIter)):
        alpha, norms = forward(spikes, nStates, dt, PI, A, B)
        beta = backward(spikes, nStates, dt, A, B, norms)
        gamma = np.zeros((nStates, nTimeSteps))
        epsilons = np.zeros((nStates, nStates, nTimeSteps-1))
        for t in range(nTimeSteps):
            if t < nTimeSteps:
                gamma[:, t] = (alpha[:, t] * beta[:, t]) / np.sum(alpha[:,t] * beta[:,t])
                epsilonNumerator = np.zeros((nStates, nStates))
                for si, sj in it.product(range(nStates), range(nStates)):
                    epsilonNumerator[si, sj] = (alpha[si, t]*A[si, sj]*
                                                beta[sj, t]*
                                                np.prod(poisson(B[:,sj],
                                                                spikes[:,t+1])))

                epsilons[:, :, t] = epsilonNumerator / np.sum(epsilonNumerator)

        # Store old parameters for convergence check
        oldPI = PI
        oldA = A
        oldB = B

        PI = gamma[:,1]
        Anumer = np.sum(epsilons, axis=2)
        Adenom = np.sum(gamma, axis=1)
        A = Anumer/Adenom
        A = A/np.sum(A, axis=1)
        B = ((spikes*gamma.T) / np.sum(gamma, axis=1).T)/dt
        B = np.max((minFR,B), axis=0)
        notConverged = isNotConverged(oldPI, oldA, oldB, PI, A, B, thresh=convergence_thresh)
        iterNum += 1
        print('Iter #%i complete.' % iterNum)


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
    for t in reversed(range(nTimeSteps-1)):
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
    def __init__(self, n_predicted_states, n_cells, max_history=500):
        self.n_states = n_predicted_states
        self.n_cells = n_cells
        self._max_history = max_history
        self.randomize()

    def randomize(self):
        nStates = self.n_states
        nCells = self.n_cells

        # Initialize transition matrix with high stay probability
        diag = np.random.normal(.99, .01, nStates)
        A = np.abs(np.random.normal(0.01/(nStates-1), 0.01, (nStates, nStates)))
        for i in range(nStates):
            A[i, i] = diag[i]
            A[i,:] = A[i,:] / np.sum(A[i,:])

        # Initialize rate matrix ("Emission" matrix)
        B = np.random.rand(nCells, nStates)

        self.transition = A
        self.emission = B
        self.inital_distribution = np.ones((nStates,)) / nStates
        self.converged = False
        self.history = None
        self.data = None
        self.dt = None

    def fit(self, spikes, dt, max_iter = 1000, convergence_thresh = 1e-4,
            parallel=False):
        if self.converged:
            return

        self.data = spikes
        self.dt = dt
        iterNum = 0
        while (not self.isConverged(convergence_thresh) and
               (iterNum < max_iter)):
            self._step(spikes, dt, parallel=parallel)

            iterNum += 1
            print('Iter #%i complete.' % iterNum)

        self.converged = True

    def _step(self, spikes, dt, parallel=False):
        if len(spikes.shape) == 2:
            spikes = np.array([spikes])

        nTrials, nCells, nTimeSteps = spikes.shape
        minFR = 1/(nTimeSteps*dt)

        A = self.transition
        B = self.emission
        PI = self.inital_distribution
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
                raise ValueError(ans)

            n_cores = mp.cpu_count() - 1
            pool = mp.get_context('spawn').Pool(n_cores)
            for i, trial in enumerate(spikes):
                pool.apply_async(wrap_baum_welch,
                                 (i, trial, nStates, dt, PI, A, B),
                                 callback=update, error_callback=error)

            pool.close()
            pool.join()
        else:
            for i, trial in enumerate(spikes):
                alpha, norms = forward(trial, nStates, dt, PI, A, B)
                beta = backward(trial, nStates, dt, A, B, norms)
                tmp_gamma, tmp_epsilons = baum_welch(trial, nStates, dt, A, B, alpha, beta)
                gammas[i, :, :] = tmp_gamma
                epsilons[i, :, :, :] = tmp_epsilons

        # Store old parameters for convergence check
        oldPI = PI
        oldA = A
        oldB = B

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

        self.transition = A
        self.emission = B
        self.inital_distribution = PI
        self.update_history(oldPI, oldA, oldB)

    def update_history(self, oldPI, oldA, oldB):
        A = self.transition
        B = self.emission
        PI = self.inital_distribution

        if self.history is None:
            self.history = {}
            self.history['A'] = [oldA]
            self.history['B'] = [oldB]
            self.history['PI'] = [PI]
            self.history['iterations'] = [0]
        else:
            self.history['A'].append(oldA)
            self.history['B'].append(oldB)
            self.history['PI'].append(oldPI)
            self.history['iterations'].append(self.history['iterations'][-1]+1)

        if len(self.history['iterations']) > self._max_history:
            nmax = self._max_history
            for k, v in self.history.items():
                self.history[k] = v[-nmax:]

    def isConverged(self, thresh):
        if self.history is None:
            return False

        oldPI = self.history['PI'][-1]
        oldA = self.history['A'][-1]
        oldB = self.history['B'][-1]

        PI = self.inital_distribution
        A = self.transition
        B = self.emission

        dPI = np.sqrt(np.sum(np.power(oldPI - PI, 2)))
        dA = np.sqrt(np.sum(np.power(oldA - A, 2)))
        dB = np.sqrt(np.sum(np.power(oldB - B, 2)))
        print('dPI = %f,  dA = %f,  dB = %f' % (dPI, dA, dB))
        if not all([x < thresh for x in [dPI, dA, dB]]):
            return False
        else:
            return True

    def get_best_path(self):
        if not self.converged:
            raise ValueError('model not yet fitted')

        spikes = self.data
        if len(spikes.shape) == 2:
            spikes = np.array([spikes])

        nTrials, nCells, nTimeSteps = spikes.shape
        bestPaths = np.zeros((nTrials, nTimeSteps))-1
        pathProbs = np.zeros((nTrials,))
        for i, trial in enumerate(spikes):
            bestPaths[i,:], pathProbs[i], _, _ = poisson_viterbi(trial, self.dt,
                                                                 self.inital_distribution,
                                                                 self.transition,
                                                                 self.emission)

        return bestPaths, pathProbs


    def plot_state_raster(self, ax=None):
        bestPaths, _ = self.get_best_path()
        data = self.data
        fig, ax = plot_state_raster(data, bestPaths, self.dt, ax=ax)
        return fig, ax

    def plot_state_rates(self, ax=None):
        rates = self.emission
        fig, ax = plot_state_rates(rates, ax=ax)
        return fig, ax



def plot_state_raster(data, stateVec, dt, ax=None):
    if len(data.shape) == 2:
        data = np.array([data])

    nTrials, nCells, nTimeSteps = data.shape
    nStates = len(np.unique(stateVec))

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
    sns.catplot(x='state', y='rate', hue='cell', data=df, kind='bar',
                palette='muted', ax=ax)

    return fig, ax

def compare_hmm_to_truth(truth_dat, hmm):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    truth_dat.plot_state_raster(ax=ax[0,0])
    truth_dat.plot_state_rates(ax=ax[1,0])
    hmm.plot_state_raster(ax=ax[0,1])
    hmm.plot_state_rates(ax=ax[1,1])
    ax[0,0].set_title('Ground Truth States')
    ax[0,1].set_title('HMM Best Decoded States')
    ax[1,0].get_legend().remove()
    ax[1,1].legend(loc='upper center', bbox_to_anchor=[-0.4, -0.6, 0.5, 0.5], ncol=5)
    fig.show()
    return fig, ax


def wrap_baum_welch(trial_id, trial_dat, nStates, dt, PI, A, B):
    alpha, norms = forward(trial_dat, nStates, dt, PI, A, B)
    beta = backward(trial_dat, nStates, dt, A, B, norms)
    tmp_gamma, tmp_epsilons = baum_welch(trial_dat, nStates, dt, A, B, alpha, beta)
    return trial_id, tmp_gamma, tmp_epsilons
