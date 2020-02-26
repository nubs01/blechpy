import numpy as np
import pandas a pd
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
import seaborn as sns


def plot_hmm(hmm, time_window=None, hmm_id=None, save_file=None):
    spikes = hmm.data
    dt = hmm.dt
    BIC, paths = hmm.get_BIC()
    nTrials, nCells, nTimeSteps = spikes.shape

    if not time_window:
        time_window = [0, nTimeSteps * dt * 1000]

    time = np.arange(time_window[0], time_window[1], dt*1000)  # Time in ms
    nStates = np.max(paths)+1
    colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, nStates)]


    fig, axes = plt.subplots(nrows=nTrials, figsize=(15,15))
    fig.subplots_adjust(right=0.9)
    y_step = np.linspace(0.05, 0.95, nCells)
    handles = []
    labels = []
    for ax, seq, trial in zip(axes, paths, spikes):
        _, leg_handles, leg_labels = plot_sequence(seq, time=time, ax=ax, colors=colors)
        for h, l in zip(leg_handles, leg_labels):
            if l not in labels:
                handles.append(h)
                labels.append(l)

        tmp = plot_raster(trial, time=time, ax=ax)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)


    axes[-1].get_xaxis().set_visible(True)
    axes[-1].set_xlabel('Time (ms)')
    mid = int(nTrials/2)
    axes[mid].legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                    fontsize=14)
    axes[mid].set_ylabel('Trials')
    title_str = 'HMM Decoded State Sequences'
    if hmm_id:
        title_str += '\n%s' % hmm_id

    axes[0].set_title(title_str)
    if save_file:
        fig.savefig(save_file)
        return
    else:
        fig.show()
        return fig, ax


def plot_sequence(seq, time=None, ax=None, y_min=0, y_max=1, colors=None):
    if ax is None:
        _, ax = plt.subplots()

    if time is None:
        time = np.arange(0, len(seq))

    nStates = np.max(seq)+1
    if colors is None:
        colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, nStates)]

    seq_windows = get_sequence_windows(seq)
    handles = {}
    for win in seq_windows:
        t_vec = [time[win[0]], time[win[1]]]
        h = ax.fill_between(t_vec, [y_min, y_min], [y_max, y_max],
                            color=colors[int(win[2])], alpha=0.4)
        if  win[2] not in handles:
            handles[win[2]] = h

    leg_handles = [handles[k] for k in sorted(handles.keys())]
    leg_labels = ['State %i' % k for k in sorted(handles.keys())]
    return ax, leg_handles, leg_labels


def get_sequence_windows(seq):
    t = 0
    out = []
    while t < len(seq):
        s = seq[t]
        tmp = np.where(seq[t:] != s)[0]
        if len(tmp) == 0:
            tmp = len(seq) - t
        else:
            tmp = tmp[0]

        out.append((t, tmp+t-1, s))
        t += tmp

    return out


def plot_raster(spikes, time=None, ax=None, y_min=0.05, y_max=0.95):
    '''Plot 2D spike raster

    Parameters
    ----------
    spikes : np.array
        2D matrix M x N where N is the number of time steps and in each bin is
        a 0 or 1, with 1 signifying the presence of a spike
    '''
    if not ax:
        _, ax = plt.subplots()

    n_rows, n_steps = spikes.shape
    if not time:
        time = np.arange(0, n_steps)

    y_steps = np.linspace(y_min, y_max, n_rows)
    for i, row in enumerate(spikes):
        idx = np.where(row == 1)[0]
        if len(idx) == 0:
            continue

        ax.plot(time[idx], row[idx]*y_steps[i], color='black', marker='|')

    return ax


def plot_hmm_rates(rates, ax=None, colors=None):
    '''Make bar plot of spike rates for each cell and state in an HMM emission
    matrix

    Parameters
    ----------
    rates: np.array
        Cell X State matrix of firing rates
    '''
    if not ax:
        _, ax = plt.subplots()

    n_cells, n_states = rates.shape

    df = pd.DataFrame(rates, columns=['state %i' % i for i in range(nStates)])
    df['cell'] = ['cell %i' % i for i in df.index]
    df = pd.melt(df, 'cell', ['state %i' % i for i in range(nStates)], 'state', 'rate')
    sns.barplot(x='state', y='rate', hue='cell', data=df,
                palette='muted', ax=ax)

    return ax


def plot_hmm_transition(transition, ax=None):
    pass


def plot_forward_probs(hmm, time=None, ax=None, colors=None):
    pass


def plot_backward_probs(hmm, time=None, ax=None, colors=None):
    pass


def plot_viterbi_probs(hmm, time=None, ax=None, colors=None):
    pass
