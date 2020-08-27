import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
import seaborn as sns



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


def get_threshold_windows(trace, thresh=0.75):
    '''Returns list of tuples with start and stop time for windows where the
    given trace is above threshold. trace can be multiple rows. returns tuples
    in fashion (start_idx, stop_idx, row)
    '''
    out = []
    if len(trace.shape) == 1:
        trace = np.array([trace])

    n_rows, n_steps = trace.shape
    for i, row in enumerate(trace):
        t = 0
        while t < n_steps:
            if row[t] >= thresh:
                tmp = np.where(row[t:] < thresh)[0]
            else:
                tmp = np.where(row[t:] >= thresh)[0]

            if len(tmp) == 0:
                tmp = len(row) - t
            else:
                tmp = tmp[0]

            if row[t] >= thresh:
                out.append((t, tmp+t-1, i))

            t += tmp

    return out


def get_hmm_plot_colors(n_states):
    colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, n_states)]
    return colors


def plot_raster(spikes, time=None, ax=None, y_min=0.05, y_max=0.95):
    '''Plot 2D spike raster

    Parameters
    ----------
    spikes : np.array
        2D matrix M x N where N is the number of time steps and in each bin is
        a 0 or 1, with 1 signifying the presence of a spike
    '''
    if not ax:
        _, ax = plt.gca()

    n_rows, n_steps = spikes.shape
    if time is None:
        time = np.arange(0, n_steps)

    y_steps = np.linspace(y_min, y_max, n_rows)
    for i, row in enumerate(spikes):
        idx = np.where(row == 1)[0]
        if len(idx) == 0:
            continue

        ax.scatter(time[idx], row[idx]*y_steps[i], color='black', marker='|')

    return ax


def make_hmm_raster(spikes, time=None, save_file=None):
    '''Create figure of spikes rasters with each trial on a seperate axis

    Parameters
    ----------
    spikes: np.array, Trials X Cells X Time array with 1s where spikes occur
    time: np.array, 1D time vector
    save_file: str, if provided figure is saved and not returned

    Returns
    -------
    plt.Figure, list of plt.Axes
    '''
    if len(spikes) == 2:
        spikes = np.array([spikes])

    n_trials, n_cells, n_steps = spikes.shape
    if time is None:
        time = np.arange(0, n_steps)

    fig, axes = plt.subplots(nrows=n_trials, figsize=(15, n_trials))
    y_step = np.linspace(0.05, 0.95, n_cells)
    for ax, trial in zip(axes, spikes):
        tmp = plot_raster(trial, time=time, ax=ax)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        if time[0] < 0:
            ax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

    axes[-1].get_xaxis().set_visible(True)
    tmp_ax = fig.add_subplot('111', frameon=False)
    tmp_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    tmp_ax.set_ylabel('Trials')
    axes[-1].set_xlabel('Time')
    axes[-1].set_ylabel('Cells', fontsize=11)
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


def plot_sequence(seq, time=None, ax=None, y_min=0, y_max=1, colors=None):
    if ax is None:
        _, ax = plt.gca()

    if time is None:
        time = np.arange(0, len(seq))

    nStates = np.max(seq)+1
    if colors is None:
        colors = [plt.cm.Set2(x) for x in np.linspace(0, 1, nStates)]

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


def plot_viterbi_paths(hmm, spikes, time=None, colors=None, axes=None, legend=True,
                       hmm_id=None, save_file=None):
    if not axes:
        fig, axes = make_hmm_raster(spikes, time=time)
    else:
        fig = axes[0].figure

    if legend:
        fig.subplots_adjust(right=0.9)  # To  make room for legend


    BIC = hmm.BIC
    paths = hmm.stat_arrays['best_sequences']
    n_trials, n_steps = paths.shape
    n_states = hmm.n_states
    if time is None:
        time = np.arange(0, n_steps)

    if not colors:
        colors = [plt.cm.Set2(x) for x in np.linspace(0,1, n_states)]

    handles = []
    labels = []
    for trial, ax in zip(paths, axes):
        _, tmp_handles, tmp_labels = plot_sequence(trial, time=time, ax=ax, colors=colors)
        for l, h in zip(tmp_labels, tmp_handles):
            if l not in labels:
                handles.append(h)
                labels.append(l)

        if time[0] != 0:
            ax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

    if legend:
        mid = int(n_trials/2)
        axes[mid].legend(handles, labels, loc='upper center',
                         bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                         fontsize=14)

    axes[-1].set_xlabel('Time (ms)')
    title_str = 'Decoded HMM Sequences'
    if hmm_id:
        title_str += '\n%s' % hmm_id

    axes[0].set_title(title_str)
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


def plot_probability_traces(traces, time=None, ax=None, colors=None, thresh=0.75,
                           smoothing=3):
    y_min=0
    y_max=1
    if ax is None:
        _, ax = plt.gca()

    n_states, n_steps = traces.shape
    if time is None:
        time = np.arange(0, n_steps)

    if not colors:
        colors = [plt.cm.Set2(x) for x in np.linspace(0, 1, n_states)]

    windows = get_threshold_windows(traces, thresh=thresh)
    handles = {}
    for win in windows:
        t_vec = [time[win[0]], time[win[1]]]
        h = ax.fill_between(t_vec, [y_min, y_min], [y_max, y_max],
                            color=colors[int(win[2])], alpha=0.4)
        if  win[2] not in handles:
            handles[win[2]] = h

    leg_handles = [handles[k] for k in sorted(handles.keys())]
    leg_labels = ['State %i' % k for k in sorted(handles.keys())]

    for line, col in zip(traces, colors):
        tmp = line
        if smoothing:
            tmp = gaussian_filter1d(tmp, smoothing)

        ax.plot(time, tmp, color=col, linewidth=2)

    return ax, leg_handles, leg_labels


def plot_forward_probs(hmm, spikes, dt, time=None, colors=None, axes=None, legend=True,
                       hmm_id=None, thresh=0.75, save_file=None):
    if not axes:
        fig, axes = make_hmm_raster(spikes, time=time)
    else:
        fig = axes[0].figure

    if legend:
        fig.subplots_adjust(right=0.9)  # To  make room for legend

    alphas, norms = hmm.get_forward_probabilities(spikes, dt)
    n_trials, n_states, n_steps = alphas.shape
    if time is None:
        time = np.arange(0, n_steps)

    if not colors:
        colors = [plt.cm.Set2(x) for x in np.linspace(0,1, n_states)]

    handles = []
    labels = []
    for trial, ax in zip(alphas, axes):
        _, tmp_handles, tmp_labels = plot_probability_traces(trial,time=time, ax=ax,
                                                             colors=colors, thresh=thresh)
        for l, h in zip(tmp_labels, tmp_handles):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    if time[0] != 0:
        ax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

    if legend:
        mid = int(n_trials/2)
        axes[mid].legend(handles, labels, loc='upper center',
                         bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                         fontsize=14)

    axes[-1].set_xlabel('Time (ms)')
    title_str = 'HMM Forward Probabilities'
    if hmm_id:
        title_str += '\n%s' % hmm_id

    axes[0].set_title(title_str)
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


def plot_backward_probs(hmm, spikes, dt, time=None, colors=None, axes=None, legend=True,
                        hmm_id=None, thresh=0.75, save_file=None):
    if not axes:
        fig, axes = make_hmm_raster(spikes, time=time)
    else:
        fig = axes[0].figure

    if legend:
        fig.subplots_adjust(right=0.9)  # To  make room for legend

    betas = hmm.get_backward_probabilities(spikes, dt)
    n_trials, n_states, n_steps = betas.shape
    if time is None:
        time = np.arange(0, n_steps)

    if not colors:
        colors = [plt.cm.Set2(x) for x in np.linspace(0,1, n_states)]

    handles = []
    labels = []
    for trial, ax in zip(betas, axes):
        _, tmp_handles, tmp_labels = plot_probability_traces(trial,time=time, ax=ax,
                                                             colors=colors, thresh=thresh)
        for l, h in zip(tmp_labels, tmp_handles):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    if time[0] != 0:
        ax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

    if legend:
        mid = int(n_trials/2)
        axes[mid].legend(handles, labels, loc='upper center',
                         bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                         fontsize=14)

    axes[-1].set_xlabel('Time (ms)')
    title_str = 'HMM Backward Probabilities'
    if hmm_id:
        title_str += '\n%s' % hmm_id

    axes[0].set_title(title_str)
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


def plot_gamma_probs(hmm, spikes=None, dt=None, time=None, colors=None, axes=None, legend=True,
                     hmm_id=None, thresh=0.75, save_file=None):
    if not axes:
        fig, axes = make_hmm_raster(spikes, time=time)
    else:
        fig = axes[0].figure

    if legend:
        fig.subplots_adjust(right=0.9)  # To  make room for legend

    gammas = hmm.stat_arrays['gamma_probabilities']
    if gammas == []:
        if spikes is None and dt is None:
            raise ValueError('Not enough info to compute gamma probabilities')

        gammas = hmm.get_gamma_probabilities(spikes, dt)

    n_trials, n_states, n_steps = gammas.shape
    if time is None:
        time = np.arange(0, n_steps)

    if not colors:
        colors = [plt.cm.Set2(x) for x in np.linspace(0,1, n_states)]

    handles = []
    labels = []
    for trial, ax in zip(gammas, axes):
        _, tmp_handles, tmp_labels = plot_probability_traces(trial,time=time, ax=ax,
                                                             colors=colors, thresh=thresh)
        for l, h in zip(tmp_labels, tmp_handles):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    if time[0] != 0:
        ax.axvline(0, color='red', linestyle='--', linewidth=3, alpha=0.8)

    if legend:
        mid = int(n_trials/2)
        axes[mid].legend(handles, labels, loc='upper center',
                         bbox_to_anchor=(0.8, .5, .5, .5), shadow=True,
                         fontsize=14)

    axes[-1].set_xlabel('Time (ms)')
    title_str = 'HMM Gamma Probabilities'
    if hmm_id:
        title_str += '\n%s' % hmm_id

    axes[0].set_title(title_str)
    if save_file:
        fig.savefig(save_file)
        return
    else:
        return fig, axes


def plot_hmm_rates(rates, axes=None, colors=None):
    '''Make bar plot of spike rates for each cell and state in an HMM emission
    matrix

    Parameters
    ----------
    rates: np.array
        Cell X State matrix of firing rates
    '''
    n_cells, n_states = rates.shape
    if axes is None:
        _, axes = plt.subplot(ncols=n_states)

    if len(axes) < n_states:
        raise ValueError('Must provided enough axes to plot each state')

    if not colors:
        colors = [plt.cm.Set2(x) for x in np.linspace(0,1, n_states)]

    df = pd.DataFrame(rates, columns=['state %i' % i for i in range(n_states)])
    df['cell'] = ['cell %i' % i for i in df.index]
    df = pd.melt(df, 'cell', ['state %i' % i for i in range(n_states)], 'state', 'rate')
    for g, ax, col in zip(df.groupby('state'), axes, colors):
        sns.barplot(data=g[1], x='rate', y='cell',
                    color='black', ax=ax)
        ax.set_title(g[0])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_facecolor(col)
        ax.patch.set_alpha(0.5)
        ax.set_yticklabels([])
        ax.tick_params(left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)


    axes[0].set_yticklabels(['Cell %i' % i for i in range(n_cells)])
    mid = int(n_states/2)
    axes[mid].set_xlabel('Firing Rate (Hz)')
    return axes


def plot_hmm_transition(transition, ax=None):
    if not ax:
        _, ax = plt.gca()

    n_states = transition.shape[0]
    labels = ['State %i' % i for i in range(n_states)]
    sns.heatmap(transition, ax=ax, cmap='plasma', cbar=True, square=True,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                cbar_kws={'shrink': 0.5})
    ax.set_ylim((0, n_states))
    ax.set_title('Transition Probabilities')
    return ax


def plot_hmm_initial_probs(PI, ax=None):
    if not ax:
        _ , ax = plt.gca()

    n_states = PI.shape[0]
    labels = ['State %i' % i for i in range(n_states)]
    PI = np.expand_dims(PI, 1)
    sns.heatmap(PI, ax=ax, cmap='plasma', cbar=True,
                yticklabels=labels, vmin=0, vmax=1)
    ax.set_ylim((0, n_states))
    ax.set_title('Initial Probabilities')
    return ax


def plot_hmm_overview(hmm, colors=None, hmm_id=None, save_file=None):
    n_states = hmm.n_states
    if not colors:
        colors = get_hmm_plot_colors(n_states)

    PI = hmm.initial_distribution
    A = hmm.transition
    B = hmm.emission
    fig, axes = plt.subplots(nrows=2, ncols=np.max((n_states,2)), figsize=(20, 15))
    if n_states > 2:
        for ax in axes[0,1:-1]:
            ax.axis('off')

    plot_hmm_initial_probs(PI, ax=axes[0,0])
    plot_hmm_transition(A, ax=axes[0,-1])
    plot_hmm_rates(B, axes=axes[1,:], colors=colors)
    mid = int(n_states/2)
    axes[1, mid].set_xlabel('')
    tmp_ax = fig.add_subplot('111', frameon=False)
    tmp_ax.tick_params(labelcolor='none', top=False, bottom=False,
                       left=False, right=False)
    tmp_ax.set_xlabel('Firing Rate (Hz)')

    fig.subplots_adjust(top=0.9)
    title_str = 'Fitted HMM Parameters'
    if hmm_id:
        title_str += '\n%s' % hmm_id

    fig.suptitle(title_str)
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


def plot_hmm_figures(hmm, spikes, dt, time, hmm_id=None, save_dir=None):
    colors = get_hmm_plot_colors(hmm.n_states)
    if hmm_id is None:
        hmm_id = hmm.hmm_id


    fig_names = ['sequences', 'forward_probabilities',
                 'backward_probabilities', 'gamma_probabilities', 'overview']
    if save_dir:
        files = {x : os.path.join(save_dir, '%s.png' % x) for x in fig_names}
    else:
        files = dict.fromkeys(fig_names, None)


    # Plot sequences
    print('Plotting Viterbi Decoded Paths...')
    plot_viterbi_paths(hmm, spikes, time=time, colors=colors,
                       hmm_id=hmm_id, save_file=files['sequences'])

    # Plot alphas
    print('Plotting Forward Probabilities...')
    plot_forward_probs(hmm, spikes, dt, time=time, colors=colors,
                       hmm_id=hmm_id, save_file=files['forward_probabilities'])
    # Plot betas
    print('Plotting Backward Probabilities...')
    plot_backward_probs(hmm, spikes, dt, time=time, colors=colors,
                        hmm_id=hmm_id, save_file=files['backward_probabilities'])

    # Plot gammas
    print('Plotting Gamma Probabilities...')
    plot_gamma_probs(hmm, spikes, dt, time=time, colors=colors,
                     hmm_id=hmm_id, save_file=files['gamma_probabilities'])

    # Plot stats: rate bar plots, transition heat map, initial probabilities
    print('Plotting HMM Overview...')
    plot_hmm_overview(hmm, colors=colors, save_file=files['overview'])

    print('Plotting Complete!')
    plt.close('all')
