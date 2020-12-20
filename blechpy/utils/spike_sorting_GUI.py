import tkinter as tk
import sys
from tkinter import ttk
import numpy as np
import matplotlib
from blechpy.utils import userIO, tk_widgets as tkw
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from blechpy.utils.tk_widgets import ScrollFrame, window_center
from blechpy.analysis.blech_clustering import SpikeSorter


class SpikeSorterGUI(ttk.Frame):
    def __init__(self, parent, spike_sorter, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.root.style = ttk.Style()
        self.root.style.theme_use('clam')
        self.root.geometry('915x800')
        self.pack(fill='both', expand=True)
        window_center(self.root)

        self.sorter = spike_sorter
        self.sorter._shell = False
        self.electrode = spike_sorter.electrode
        self.initUI()

    def initUI(self):
        # Make layout
        row1 = ttk.Frame(self)
        row2 = ttk.Frame(self)

        figs = ttk.Frame(row2)
        ui = ttk.Frame(row2)

        solution_row = ttk.Frame(ui)
        click_row = ttk.Frame(ui)

        checks = ttk.Frame(click_row)
        buttons = ttk.Frame(click_row)

        buttons.pack(side='right', padx=10)
        checks.pack(side='right', padx=10)

        solution_row.pack(side='top', anchor='n', pady=30)
        click_row.pack(side='top', anchor='n')

        figs.pack(side='left', fill='both', expand=True)
        ui.pack(side='left')

        row1.pack(side='top')
        row2.pack(side='top', fill='both', expand=True, anchor='n')

        # Make title
        title = tk.Label(row1, text='Electrode %i' % self.electrode)
        title.pack(side='top', fill='x')

        # Make user interface pane

        # Select solution
        solutions = self.sorter.get_possible_solutions()
        self._solution_var = tk.IntVar(self, max(solutions))
        solution_drop = ttk.OptionMenu(solution_row, self._solution_var,
                                       *solutions)
        solution_drop.pack(side='right')
        ttk.Label(solution_row, text='Solution Clusters: ').pack(side='right')

        self._solution_var.trace('w', self.change_solution)

        # Set sorter to max solutions
        self.sorter.set_active_clusters(self._solution_var.get())

        # Check boxes
        cluster_choices = list(range(self._solution_var.get()))
        self._check_bar = CheckBar(checks, cluster_choices)
        ttk.Label(checks, text='Clusters:').pack(side='top', pady=10)
        self._check_bar.pack()

        # Buttons
        merge = ttk.Button(buttons, text='Merge', command=self.merge_clusters)
        split = ttk.Button(buttons, text='Split', command=self.split_clusters)
        splitUMAP = ttk.Button(buttons, text='UMAP Split (Slow)',
                               command=self.umap_split_clusters)
        save = ttk.Button(buttons, text='Save Cells', command=self.save)
        viewWaves = ttk.Button(buttons, text='View Waves', command=self.view_waves)
        # viewRecWaves = ttk.Button(buttons, text='View Waves by Rec',
        #                           command=self.view_waves_by_rec)
        viewTimeWaves = ttk.Button(buttons, text='View Waves over Time',
                                  command=self.view_waves_over_time)
        viewPCA = ttk.Button(buttons, text='View PCA', command=self.view_pca)
        viewUMAP = ttk.Button(buttons, text='View UMAP', command=self.view_umap)
        viewWAVELET = ttk.Button(buttons, text='View Wavelets', command=self.view_wavelets)
        viewRaster = ttk.Button(buttons, text='View Raster', command=self.view_raster)
        viewISI = ttk.Button(buttons, text='View ISI', command=self.view_ISI)
        viewXCORR = ttk.Button(buttons, text='View XCorr', command=self.view_xcorr)
        viewACORR = ttk.Button(buttons, text='View AutoCorr', command=self.view_acorr)
        discard = ttk.Button(buttons, text='Discard Clusters', command=self.discard_clusters)
        self._undo_button = ttk.Button(buttons, text='Undo', command=self.undo)
        merge.pack(side='top', fill='x', pady=5)
        split.pack(side='top', fill='x', pady=5)
        splitUMAP.pack(side='top', fill='x', pady=5)
        save.pack(side='top', fill='x', pady=5)
        viewWaves.pack(side='top', fill='x', pady=5)
        # viewRecWaves.pack(side='top', fill='x', pady=5)
        viewTimeWaves.pack(side='top', fill='x', pady=5)
        viewPCA.pack(side='top', fill='x', pady=5)
        viewUMAP.pack(side='top', fill='x', pady=5)
        viewWAVELET.pack(side='top', fill='x', pady=5)
        viewRaster.pack(side='top', fill='x', pady=5)
        viewISI.pack(side='top', fill='x', pady=5)
        viewACORR.pack(side='top', fill='x', pady=5)
        viewXCORR.pack(side='top', fill='x', pady=5)
        discard.pack(side='top', fill='x', pady=5)
        self._undo_button.pack(side='top', fill='x', pady=5)
        self._undo_button.config(state='disabled')

        self._ui_frame = ui

        # Figures
        self._wavepane = WaveformPane(figs)
        self._wavepane.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        self.update()


    def update(self):
        if self.sorter._last_action is not None:
            self._undo_button.config(text='Undo ' + self.sorter._last_action,
                                     state='normal')
        else:
            self._undo_button.config(text='Undo', state='disabled')

        # Check active clusters
        clusters = list(range(len(self.sorter._active)))

        # Update cluster checkbar
        self._check_bar.updateChoices(clusters)

        # Update waveforms
        wave_dict = {}
        for i in clusters:
            wave_dict[i] = self.sorter.get_mean_waveform(i)

        self._wavepane.update(wave_dict)

    def undo(self):
        self.sorter.undo()
        self.update()

    def change_solution(self, *args):
        solutions = self._solution_var.get()
        self.sorter.set_active_clusters(solutions)
        self.update()

    def merge_clusters(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) < 2:
            return

        self.sorter.merge_clusters(chosen)
        self.update()

    def umap_split_clusters(self, *args):
        self.split_clusters(umap=True)

    def split_clusters(self, *args, umap=False):
        chosen = self._check_bar.get_selected()
        if len(chosen) != 1:
            return

        if isinstance(chosen, list):
            chosen = chosen[0]

        params = {'n_iterations': 1000,
                  'n_restarts': 10,
                  'thresh': 10e-6,
                  'n_clusters': int}

        popup = userIO.fill_dict_popup(params, master=self.root,
                                       prompt='Input parameters for splitting')
        self.disable_all()
        self.root.wait_window(popup.root)
        self.enable_all()
        params = popup.output
        if popup.cancelled or params['n_clusters'] is None:
            return

        new_clusts = self.sorter.split_cluster(chosen, params['n_iterations'],
                                               params['n_restarts'],
                                               params['thresh'],
                                               params['n_clusters'],
                                               store_split=True,
                                               umap=umap)

        choices = ['%i' % i for i in range(len(new_clusts))]
        popup = tkw.ListSelectPopup(choices, self.root,
                                    'Select split clusters to keep.\n'
                                    'Cancel to undo split.', multi_select=True)
        self.disable_all()
        self.root.wait_window(popup.root)
        self.enable_all()
        chosen = list(map(int, popup.output))

        self.sorter.set_split(chosen)
        self.update()

    def save(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        cell_types = {}
        for i in chosen:
            cell_types[i] = {'single_unit': False, 'pyramidal': False,
                             'interneuron': False}

        popup  = userIO.fill_dict_popup(cell_types, master=self.root)
        self.disable_all()
        self.root.wait_window(popup.root)
        self.enable_all()
        cell_types = popup.output
        if popup.cancelled:
            return

        single = [cell_types[i]['single_unit'] for i in sorted(cell_types.keys())]
        pyramidal = [cell_types[i]['pyramidal'] for i in sorted(cell_types.keys())]
        interneuron = [cell_types[i]['interneuron'] for i in sorted(cell_types.keys())]
        self.sorter.save_clusters(chosen, single, pyramidal, interneuron)
        self.update()

    def view_waves(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.plot_clusters_waveforms(chosen)

    def view_waves_by_rec(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) != 1:
            return

        if isinstance(chosen, list):
            chosen = chosen[0]

        self.sorter.plot_cluster_waveforms_by_rec(chosen)

    def view_waves_over_time(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) != 1:
            return

        if isinstance(chosen, list):
            chosen = chosen[0]

        params = {'interval': int}
        popup = userIO.fill_dict_popup(params, master=self.root,
                                       prompt='Input time interval for segments (in seconds)')
        self.disable_all()
        self.root.wait_window(popup.root)
        self.enable_all()
        params = popup.output
        if popup.cancelled or params['interval'] is None:
            return

        self.sorter.plot_cluster_waveforms_over_time(chosen, params['interval'])

    def view_pca(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.plot_clusters_pca(chosen)

    def view_umap(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.disable_all()
        self.sorter.plot_clusters_umap(chosen)
        self.enable_all()

    def view_wavelets(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.disable_all()
        self.sorter.plot_clusters_wavelets(chosen)
        self.enable_all()

    def view_ISI(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.plot_clusters_ISI(chosen)

    def view_raster(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.plot_clusters_raster(chosen)

    def view_acorr(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.plot_clusters_acorr(chosen)

    def view_xcorr(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.plot_clusters_xcorr(chosen)

    def discard_clusters(self, *args):
        chosen = self._check_bar.get_selected()
        if len(chosen) == 0:
            return

        self.sorter.discard_clusters(chosen)
        self.update()

    def undo_save(self, *args):
        self.sorter.undo_last_save()
        self.update()

    def cstate(self,state,widget=None):
        if widget is None:
            widget = self
        if widget.winfo_children:
            for w in widget.winfo_children():
                try:
                    w.state((state,))
                except:
                    pass
                self.cstate(state,widget=w)

    def enable_all(self):
        self.cstate('!disabled')

    def disable_all(self):
        self.cstate('disabled')



class CheckBar(ttk.Frame):
    def __init__(self, parent=None, choices=[]):
        tk.Frame.__init__(self, parent)
        self.choices = choices
        title = ttk.Label(self, text='Clusters')
        self.choice_rows = []
        self.choice_vars = []
        self.updateChoices()

    def updateChoices(self, new_choices=[]):
        if len(new_choices) > 0:
            self.choices = new_choices

        if len(self.choice_rows) > 0:
            for row in self.choice_rows:
                row.destroy()

        self.choice_vars = [tk.IntVar(self, 0) for i in self.choices]
        self.choice_rows = []
        for choice, var in zip(self.choices, self.choice_vars):
            check = tk.Checkbutton(self, text=str(choice), variable=var)
            check.pack(fill='x', anchor='n', pady=5)
            self.choice_rows.append(check)

    def get_selected(self):
        checked = [i.get() for i in self.choice_vars]
        out = [self.choices[i] for i, j in enumerate(checked) if j==1]
        return out


def make_waveform_plot(wave, wave_std, n_waves=None,index=None):
    minima = min(wave)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.xaxis.set_tick_params(bottom=False, top=False, labelbottom=False)
    ax.yaxis.set_tick_params(bottom=False, top=False, labelbottom=False)
    fig.tight_layout()
    X = list(range(len(wave)))
    ax.fill_between(X, [i+j for i,j in zip(wave, wave_std)],
                    [i-j for i,j in zip(wave, wave_std)],
                    alpha=0.4)
    ax.plot(X, wave, linewidth=3)
    ax.autoscale(axis='x', tight=True)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    tmp_str = 'Amp: %0.1f' % minima
    if n_waves:
        tmp_str += '\nN Waves: %i' % n_waves

    ax.text(xlim[1]-200, ylim[0] + 0.1*abs(ylim[0]), tmp_str,
            fontsize=10)

    if index is not None:
        ax.text(xlim[0] + 10, ylim[0] + 0.05*abs(ylim[0]), str(index),
                fontweight='bold', fontsize=14)

    return fig


class WaveformPane(ttk.Frame):
    def __init__(self, parent, wave_dict=None, **kwargs):
        self._wave_dict = wave_dict
        tk.Frame.__init__(self, parent, **kwargs)
        self.parent = parent
        self._fig_rows = []
        self._all_figs = []
        self._fig_canvases = []
        self.initUI()

    def initUI(self):
        self.scrollpane = ScrollFrame(self)
        self.scrollpane.pack(fill='both', expand=True, padx=10, pady=10)
        self.update()

    def update(self, wave_dict=None):
        if wave_dict is not None:
            self._wave_dict = wave_dict

        if len(self._fig_rows) > 0:
            for row in self._fig_rows:
                row.destroy()

        if len(self._all_figs) > 0:
            for f in self._all_figs:
                plt.close(f)

        if len(self._fig_canvases) > 0:
            for canvas in self._fig_canvases:
                canvas.get_tk_widget().destroy()

        self._fig_rows = []
        self._all_figs = []
        self._fig_canvases = []

        if self._wave_dict is None:
            return

        row = None
        for k, v in self._wave_dict.items():
            wave = v[0]
            wave_std = v[1]
            n_waves = v[2]
            fig = make_waveform_plot(wave, wave_std, n_waves=n_waves, index=k)
            self._all_figs.append(fig)

            if row is None:
                row = ttk.Frame(self.scrollpane.viewport)
                row.pack(side='top', fill='x')
                self._fig_rows.append(row)
                side = 'left'
                delFlag = False
            else:
                side = 'right'
                delFlag = True

            canvas = FigureCanvasTkAgg(fig, row)
            canvas.draw()
            canvas.get_tk_widget().pack(side=side)
            self._fig_canvases.append(canvas)
            if delFlag:
                row = None

        self.scrollpane.bind_children_to_mouse()


class DummySorter(object):
    def __init__(self, electrode, shell=False):
        self.electrode = electrode
        # Match recording directory ordering to clustering
        self._active = None
        self._previous=None
        self._waves = {}
        self._shell = shell

    def set_active_clusters(self, solution_num):
        clusters = [i for i in range(solution_num)]
        self._active = clusters

    def save_clusters(self, target_clusters, single_unit, pyramidal, interneuron):
        '''Saves active clusters as cells, write them to the h5_files in the
        appropriate recording directories

        Parameters
        ----------
        target_clusters: list of int
            indicies of active clusters to save
        single_unit : list of bool
            elements in list must correspond to elements in active clusters
        pyramidal : list of bool
        interneuron : list of bool
        '''
        if self._active is None:
            return

        clusters = [self._active[i] for i in target_clusters]
        for clust, single, pyr, intr in zip(clusters, single_unit,
                                            pyramidal, interneuon):
            out_str = ['Saved cluster %i.' % clust]
            if pyr:
                out_str.append('Pyramidal')

            if intr:
                out_str.append('Interneuron')

            if single:
                out_str.append('Single-Unit')

            print(' '.join(out_str))

        self._active = [self._active[i] for i in range(len(self._active))
                        if i not in target_clusters]
        self._previous = clusters

    def undo_last_save(self):
        if self._previous is None:
            return

        last_saved = self._last_saved
        self._active.extend(self._previous)
        self._previous = None

    def split_cluster(self, target_clust, n_iter, n_restart, thresh, n_clust):
        '''splits the target active cluster using a GMM
        '''
        if target_clust >= len(self._active):
            raise ValueError('Invalid target. Only %i active clusters' % len(self._active))

        cluster = self._active.pop(target_clust)
        new_clusts = [cluster*10+i for i in range(n_clust)]
        selection_list = ['all'] + ['%i' % i for i in range(len(new_clusts))]
        prompt = 'Select split clusters to keep'
        ans = userIO.select_from_list(prompt, selection_list,
                                      multi_select=True, shell=self._shell)
        if ans is None or 'all' in ans:
            print('Reset to before split')
            self._active.insert(target_cluster, cluster)
        else:
            self._waves.pop(clusters)
            keepers = [new_clusts[int(i)] for i in ans]
            self._active.extend(keepers)

    def merge_clusters(self, target_clusters):
        if any([i >= len(self._active) for i in target_clusters]):
            raise ValueError('Target cluster is out of range.')

        clusters = [self._active[i] for i in target_clusters]
        new_clust = sum([pow(10,i)*j for i,j in enumerate(clusters)])
        self._active = [self._active[i] for i in range(len(self._active))
                        if i not in target_clusters]
        for c in clusters:
            self._waves.pop(c)

        self._active.append(new_clust)

    def discard_clusters(self, target_clusters):
        self._active = [self._active[i] for i in range(len(self._active))
                        if i not in target_clusters]
        for i in target_clusters:
            self._waves.pop(i)

    def plot_clusters_waveforms(self, target_clusters):
        pass

    def plot_clusters_pca(self, target_clusters):
        pass

    def plot_clusters_raster(self, target_clusters):
        pass

    def plot_clusters_ISI(self, target_clusters):
        pass

    def get_mean_waveform(self, target_cluster):
        '''Returns mean waveform of target_cluster in active clusters. Also
        returns SEM of waveforms
        '''
        cluster = self._active[target_cluster]
        if self._waves.get(cluster) is not None:
            return self._waves[cluster][0], self._waves[cluster][1]

        amp = np.random.randint(-60, -20, 1)[0]
        start = np.random.randint(0, 20, 1)[0]
        rebound = np.random.randint(-10,40,1)[0]
        steps_to_rise = np.random.randint(5,30,1)[0]
        tmp = np.array([start, start, amp, rebound, start])
        xp = np.array([0, 10, 15, 15+steps_to_rise, 45])
        x = np.arange(0,45)
        wave = np.interp(x,xp,tmp)
        wave_sem = np.random.randint(1,10,45)
        self._waves[cluster] = (wave, wave_sem)
        return wave, wave_sem

    def get_possible_solutions(self):
        return list(range(8))

def launch_sorter_GUI(sorter):
    root = tk.Tk()
    sorter_GUI = SpikeSorterGUI(root, sorter)
    return root, sorter_GUI

