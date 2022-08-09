import os
import shutil
from tqdm import tqdm
import multiprocessing
import numpy as np
import pandas as pd
from itertools import combinations
from blechpy import dio
from blechpy.datastructures.objects import data_object, load_dataset
from blechpy.utils import userIO, print_tools as pt, write_tools as wt, spike_sorting_GUI as ssg
from blechpy.analysis import held_unit_analysis as hua, blech_clustering as bclust
from blechpy.plotting import data_plot as dplt
from blechpy.utils.decorators import Logger


class experiment(data_object):
    def __init__(self, exp_dir=None, exp_name=None, shell=False, order_dict=None):
        '''Setup for analysis across recording sessions

        Parameters
        ----------
        exp_dir : str (optional)
            path to directory containing all recording directories
            if None (default) is passed then a popup to choose file
            will come up
        shell : bool (optional)
            True to use command-line interface for user input
            False (default) for GUI
        '''
        if 'SSH_CONNECTION' in os.environ:
            shell = True

        super().__init__('experiment', root_dir=exp_dir, data_name=exp_name, shell=shell)

        fd = [os.path.join(exp_dir, x) for x in os.listdir(exp_dir)]
        file_dirs = [x for x in fd if (os.path.isdir(x) and
                                       dio.h5io.get_h5_filename(x) is not None)]
        if len(file_dirs) == 0:
            q = userIO.ask_user('No recording directories with h5 files found '
                               'in experiment directory\nContinue creating'
                               'empty experiment?', shell=shell)
            if q == 0:
                return

        self.recording_dirs = file_dirs
        self._order_dirs(shell=shell, order_dict=order_dict)

        rec_names = [os.path.basename(x) for x in self.recording_dirs]
        el_map = None
        rec_labels = {}
        for rd in self.recording_dirs:
            dat = load_dataset(rd)
            if dat is None:
                raise FileNotFoundError('No dataset.p object found for in %s' % rd)
            elif el_map is None:
                el_map = dat.electrode_mapping.copy()

            rec_labels[dat.data_name] = rd

        self.rec_labels = rec_labels

        self.electrode_mapping = el_map
        self._setup_taste_map()

        save_dir = os.path.join(self.root_dir, '%s_analysis' % self.data_name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.analysis_dir = save_dir
        self.save()

    def _change_root(self, new_root=None):
        old_root = self.root_dir
        new_root = super()._change_root(new_root)
        self.recording_dirs = [x.replace(old_root, new_root)
                               for x in self.recording_dirs]
        self.rec_labels = {k: v.replace(old_root, new_root)
                           for k,v in self.rec_labels.items()}
        return new_root

    def __str__(self):
        out = [super().__str__()]
        out.append('Analysis Directory: %s' % self.analysis_dir)
        out.append('Recording Directories :')
        out.append(pt.print_dict(self.rec_labels, tabs=1))
        out.append('\nTaste Mapping :')
        out.append(pt.print_dict(self.taste_map, tabs=1))
        out.append('\nElectrode Mapping\n----------------')
        out.append(pt.print_dataframe(self.electrode_mapping))
        if hasattr(self, 'held_units'):
            out.append('\nHeld Units :')
            out.append(pt.print_dataframe(self.held_units.drop(columns=['J3'])))

        return '\n'.join(out)

    def _order_dirs(self, shell=False, order_dict=None):
        '''set order of redcording directories
        '''
        if 'SSH_CONNECTION' in os.environ:
            shell = True

        if self.recording_dirs == []:
            return

        if order_dict is None:
            self.recording_dirs = [x[:-1] if x.endswith(os.sep) else x
                                   for x in self.recording_dirs]
            top_dirs = {os.path.basename(x): os.path.dirname(x)
                        for x in self.recording_dirs}
            file_dirs = list(top_dirs.keys())
            order_dict = dict.fromkeys(file_dirs, 0)
            tmp = userIO.dictIO(order_dict, shell=shell)
            order_dict = userIO.fill_dict(order_dict,
                                          ('Set order of recordings (1-%i)\n'
                                           'Leave blank to delete directory'
                                           ' from list') % len(file_dirs),
                                          shell)
            if order_dict is None:
                return

            file_dirs = [k for k, v in order_dict.items()
                         if v is not None and v != 0]
            file_dirs = sorted(file_dirs, key=order_dict.get)
            file_dirs = [os.path.join(top_dirs.get(x), x) for x in file_dirs]
        else:
            file_dirs = sorted(self.recording_dirs, key=order_dict.get) #TODO: idk what is hapening with these key orders this is fucked
            
            
        self.order_dict = order_dict
        self.recording_dirs = file_dirs

    def _setup_taste_map(self):
        rec_dirs = self.recording_dirs
        rec_labels = self.rec_labels
        tastants = []
        for rd in rec_dirs:
            dat = load_dataset(rd)
            tmp = dat.dig_in_mapping
            tastants.extend(tmp['name'].to_list())

        tastants = np.unique(tastants)
        taste_map = {}
        for rl, rd in rec_labels.items():
            dat = load_dataset(rd)
            din = dat.dig_in_mapping
            for t in tastants:
                if taste_map.get(t) is None:
                    taste_map[t] = {}

                tmp = din['channel'][din['name'] == t]
                if not tmp.empty:
                    taste_map[t][rl] = tmp.values[0]
                    
        self.taste_map = taste_map

    def add_recording(self, new_dir=None, shell=None):
        '''Add recording directory to experiment

        Parameters
        ----------
        new_dir : str (optional)
            full path to new directory to add to recording dirs
        shell : bool (optional)
            True for command-line interface for user input
            False (default) for GUI
            If not passed then the preference set upon object creation is used
        '''
        if 'SSH_CONNECTION' in os.environ:
            shell = True
        elif shell is None:
            shell = False

        if new_dir is None:
            new_dir = userIO.get_filedirs('Select recoring directory',
                                          root=self.root_dir, shell=shell)

        if not os.path.isdir(new_dir):
            raise NotADirectoryError('%s must be a valid directory' % new_dir)

        if not any([x.endswith('.h5') for x in os.listdir(new_dir)]):
            raise FileNotFoundError('No .h5 file found in %s' % new_dir)

        if not any([x.endswith('dataset.p') for x in os.listdir(new_dir)]):
            raise FileNotFoundError('*_dataset.p file not found in %s' % new_dir)

        if new_dir.endswith('/'):
            new_dir = new_dir[:-1]

        label = userIO.get_user_input('Enter label for recording %s' %
                                      os.path.basename(new_dir), shell=shell)

        self.recording_dirs.append(new_dir)
        self.rec_labels[label] = new_dir
        self._order_dirs(shell=shell)
        self._setup_taste_map()
        print('Added recording: %s')
        self.save()

    def remove_recording(self, rec_dir=None, shell=None):
        '''Remove recording directory from experiment

        Parameters
        ----------
        rec_dir : str (optional)
            full path or label of the directory to remove
        shell : bool (optional)
            True for command-line interface for user input. Default for SSH
            False (default) for GUI
            If not passed then the preference set upon object creation is used
        '''
        if 'SSH_CONNECTION' in os.environ:
            shell = True
        elif shell is None:
            shell = False

        if rec_dir is None:
            rec_dir = userIO.select_from_list('Choose recording to remove\n'
                                              'Leave blank to cancel',
                                              list(self.rec_labels.keys(())),
                                              shell=shell)
            if rec_dir is None:
                return

        if os.path.isabs(rec_dir):
            if rec_dir.endswith('/'):
                rec_dir = rec_dir[:-1]

            idx = list(self.rec_labels.values()).index(rec_dir)  # throws ValueError
            key = list(self.rec_labels.keys())[idx]
        else:
            key = rec_dir
            rec_dir = self.rec_labels.get(key)
            if rec_dir is None:
                raise ValueError('%s is not in recording dirs' % key)

        self.rec_labels.pop(key)
        self.recording_dirs.pop(rec_dir)
        self._setup_taste_map()
        print('Removed recording: %s' % rec_dir)
        self.save()

    @Logger('Detecting held units')
    def detect_held_units(self, percent_criterion=95, raw_waves=False, shell=False):
        '''Determine which units are held across recording sessions
        Grabs single units from each recording and compares consecutive
        recordings to determine if units were held

        Parameters
        ----------
        percent_criterion : float
            percentile (0-100) of intra_J3 below which to accept units as held
            5.0 (default) for 95th percentile
            lower number is stricter criteria

        shell : bool (optional)
            True for command-line interface for user input
            False (default) for GUI
        '''
        if 'SSH_CONNECTION' in os.environ:
            shell = True

        save_dir = os.path.join(self.analysis_dir, 'held_unit_detection')
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        os.mkdir(save_dir)

        rec_dirs = self.recording_dirs
        rec_labels = self.rec_labels
        rec_names = list(rec_labels.keys())

        print('Detecting held units for :')
        print('\t' + '\n\t'.join(rec_names))
        print('Saving output to : %s' % save_dir)

        held_df, intra_J3, inter_J3 = hua.find_held_units(rec_dirs,
                                                          percent_criterion,
                                                          raw_waves=raw_waves)

        rl_dict = {os.path.basename(v) : k for k, v in self.rec_labels.items()}
        held_df = held_df.rename(columns=rl_dict)

        em = self.electrode_mapping
        held_df = held_df.apply(lambda x: self._assign_area(x), axis=1)


        self.held_units = held_df
        self.J3_values = {'intra_J3': intra_J3,
                          'inter_J3': inter_J3}

        # Write dataframe of held units to text file
        df_file = os.path.join(save_dir, 'held_units_table.txt')
        json_file = os.path.join(save_dir, 'held_units.json')
        held_df.to_json(json_file, orient='records', lines=True)

        # Print table of held unti to text file, separate tables by pairs of recordings
        rec_pairs = [(rec_names[i], rec_names[i+1]) for i in range(len(rec_names) - 1)]
        with open(df_file, 'w') as f:
            for rec1, rec2 in rec_pairs:
                tmp_df = held_df.copy()
                exc = [x for x in rec_names if x not in [rec1, rec2]]
                tmp_df = tmp_df.drop(columns=['J3', *exc]).dropna()
                print('Units held from %s to %s\n----------' % (rec1, rec2), file=f)
                print(pt.print_dataframe(tmp_df, tabs=1), file=f)
                print('', file=f)

        np.save(os.path.join(save_dir, 'intra_J3'), np.array(intra_J3))
        np.save(os.path.join(save_dir, 'inter_J3'), np.array(inter_J3))

        # For each held unit, plot waveforms side by side
        dplt.plot_held_units(rec_dirs, held_df, save_dir, rec_names=rec_names)

        # Plot intra and inter J3
        dplt.plot_J3s(intra_J3, inter_J3, save_dir, percent_criterion)
        self.save()

    def _assign_area(self, row):
        data_dir = self.root_dir
        em = self.electrode_mapping
        rec = None
        unit = None
        for k, v in self.rec_labels.items():
            if not pd.isna(row[k]):
                rec = v
                unit = row[k]
                break

        if rec is None:
            row['area'] = float('nan')
            row['electrode'] = float('nan')
            return row

        unit_num = dio.h5io.parse_unit_number(unit)
        descrip = dio.h5io.get_unit_descriptor(rec, unit_num)
        electrode = descrip['electrode_number']
        area = em.query('Electrode == @electrode')['area'].values[0]

        row['electrode'] = electrode
        row['area'] = area
        return row

    @Logger('Running Spike Clustering')
    def cluster_spikes(self, data_quality=None, multi_process=True,
                       n_cores=None, custom_params=None, umap=False):
        '''Write clustering parameters to file and
        Run blech_process on each electrode using GNU parallel

        Parameters
        ----------
        data_quality : {'clean', 'noisy', None (default)}
            set if you want to change the data quality parameters for cutoff
            and spike detection before running clustering. These parameters are
            automatically set as "clean" during initial parameter setup
        accept_params : bool, False (default)
            set to True in order to skip popup confirmation of parameters when
            running
        '''
        clustering_params = None
        if custom_params:
            clustering_params = custom_params
        elif data_quality:
            tmp = dio.params.load_params('clustering_params', self.root_dir,
                                         default_keyword=data_quality)
            if tmp:
                clustering_params = tmp
            else:
                raise ValueError('%s is not a valid data_quality preset. Must '
                                 'be "clean" or "noisy" or None.')


        # Get electrodes, throw out 'dead' electrodes
        em = self.electrode_mapping
        if 'dead' in em.columns:
            electrodes = em.Electrode[em['dead'] == False].tolist()
        else:
            electrodes = em.Electrode.tolist()


        # Setup progress bar
        pbar = tqdm(total = len(electrodes))
        def update_pbar(ans):
            pbar.update()


        # get clustering params
        rec_dirs = list(self.rec_labels.values())
        if clustering_params is None:
            dat =  load_dataset(rec_dirs[0])
            clustering_params = dat.clustering_params.copy()

        print('\nRunning Blech Clust\n-------------------')
        print('Parameters\n%s' % pt.print_dict(clustering_params))

        # Write clustering params to recording directories & check for spike detection
        spike_detect = True
        for rd in rec_dirs:
            dat = load_dataset(rd)
            if dat.process_status['spike_detection'] == False:
                raise FileNotFoundError('Spike detection has not been run on %s' % rd)

            dat.clustering_params = clustering_params
            wt.write_params_to_json('clustering_params', rd, clustering_params)
            # dat.save()

        # Run clustering
        if not umap:
            clust_objs = [bclust.BlechClust(rec_dirs, x, params=clustering_params)
                          for x in electrodes]
        else:
            clust_objs = [bclust.BlechClust(rec_dirs, x,
                                            params=clustering_params,
                                            data_transform=bclust.UMAP_METRICS,
                                            n_pc=5)
                          for x in electrodes]

        if multi_process:
            if n_cores is None or n_cores > multiprocessing.cpu_count():
                n_cores = multiprocessing.cpu_count() - 1

            pool = multiprocessing.get_context('spawn').Pool(n_cores)
            for x in clust_objs:
                pool.apply_async(x.run, callback=update_pbar)

            pool.close()
            pool.join()
        else:
            for x in clust_objs:
                res = x.run()
                update_pbar(res)

        pbar.close()

        for rd in rec_dirs:
            dat = load_dataset(rd)
            dat.process_status['spike_clustering'] = True
            dat.process_status['cleanup_clustering'] = False
            # dat.save()

        # self.save()
        print('Clustering Complete\n------------------')

    def sort_spikes(self, electrode=None, shell=False):
        if electrode is None:
            electrode = userIO.get_user_input('Electrode #: ', shell=shell)
            if electrode is None or not electrode.isnumeric():
                return

            electrode = int(electrode)

        rec_dirs = list(self.rec_labels.values())
        for rd in rec_dirs:
            dat = load_dataset(rd)
            if not dat.process_status['cleanup_clustering']:
                dat.cleanup_clustering()

            dat.process_status['sort_units'] = True

        sorter = bclust.SpikeSorter(rec_dirs, electrode, shell=shell)
        if not shell:
           root, sorter_GUI = ssg.launch_sorter_GUI(sorter)
           return root, sorter_GUI
        else:
            print('No shell UI yet')
            return
        
    #def plot_held_session_rasters():
    def get_electrode_unit_counts(self):
        all_unit_table = pd.DataFrame()
        for rd in self.recording_dirs:
            dat = load_dataset(rd)
            nm = dat.data_name
            unit_table = dat.get_unit_table()
            unit_table['rec'] = nm
            all_unit_table = all_unit_table.append(unit_table)
            
        out = all_unit_table.groupby(['electrode','rec']).size().reset_index(name = 'n_units')
        return out
            
    def export_electrode_unit_counts(self):
        uc = self.get_electrode_unit_counts()
        filename = os.path.join(self.root_dir,'electrode_unit_counts.csv')
        uc.to_csv(filename)
        

