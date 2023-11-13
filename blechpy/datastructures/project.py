import os
import pandas as pd
from blechpy.datastructures.objects import data_object, load_experiment, load_dataset
from blechpy.utils import write_tools as wt
from blechpy.utils.decorators import Logger
from blechpy.utils import userIO
from blechpy.analysis import poissonHMM as phmm
from joblib import Parallel, delayed
import os
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

#import glob

class project(data_object):

    def __init__(self, proj_dir=None, proj_name=None, exp_dirs=None,
                 exp_groups=None, params=None, shell=False):
        if 'SSH_CONNECTION' in os.environ:
            shell = True


        # Setup basics
        super().__init__('project', data_name=proj_name,
                         root_dir=proj_dir, shell=shell)

        if exp_dirs is None:
            exp_dirs = userIO.get_filedirs('Select experiment directories',
                                           multi=True, shell=shell)

        exp_names = []
        for ed in exp_dirs:
            exp = load_experiment(ed)
            if exp is None:
                raise FileNotFoundError('No experiment.p file found for %s' % ed)

            exp_names.append(exp.data_name)

        # Get experiment groups
        if exp_groups is None:
            exp_groups = userIO.get_labels(exp_names, 'Label Experiment Groups')

        self._exp_info = pd.DataFrame({'exp_name': exp_names,
                                       'exp_group': exp_groups,
                                       'exp_dir': exp_dirs})
        
        self.rec_info= self.get_rec_info()

        # Make list of all major files managed by this object
        self._files = {'params':
                       os.path.join(self.root_dir,
                                    self.data_name+'_analysis_params.json')}

        # Check which files exist
        status = self._file_check()
        if status['params'] and params is None:
            self._params = wt.read_dict_from_json(self._files['params'])
        elif params is not None:
            self._params = params
            wt.write_dict_to_json(params, self._files['params'])
        else:
            # TODO: Load defaults and allow user edit
            pass
        

        self.save()

    def _file_check(self):
        '''Iterates though files and checks for their existence
        '''
        out = dict.fromkeys(self._files.keys(), False)
        for k, v in self._files.items():
            if os.path.isfile(v):
                out[k] = True

        return out

    def _change_root(self, new_root=None, deep=True):
        if new_root is None:
            raise ValueError('Must specify new root directory')
        old_root = self.root_dir
        new_root = super()._change_root(new_root)
        def swap(x):
            return x.replace(old_root, new_root)

        for k in self._files.keys():
            self._files[k] = self._files[k].replace(old_root, new_root)

        self._exp_info['exp_dir'] = self._exp_info['exp_dir'].apply(swap)
        self.rec_info['exp_dir'] = self.rec_info['exp_dir'].apply(swap)
        self.rec_info['rec_dir'] = self.rec_info['rec_dir'].apply(swap)
        if deep:
            self.change_rec_roots()
            self.change_exp_roots()


    def __str__(self):
        out = [super().__str__()]
        out.append('\n-----------\nExperiments\n-----------')
        out.append(self._exp_info.to_string())
        out.append('\n-----\nFiles\n-----')
        out.append('\t' + '\n\t'.join(['%s : %s' % (k,v) for k,v in self._files.items()]))

        return '\n'.join(out)

    @Logger('Adding experiment')
    def add_experiment(self, exp_dir, exp_group=None, shell=False):
        exp = load_experiment(exp_dir)
        if exp is None:
            raise FileNotFoundError('No experiment.p file found in %s' % exp_dir)

        exp_name = exp.data_name

        if exp_name in self._exp_info['exp_name']:
            raise KeyError('%s already in project.' % exp_name)

        if exp_group is None:
            opts = self._exp_info['exp_group'].unique()
            opts_str = '{' + ', '.join(opts) + '}'
            q_str = ('Enter experiment group for %s:\nExisting Groups: %s' %
                     (exp_name, opts_str))
            exp_group = userIO.get_user_input(q_str, shell=shell)

        self._exp_info = self._exp_info.append({'exp_name': exp_name,
                                                'exp_group': exp_group,
                                                'exp_dir': exp_dir},
                                               ignore_index=True)
        print('Experiment %s added to project.\n\tExperiment Group: %s\n\t'
              ' Experiment Directory: %s' % (exp_name, exp_group, exp_dir))
        self.save()

    @Logger('Removing Experiment')
    def remove_experiment(self, exp_name):
        df = self._exp_info
        idx = df.query('exp_name == @exp_name').index
        if len(idx) == 0:
            print('Tried to drop %s. Experiment not found in project' % exp_name)
        else:
            print('Dropping experiment %s from project.\n%s\nRemoved' % (exp_name, df.loc[idx]))

        self._exp_info = df.drop(index=idx)
        self.save()
        
    def make_rec_info_table(self):
        self.rec_info = self.get_rec_info()
        self.save()
        
    def get_trial_info(self):
        dflist = []
        rec_info = self.get_rec_info()
        
        for i, row in rec_info.iterrows():
            rd = row['rec_dir']
            dat = load_dataset(rd)
            dit = dat.dig_in_trials
            dit['taste_trial'] = True
            dit['taste_trial'] = dit.groupby(['channel']).taste_trial.cumsum()
            dit['rec_dir'] = rd
            dflist.append(dit)
        outdf = pd.concat(dflist)
        outdf= outdf.rename(columns = {'trial':'session_trial'})
        
        return(outdf)
        
    def get_rec_info(self): 
        rec_info = pd.DataFrame(columns=['exp_name',
                                         'rec_name',
                                         'exp_group',
                                         'rec_num',
                                         'rec_group',
                                         'exp_dir',
                                         'rec_dir'])
        for i, row in self._exp_info.iterrows():
            exp_name = row['exp_name']
            exp_group = row['exp_group']
            exp_dir = row['exp_dir']
            exp = load_experiment(exp_dir)
            for rec_name, rec_dir in exp.rec_labels.items():
                try: 
                    rec_num = exp.order_dict.get(rec_name)
                    rec_group = str(exp_group)+'_'+str(rec_num)
                    info_row = {'exp_name': exp_name,
                                 'rec_name': rec_name, 
                                 'exp_group': exp_group,
                                 'rec_num': rec_num,
                                 'rec_group': rec_group,
                                 'exp_dir': exp_dir,
                                 'rec_dir': rec_dir}
                    rec_info = rec_info.append(info_row, ignore_index=True)
                except:
                    print('error: session number not in experiment ', rec_name)
                
        return rec_info

    def change_rec_roots(self):
        for i, row in self.rec_info.iterrows():
            dat = load_dataset(row['rec_dir'])
            dat._change_root(row['rec_dir'])
            dat.save()

    def change_exp_roots(self):
        for i, row in self._exp_info.iterrows():
            exp = load_experiment(row['exp_dir'])
            exp._change_root(row['exp_dir'])
            exp.save()

    def make_raster_plots(self):
        rec_info = self.rec_info
        for i, row in rec_info.iterrows():
            dat = load_dataset(row['rec_dir'])
            dat.make_raster_plots()
    
    def make_ensemble_raster_plots(self):
        rec_info = self.rec_info
        for i, row in rec_info.iterrows():
            dat = load_dataset(row['rec_dir'])
            dat.make_ensemble_raster_plots()

    def make_rate_arrays(self, overwrite=True, parallel=False):
        rec_info = self.rec_info
        def run_make_rate_arrays(rec_dir):
            print("Making rate arrays for %s" % rec_dir)
            dat = load_dataset(rec_dir)
            dat.make_rate_arrays(overwrite, parallel=True)
            print("Rate arrays made for %s" % rec_dir)

        rec_dirs = rec_info.rec_dir
        if parallel==False:
            for i in rec_dirs:
                run_make_rate_arrays(i)
        elif parallel==True:
            Parallel(n_jobs=-1)(delayed(run_make_rate_arrays)(i) for i in rec_dirs)
            
    #idk if this would actually work, I will need to figure this one out
    def apply_dat_function(self, function):
        rec_info = self.rec_info
        for i, row in rec_info.iterrows():
            dat = load_dataset(row['rec_dir'])
            dat.function()
            
    def plot_hmms(self):
        '''
        Plots hmms under each recording analysis directory in your project. Requires that you have already run HMMs

        '''
        rec_info = self.get_rec_info()
        rec_dirs = rec_info.rec_dir

        def load_plot_hmm(rec_dir):
            handler = phmm.HmmHandler(rec_dir)
            handler.plot_saved_models()
            
        n_cpu = os.cpu_count()
            
        Parallel(n_jobs=n_cpu-1)(delayed(load_plot_hmm)(i) for i in rec_dirs)

    def export_portable_copy(self, dest_dir=None, extensions=None):
        if extensions is None:
            extensions = ['.hdf5', '.p', '.h5', '.log']

        if dest_dir is None:
            dest_dir = select_directory_via_gui("Select Destination Directory")

        # Add '_copy' to the destination directory name
        dest_dir = os.path.join(dest_dir, os.path.basename(self.root_dir) + "_copy")
        print("Copying to %s" % dest_dir)

        # Ask for user confirmation
        confirmation = input("Do you want to copy the files to the above directory? (y/n): ").strip().lower()
        if confirmation != 'y':
            print("Copying operation aborted.")
            return

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for file in filenames:
                if file.endswith(tuple(extensions)):
                    source_file_path = os.path.join(dirpath, file)

                    # Reconstruct the path in the destination directory
                    relative_path = os.path.relpath(dirpath, self.root_dir)
                    dest_file_path = Path(dest_dir) / relative_path / file

                    # Ensure the directory exists
                    dest_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy the file
                    shutil.copy2(source_file_path, dest_file_path)
                    print("Copied %s to %s" % (source_file_path, dest_file_path))

    def get_rec_dir_list(self):
        rec_info = self.rec_info
        rec_dirs = list(rec_info.rec_dir)
        return rec_dirs
# Example usage

def select_directory_via_gui(title="Select a directory"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title=title)  # Open the directory selection dialog
    return directory