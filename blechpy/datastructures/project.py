import os
import pandas as pd
from blechpy.datastructures.objects import data_object, load_experiment
from blechpy.utils.decorators import Logger
from blechpy.utils import userIO

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

    def _change_root(self, new_root=None):
        old_root = self.root_dir
        new_root = super()._change_root(new_root)
        def swap(x):
            return x.replace(old_root, new_root)

        for k in self._files.keys():
            self._files[k] = self._files[k].replace(old_root, new_root)

        self._exp_info['exp_dir'] = self._exp_info['exp_dir'].apply(swap)

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
