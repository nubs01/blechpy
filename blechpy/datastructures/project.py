import os
from blechpy.datastructures.object import data_object

class project(data_object):

    def __init__(self, proj_name=None, proj_dir=None, exp_dirs=None,
                 exp_names=None, exp_groups=None, shell=False):
        if 'SSH_CONNECTION' in os.environ:
            shell = True


        self.super().__init__('project', data_name=proj_name,
                              root_dir=proj_dir, shell=shell)

        if exp_dirs is None:
            exp_dirs = userIO.get_filedirs('Select experiment directories',
                                           multi=True, shell=shell)

        if exp_names is None:
            exp_names = [os.path.basename(x) for x in exp_dirs]

        if exp_groups is None:
            exp_groups = userIO.get_labels('Label experiment groups', exp_names)

        self._exp_info = pd.DataFrame({'exp_name': exp_names,
                                       'exp_group': exp_groups,
                                       'exp_dir': exp_dirs})
        self._files = {'params':
                       os.path.join(self.root_dir,
                                    self.data_name+'_analysis_params.json')}

    def add_experiment(self, exp_dir, shell=False):
        pass
