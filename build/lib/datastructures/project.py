import os

class project(object):

    def __init__(self, proj_name, exp_dirs=None, shell=False):
        if 'SSH_CONNECTION' in os.environ:
            shell = True

        if exp_dirs is None:
            exp_dirs = userIO.get_filedirs('Select experiment directories',
                                           multi=True, shell=shell)

       exp_names = [os.path.basename(x) for x in exp_dirs]
       self.project_name = proj_name
       self.exp_groups = userIO.get_labels('Label experiment groups', exp_names)
       self.exp_names = exp_names
       self.exp_dirs = exp_dirs

    def add_experiment(self, exp_dir, shell=False):
        pass
