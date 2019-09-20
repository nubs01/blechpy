from blechpy.datastructures.objects import data_object, load_dataset
from blechpy.utils import userIO, print_tools as pt


class experiment(data_object):

    def __init__(self, exp_dir=None, exp_name=None, shell=False):
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

        super().__init__('experiment', exp_dir, exp_name)

        fd = [os.path.join(exp_dir, x) for x in os.listdir(exp_dir)]
        file_dirs = [x for x in fd if (os.path.isdir(x) and
                                       dio.h5io.get_h5_filenames(x) is not None)]
        if file_dirs == []:
            q = uerIO.ask_user('No recording directories with h5 files found '
                               'in experiment directory\nContinue creating'
                               'empty experiment?', shell=shell)
            if q == 0:
                return

        self.recording_dirs = file_dirs
        self._order_dirs(shell)

        dat = load_dataset(file_dirs[0])
        self.electrode_mapping = dat.electrode_mapping.copy()

    def _change_root(self, new_root=None):
        old_root = self.root_dir
        new_root = super()._change_root(new_root)
        self.recording_dirs = [x.replace(old_root, new_root)
                               for x in self.recording_dirs]
        return new_root

    def __str__(self):
        out = super().__str__()
        rd_str = 'Recording Directories :\n    ' + '\n    '.join(self.recording_dirs)
        el_str = ('\nElectrode Mapping\n-----------------'
                  + pt.print_dataframe(self.electrode_mapping))
        return out + rd_str + el_str

    def _order_dirs(self, shell=None):
        '''set order of redcording directories
        '''
        if 'SSH_CONNECTION' in os.environ:
            shell = True

        if self.recording_dirs == []:
            return

        self.recording_dirs = [x[:-1] if x.endswith('/') else x
                               for x in self.recording_dirs]
        top_dirs = {os.path.basename(x): os.path.dirname(x)
                    for x in self.recording_dirs}
        file_dirs = list(top_dirs.keys())
        order_dict = dict.fromkeys(file_dirs, 0)
        tmp = userIO.dictIO(order_dict, shell=shell)
        order_dict = userIO.fill_dict(order_dict,('Set order of recordings (1-%i)\n'
                                                  'Leave blank to delete directory'
                                                  ' from list') % len(file_dirs),
                                      shell)
        if order_dict is None:
            return

        file_dirs = [k for k, v in order_dict.items()
                     if v is not None and v != 0]
        file_dirs = sorted(file_dirs, key=order_dict.get)
        file_dirs = [os.path.join(top_dirs.get(x), x) for x in file_dirs]
        self.recording_dirs = file_dirs

    def add_dir(self, new_dir=None, shell=None):
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
        if shell is None:
            shell = self.shell

        if new_dir is None:
            if shell:
                new_dir = input('Full path to new directory:  ')
            else:
                new_dir = eg.diropenbox('Select new recording directory',
                                        'Add Recording Directory')

        if os.path.isdir(new_dir):
            self.recording_dirs.append(new_dir)
        else:
            raise NotADirectoryError('new directory must be a valid full'
                                     ' path to a directory')

        self._order_dirs(shell=shell)
