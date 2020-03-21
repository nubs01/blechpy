import tkinter as tk
from tkinter import ttk
from collections import Mapping
from copy import deepcopy
from blechpy.utils import print_tools as pt
import easygui as eg
import sys
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import prompt as pt_prompt
from prompt_toolkit.completion import PathCompleter


def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def get_dict_shell_input(dat, type_dict=None, tabstop='', prompt=None):
    if prompt:
        print('----------\n%s\n----------' % prompt)

    output = deepcopy(dat)
    if type_dict is None:
        type_dict = make_type_dict(dat)

    for k, v in dat.items():
        if isinstance(v, Mapping) and v != {}:
            print('%s%s\n%s----------' % (tabstop, k, tabstop))
            tmp = get_dict_shell_input(v, type_dict[k], tabstop+'    ')
            output[k] = deepcopy(tmp)
            continue

        prompt = '%s%s (%s) ' % (tabstop, k, type_dict[k].__name__)
        if type_dict[k] is list:
            prompt += '(comma-separated)'

        prompt += ' : '
        if any([isinstance(v, x) for x in [str, list, int, float]]):
            default_val = str(v).replace('[','').replace(']','')
            tmp = pt_prompt(prompt, default=default_val)
        else:
            tmp = pt_prompt(prompt)

        if tmp == 'abort':
            return None

        output[k] = convert_str_to_type(tmp, type_dict[k])

    return output


def make_type_dict(dat):
    type_dict = deepcopy(dat)
    for k, v in dat.items():
        if isinstance(v, Mapping) and v != {}:
            tmp = make_type_dict(v)
        elif isinstance(v, type):
            tmp = v
        elif v is None or v == {}:
            tmp = str
        else:
            tmp = type(v)

        type_dict[k] = tmp

    return type_dict


def convert_str_to_type(item, dtype):
    if dtype is list:
        tmp = item.split(',')
        out = []
        hold = ''
        for x in tmp:
            if x.endswith('\\'):
                hold = x.replace('\\', ',')
            else:
                out.append(hold+x)
                hold = ''
        return out

    if dtype is bool:
        if item.isnumeric():
            item = int(item)
        else:
            if item.lower() == 'false' or item.lower() == 'n':
                item = 0
            elif item.lower() == 'true' or item.lower() == 'y':
                item = 1
            elif item == '':
                return None
            else:
                raise ValueError('Boolean inputs must be true, false, '
                                 'y, n, 1 or 0')

    if item == '' and dtype is not str:
        return None

    return dtype(item)


def fill_dict(data, prompt=None, shell=False):
    if 'SSH_CONNECTION' in os.environ:
        shell = True

    filler = dictIO(data, shell=shell)
    out= filler.fill_dict(prompt=prompt)
    return out


class dictIO(object):
    def __init__(self, data, types=None, shell=False):
        if 'SSH_CONNECTION' in os.environ:
            shell = True

        if not isinstance(data, Mapping):
            raise TypeError(('%s is invalid data type. Requires extension of'
                             'Mapping such as dict.') % type(data))
        if types is None:
            self._types = make_type_dict(data)
        elif len(types.keys()) != len(data.keys()):
            raise ValueError(('types must be a dict with keys matching data'
                              ' and values as types or be set to None to '
                              'automatically detect data types'
                              'or default to string input'))
        else:
            self._types = types

        self._storage = deepcopy(data)
        self._output = deepcopy(data)
        self._shell = shell

    def fill_dict(self, prompt=None):
        if self._shell:
            self.fill_dict_shell(prompt=prompt)
        else:
            self.fill_dict_gui(prompt=prompt)

        return self.get_dict()

    def fill_dict_shell(self, prompt=None):
        original = sys.stdout
        sys.stdout = sys.__stdout__
        out = get_dict_shell_input(self._storage, prompt=prompt)
        self._output = deepcopy(out)
        sys.stdout = original

    def get_dict(self):
        return deepcopy(self._output)

    def fill_dict_gui(self, prompt=None):
        output = deepcopy(self._storage)
        type_dict = self._types
        root = tk.Tk()
        root.style = ttk.Style()
        root.style.theme_use('clam')
        self.root = root

        if prompt:
            prompt_label = ttk.Label(root, text=prompt)
            prompt_label.pack(side='top', fill='x', expand='True')
            ttk.Separator(root, orient='horizontal').pack(side='top', fill='x',
                                                          expand=True, pady=10)

        dict_pane = dict_fill_pane(root, output, type_dict)
        dict_pane.pack(side='top', fill='both', expand=True)
        self.dict_pane = dict_pane
        line = ttk.Frame(root)
        submit = ttk.Button(line, text='Submit', command=self.submit)
        cancel = ttk.Button(line, text='Cancel', command=self.cancel)
        submit.pack(side='left')
        cancel.pack(side='right')
        line.pack(side='bottom', anchor='e', pady=5)
        center(root)
        root.mainloop()

    def submit(self):
        self._output = self.dict_pane.get_values()
        self.root.destroy()

    def cancel(self):
        self._output = None
        self.root.destroy()


class dict_fill_pane(ttk.Frame):
    def __init__(self, root, data, type_dict, tabstop='', *args, **kwargs):
        ttk.Frame.__init__(self, root, *args, **kwargs)
        self.data = deepcopy(data)
        self.val_dict = deepcopy(type_dict)
        self.type_dict = deepcopy(type_dict)
        row = 0
        for k, v in data.items():
            t = type_dict[k]
            if isinstance(v, Mapping) and v != {}:
                sub_pane = dict_fill_pane(self, v, t, tabstop+'    ')
                label = ttk.Label(self, text='%s%s' % (tabstop, k),
                                  foreground='red')
                self.val_dict[k] = sub_pane
                label.grid(row=row, column=0, sticky='w')
                sub_pane.grid(row=row+1, column=0, sticky='nw')
                row += 2
            else:
                line = ttk.Frame(self)
                prompt = '%s%s (%s) ' % (tabstop, k, t.__name__)
                if t is list:
                    prompt += '(comma-separated)'
                if all([v is not x for x in [None, [], {}, '']]):
                    if isinstance(v, type):
                        if t is bool:
                            default = False
                        else:
                            default = ''

                    elif t is bool:
                        default = v
                    else:
                        default = str(v)
                        if t is list:
                            default = default[1:-1]
                else:
                    default = ''

                prompt += ' : '
                label = ttk.Label(line, text=prompt)
                if t is bool:
                    if isinstance(v, type):
                        val = False
                    else:
                        val = v

                    var = tk.BooleanVar(self, value=val)
                    entry = ttk.Checkbutton(line, text='', variable=var)
                else:
                    var = tk.StringVar(self, value=default)
                    entry = ttk.Entry(line, textvariable=var)

                self.val_dict[k] = var
                label.pack(side='left')
                entry.pack(side='right')
                line.grid(row=row, column=0, sticky='ew')
                row += 1

    def get_value_vars(self):
        return self.val_dict

    def get_values(self):
        output = deepcopy(self.data)
        vals = self.val_dict
        types = deepcopy(self.type_dict)
        for k, v in vals.items():
            t = types[k]
            if isinstance(v, ttk.Frame):
                output[k] = v.get_values()
            elif isinstance(v, tk.BooleanVar):
                output[k] = v.get()
            else:
                output[k] = convert_str_to_type(v.get(), t)

        return deepcopy(output)


def ask_user(msg, choices=['No', 'Yes'], shell=False):
    '''Ask the user a question with certain choices

    Parameters
    ----------
    msg : str, message to show user
    choices : list or tuple with choices (optional)
        default is ('No', 'Yes')
    shell : bool (optional)
        True is command line interface for input
        False (default) for GUI

    Returns
    -------
    int : index of users choice
    '''
    if 'SSH_CONNECTION' in os.environ:
        shell = True

    if shell:
        original = sys.stdout
        sys.stdout = sys.__stdout__
        print(msg)
        for i, c in enumerate(choices):
            print('%i) %s' % (i, c))
        idx = input('Enter number of choice >>  ')
        if idx == '' or not idx.isnumeric():
            out = None
        else:
            out = int(idx)

        sys.stdout = original
        return out
    else:
        idx = eg.indexbox(msg, choices=choices)
        return idx


def get_user_input(msg, default=None, shell=False):
    '''Get single input from user.

    Parameters
    ----------
    msg : str, prompt
    default : str (optional)
        value returned if user enters nothing.
        default is None
    shell : bool (optional)
        True for CLI, False (default) for GUI
    '''
    if 'SSH_CONNECTION' in os.environ:
        shell = True

    if shell:
        original = sys.stdout
        sys.stdout = sys.__stdout__
        try:
            prompt = msg + ' '
            #if default is not None:
            #    prompt += '(default=%s) ' % default
            if default is None:
                default = ''

            prompt += ' : '
            out = pt_prompt(prompt, default=default)
            if out == '':
                out = default

        except EOFError:
            out = None

        sys.stdout = original
        return out
    else:
        out = eg.enterbox(msg, default=default)

        return out


def select_from_list(prompt, items, title='', multi_select=False, shell=False):
    '''makes a popup for list selections, can be multichoice or single choice
    default is single selection

    Parameters
    ----------
    prompt : str, prompt for selection dialog
    item : list, list of items to be selected from
    title : str (optional), title of selection dialog
    multi_select : bool (optional)
        whether multiple selection is permitted,
        default False
    shell : bool (optional)
        True for command-line interface
        False (default) for GUI

    Returns
    -------
    str (if multi_select=False): string of selected choice
    list (if multi_select=True): list of strings that were selected
    '''
    if 'SSH_CONNECION' in os.environ:
        shell = True

    if not isinstance(items, list):
        raise TypeError("Passed %s, but expected <class 'list'>" % type(items))

    if shell:
        original = sys.stdout
        sys.stdout = sys.__stdout__
        print(prompt)
        for i, item in enumerate(items):
            print('%i) %s' % (i, item))

        if multi_select:
            idx = input('Enter selection numbers comma-separated:\n>>  ')
            if idx == '':
                out = None
            else:
                idx = [int(x) for x in idx.split(',')]
                out = [items[x] for x in idx]

        else:
            idx = input('Enter selection number:  ')
            if idx == '':
                out = None
            else:
                out = items[int(idx)]

        sys.stdout = original
        return out
    else:
        if multi_select is False:
            choice = eg.choicebox(prompt, title, items)
        else:
            choice = eg.multchoicebox(prompt, title, items, None)

        return choice


def tell_user(msg, shell=False):
    '''Tells users a message, even if top level function is outputing to a
    loggin file

    Parameters
    ----------
    msg : str, message
    shell : bool (optional)
        True for command-line, False (default) for GUI
    '''
    if 'SHH_CONNECTION' in os.environ:
        shell = True

    if shell:
        original = sys.stdout
        sys.stdout = sys.__stdout__
        print(msg)
        sys.stdout = original
        return True
    else:
        eg.msgbox(msg)
        return True


def confirm_parameter_dict(params, prompt, shell=False):
    '''Shows user a dictionary and asks them to confirm that the values are
    correct. If not they have an option to edit the dict.

    Parameters
    ----------
    params: dict
        values in dict can be int, float, str, bool, list, dict or None
    prompt: str
        prompt to show user
    shell : bool (optional)
        True to use command line interface
        False (default) for GUI

    Returns
    -------
    dict
       lists are returned as lists of str so other types m ust be cast manually
       by  user
    '''
    prompt = ('----------\n%s\n----------\n%s\nAre these parameters good?' %
              (prompt, pt.print_dict(params)))
    q = ask_user(prompt, choices=['Yes', 'Edit', 'Cancel'], shell=shell)
    if q == 2:
        return None
    elif q == 0:
        return params
    else:
        new_params = fill_dict(params, 'Enter new values:', shell=shell)
        return new_params


def get_labels(items, prompt='', shell=False):
    '''Gets user input corresponding to items in items

    Parameters
    ----------
    items : list of str
    prompt: str
    shell : bool
        True for command-line interface. False (default) for GUI.
        Automatically sets to True when connected via ssh

    Returns
    -------
    list of str
    '''
    if 'SSH_CONNECTION' in os.environ:
        shell=True

    tmp = dict.fromkeys(items,'')
    tmp = fill_dict(tmp, prompt, shell)
    out = [tmp[x] for x in items]
    return out


def get_filedirs(prompt='', root=None, multi=False, shell=False):
    '''Queries user for directory or multiple directories

    Parameters
    ----------
    prompt : str
    root : str, where to start file chooser
    multi : bool, whether to query for multiple directories
    shell : bool
        True for command-line interface, False (default) for GUI
        forced True if ssh

    Returns
    -------
    str or list of str
    '''
    if 'SSH_CONNECTION' in os.environ:
        shell=True

    path_completer = PathCompleter(only_directories=True)
    history = InMemoryHistory()

    go = True
    out = []
    if shell:
        print(prompt)
        if multi:
            print('Leave blank or ctrl-c to terminate collection')

        print('----------')

    while go:
        if shell:
            if root is None:
                root = ''

            try:
                tmp = pt_prompt(' >> ', completer=path_completer, history=history,
                                auto_suggest=AutoSuggestFromHistory(), default=root)
            except KeyboardInterrupt:
                break

        else:
            tmp = eg.diropenbox(prompt + ' (cancel to stop collection)', default=root)

        if tmp == '' or tmp is None:
            go = False
        elif os.path.isdir(tmp):
            root = os.path.dirname(tmp)
            out.append(tmp)
        else:
            print('Must enter a valid path to a directory')

        if not multi:
            out = out[0]
            break

    if out == [] or out == '':
        out = None

    return out


def get_files(prompt='', root=None, filetypes=None, multi=False, shell=False):
    '''Queries user for file or multiple files

    Parameters
    ----------
    prompt : str
    root : str, where to start file chooser
    multi : bool, whether to query for multiple directories
    filetypes : list of str, list of acceptable file suffixes
    shell : bool
        True for command-line interface, False (default) for GUI
        forced True if ssh

    Returns
    -------
    str or list of str
    '''
    if 'SSH_CONNECTION' in os.environ:
        shell=True

    if filetypes is not None:
        file_filter = lambda x: (any([x.endswith(y) for y in filetypes]) or os.path.isdir(x))
    else:
        file_filter = None

    path_completer = PathCompleter(file_filter=file_filter)
    history = InMemoryHistory()

    go = True
    out = []
    if shell:
        print(prompt)
        if multi:
            print('Leave blank or ctrl-c to terminate collection')

        print('----------')

    while go:
        if shell:
            if root is None:
                root = ''

            try:
                tmp = pt_prompt(' >> ', completer=path_completer, history=history,
                                auto_suggest=AutoSuggestFromHistory(), default=root)
            except KeyboardInterrupt:
                break

        else:
            tmp = eg.fileopenbox(prompt + ' (cancel to stop collection)',
                                 default=root, multiple=multi,
                                 filetypes=filetypes)

        if tmp == '' or tmp is None:
            go = False
        elif isinstance(tmp, list) and all([os.path.isfile(x) for x in tmp]):
            root = os.path.dirname(tmp[0])
            out.extend(tmp)
        elif os.path.isfile(tmp):
            root = os.path.dirname(tmp)
            out.append(tmp)
        else:
            print('Must enter a valid path to file')

        if not multi:
            out = out[0]
            break

    if out == [] or out == '':
        out = None

    return out


class fill_dict_popup(object):
    def __init__(self, data, master=None, prompt=None):
        if master:
            root = self.root = tk.Toplevel(master)
        else:
            root = self.root = tk.Tk()

        root.style = ttk.Style()
        root.style.theme_use('clam')
        self.data = data

        if prompt:
            prompt_label = ttk.Label(root, text=prompt)
            prompt_label.pack(side='top', fill='x', expand='True')
            ttk.Separator(root, orient='horizontal').pack(side='top', fill='x',
                                                         expand=True, pady=10)

        type_dict = make_type_dict(data)
        dict_pane = dict_fill_pane(root, data, type_dict)
        dict_pane.pack(side='top', fill='both', expand=True)
        self.dict_pane = dict_pane
        line = ttk.Frame(root)
        submit = ttk.Button(line, text='Submit', command=self.submit)
        cancel = ttk.Button(line, text='Cancel', command=self.cancel)
        submit.pack(side='left')
        cancel.pack(side='right')
        line.pack(side='bottom', anchor='e', pady=5)
        center(root)

        self.output = deepcopy(data)
        self.cancelled = False

        if master is None:
            root.mainloop()

    def submit(self):
        output = self.dict_pane.get_values()
        for k,v in output.items():
            self.output[k] = v

        self.root.destroy()

    def cancel(self):
        self.cancelled = True
        self.root.destroy()


def new_fill_dict(data, prompt=None, shell=False, gui_root=None):
    pass
