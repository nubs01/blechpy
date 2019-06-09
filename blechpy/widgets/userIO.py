import tkinter as tk
from tkinter import ttk
from collections.abc import Mapping
from copy import deepcopy

class fill_dict(object):
    def __init__(self,data,types=None,shell=False):
        if not isinstance(data,Mapping):
            raise TypeError(('%s is invalid data type. Requires extension of'
                    'Mapping such as dict.') % type(data))
        if types is None:
            self._types = make_type_dict(data)
        elif len(types.keys()) != len(data.keys()):
            raise ValueError(('types must be a dict with keys matching data'
            ' and values as types or be set to None to automatically detect data types'
            'or default to string input'))
        else:
            self._types = types
            
        self._storage = deepcopy(data)
        self._output = deepcopy(data)
        self._shell = shell

    def fill_dict(self):
        if self._shell:
            self.fill_dict_shell()
        else:
            self.fill_dict_gui()

    def fill_dict_shell(self):
        out = get_dict_shell_input(self._storage)
        self._output = deepcopy(out)

    def get_dict(self):
        return deepcopy(self._output)

    def get_dict_shell_input(dat,type_dict=None,tabstop=''):
        output = deepcopy(dat)
        if type_dict==None:
            type_dict = make_type_dict(dat)

        for k,v in dat.items():
            if isinstance(v,Mapping) and v!={}:
                print('%s%s\n%s----------\n' % (tabstop,k,tabstop))
                tmp = get_dict_shell_input(v,type_dict[k],tabstop+'    ')
                output[k] = deepcopy(tmp)
                continue
            prompt = '%s%s (%s) ' % (tabstop,k,type_dict[k].__name__)
            if type_dict[k] is list:
                prompt += '(comma-separated)'
            if all([v is not x for x in [None,[],{},'']]):
                prompt += '[default = %s]' % v
            prompt += ' : '
            tmp = input(prompt)
            output[k] = convert_str_to_type(tmp,type_dict[k])
        return output

    def make_type_dict(dat):
        type_dict = deepcopy(dat)
        for k,v in dat.items():
            if isinstance(v,Mapping) and v!={}:
                tmp = make_type_dict(v)
            else:
                tmp = type(v)
            if tmp is type(None) or v=={}:
                tmp = str
            type_dict[k] = tmp
        return type_dict

    def convert_str_to_type(item,dtype):
        if dtype is list:
            tmp = item.split(',')
            out = []
            hold = ''
            for x in tmp:
                if x.endswith('\\'):
                    hold = x.replace('\\',',')
                else:
                    out.append(hold+x)
                    hold=''
            return out
        return dtype(item)

    def fill_dict_gui(self):
        output = deepcopy(self._storage)
        type_dict = self._types
        root= tk.Tk()
        dict_pane,var_dict = dict_fill_pane(root,output,type_dict,tablevel=0)


class dict_fill_pane(ttk.Frame):
    def __init__(self,root,data,type_dict,tablevel=0,*args,**kwargs):
        ttk.Frame.__init__(self,root,*args,**kwargs)
        self.data = deepcopy(data)
        self.val_dict = deepcopy(type_dict)
        self.type_dict = deepcopy(type_dict)
        
        for k,v in data.items():
            t = type_dict[k]
            if isinstance(v,Mapping) and v!={}:
                sub_pane,val_dict = [2,3]

