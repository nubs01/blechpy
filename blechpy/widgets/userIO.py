import tkinter as tk
from tkinter import ttk
from collections.abc import Mapping
from copy import deepcopy

def get_dict_shell_input(dat,type_dict=None,tabstop=''):
    output = deepcopy(dat)
    if type_dict==None:
        type_dict = make_type_dict(dat)

    for k,v in dat.items():
        if isinstance(v,Mapping) and v!={}:
            print('%s%s\n%s----------' % (tabstop,k,tabstop))
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
        if tmp=='abort':
            return None
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
    if dtype is bool:
        if item.isnumeric():
            item = int(item)
        else:
            if item.lower()=='false' or item.lower()=='n':
                item = 0
            elif item.lower()=='true' or item.lower()=='y':
                item = 1
            elif item=='':
                return None
            else:
                raise ValueError('Boolean inputs must be true, false, y, n, 1 or 0')
    if item=='' and dtype is not str:
        return None

    return dtype(item)

class dictIO(object):
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


    def fill_dict_gui(self):
        output = deepcopy(self._storage)
        type_dict = self._types
        root= tk.Tk()
        root.style = ttk.Style()
        root.style.theme_use('clam')
        self.root = root
        dict_pane = dict_fill_pane(root,output,type_dict)
        dict_pane.pack(side='top',fill='both',expand=True)
        self.dict_pane = dict_pane
        line = ttk.Frame(root)
        submit = ttk.Button(line,text='Submit',command=self.submit)
        cancel = ttk.Button(line,text='Cancel',command=self.cancel)
        submit.pack(side='left')
        cancel.pack(side='right')
        line.pack(side='bottom',anchor='e')
        root.mainloop()
    
    def submit(self):
        self._output = self.dict_pane.get_values()
        self.root.destroy()
    
    def cancel(self):
        self._output = None
        self.root.destroy()

class dict_fill_pane(ttk.Frame):
    def __init__(self,root,data,type_dict,tabstop='',*args,**kwargs):
        ttk.Frame.__init__(self,root,*args,**kwargs)
        self.data = deepcopy(data)
        self.val_dict = deepcopy(type_dict)
        self.type_dict = deepcopy(type_dict)
        row = 0
        for k,v in data.items():
            t = type_dict[k]
            if isinstance(v,Mapping) and v!={}:
                sub_pane = dict_fill_pane(self,v,t,tabstop+'    ')
                val_dict = sub_pane.get_value_vars()
                label = ttk.Label(self,text='%s%s' % (tabstop,k),foreground='red')
                self.val_dict[k] = sub_pane
                label.grid(row=row,column=0,sticky='w')
                sub_pane.grid(row=row+1,column=0,sticky='nw')
                row+=2
            else:
                line = ttk.Frame(self)
                prompt = '%s%s (%s) ' % (tabstop,k,t.__name__)
                if t is list:
                    prompt += '(comma-separated)'
                if all([v is not x for x in [None,[],{},'']]):
                    if t is bool:
                        default = v
                    else:
                        default = str(v)
                        if t is list:
                            default = default[1:-1]
                else:
                    default = ''
                prompt += ' : '
                label = ttk.Label(line,text=prompt)
                if t is bool:
                    var = tk.BooleanVar(self,value=v)
                    entry = ttk.Checkbutton(line,text='',variable=var)
                else:
                    var = tk.StringVar(self,value=default)
                    entry = ttk.Entry(line,textvariable=var)
                self.val_dict[k] = var
                label.pack(side='left')
                entry.pack(side='right')
                line.grid(row=row,column=0,sticky='ew')
                row+=1

    def get_value_vars(self):
        return self.val_dict

    def get_values(self):
        output = deepcopy(self.data)
        vals = self.val_dict
        types = deepcopy(self.type_dict)
        for k,v in vals.items():
            t = types[k]
            if isinstance(v,ttk.Frame):
                output[k] = v.get_values()
            elif isinstance(v,tk.BooleanVar):
                output[k] = v.get()
            else:
                output[k] = convert_str_to_type(v.get(),t)
        return deepcopy(output)
