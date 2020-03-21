import tkinter as tk
from tkinter import ttk
import sys

class ScrollFrame(ttk.Frame):
    def __init__(self,parent,*args,**kwargs):
        ttk.Frame.__init__(self,parent,*args,**kwargs)
        self.parent= parent
        self.initUI()

    def initUI(self):
        # Create canvas with scroll bars (x & y)
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side='top',fill='both',expand=True)
        self.xscrollbar = ttk.Scrollbar(self.canvas_frame,orient='horizontal')
        self.canvas = tk.Canvas(self.canvas_frame,relief='sunken')
        self.canvas.config(scrollregion=(0,0,500,9000))
        self.xscrollbar.config(command=self.canvas.xview)
        self.yscrollbar = ttk.Scrollbar(self.canvas_frame,orient='vertical')
        self.yscrollbar.config(command=self.canvas.yview)
        self.canvas.config(yscrollcommand=self.yscrollbar.set,xscrollcommand=self.xscrollbar.set)
        self.yscrollbar.pack(side='right',fill='y')
        self.xscrollbar.pack(side='bottom',fill='x')
        self.canvas.pack(side='left',fill='both',expand=True)
        
        self.viewport = ttk.Frame(self.canvas)
        self.viewport.parent=self
        self.canvas.create_window((0,0),anchor='nw',window=self.viewport,tags='self.viewport')
        self.bind_children_to_mouse()
        self.viewport.bind("<Configure>",self.onFrameConfigure)

    def onFrameConfigure(self,event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_mousewheel(self,event):
        if sys.platform=='linux':
            if event.num==4:
                delta = -1
            else:
                delta = 1
        else:
            delta = -1*int(event.delta/120)
        self.canvas.yview_scroll(delta,'units')

    def bind_children_to_mouse(self,scroll_func=None):
        if scroll_func is None:
            scroll_func = self._on_mousewheel

        children = self.winfo_children()
        for child in children:
            if child.winfo_children():
                children.extend(child.winfo_children())
            if sys.platform=='linux':
                child.bind('<Button-4>',scroll_func)
                child.bind('<Button-5>',scroll_func)
            else:
                child.bind('<MouseWheel>',scroll_func)

def window_center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


class ListSelectFrame(ttk.Frame):
    def __init__(self, parent, choices, multi_select=False):
        self.root = parent
        self.choices = choices
        ttk.Frame.__init__(self, parent)
        if multi_select:
            self.select_mode = select = 'extended'
        else:
            self.select_mode = select = 'single'

        scroll = ttk.Scrollbar(self, orient='vertical')
        self.listbox = tk.Listbox(self, exportselection=False,
                                  selectmode=select, width=15,
                                  yscrollcommand=scroll.set)
        scroll.config(command=self.listbox.yview)
        self.listbox.pack(side='left', fill='both', expand=True)
        scroll.pack(side='left',fill='y')

        for item in choices:
            self.listbox.insert('end', item)

    def get_selection(self):
        idx = map(int, self.listbox.curselection())
        chosen = [self.choices[i] for i in idx]
        return chosen

class ListSelectPopup(object):
    def __init__(self, choices, master=None, prompt=None, multi_select=False):
        if master is None:
            self.root = root = tk.Tk()
        else:
            self.root = root = tk.Toplevel()

        root.style = ttk.Style()
        root.style.theme_use('clam')
        self.output = []
        self.cancelled = False

        if multi_select:
            txt = '(shift or ctrl to select multiple)'
            if prompt:
                prompt += '\n' + txt
            else:
                prompt = txt

        if prompt:
            prompt_label = ttk.Label(root, text=prompt)
            prompt_label.pack(side='top', fill='x', expand='True')
            ttk.Separator(root, orient='horizontal').pack(side='top', fill='x',
                                                          expand=True, pady=10)

        self.listframe = ListSelectFrame(root, choices, multi_select=multi_select)
        self.listframe.pack(side='top', fill='both', expand=True)
        line = ttk.Frame(root)
        submit = ttk.Button(line, text='Submit', command=self.submit)
        cancel = ttk.Button(line, text='Cancel', command=self.cancel)
        submit.pack(side='left')
        cancel.pack(side='right')
        line.pack(side='bottom', anchor='e', pady=5)
        window_center(root)

        if master is None:
            root.mainloop()

    def submit(self):
        self.output = self.listframe.get_selection()
        self.root.destroy()

    def cancel(self):
        self.cancelled = True
        self.root.destroy()
