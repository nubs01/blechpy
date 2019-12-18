import tkinter as tk
from tkinter import ttk

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
