from .analysis import dataset
from . import dio,plotting, data_print as dp
import easygui as eg
import os, pickle

def load_dataset(file_name=None):
    '''Loads dataset processing metadata object from dataset.p file

    Parameters
    ----------
    file_name : str (optional), absolute path to file, if not given file
                chooser is displayed

    Returns
    -------
    blechpy.analysis.dataset : processing metadata object

    Throws
    ------
    FileNotFoundError : if file_name is not a file
    '''
    if file_name is None:
        file_name = eg.fileopenbox('Choose dataset.p file', \
                                    'Choose file',filetypes=['.p'])
    if not os.path.isfile(file_name):
        raise FileNotFoundError('%s is not a valid filename' % file_name)

    with open(file_name,'rb') as f:
        dat = pickle.load(f)
    return dat

