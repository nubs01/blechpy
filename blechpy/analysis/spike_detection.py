from blechpy.dio import h5io
import numpy as np

def detect_spikes(h5_file, el, params):
    pass

class Test(object):
    def __init__(self, data_dir, el, params, h5_file=None):
        self.electrode = el
        self.data_dir = data_dir
        if h5_file is None:
            h5_file = h5io.get_h5_filename(data_dir)

        self.h5_file = h5_file
        self.params = params

    def run(self):
        print(f'Running {self.electrode}')
        return self.electrode


def run_clust(sd):
    return sd.run()
