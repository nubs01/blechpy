from blechpy.analysis import multiple_recordings as multi, dataset
import numpy as np
from blechpy.widgets import userIO
import os


def make_summary_figures(rec_dirs, anim_groups=None, shell=False):
    animals = [os.path.basename(x) for x in rec_dirs]
    anim_groups = userIO.fill_dict(dict.fromkeys(animals,''),
                                   prompt='Enter animal groups names',
                                   shell=shell)
    if anim_groups is None:
        return

    groups = set(anim_groups.values())

    # Collect: number of taste response neurons pre & post CTA (Saccharin sessions), # of neurons pre & post, # of held neurons between pre & post, # of changed neurons
    # For each animal get # of significantly different held units at each time point



