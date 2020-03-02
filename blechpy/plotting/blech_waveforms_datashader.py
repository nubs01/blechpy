# yaml.load is deprecated so hide the warnings
import os
import warnings
import yaml
warnings.simplefilter('ignore',category=yaml.YAMLLoadWarning)

# Import stuff
import datashader as ds
import datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread
import shutil

# A function that accepts a numpy array of waveforms and creates a datashader image from them
def waveforms_datashader(waveforms):
    if waveforms.shape[0]==0:
        return None
    # Make a pandas dataframe with two columns, x and y, holding all the data. The individual waveforms are separated by a row of NaNs

    # First downsample the waveforms 10 times (to remove the effects of 10 times upsampling during de-jittering)
    waveforms = waveforms[:, ::10]
    x_values = np.arange(len(waveforms[0])) + 1
    # Then make a new array of waveforms - the last element of each waveform is a NaN
    new_waveforms = np.zeros((waveforms.shape[0], waveforms.shape[1] + 1))
    new_waveforms[:, -1] = np.nan
    new_waveforms[:, :-1] = waveforms

    # Now make an array of x's - the last element is a NaN
    x = np.zeros(x_values.shape[0] + 1)
    x[-1] = np.nan
    x[:-1] = x_values

    # Now make the dataframe
    df = pd.DataFrame({'x': np.tile(x, new_waveforms.shape[0]), 'y': new_waveforms.flatten()})

    # Produce a datashader canvas
    canvas = ds.Canvas(x_range = (np.min(x_values), np.max(x_values)),
                       y_range = (df['y'].min() - 10, df['y'].max() + 10),
                       plot_height=1200, plot_width=1600)
    # Aggregate the data
    agg = canvas.line(df, 'x', 'y', ds.count())
    # Transfer the aggregated data to image using log transform and export the temporary image file
    img = tf.shade(agg, how='eq_hist')
    img = tf.set_background(img, 'white')

    # Figure sizes chosen so that the resolution is 100 dpi
    fig,ax = plt.subplots(1, 1, figsize = (12,8), dpi = 200)
    # Start plotting
    ax.imshow(img.to_pil())
    # Set ticks/labels - 10 on each axis
    ax.set_xticks(np.linspace(0, 1600, 10))
    ax.set_xticklabels(np.floor(np.linspace(np.min(x_values), np.max(x_values), 10)))
    ax.set_yticks(np.linspace(0, 1200, 10))
    ax.set_yticklabels(np.floor(np.linspace(df['y'].max() + 10, df['y'].min() - 10, 10)))

    # Delete the dataframe
    del df, waveforms, new_waveforms

    # Return and figure and axis for adding axis labels, title and saving the file
    return fig, ax

