from blechpy.analysis.blech_clustering import SpikeCluster
import numpy as np

def threshold_spikes(cluster, bottom=None, top=None):
    '''Removes spikes from cluster above or below the given thresholds

    Parameters
    ----------
    cluster : SpikeCluster
    bottom : float, cut spikes that dip below this
    top : float, cut spikes that go above this

    Returns
    -------
    SpikeCluster
    '''
    waves = cluster['spike_waveforms']
    if top is None:
        top = np.max(waves)

    if bottom is None:
        bottom = np.min(waves)

    idx = np.unique(np.where((waves < bottom) | (waves > top))[0])
    cluster.delete_spikes(idx, msg='Thresholded spikes between %g and %g'
                          % (bottom,top))
