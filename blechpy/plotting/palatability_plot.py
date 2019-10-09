# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from scipy.ndimage.filters import gaussian_filter1d
import pylab as plt
from blechpy.utils import userIO
from blechpy import dio


default_plot_params = {'p-value': 0.01, 'num_consecutive_bins': 5,
                       'plot_time_start': -1500, 'plot_time_end': 2500,
                       'smoothing_sigma': 5}

# Ask the user for the hdf5 files that need to be plotted together
def plot_palatability_identity(rec_dirs=None, out_dir=None, params=None, shell=False):
    #TODO: Make shell compatible
    #TODO: Make userIO directory selection
    if rec_dirs is None:
        rec_dirs = []
        while True:
            dir_name = easygui.diropenbox(msg = 'Choose a directory with a '
                                          'hdf5 file, hit cancel to stop '
                                          'choosing')
            if dir_name is None:
                break
            else:
                rec_dirs.append(dir_name)

    if out_dir is None:
        if len(rec_dirs) == 1:
            out_dir = os.path.join(rec_dirs[0], 'palatability_identity_plots')
        else:
            out_dir = easygui.diropenbox('Select a directory for output of i'
                                         'palatability/identity plots')
            if out_dir is None:
                print('Must select output directory for plots....quitting')
                return

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if params is None or params.keys() != default_plot_params.keys():
        params = default_plot_params.copy()
        params = userIO.confirm_parameter_dict(params, ('Palatability/Identity'
                                                        ' Plotting Parameters'
                                                        '\nTimes in ms'),
                                               shell=shell)

    if params is None:
        return

    p_values = [params['p-value'], params['num_consecutive_bins']]
    sigma = params['smoothing_sigma']
    time_limits = [params['plot_time_start'], params['plot_time_end']]

    # Now run through the directories, and pull out the data
    unique_lasers = []
    r_pearson = []
    r_spearman = []
    r_isotonic = []
    p_pearson = []
    p_spearman = []
    p_identity = []
    lda_palatability = []
    lda_identity = []
    taste_cosine_similarity = []
    taste_euclidean_distance = []
    #taste_mahalanobis_distance = []
    pairwise_NB_identity = []
    p_discriminability = []
    pre_stim = []
    win_params = []
    id_pal_regress = []
    taste_responsiveness = []
    bin_times = []
    num_units = 0
    for dir_name in rec_dirs:
        h5_file = dio.h5io.get_h5_filename(dir_name)

        # Open the hdf5 file
        with tables.open_file(h5_file, 'r') as hf5:

            # Pull the data from the /ancillary_analysis node
            unique_lasers.append(hf5.root.ancillary_analysis.laser_combination_d_l[:])
            r_pearson.append(hf5.root.ancillary_analysis.r_pearson[:])
            r_spearman.append(hf5.root.ancillary_analysis.r_spearman[:])
            r_isotonic.append(hf5.root.ancillary_analysis.r_isotonic[:])
            p_pearson.append(hf5.root.ancillary_analysis.p_pearson[:])
            p_spearman.append(hf5.root.ancillary_analysis.p_spearman[:])
            p_identity.append(hf5.root.ancillary_analysis.p_identity[:])
            lda_palatability.append(hf5.root.ancillary_analysis.lda_palatability[:])
            lda_identity.append(hf5.root.ancillary_analysis.lda_identity[:])
            taste_cosine_similarity.append(hf5.root.ancillary_analysis.taste_cosine_similarity[:])
            taste_euclidean_distance.append(hf5.root.ancillary_analysis.taste_euclidean_distance[:])
            pairwise_NB_identity.append(hf5.root.ancillary_analysis.pairwise_NB_identity[:])
            p_discriminability.append(hf5.root.ancillary_analysis.p_discriminability[:])
            id_pal_regress.append(hf5.root.ancillary_analysis.id_pal_regress[:])
            bin_times.append(hf5.root.ancillary_analysis.bin_times[:])
            taste_responsiveness.append(hf5.root.ancillary_analysis.taste_responsiveness[:])
            # Reading single values from the hdf5 file seems hard, needs the read() method to be called
            pre_stim.append(hf5.root.ancillary_analysis.pre_stim.read())
            win_params.append(hf5.root.ancillary_analysis.params[:])
            # Also maintain a counter of the number of units in the analysis
            num_units += hf5.root.ancillary_analysis.palatability.shape[1]


    # Check if the number of laser activation/inactivation windows is same
    # across files, raise an error and quit if it isn't
    if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
        pass
    else:
        print("Number of inactivation/activation windows doesn't seem to be "
              "the same across days. Please check and try again")
        return

    # Now first set the ordering of laser trials straight across data files
    laser_order = []
    for i in range(len(unique_lasers)):
        # The first file defines the order
        if i == 0:
            laser_order.append(np.arange(unique_lasers[i].shape[0]))
        # And everyone else follows
        else:
            this_order = []
            for j in range(unique_lasers[i].shape[0]):
                for k in range(unique_lasers[i].shape[0]):
                    if np.array_equal(unique_lasers[0][j, :], unique_lasers[i][k, :]):
                        this_order.append(k)
            laser_order.append(np.array(this_order))

    # Now join up all the data into big numpy arrays, maintaining the laser order defined in laser_order
    # If there's only one data file, set the final arrays to the only array read in
    if len(laser_order) == 1:
        r_pearson = r_pearson[0]
        r_spearman = r_spearman[0]
        r_isotonic = r_isotonic[0]
        p_pearson = p_pearson[0]
        p_spearman = p_spearman[0]
        p_identity = p_identity[0]
        lda_palatability = lda_palatability[0]
        lda_identity = lda_identity[0]
        taste_cosine_similarity = taste_cosine_similarity[0]
        taste_euclidean_distance = taste_euclidean_distance[0]
        pairwise_NB_identity = pairwise_NB_identity[0]
        p_discriminability = p_discriminability[0]
        id_pal_regress = id_pal_regress[0]
        taste_responsiveness = taste_responsiveness[0]
        bin_times = bin_times[0]
    else:
        r_pearson = np.concatenate(tuple(r_pearson[i][laser_order[i], :, :]
                                         for i in range(len(r_pearson))), axis = 2)
        r_spearman = np.concatenate(tuple(r_spearman[i][laser_order[i], :, :]
                                          for i in range(len(r_spearman))), axis = 2)
        r_isotonic = np.concatenate(tuple(r_isotonic[i][laser_order[i], :, :]
                                          for i in range(len(r_isotonic))), axis = 2)
        p_pearson = np.concatenate(tuple(p_pearson[i][laser_order[i], :, :]
                                         for i in range(len(p_pearson))), axis = 2)
        p_spearman = np.concatenate(tuple(p_spearman[i][laser_order[i], :, :]
                                          for i in range(len(p_spearman))), axis = 2)
        p_identity = np.concatenate(tuple(p_identity[i][laser_order[i], :, :]
                                          for i in range(len(p_identity))), axis = 2)
        taste_responsiveness = np.concatenate(tuple(taste_responsiveness[i][:, :, :]
                                                    for i in range(len(taste_responsiveness))), axis = 1)
        id_pal_regress = np.concatenate(tuple(id_pal_regress[i][laser_order[i], :, :]
                                              for i in range(len(id_pal_regress))), axis = 2)
        lda_palatability = np.stack(tuple(lda_palatability[i][laser_order[i], :]
                                          for i in range(len(lda_palatability))), axis = -1)
        lda_identity = np.stack(tuple(lda_identity[i][laser_order[i], :]
                                      for i in range(len(lda_identity))), axis = -1)
        taste_cosine_similarity = np.stack(tuple(taste_cosine_similarity[i][laser_order[i], :]
                                                 for i in range(len(taste_cosine_similarity))), axis = -1)
        taste_euclidean_distance = np.stack(tuple(taste_euclidean_distance[i][laser_order[i], :]
                                                  for i in range(len(taste_euclidean_distance))), axis = -1)
        pairwise_NB_identity = np.stack(tuple(pairwise_NB_identity[i][laser_order[i], :, :, :]
                                              for i in range(len(pairwise_NB_identity))), axis = -1)
        p_discriminability = np.concatenate(tuple(p_discriminability[i][laser_order[i], :, :]
                                                  for i in range(len(p_discriminability))), axis = 4)
        bin_times = np.stack(tuple(x for x in bin_times))

        # Now average the lda and distance results along the last axis (i.e across sessions)
        lda_palatability = np.mean(lda_palatability, axis = 2)
        lda_identity = np.mean(lda_identity, axis = 2)
        taste_cosine_similarity = np.mean(taste_cosine_similarity, axis = -1)
        taste_euclidean_distance = np.mean(taste_euclidean_distance, axis = -1)
        pairwise_NB_identity = np.mean(pairwise_NB_identity, axis = -1)

    def out_file(fn):
        return os.path.join(out_dir, fn)

    # Get the x array for all the plotting
    # x = np.arange(0, r_pearson.shape[1]*params[0][1], params[0][1]) - pre_stim[0]
    if len(bin_times.shape) == 1:
        x = bin_times
    else:
        x = bin_times[0]

    plot_indices = np.where((x >= time_limits[0]) & (x <= time_limits[1]))[0]

    # Save all these arrays in the output directory
    np.save(out_file('r_pearson.npy'), r_pearson)
    np.save(out_file('r_spearman.npy'), r_spearman)
    np.save(out_file('r_isotonic.npy'), r_isotonic)
    np.save(out_file('p_pearson.npy'), p_pearson)
    np.save(out_file('p_spearman.npy'), p_spearman)
    np.save(out_file('p_identity.npy'), p_identity)
    np.save(out_file('lda_palatability.npy'), lda_palatability)
    np.save(out_file('lda_identity.npy'), lda_identity)
    np.save(out_file('unique_lasers.npy'), unique_lasers)
    np.save(out_file('taste_cosine_similarity.npy'), taste_cosine_similarity)
    np.save(out_file('taste_euclidean_distance.npy'), taste_euclidean_distance)
    np.save(out_file('p_discriminability.npy'), p_discriminability)
    np.save(out_file('taste_responsiveness.npy'), taste_responsiveness)
    np.save(out_file('palatability_bin_times.npy'), bin_times)

    # Plot the r_squared values together first (for the different laser conditions)
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(r_pearson.shape[0]):
        plt.plot(x[plot_indices],
                 np.mean(r_pearson[i, plot_indices, :]**2, axis = 1),
                 linewidth = 3.0,
                 label = ('Dur:%ims, Lag:%ims'
                          % (unique_lasers[0][i, 0], unique_lasers[0][i, 1])))

    plt.title('Pearson $r^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Average Pearson $r^2$')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Pearson correlation-palatability.png'), bbox_inches = 'tight')
    plt.close('all')

    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(r_spearman.shape[0]):
        plt.plot(x[plot_indices], np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Spearman $rho^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Average Spearman $rho^2$')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Spearman correlation-palatability.png'), bbox_inches = 'tight')
    plt.close('all')

    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(r_isotonic.shape[0]):
        plt.plot(x[plot_indices], np.median(r_isotonic[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Isotonic $R^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Median Isotonic $R^2$')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Isotonic correlation-palatability.png'), bbox_inches = 'tight')
    plt.close('all')

    # Plot a Gaussian-smoothed version of the r_squared values as well
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(r_pearson.shape[0]):
        plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), sigma = sigma), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Pearson $r^2$ with palatability ranks, smoothing std:%1.1f' % sigma + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Average Pearson $r^2$')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Pearson correlation-palatability-smoothed.png'), bbox_inches = 'tight')
    plt.close('all')

    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(r_spearman.shape[0]):
        plt.plot(x[plot_indices], gaussian_filter1d(np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), sigma = sigma), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Spearman $rho^2$ with palatability ranks, smoothing std:%1.1f' % sigma + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Average Spearman $rho^2$')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Spearman correlation-palatability-smoothed.png'), bbox_inches = 'tight')
    plt.close('all')

    # Now plot the r_squared values separately for the different laser conditions
    for i in range(r_pearson.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.errorbar(x[plot_indices], np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), yerr = np.std(r_pearson[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_pearson.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Pearson $r^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Average Pearson $r^2$')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Pearson correlation-palatability,laser_condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    for i in range(r_spearman.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.errorbar(x[plot_indices], np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), yerr = np.std(r_spearman[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_spearman.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Spearman $rho^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Average Spearman $rho^2$')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Spearman correlation-palatability,laser_condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    # Now plot the absolute values of the coefficients from the multiple regression of palatability and identity
    # First identity together for the different laser conditions
    # Plot identity first
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(id_pal_regress.shape[0]):
        plt.plot(x[plot_indices], np.mean(np.abs(id_pal_regress[i, plot_indices, :, 0]), axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Identity coeff from multiple regression' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Average Identity coefficient')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Multiple regression-identity.png'), bbox_inches = 'tight')
    plt.close('all')

    # And then palatability
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(id_pal_regress.shape[0]):
        plt.plot(x[plot_indices], np.mean(np.abs(id_pal_regress[i, plot_indices, :, 1]), axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Palatability coeff from multiple regression' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Average Palatability coefficient')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Multiple regression-palatability.png'), bbox_inches = 'tight')
    plt.close('all')

    # Now plot the multiple regression coefficients separately for the different laser conditions
    # Identity first
    for i in range(id_pal_regress.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.errorbar(x[plot_indices], np.mean(np.abs(id_pal_regress[i, plot_indices, :, 0]), axis = 1), yerr = np.std(np.abs(id_pal_regress[i, plot_indices, :, 0]), axis = 1)/np.sqrt(id_pal_regress.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Multiple regression identity, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Average Identity coefficient')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Multiple regression-identity,laser_condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')
    # Then palatability
    for i in range(id_pal_regress.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.errorbar(x[plot_indices], np.mean(id_pal_regress[i, plot_indices, :, 1]**2, axis = 1), yerr = np.std(np.abs(id_pal_regress[i, plot_indices, :, 0]), axis = 1)/np.sqrt(id_pal_regress.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Multiple regression palatability, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Average Palatability coefficient')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Multiple regression-palatability,laser_condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    # Now plot the p values together using the significance criterion specified by the user
    # Make a final p array - this will store 1s if x consecutive time bins have significant p values (these parameters are specified by the user)
    p_pearson_final = np.zeros(p_pearson.shape)
    p_spearman_final = np.zeros(p_spearman.shape)
    p_identity_final = np.zeros(p_identity.shape)
    for i in range(p_pearson_final.shape[0]):
        for j in range(p_pearson_final.shape[1]):
            for k in range(p_pearson_final.shape[2]):
                if (j < p_pearson_final.shape[1] - p_values[1]):
                    if all(p_pearson[i, j:j + p_values[1], k] <= p_values[0]):
                        p_pearson_final[i, j, k] = 1
                    if all(p_spearman[i, j:j + p_values[1], k] <= p_values[0]):
                        p_spearman_final[i, j, k] = 1
                    if all(p_identity[i, j:j + p_values[1], k] <= p_values[0]):
                        p_identity_final[i, j, k] = 1

    # Also put the p_discriminability values together with the same rule as above
    p_discriminability_final = np.zeros(p_discriminability.shape)
    for i in range(p_discriminability.shape[0]):
        for j in range(p_discriminability.shape[1]):
            for k in range(p_discriminability.shape[2]):
                for l in range(p_discriminability.shape[3]):
                    for m in range(p_discriminability.shape[4]):
                        if (j < p_discriminability.shape[1] - p_values[1]):
                            if all(p_discriminability[i, j:j + p_values[1], k, l, m] <= p_values[0]):
                                p_discriminability_final[i, j, k, l, m] = 1.0

    # Plot the p_discriminability values separately for each taste and laser condition
    for i in range(p_discriminability_final.shape[0]):
        for j in range(p_discriminability_final.shape[2]):
            fig = plt.figure(figsize=(12.8,7.2),dpi=100)
            for k in range(p_discriminability.shape[3]):
                plt.plot(x[plot_indices], np.mean(p_discriminability_final[i, plot_indices, j, k, :], axis = -1), linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
            plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]) + ' ' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
            plt.xlabel('Time from stimulus (ms)')
            plt.ylabel('Fraction of significant neurons')
            plt.legend(loc = 'upper left', fontsize = 15)
            plt.tight_layout()
            fig.savefig(out_file('Taste %i discriminability p values-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1])), bbox_inches = 'tight')
            plt.close("all")


    # Now first plot the p values together for the different laser conditions
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(p_pearson_final.shape[0]):
        plt.plot(x[plot_indices], np.mean(p_pearson_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Fraction of significant neurons')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Pearson correlation p values-palatability.png'), bbox_inches = 'tight')
    plt.close('all')

    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(p_spearman_final.shape[0]):
        plt.plot(x[plot_indices], np.mean(p_spearman_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Fraction of significant neurons')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Spearman correlation p values-palatability.png'), bbox_inches = 'tight')
    plt.close('all')

    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(p_identity_final.shape[0]):
        plt.plot(x[plot_indices], np.mean(p_identity_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Fraction of significant neurons')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('ANOVA p values-identity.png'), bbox_inches = 'tight')
    plt.close('all')

    # Now plot them separately for every laser condition
    for i in range(p_pearson_final.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.plot(x[plot_indices], np.mean(p_pearson_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Fraction of significant neurons')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Pearson correlation p values-palatability,laser condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    for i in range(p_spearman_final.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.plot(x[plot_indices], np.mean(p_spearman_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Fraction of significant neurons')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Spearman correlation p values-palatability,laser condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    for i in range(p_identity_final.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.plot(x[plot_indices], np.mean(p_identity_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'threshold:%.02f, consecutive windows:%i' % (p_values[0], p_values[1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Fraction of significant neurons')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('ANOVA p values-identity,laser condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    # Now plot the LDA results for palatability and identity together for the different laser conditions
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(lda_palatability.shape[0]):
        plt.plot(x[plot_indices], lda_palatability[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Units:%i, Window (ms):%i, Step (ms):%i, palatability LDA' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Fraction correct trials')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Palatability_LDA.png'), bbox_inches = 'tight')
    plt.close('all')

    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    for i in range(lda_identity.shape[0]):
        plt.plot(x[plot_indices], lda_identity[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
    plt.title('Units:%i, Window (ms):%i, Step (ms):%i, identity LDA' % (num_units, win_params[0][0], win_params[0][1]))
    plt.xlabel('Time from stimulus (ms)')
    plt.ylabel('Fraction correct trials')
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.tight_layout()
    fig.savefig(out_file('Identity_LDA.png'), bbox_inches = 'tight')
    plt.close('all')

    # Now plot the LDA results separately for each laser condition
    for i in range(lda_palatability.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.plot(x[plot_indices], lda_palatability[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Units:%i, Window (ms):%i, Step (ms):%i, palatability LDA' % (num_units, win_params[0][0], win_params[0][1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Fraction correct trials')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Palatability_LDA,laser_condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    for i in range(lda_identity.shape[0]):
        fig = plt.figure(figsize=(12.8,7.2),dpi=100)
        plt.errorbar(x[plot_indices], lda_identity[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
        plt.title('Units:%i, Window (ms):%i, Step (ms):%i, identity LDA' % (num_units, win_params[0][0], win_params[0][1]))
        plt.xlabel('Time from stimulus (ms)')
        plt.ylabel('Fraction correct trials')
        plt.legend(loc = 'upper left', fontsize = 15)
        plt.tight_layout()
        fig.savefig(out_file('Identity_LDA,laser_condition%i.png' % (i+1)), bbox_inches = 'tight')
        plt.close('all')

    # Plot the taste cosine similarity and distance plots for every laser condition and taste
    # Start with cosine similarity
    for i in range(taste_cosine_similarity.shape[0]):
        for j in range(taste_cosine_similarity.shape[2]):
            fig = plt.figure(figsize=(12.8,7.2),dpi=100)
            for k in range(taste_cosine_similarity.shape[3]):
                plt.plot(x[plot_indices], taste_cosine_similarity[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
            plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
            plt.xlabel('Time from stimulus (ms)')
            plt.ylabel('Average cosine similarity')
            plt.legend(loc = 'upper left', fontsize = 15)
            plt.tight_layout()
            fig.savefig(out_file('Taste %i similarity values-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1])), bbox_inches = 'tight')
            plt.close("all")

    # Now do the distances
    for i in range(taste_euclidean_distance.shape[0]):
        for j in range(taste_euclidean_distance.shape[2]):
            fig = plt.figure(figsize=(12.8,7.2),dpi=100)
            for k in range(taste_euclidean_distance.shape[3]):
                plt.plot(x[plot_indices], taste_euclidean_distance[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
            plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
            plt.xlabel('Time from stimulus (ms)')
            plt.ylabel('Average Euclidean distance')
            plt.legend(loc = 'upper left', fontsize = 15)
            plt.tight_layout()
            fig.savefig(out_file('Taste %i Euclidean distances-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1])), bbox_inches = 'tight')
            plt.close("all")

    for i in range(pairwise_NB_identity.shape[0]):
        for j in range(pairwise_NB_identity.shape[2]):
            fig = plt.figure(figsize=(12.8,7.2),dpi=100)
            for k in range(pairwise_NB_identity.shape[3]):
                plt.plot(x[plot_indices], pairwise_NB_identity[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
            plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, win_params[0][0], win_params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
            plt.xlabel('Time from stimulus (ms)')
            plt.ylabel('Average Pairwise Identity Accuracy (Naive Bayes)')
            plt.legend(loc = 'upper left', fontsize = 15)
            plt.tight_layout()
            fig.savefig(out_file('Taste %i Pairwise Identity NB-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1])), bbox_inches = 'tight')
            plt.close("all")

    # Plot the fraction of taste responsive neurons across time bins - this does not pay attention to laser conditions (to look at CTA data as in Grossman et al., 2008)
    fig = plt.figure(figsize=(12.8,7.2),dpi=100)
    plt.plot(taste_responsiveness[:, 0, 1], np.mean(taste_responsiveness[:, :, 0], axis = 1), linewidth = 3.0)
    plt.title('Units:%i' % (num_units))
    plt.xlabel('Time bin marker (ms)')
    plt.ylabel('Fraction of significant neurons')
    plt.tight_layout()
    fig.savefig(out_file('taste_responsiveness_p_values.png'), bbox_inches = 'tight')
    plt.close('all')
