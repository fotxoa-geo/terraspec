import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd
import numpy as np
from utils.envi import envi_to_array, load_band_names
from utils.create_tree import create_directory
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import geopandas as gp
from glob import glob
from utils.results_utils import r2_calculations, band_depth_group_aggregate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from isofit.core.sunposition import sunpos
from datetime import datetime, timezone
from tetracorder.aggregator import unique_file_fractions
import tetracorder.tetracorder_engine as tetracorder
from scipy.interpolate import interp1d
import spectral.io.envi as envi
from emit_utils.file_checks import envi_header
import logging
from utils.spectra_utils import spectra
import matplotlib.image as mpimg
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#from mpl_toolkits.basemap import Basemap
from p_tqdm import p_map
from matplotlib.ticker import MultipleLocator
from utils.slpit_download import load_pickle, save_pickle
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
from functools import partial
from matplotlib.ticker import MaxNLocator
from matplotlib import colors, ticker
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from itertools import product
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

mineral_groups = {'Calcite': 'Carbonates',
                  'Chlorite': 'Chlorite',
                  'Dolomite': 'Carbonates',
                  'Goethite-Nano': 'Fe Oxides',
                  'Goethite-Fine': 'Fe Oxides',
                  'Goethite-Med' : 'Fe Oxides',
                  'Goethite-Large': 'Fe Oxides',
                  'Gypsum-Fine': 'Carbonates',
                  'Gypsum-Coarse': 'Carbonates',
                  'Hematite-Nano': 'Fe Oxides',
                  'Hematite-Fine': 'Fe Oxides',
                  'Hematite-Med': 'Fe Oxides',
                  'Hematite-Large': 'Fe Oxides',
                  'Illite+Muscovite': 'Clays',
                  'Kaolinite': 'Clays',
                  'Montmorillonite': 'Clays',
                  'Vermiculite': 'Clays',
                  'Quartz+Feldspar': 'Quartz+Feldspar'}


minerals_to_exclude = [0, 1, 13, 15, 20, 21, 22, 25, 28, 29, 37, 38, 40, 41, 49, 56, 57, 60, 82, 83, 94,
                       96, 97, 98, 99, 100, 105, 106, 135, 136, 182, 144, 148, 152, 194, 196, 228, 234,
                       238, 270, 271]


def tetracorder_library(mineral_index, fig_directory, group):

    if int(mineral_index) not in [0, -9999]:
        decoded_expert = tcs.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                              log_file=None, log_level='INFO')

        SPECTRAL_REFERENCE_LIBRARY = {'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
                                      'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi')}

        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
        df_mineral_matrix = df_mineral_matrix.fillna(-9999)
        record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral_index), 'Record'].iloc[0]
        filename = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Filename'].iloc[0]
        ref_library = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Library'].iloc[0]
        name = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Name'].iloc[0]

        # load library
        item = SPECTRAL_REFERENCE_LIBRARY[ref_library]
        library = envi.open(envi_header(item), item)
        library_reflectance = library.spectra.copy()
        library_records = [int(q) for q in library.metadata['record']]

        mineral_reflectance = library_reflectance[library_records.index(record), :]

        normalized_group_name = os.path.normpath(filename.split('.depth.gz')[0])
        expert_file_selection = decoded_expert[normalized_group_name]['features']
        hdr = envi.read_envi_header(envi_header(item))

        wavelengths = np.array([float(q) for q in hdr['wavelength']])

        plt.plot(wavelengths, mineral_reflectance)

        # plot the features
        for _cont_feat, cont_feat in enumerate(expert_file_selection):
            feature = cont_feat['continuum']

            left_inds = np.where(np.logical_and(wavelengths >= feature[0], wavelengths <= feature[1]))[0]
            left_x = wavelengths[int(left_inds.mean())]
            left_y_obs = mineral_reflectance[left_inds].mean()

            right_inds = np.where(np.logical_and(wavelengths >= feature[2], wavelengths <= feature[3]))[0]

            if right_inds.size == 0:
                right_inds = spectra.nearest_index_to_wavelength(wavelengths=wavelengths, target_wavelength=(feature[2] + feature[3]) / 2)  # this takes the mean of right bounds
            else:
                pass

            right_x = wavelengths[int(right_inds.mean())]
            right_y_obs = mineral_reflectance[right_inds].mean()

            feature_inds = np.logical_and(wavelengths >= feature[0], wavelengths <= feature[3])
            continuum_obs = interp1d([left_x, right_x], [left_y_obs, right_y_obs], bounds_error=False,
                                     fill_value='extrapolate')(wavelengths)

            h_x = continuum_obs - mineral_reflectance

            # Fill the areas where g(x) > f(x) and f(x) > g(x)
            abs_integral = np.trapz(h_x[feature_inds], wavelengths[feature_inds])

            plt.plot(wavelengths[feature_inds], continuum_obs[feature_inds], label=f'Area: {abs_integral:.3f}; Wvl: {left_x:.2f},{right_x:.2f}')
            plt.fill_between(wavelengths, continuum_obs, mineral_reflectance, where=feature_inds, color='green', alpha=0.3)

        plt.ylim(0,1)
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(fig_directory, f'{int(mineral_index)}.png'))
        plt.clf()
        plt.close()


def simplify_legend(handles, labels):
    unique_labels = {}
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels[label] = handles[i]

    return unique_labels


def cont_rem(wavelengths, reflectance, feature):
    left_inds = np.where(np.logical_and(wavelengths >= feature[0], wavelengths <= feature[1]))[0]
    left_x = wavelengths[int(left_inds.mean())]
    left_y = reflectance[left_inds].mean()

    right_inds = np.where(np.logical_and(wavelengths >= feature[2], wavelengths <= feature[3]))[0]
    right_x = wavelengths[int(right_inds.mean())]
    right_y = reflectance[right_inds].mean()

    feature_inds = np.logical_and(wavelengths >= feature[0], wavelengths <= feature[3])

    continuum = interp1d([left_x, right_x], [left_y, right_y],
                         bounds_error=False, fill_value='extrapolate')(wavelengths)
    depths = reflectance[feature_inds] / continuum[feature_inds]
    return depths, wavelengths[feature_inds]



def standardize_cont_feature(pure_signal, mixed_singal):
    min_pure = np.min(pure_signal)
    max_pure = np.max(pure_signal)

    min_mixed = np.min(mixed_singal)
    max_mixed = np.max(mixed_singal)

    pure_normalized = (pure_signal - min_pure) / (max_pure - min_pure)
    pure_scaled = pure_normalized * (max_mixed - min_mixed) + min_mixed

    return pure_scaled





class tetracorder_figures:
    def __init__(self, base_directory: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs, legend_text):

        self.base_directory = base_directory
        self.simulation_output_directory = os.path.join(base_directory, 'simulation', 'output')
        self.slpit_output_directory = os.path.join(base_directory, 'slpit', 'output')
        self.output_directory = os.path.join(base_directory, 'tetracorder', 'output')

        self.synthetic_rfls = os.path.join(self.output_directory, 'synthethic_rfls')
        self.libraries_output = os.path.join(self.output_directory, 'libraries')
        self.fig_directory = os.path.join(base_directory, 'tetracorder', 'figures')

        #create_directory(os.path.join(self.fig_directory, 'plot_minerals'))
        #create_directory(os.path.join(self.fig_directory, 'field_continuum'))
        #self.cont_field_figs_directory = os.path.join(self.fig_directory, 'field_continuum')

        self.major_axis_fontsize = major_axis_fontsize
        self.minor_axis_fontsize = minor_axis_fontsize
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.fig_height = fig_height
        self.fig_width = fig_width
        self.linewidth = linewidth
        self.sig_figs = sig_figs
        self.legend_text = legend_text

        self.wvls, self.fwhms = spectra.load_wavelengths(sensor='emit')


    def classification_rates(self):
        group_dict = {'g1': 1, 'g2': 3}
        from sklearn.metrics import multilabel_confusion_matrix



        for group in ['g1', 'g2']:
            soil_fractions = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            soil_fractions = np.round(soil_fractions, 2)

            # # mineral detection from tetracorder on soil only
            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, group_dict[group]].astype(int)

            # this is sim spectra from tetracorder output (mixed reflectance)
            simulated_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                               f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                               f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - vegetation removed (veg derived with emc2)
            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - veg removed with rock fractions
            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_unmix_with_rock_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_unmix_with_rock_tc_a_min'))[:, :21, group_dict[group]].astype(int)


            # aggregated confusion matrix
            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(path_to_tetracorder_minerals=os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                                                                            f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                                                                            f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]
            flat_truth = truth_array.flatten()
            flat_fractions = soil_fractions.flatten()

            for _sim, sim in enumerate([simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id, veg_extracted_rock_mineral_id]):
                # Flatten your arrays for processing
                flat_sim = sim.flatten()

                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                y_pred_bin = mlb.transform(y_pred_labels)
                classes = mlb.classes_

                # 2. Get Unique Fraction Values
                unique_fractions = np.unique(flat_fractions)
                unique_fractions.sort()

                # 3. Setup Plotting Grid
                cols = 2
                rows = (len(classes) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4 + 1), sharex=True)
                axes = axes.flatten()

                colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']  # TP, FP, FN, TN
                labels = ['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)']

                # 4. Loop through Classes
                for i, class_name in enumerate(classes):
                    ax = axes[i]
                    tp_v, fp_v, fn_v, tn_v = [], [], [], []

                    for val in unique_fractions:
                        mask = (flat_fractions == val)
                        yt, yp = y_true_bin[mask, i], y_pred_bin[mask, i]

                        tp_v.append(np.sum((yt == 1) & (yp == 1)))
                        fp_v.append(np.sum((yt == 0) & (yp == 1)))
                        fn_v.append(np.sum((yt == 1) & (yp == 0)))
                        tn_v.append(np.sum((yt == 0) & (yp == 0)))

                    bottom = np.zeros(len(unique_fractions))
                    x_ticks = [f"{v:.2f}" for v in unique_fractions]

                    # Store the bars so we can pull the handles for the legend
                    bars = []
                    for data, color, label in zip([tp_v, fp_v, fn_v, tn_v], colors, labels):
                        b = ax.bar(x_ticks, data, bottom=bottom, color=color, label=label, width=0.8)
                        bars.append(b)
                        bottom += data

                    ax.set_title(f"Class: {class_name}", fontweight='bold')
                    ax.set_ylabel("Count")
                    ax.set_ylim(0, 70000)

                # 5. Legend Fix: Move everything down and put legend in the clear space
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                # subplots_adjust creates a 15% gap at the top of the figure
                fig.subplots_adjust(top=0.85, hspace=0.4)

                # We grab the labels/handles from the very last active axis
                handles, labels = ax.get_legend_handles_labels()

                # Place legend in the middle of that 15% gap
                fig.legend(handles, labels,
                           loc='upper center',
                           bbox_to_anchor=(0.5, 0.95),
                           ncol=4,
                           fontsize=12,
                           frameon=True,
                           facecolor='white',
                           edgecolor='gray')

                plt.xticks(rotation=45)
                plt.savefig(os.path.join(self.fig_directory, f"{_sim}_{group}_identification_rate.png"),
                            bbox_inches='tight')
                plt.clf()
                plt.close()


    def f1_score_matrix_detailed(self):

        group_dict = {
            'g1': 1,
            'g2': 3}
        bd_group_dict = {
            'g1': 0,
            'g2': 2}

        for group in ['g1', 'g2']:
            soil_fractions = envi_to_array(
                os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            soil_fractions = np.round(soil_fractions, 2)

            # # mineral detection from tetracorder on soil only
            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, group_dict[group]].astype(int)

            # this is sim spectra from tetracorder output (mixed reflectance)
            simulated_mineral_id_from_tetracorder = \
            envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                       f'tetracorder_{group}_simulation_spectra_global_lib',
                                       f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - vegetation removed (veg derived with emc2)
            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - veg removed with rock fractions
            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_unmix_with_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_unmix_with_rock_tc_a_min'))[:, :21, group_dict[group]].astype(int)

            # aggregated confusion matrix
            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                                  f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                                  f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]
            flat_truth = truth_array.flatten()
            flat_fractions = soil_fractions.flatten()

            for _sim, sim in enumerate([simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id, veg_extracted_rock_mineral_id]):
                # Flatten your arrays for processing
                flat_sim = sim.flatten()

                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                y_pred_bin = mlb.transform(y_pred_labels)
                classes = mlb.classes_

                # 2. Get Unique Fraction Values
                unique_fractions = np.unique(flat_fractions)
                unique_fractions.sort()

                # 3. Setup Plotting Grid
                cols = 2
                rows = (len(classes) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4 + 1), sharex=True)
                axes = axes.flatten()

                # 4. Loop through Classes
                for i, class_name in enumerate(classes):
                    ax = axes[i]
                    f1_values = []
                    x_ticks = [f"{v:.2f}" for v in unique_fractions]

                    for val in unique_fractions:
                        mask = (flat_fractions == val)
                        # Calculate F1 Score
                        score = f1_score(y_true_bin[mask, i], y_pred_bin[mask, i], zero_division=0)
                        f1_values.append(score)

                    # Plot as a line chart
                    ax.plot(x_ticks, f1_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)

                    ax.set_title(f"Class: {class_name}", fontweight='bold')
                    ax.set_ylabel("F1 Score")
                    ax.set_ylim(-0.05, 1.05)  # Add slight padding below 0
                    ax.grid(True, linestyle='--', alpha=0.7)

                # 5. Cleanup Empty Axes
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                # Adjust layout
                fig.subplots_adjust(top=0.9, hspace=0.4)
                plt.xticks(rotation=45)

                plt.savefig(os.path.join(self.fig_directory, f"{_sim}_{group}_F1_score.png"),
                            bbox_inches='tight')
                plt.clf()
                plt.close()


    def fraction_error(self):


        for group in ['g1', 'g2']:
            truth = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            emc = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', 'emc2',
                                             f'global_lib_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, :, 2]
            rock = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', 'emc2',
                                              f'unmix_with_rock_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, :, 2]

            # Calculate absolute differences (Pixels x Increments)
            diff_emc = np.abs(truth - emc)
            diff_rock = np.abs(truth - rock)

            # Calculate Mean and Std Deviation along the pixel axis (axis=0)
            mae_emc = np.mean(diff_emc, axis=0)
            std_emc = np.std(diff_emc, axis=0)

            mae_rock = np.mean(diff_rock, axis=0)
            std_rock = np.std(diff_rock, axis=0)

            # Use the first row for X-axis values (assuming they are consistent across all rows)
            x = truth[0, :]

            # Plotting
            plt.figure(figsize=(10, 6))

            # Plot EMC2
            plt.plot(x, mae_emc, label='EMC$^2$ MAE', color='blue')
            plt.fill_between(x, mae_emc - std_emc, mae_emc + std_emc, color='blue', alpha=0.2)

            # Plot EMC2 + ROCK
            plt.plot(x, mae_rock, label='EMC$^2$ + ROCK MAE', color='orange')
            plt.fill_between(x, mae_rock - std_rock, mae_rock + std_rock, color='orange', alpha=0.2)

            plt.axhline(y=0.10)

            plt.xlabel("Simulated Soil Fraction")
            plt.ylabel("Mean Absolute Error")
            plt.legend()
            plt.title(f"Group {group} - Error with 1-$\sigma$ Bounds")

            plt.show()
            plt.clf()
            plt.close()

    def mae_by_classification(self):
        group_dict = {
            'g1': 1,
            'g2': 3}
        bd_group_dict = {
            'g1': 0,
            'g2': 2}

        for group in ['g1', 'g2']:
            soil_fractions = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            soil_fractions = np.round(soil_fractions, 2)

            # # mineral detection from tetracorder on soil only
            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, group_dict[group]].astype(int)

            # this is sim spectra from tetracorder output (mixed reflectance)
            simulated_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                           f'tetracorder_{group}_simulation_spectra_global_lib',
                                           f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - vegetation removed (veg derived with emc2)
            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - veg removed with rock fractions
            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_unmix_with_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_unmix_with_rock_tc_a_min'))[:, :21, group_dict[group]].astype(int)

            emc = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', 'emc2',
                                             f'global_lib_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, :, 2]
            rock = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', 'emc2',
                                              f'unmix_with_rock_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, :, 2]

            # aggregated confusion matrix
            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                                               f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                                               f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]

            for _sim, sim in enumerate([simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id, veg_extracted_rock_mineral_id]):
                # Flatten everything for processing
                flat_sim = sim.flatten()
                flat_truth = truth_array.flatten()
                flat_fractions = soil_fractions.flatten()

                if _sim in [0, 1]:
                    flat_pred_fractions = emc.flatten()
                else:
                    flat_pred_fractions = rock.flatten()

                # Mineral Classification Binarization
                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]
                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                y_pred_bin = mlb.transform(y_pred_labels)
                classes = mlb.classes_

                unique_fractions = np.unique(flat_fractions)

                # Setup Plotting Grid
                cols = 2
                rows = (len(classes) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4 + 1), sharex=True)
                axes = axes.flatten()

                colors = {'TP': '#2ca02c', 'FP': '#d62728', 'FN': '#ff7f0e', 'TN': '#1f77b4'}

                for i, class_name in enumerate(classes):
                    ax = axes[i]

                    # Dictionary to store MAE per category per fraction increment
                    mae_data = {'TP': [], 'FP': [], 'FN': [], 'TN': []}

                    for val in unique_fractions:
                        mask = (flat_fractions == val)
                        yt, yp = y_true_bin[mask, i], y_pred_bin[mask, i]

                        # Sub-masks for categories
                        masks = {
                            'TP': (yt == 1) & (yp == 1),
                            'FP': (yt == 0) & (yp == 1),
                            'FN': (yt == 1) & (yp == 0),
                            'TN': (yt == 0) & (yp == 0)
                        }

                        for cat in ['TP', 'FP', 'FN', 'TN']:
                            cat_mask = masks[cat]
                            if np.any(cat_mask):
                                # Calculate MAE for this subset: |Truth - Estimated|
                                subset_error = np.abs(
                                    flat_fractions[mask][cat_mask] - flat_pred_fractions[mask][cat_mask])
                                mae_data[cat].append(np.mean(subset_error))
                            else:
                                mae_data[cat].append(np.nan)  # No data for this bin

                    # Plotting the MAE
                    for cat, color in colors.items():
                        ax.plot(unique_fractions, mae_data[cat], marker='o', color=color, label=cat, alpha=0.7)

                    ax.set_title(f"Class: {class_name}", fontweight='bold')
                    ax.set_ylabel("MAE of Soil Fraction")
                    ax.set_ylim(0, 0.4)
                    ax.grid(True, linestyle='--', alpha=0.5)

                # Clean up unused axes
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                # Legend
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
                fig.subplots_adjust(top=0.85, hspace=0.4)

                plt.savefig(os.path.join(self.fig_directory, f"{_sim}_{group}_MAE_by_class_perf.png"),
                            bbox_inches='tight')
                plt.close()


    def band_depth_no_reclaimer(self):
        group_dict = {
            'g1': 1,
            'g2': 3}
        bd_group_dict = {
            'g1': 0,
            'g2': 2}

        for group in ['g1', 'g2']:
            soil_fractions = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            soil_fractions = np.round(soil_fractions, 2)

            # # mineral detection from tetracorder on soil only
            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, group_dict[group]].astype(int)

            # this is sim spectra from tetracorder output (mixed reflectance)
            simulated_mineral_id_from_tetracorder = \
            envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                       f'tetracorder_{group}_simulation_spectra_global_lib',
                                       f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - vegetation removed (veg derived with emc2)
            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # this is sim spectra - veg removed with rock fractions
            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_unmix_with_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_unmix_with_rock_tc_a_min'))[:, :21, group_dict[group]].astype(int)

            # this is sim spectra from tetracorder output (mixed reflectance)
            soil_mineral_bd_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, bd_group_dict[group]]


            mixed_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                  f'tetracorder_{group}_simulation_spectra_global_lib',
                                                  f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, bd_group_dict[group]]

            # this is sim spectra - vegetation removed (veg derived with emc2)
            veg_extracted_mineral_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, bd_group_dict[group]]

            # this is sim spectra - veg removed with rock fractions
            veg_extracted_rock_mineral_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_unmix_with_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_unmix_with_rock_tc_a_min'))[:, :21, bd_group_dict[group]]

            # aggregated confusion matrix
            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                                               f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                                               f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            # this is the truth for mineral id's!
            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]

            truth_array_bd = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])) * -9999
            truth_array_bd[:] = soil_mineral_bd_from_tetracorder[:, np.newaxis]
            truth_array_bd[:, 0] = mixed_bd[:, 0]

            flat_fractions = soil_fractions.flatten()
            flat_bd = truth_array_bd.flatten()
            unique_fractions = np.unique(flat_fractions)

            for _sim, sim in enumerate([simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id, veg_extracted_rock_mineral_id]):
                # Flatten everything for processing
                flat_sim = sim.flatten()
                flat_truth = truth_array.flatten()

                if _sim == 0:
                    flat_pred_bd = mixed_bd.flatten()
                elif _sim == 1:
                    flat_pred_bd = veg_extracted_mineral_bd.flatten()
                else:
                    flat_pred_bd = veg_extracted_rock_mineral_bd.flatten()

                # Mineral Classification Binarization
                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]
                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                y_pred_bin = mlb.transform(y_pred_labels)
                classes = mlb.classes_

                # Setup Plotting Grid
                cols = 2
                rows = (len(classes) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4 + 1), sharex=True)
                axes = axes.flatten()

                colors = {'TP': '#2ca02c', 'FP': '#d62728', 'FN': '#ff7f0e', 'TN': '#1f77b4'}

                for i, class_name in enumerate(classes):
                    ax = axes[i]

                    # Dictionary to store MAE per category per fraction increment
                    mae_data = {'TP': [], 'FP': [], 'FN': [], 'TN': []}

                    for val in unique_fractions:
                        mask = (flat_fractions == val)
                        yt, yp = y_true_bin[mask, i], y_pred_bin[mask, i]

                        # Sub-masks for categories
                        masks = {
                            'TP': (yt == 1) & (yp == 1),
                            'FP': (yt == 0) & (yp == 1),
                            'FN': (yt == 1) & (yp == 0),
                            'TN': (yt == 0) & (yp == 0)
                        }

                        for cat in ['TP', 'FP', 'FN', 'TN']:
                            cat_mask = masks[cat]
                            if np.any(cat_mask):
                                # Calculate MAE for this subset: |Truth - Estimated|
                                subset_error = np.abs(flat_bd[mask][cat_mask] - flat_pred_bd[mask][cat_mask])
                                mae_data[cat].append(np.mean(subset_error))
                            else:
                                mae_data[cat].append(np.nan)  # No data for this bin

                    # Plotting the MAE
                    for cat, color in colors.items():
                        ax.plot(unique_fractions, mae_data[cat], marker='o', color=color, label=cat, alpha=0.7)

                    ax.set_title(f"Class: {class_name}", fontweight='bold')
                    ax.set_ylabel("MAE of Band Depth")
                    ax.set_ylim(0, 0.4)
                    ax.grid(True, linestyle='--', alpha=0.5)

                # Clean up unused axes
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                # Legend
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
                fig.subplots_adjust(top=0.85, hspace=0.4)

                plt.savefig(os.path.join(self.fig_directory, f"{_sim}_{group}_MAE_bd.png"),
                            bbox_inches='tight')
                plt.clf()
                plt.close()


    def spectral_residual_by_class(self):
        # --- Configuration ---
        group_dict = {'g1': 1, 'g2': 3}
        target_covers = [1.0, 0.75, 0.5, 0.25, 0.0]
        cat_list = ['TP', 'FP', 'FN', 'TN']

        wvls, fwhm = spectra.load_wavelengths(sensor='emit')
        good_wvls = spectra.get_good_bands_mask(wvls, wavelength_pairs=None)
        wvls[~good_wvls] = np.nan

        for group in ['g1', 'g2']:
            # 1. Load Fractions
            soil_fractions = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                        f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            soil_fractions = np.round(soil_fractions, 2)
            flat_fractions = soil_fractions.flatten()
            unique_fractions = np.unique(flat_fractions)

            # Find indices for target covers (100%, 75%, 50%, 25%, 0%)
            plot_indices = [np.abs(unique_fractions - t).argmin() for t in target_covers]

            # 2. Load Mineral IDs (for classification logic)
            # Soil-only Truth IDs
            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, group_dict[group]].astype(int)

            # Simulated Mixed IDs
            simulated_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                               f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                               f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, group_dict[group]].astype(int)

            # Veg Extracted IDs (EMC2)
            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # Veg Extracted IDs (Unmix with Rock)
            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_global_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_global_rock_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # 3. Load Spectra (Measurements - Must be Float)
            truth_array_spectra = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_spectra'))

            rho_hat_veg_ext = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'recon_rho_tetracorder_{group}_simulation_spectra_global_lib'))

            rho_hat_veg_ext_with_rock = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'recon_rho_tetracorder_{group}_simulation_spectra_global_rock'))

            # 4. Construct Truth Arrays
            # Classification Truth
            truth_array_ids = np.full(soil_fractions.shape, -9999, dtype=int)
            truth_array_ids[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array_ids[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]

            # Get Mineral Reclassification map
            mineral_class, _ = spectra.get_mineral_reclassification(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                                 f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                                 f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            # 5. Evaluate each Simulation (Mixed, Veg-Ext, Rock-Ext)
            sim_labels = ['Veg_Extracted', 'Veg_Extracted_with_Rock']
            for _sim, sim_ids in enumerate([veg_extracted_mineral_id, veg_extracted_rock_mineral_id]):

                # Select and Reshape Predicted Spectra
                if _sim == 0:
                    rho_hat = rho_hat_veg_ext
                else:
                    rho_hat = rho_hat_veg_ext_with_rock

                # MultiLabel Binarization for Classification Metrics
                y_true_flat = [mineral_class.get(i, ['other']) for i in truth_array_ids.flatten()]
                y_pred_flat = [mineral_class.get(i, ['other']) for i in sim_ids.flatten()]

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_flat)
                y_pred_bin = mlb.transform(y_pred_flat)
                classes = mlb.classes_

                # 6. Plotting Grid: Rows = Minerals, Cols = TP, FP, FN, TN
                n_classes = len(classes)
                fig, axes = plt.subplots(n_classes, 4, figsize=(30, n_classes * 8), sharex=True)
                if n_classes == 1: axes = axes[np.newaxis, :]

                line_colors = cm.viridis(np.linspace(0, 1, len(target_covers)))

                for i, class_name in enumerate(classes):
                    # Extract boolean vectors for this class across all pixels
                    yt_class = y_true_bin[:, i]
                    yp_class = y_pred_bin[:, i]

                    for j, cat in enumerate(cat_list):
                        ax = axes[i, j]

                        # Create category mask for entire class
                        if cat == 'TP':
                            cat_mask_all = (yt_class == 1) & (yp_class == 1)
                        elif cat == 'FP':
                            cat_mask_all = (yt_class == 0) & (yp_class == 1)
                        elif cat == 'FN':
                            cat_mask_all = (yt_class == 1) & (yp_class == 0)
                        elif cat == 'TN':
                            cat_mask_all = (yt_class == 0) & (yp_class == 0)

                        # Iterate target fractions
                        for idx, target_idx in enumerate(plot_indices):
                            frac_val = unique_fractions[target_idx]
                            frac_mask = (flat_fractions == frac_val)

                            # Intersect Category and Fraction
                            combined_mask = frac_mask & cat_mask_all

                            if np.any(combined_mask):
                                # Reshape to spatial dimensions to slice 3D spectra
                                spatial_mask = combined_mask.reshape(soil_fractions.shape)
                                print(spatial_mask.shape)
                                t_spec = truth_array_spectra[spatial_mask]
                                p_spec = rho_hat[spatial_mask]

                                residuals = (t_spec - p_spec)/t_spec

                                minerals_identified_in_cat = np.unique(truth_array_ids[spatial_mask])
                                minerals_identified_in_cat = minerals_identified_in_cat[minerals_identified_in_cat != 0]

                                cmap = plt.get_cmap('tab20')
                                num_minerals = len(minerals_identified_in_cat)
                                colors = [cmap(i % 20) for i in range(num_minerals)]

                                for _mineral_idx, mineral_idx in enumerate(minerals_identified_in_cat):
                                    expert_file_system_selection, group_name = spectra.get_mineral_information(mineral_idx)
                                    mineral_color = colors[_mineral_idx]

                                    for _cont_feat, cont_feat in enumerate(expert_file_system_selection):
                                        feature= cont_feat['continuum']
                                        ax.axvspan(feature[0] * 1000, feature[-1] *1000, color=mineral_color, alpha=0.3,
                                                   label=group_name if _cont_feat == 0 else "")

                                # 1. Calculate Mean (The line)
                                mean_res = np.mean(residuals, axis=0)

                                # 2. Calculate Standard Deviation (The 1-sigma spread)
                                std_res = np.std(residuals, axis=0)

                                ax.plot(wvls, mean_res, color=line_colors[idx],
                                        label=f"{int(frac_val * 100)}% Soil", alpha=0.8)

                                # Plot the 1-Sigma Shaded Region
                                ax.fill_between(wvls,
                                                mean_res - std_res,  # Lower bound (clipped at 0)
                                                mean_res + std_res,  # Upper bound
                                                color=line_colors[idx],
                                                alpha=0.2)  # Light transparency for overlapping regions

                        # Titles and Formatting
                        if i == 0: ax.set_title(f"CATEGORY: {cat}", fontweight='bold', fontsize=14)
                        if j == 0: ax.set_ylabel(f"{class_name}\nAbs Residual", fontweight='bold')
                        ax.set_ylim(-0.6, 0.6)
                        ax.grid(True, linestyle='--', alpha=0.4)

                # Legend and Final Layout
                unique_map = {}

                # 2. Iterate through every axis in the grid
                for ax in axes.flatten():
                    handles, labels = ax.get_legend_handles_labels()
                    # Update the dictionary with new labels as they are found
                    for lbl, hndl in zip(labels, handles):
                        if lbl not in unique_map:
                            unique_map[lbl] = hndl

                sorted_labels = sorted(unique_map.keys())

                # 4. Retrieve the corresponding handles in sorted order
                sorted_handles = [unique_map[lbl] for lbl in sorted_labels]

                fig.legend(sorted_handles, sorted_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=12)
                fig.text(0.5, 0.05, 'Wavelength(nm)', ha='center', fontsize=12)
                plt.tight_layout(rect=[0, 0.08, 1, 0.98])
                out_name = f"Sim{_sim}_{group}_{sim_labels[_sim]}_Spectral_Full_Confusion.png"
                plt.savefig(os.path.join(self.fig_directory, out_name))
                plt.close()
                print('Done!')

    def veg_correction_fig(self):
        group_dict = {'g1': 0, 'g2': 2}
        bd_group_dict = {
            'g1': 1,
            'g2': 3}

        for group in ['g1', 'g2']:
            fractions = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_fractions'))[:, :, 2]

            # this is simulated spectra corrections
            bds = envi_to_array(os.path.join(self.veg_correction_dir, f'tetracorder_{group}_simulated_spectra'))
            bds[bds == -9999.] = np.nan

            bd_library = bds[:, :, 2]
            bd = bds[:, :, 3]
            bd_prime = bds[:, :, 4]
            bd_prime_prime = bds[:, :, 5]

            # this is for unmixed simulated spectra - Bd '
            bd_emc2 = envi_to_array(os.path.join(self.veg_correction_dir, f'tetracorder_{group}_emc2_spectra'))
            bd_emc2[bd_emc2 == -9999.] = np.nan
            bd_emc2_prime = bd_emc2[:, :, 4]

            # this is for unmixed simulated spectra - Bd''
            bd_emc2_prime_prime = bd_emc2[:, :, 5]

            # load correct mineral IDs @ 100 % soil
            true_mineral_id = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 20, bd_group_dict[group]]
            true_mineral_ids = np.zeros((fractions.shape[0], fractions.shape[1]))
            true_mineral_ids[:] = true_mineral_id[:, np.newaxis]
            true_mineral_ids[:, 0] = 0

            # load simulated spectra mineral IDs
            sim_mineral_detections = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, bd_group_dict[group]]
            bd_from_tetracorder = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21, group_dict[group]]

            true_bd = bd[:, 20] # bd_from_tetracorder[:, 20]
            truth_array = np.zeros((fractions.shape[0], fractions.shape[1]))
            truth_array[:] = true_bd[:, np.newaxis]
            truth_array[:, 0] = 0

            # aggregated ids
            mineral_class, df_minerals = spectra.get_mineral_reclassification(path_to_tetracorder_minerals=os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_minerals'))
            truth_category_array = np.full(fractions.shape, 'Other', dtype=object)

            for value, category in mineral_class.items():
                truth_category_array[true_mineral_ids == value] = category

            simulated_category_array = np.full(fractions.shape, 'Other', dtype=object)

            for value, category in mineral_class.items():
                simulated_category_array[sim_mineral_detections == value] = category

            # flatten arrays
            a_flat = truth_category_array.flatten()
            b_flat = simulated_category_array.flatten()
            fractions_flat = fractions.flatten()
            bd_flat = bd.flatten()
            bd_prime_flat = bd_prime.flatten()
            bd_prime_prime_flat = bd_prime_prime.flatten()
            bd_emc2_prime_flat = bd_emc2_prime.flatten()
            bd_emc2_prime_prime_flat = bd_emc2_prime_prime.flatten()
            bd_from_tetracorder_flat = bd_from_tetracorder.flatten()
            truth_array_flat = truth_array.flatten()

            # Generate confusion matrix
            labels = sorted(set(a_flat) | set(b_flat))  # Ensures all labels appear

            # create figure
            fig, axes = plt.subplots(nrows=1, ncols=len(labels), figsize=(2 * len(labels), 4), constrained_layout=True)

            for ax, label in zip(axes, labels):
                if label in ['Other', 'No Detection', 'Vegetation']:
                    continue

                mask = (a_flat == label) & (b_flat == label)
                fractions_in_mask = fractions_flat[mask]

                bd_in_mask = bd_flat[mask]
                bd_prime_in_mask = bd_prime_flat[mask]
                bd_prime_prime_in_mask = bd_prime_prime_flat[mask]
                bd_emc2_prime_in_mask = bd_emc2_prime_flat[mask]
                bd_emc2_prime_prime_in_mask = bd_emc2_prime_prime_flat[mask]
                bd_from_tetracorder_in_mask = bd_from_tetracorder_flat[mask]
                truth_in_mask = truth_array_flat[mask]

                n = len(fractions_in_mask)

                box_data = []
                unique_fractions = np.sort(np.unique(fractions_in_mask))
                mae_bd = []
                mae_bd_prime = []
                mae_bd_prime_prime = []
                mae_bd_emc2_prime = []
                mae_bd_emc2_prime_prime = []
                mae_bd_from_tetracorder = []

                for fraction_bin in unique_fractions:
                    fraction_bin_mask = fractions_in_mask == fraction_bin
                    error_bd_bin = truth_in_mask[fraction_bin_mask] - bd_in_mask[fraction_bin_mask]
                    error_bd_prime_bin = truth_in_mask[fraction_bin_mask] - bd_prime_in_mask[fraction_bin_mask]
                    error_bd_prime_prime = truth_in_mask[fraction_bin_mask] - bd_prime_prime_in_mask[fraction_bin_mask]
                    error_bd_emc2_prime = truth_in_mask[fraction_bin_mask] - bd_emc2_prime_in_mask[fraction_bin_mask]
                    error_bd_emc2_prime_prime = truth_in_mask[fraction_bin_mask] - bd_emc2_prime_prime_in_mask[fraction_bin_mask]
                    error_bd_from_tetracorder = truth_in_mask[fraction_bin_mask] - bd_from_tetracorder_in_mask[fraction_bin_mask]

                    mae_bd_bin = np.nanmean(np.abs(error_bd_bin))
                    mae_bd_prime_bin = np.nanmean(np.abs(error_bd_prime_bin))
                    mae_bd_prime_prime_bin = np.nanmean(np.abs(error_bd_prime_prime))
                    mae_bd_emc2_prime_bin = np.nanmean(np.abs(error_bd_emc2_prime))
                    mae_bd_emc2_prime_prime_bin = np.nanmean(np.abs(error_bd_emc2_prime_prime))
                    mae_bd_from_tetracorder_bin = np.nanmean(np.abs(error_bd_from_tetracorder))

                    mae_bd.append(mae_bd_bin)
                    mae_bd_prime.append(mae_bd_prime_bin)
                    mae_bd_prime_prime.append(mae_bd_prime_prime_bin)
                    mae_bd_emc2_prime.append(mae_bd_emc2_prime_bin)
                    mae_bd_emc2_prime_prime.append(mae_bd_emc2_prime_prime_bin)
                    mae_bd_from_tetracorder.append(mae_bd_from_tetracorder_bin)

                    # if error_bin.size > 0:
                    #     box_data.append(error_bin)
                    #     #mae.append(mae_bin)
                    # else:
                    #     box_data.append([np.nan])  # Keep alignment even if empty (unlikely)
                    #     #mae.append([np.nan])

                # ax.boxplot(
                #     box_data,
                #     positions=unique_fractions,
                #     widths=0.025,
                #     showfliers=True,
                #     patch_artist=True,
                #     boxprops=dict(facecolor='lightgray', color='black'),
                #     medianprops=dict(color='red'),
                #     whiskerprops=dict(color='black'),
                #     capprops=dict(color='black'),
                #     flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor='black')
                # )

                ax.plot(unique_fractions, mae_bd, color='red', linewidth=1.5, label='Bd')
                ax.plot(unique_fractions, mae_bd_prime, color='green', linewidth=1.5, label="Bd'")
                ax.plot(unique_fractions, mae_bd_prime_prime, color='green', linewidth=3.5, label="Bd''", linestyle='dashed')
                ax.plot(unique_fractions, mae_bd_emc2_prime, color='blue', linewidth=1.5, label="E(MC)$^2$ - Bd'")
                ax.plot(unique_fractions, mae_bd_emc2_prime_prime, color='orange', linewidth=1.5, label="E(MC)$^2$ - BD''")
                ax.plot(unique_fractions, mae_bd_from_tetracorder, color='blue', linewidth=1.5, label="Tetracorder", linestyle='dashed')

                # ✅ Labels and limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 0.2)
                ax.set_xlabel('Soil Fraction')
                ax.set_ylabel('Mean Absolute Error')
                ax.set_title(f'Truth = Pred = {label}\nn = {n}')
                ax.legend()

                # ✅ Format x-ticks with two decimals
                ax.set_xticks(unique_fractions)
                ax.set_xticklabels([f'{x:.2f}' for x in unique_fractions], rotation=90)

           # plt.tight_layout()
            plt.savefig(os.path.join(self.fig_directory, f"{group}_mae_band_depths.png"),
                        bbox_inches='tight')
            plt.clf()
            plt.close()


    def veg_correction_by_mineral(self):
        group_dict = {'g1': 0, 'g2': 2}

        df_mineral_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
        df_mineral_matrix = df_mineral_matrix.fillna(-9999)

        for group in ['g1', 'g2']:
            fractions = envi_to_array(os.path.join(self.sim_spectra_directory,
                                                   f'tetracorder_{group}_simulation_fractions'))

            create_directory(os.path.join(self.fig_directory, f'{group}_veg_correction_mineral'))
            fig_directory_tetracorder = os.path.join(self.fig_directory, f'{group}_veg_correction_mineral')

            unique_x_val = np.unique(fractions[0, :, 2]) * 100

            # this is soil from tetracorder output
            bd_tetra = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, :21, group_dict[group]]
            bd_tetra[bd_tetra == 0] = np.nan

            bd_tetra_minerals = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:,:21, group_dict[group] + 1]

            bad_minerals = np.array(minerals_to_exclude)
            bd_tetra_minerals[np.isin(bd_tetra_minerals, bad_minerals)] = np.nan
            bd_tetra_minerals[np.isnan(bd_tetra)] = np.nan

            unique_minerals = np.unique(bd_tetra_minerals)
            unique_minerals = unique_minerals[~np.isnan(unique_minerals)]

            # this is the simulated spectra w/ corrections
            bd_hat = envi_to_array(os.path.join(self.veg_correction_dir, f'tetracorder_{group}_vegetation_correction'))
            bd_hat[bd_hat == -9999.] = np.nan

            # this is sim spectra from tetracorder output w/out corrections
            bd_tetra_sim = envi_to_array(os.path.join(self.sa_outputs,
                                                      f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, :21,
                           group_dict[group]]
            bd_tetra_sim[bd_tetra_sim == 0] = np.nan

            bd_tetra_sim_minerals = envi_to_array(os.path.join(self.sa_outputs,
                                                               f'tetracorder_{group}_simulation_spectra_augmented_min'))[
                                    :, :21, group_dict[group] + 1]
            bd_tetra_sim_minerals[np.isin(bd_tetra_sim_minerals, bad_minerals)] = np.nan
            bd_tetra_sim_minerals[np.isnan(bd_tetra_sim)] = np.nan

            # this is the sma vegetation reconstructed signal!
            bd_sma = envi_to_array(
                os.path.join(self.veg_correction_dir, f'tetracorder_{group}_vegetation_correction_sma'))
            bd_sma[bd_sma == -9999.] = np.nan

            for mineral in unique_minerals:
                mineral_mask = (bd_tetra_minerals == mineral)

                tetra_soil = bd_tetra.copy()
                tetra_soil[mineral_mask] = np.nan

                tetra_sim = bd_tetra_sim.copy()
                tetra_sim[mineral_mask] = np.nan

                tetra_veg_correction = bd_hat.copy()
                tetra_veg_correction[mineral_mask] = np.nan

                tetra_sma = bd_sma.copy()
                tetra_sma[mineral_mask] = np.nan

                # create lists to store the means
                mean_y_tetra = []
                mean_y_tetra_sim = []
                mean_y_hat = []
                mean_y_sma = []

                # create lists to store standard deviations
                std_y_tetra = []
                std_y_tetra_sim = []
                std_y_hat = []
                std_y_sma = []

                for _col in range(tetra_soil.shape[1]):
                    mean_y_tetra.append(np.nanmean(tetra_soil[:, _col] - tetra_soil[:, _col]))
                    std_y_tetra.append(np.nanstd(tetra_soil[:, _col] - tetra_soil[:, _col]))

                    mean_y_tetra_sim.append(np.nanmean(tetra_soil[:, _col] - tetra_sim[:, _col]))
                    std_y_tetra_sim.append(np.nanstd(tetra_soil[:, _col] - tetra_sim[:, _col]))

                    mean_y_hat.append(np.nanmean(tetra_soil[:, _col] - tetra_veg_correction[:, _col, 6]))
                    std_y_hat.append(np.nanstd(tetra_soil[:, _col] - tetra_veg_correction[:, _col, 6]))

                    mean_y_sma.append(np.nanmean(tetra_soil[:, _col] - tetra_sma[:, _col, 6]))
                    std_y_sma.append(np.nanstd(tetra_soil[:, _col] - tetra_sma[:, _col, 6]))

                record = df_mineral_matrix.loc[df_mineral_matrix['Index'] == int(mineral), 'Record'].iloc[0]
                name = df_mineral_matrix.loc[df_mineral_matrix['Record'] == record, 'Name'].iloc[0]

                # make figure
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.plot(unique_x_val, np.absolute(mean_y_tetra),
                        label='Tetracorder$_{soil}$', linestyle='solid', color='red')

                ax.errorbar(unique_x_val, np.absolute(mean_y_tetra_sim), yerr=std_y_tetra_sim, fmt='o',
                            label='Tetracorder$_{mixed}$', linestyle='solid', color='purple', capsize=8,
                            ecolor='purple')

                ax.errorbar(unique_x_val, np.absolute(mean_y_hat), yerr=std_y_hat, fmt='o',
                            label='Tetracorder$_{vegetation corrected}$', linestyle='solid', color='green', capsize=6,
                            ecolor='green')

                ax.errorbar(unique_x_val, np.absolute(mean_y_sma), yerr=std_y_sma, fmt='o',
                            label='Tetracorder$_{sma vegetation corrected}$', linestyle='solid', color='blue',
                            capsize=4, ecolor='blue')

                ax.set_xlabel('% Soil Cover')
                ax.set_ylabel('Bd - Band Depth')
                ax.legend(loc='upper right')
                ax.set_title(name)

                ax.set_aspect('auto')

                # major ticks every 10 units
                major_ticks = range(0, 101, 10)
                ax.set_xticks(major_ticks)

                # Minor ticks every 5 units
                minor_ticks = range(0, 101, 5)
                ax.set_xticks(minor_ticks, minor=True)

                # set tick labels for x-axis
                ax.set_xticklabels(major_ticks)

                # major ticks every 10 units - y-axis
                major_ticks = np.arange(0, 0.25, 0.05)
                ax.set_yticks(major_ticks)

                # Minor ticks every 5 units
                minor_ticks = np.arange(0, 0.25, 0.01)
                ax.set_yticks(minor_ticks, minor=True)

                ax.set_yticklabels(major_ticks)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.tight_layout()
                plt.savefig(os.path.join(fig_directory_tetracorder, f'{int(mineral)}-bd-veg-corrections.png'))
                plt.clf()
                plt.close()


    def slpit_bd(self):
        def signal_reconstruct(complete_fractions_array, user_em, df_unmix, spectra_start):
            unmix_library_array = df_unmix.iloc[:, spectra_start:].to_numpy()

            df_unmix['level_1'] = df_unmix['level_1'].str.lower()
            min_em_index = np.min(df_unmix[df_unmix['level_1'] == user_em].index)
            max_em_index = np.max(df_unmix[df_unmix['level_1'] == user_em].index)

            unmix_library_array = unmix_library_array[min_em_index:max_em_index + 1, :]
            complete_fractions_array = complete_fractions_array[0, 0, min_em_index:max_em_index + 1]
            spectra_grid = np.zeros((len(self.wvls)))

            if np.sum(complete_fractions_array) == 0:
                return spectra_grid
            else:
                spectra_grid = np.average(unmix_library_array, weights=complete_fractions_array, axis=0)
                return spectra_grid

        def band_depth_row(row, output_directory):
            plot = row['Name']

            emit_filetime = row['EMIT DATE']

            df_parallel_rows = []

            for em_lib_type in ['local', 'global']:
                if em_lib_type in ['local']:
                    df_unmix = pd.read_csv(os.path.join(self.slpit_output_directory, 'spectral_transects', 'endmembers',
                                                        f'{plot.replace("SPEC", "Spectral").replace(" ", "")}-emit.csv'))
                    spectra_start = 10
                else:
                    # construct rho of ems
                    df_unmix = pd.read_csv(os.path.join(self.simulation_output_directory, 'endmember_libraries',
                                                        'convex_hull__n_dims_4_unmix_library.csv'))
                    spectra_start = 7

                # load reflectances
                emit_rfl = envi_to_array(os.path.join(self.aug_directory, f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented'))[0, 0, :]
                slpit_rfl = envi_to_array(os.path.join(self.aug_directory,f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented'))[0, 0, :]
                contact_rfl = envi_to_array(os.path.join(self.aug_directory, f'{plot.replace(" ", "").replace("SPEC", "Spectral")}-emit_ems_augmented'))[0, 0, :]

                # rebuild vegetation signals
                emit_fractions_complete = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                                     f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_{em_lib_type}_complete_fractions'))
                slpit_fractions_complete = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                                      f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented_{em_lib_type}_complete_fractions'))

                emit_pgv = signal_reconstruct(emit_fractions_complete, user_em='pv', df_unmix=df_unmix,
                                              spectra_start=spectra_start)
                slipt_pgv = signal_reconstruct(slpit_fractions_complete, user_em='pv', df_unmix=df_unmix,
                                               spectra_start=spectra_start)
                emit_pnpv = signal_reconstruct(emit_fractions_complete, user_em='npv', df_unmix=df_unmix,
                                               spectra_start=spectra_start)
                slipt_pnpv = signal_reconstruct(slpit_fractions_complete, user_em='npv', df_unmix=df_unmix,
                                                spectra_start=spectra_start)

                # load transect fractions
                emit_fractions = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                            f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_{em_lib_type}_fractional_cover'))[0, 0, :]
                slpit_fractions = envi_to_array(os.path.join(self.output_directory, 'fractions', 'sma',
                                                             f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented_{em_lib_type}_fractional_cover'))[0, 0, :]
                # load the mineral indices
                emit_mineral_indexs = envi_to_array(os.path.join(self.output_directory, 'spectral_abundance',
                                                                 f'{plot.replace(" ", "")}_RFL_{emit_filetime}_pixels_augmented_min'))[0, 0, :]
                slpit_mineral_indexs = envi_to_array(os.path.join(self.output_directory, 'spectral_abundance',
                                                                  f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_transect_augmented_min'))[0, 0, :]
                contact_probe_indices = envi_to_array(os.path.join(self.output_directory, 'spectral_abundance',
                                                                   f'{plot.replace(" ", "").replace("SPEC", "Spectral")}-emit_ems_augmented_min'))[0, 0, :]

                for _group, group in enumerate(['g1', 'g2']):
                    index_dict = {0: 1, 1: 3}
                    slpit_mineral_index = slpit_mineral_indexs[index_dict[_group]]
                    emit_mineral_index = emit_mineral_indexs[index_dict[_group]]
                    contact_probe_index = contact_probe_indices[index_dict[_group]]

                    index_src = ['Contact']
                    for _index, index in enumerate([contact_probe_index]):  # emit_mineral_index, slpit_mineral_index]):
                        # # contact probe
                        contact_probe_bd = spectra.mineral_group_retrival(mineral_index=index,
                                                                          spectra_observed=contact_rfl,
                                                                          npv_fraction=slpit_fractions[0],
                                                                          gv_fraction=slpit_fractions[1],
                                                                          pnpv=slipt_pnpv, pgv=slipt_pgv,
                                                                          soil_fraction=slpit_fractions[2], psoil=None,
                                                                          exclude_minerals=False, plot=True, output_directory=output_directory, plot_info=f'{em_lib_type}-{group}-{plot}-contact_')

                        # # slpit data
                        slpit_bd = spectra.mineral_group_retrival(mineral_index=index, spectra_observed=slpit_rfl,
                                                                  npv_fraction=slpit_fractions[0],
                                                                  gv_fraction=slpit_fractions[1], pnpv=slipt_pnpv,
                                                                  pgv=slipt_pgv, soil_fraction=slpit_fractions[2],
                                                                  psoil=None, exclude_minerals=False, plot=True, output_directory=output_directory, plot_info=f'{em_lib_type}-{group}-{plot}-slpit_')

                        # emit data
                        emit_band_bd = spectra.mineral_group_retrival(mineral_index=index, spectra_observed=emit_rfl,
                                                                      npv_fraction=emit_fractions[0],
                                                                      gv_fraction=emit_fractions[1],
                                                                      pnpv=emit_pnpv, pgv=emit_pgv,
                                                                      soil_fraction=emit_fractions[2], psoil=None,
                                                                      exclude_minerals=False, plot=True, output_directory=output_directory, plot_info=f'{em_lib_type}-{group}-{plot}-emit_')
                        # append results
                        df_row = ([plot, group, em_lib_type, index_src[_index], np.int64(index)]
                                  + list(contact_probe_bd[2:]) + list(emit_band_bd[2:]) + list(slpit_bd[2:])
                                  + list(emit_fractions) + list(slpit_fractions))
                        df_parallel_rows.append(df_row)

            df_parallel = pd.DataFrame(df_parallel_rows)

            return df_parallel

        df = pd.DataFrame(gp.read_file(os.path.join('gis', "Observation.json")))
        df = df.sort_values('Name')
        df['Team'] = df['Name'].str.split('-').str[0].str.strip()
        df = df[df['Team'] != 'THERM']
        df['Plot_num'] = df['Name'].str.split('-').str[1].str.strip().astype(int)
        df = df[df['Plot_num'] <= 60]

        results = p_map(partial(band_depth_row, output_directory=self.cont_field_figs_directory), [row for _, row in df.iterrows()],
                        **{"desc": "\t\t\tcalculating band depths on field data", "ncols": 150})



        # these are labels for the csv
        data_src_label = ['Contact', "EMIT", 'SLPIT']
        bd_labels = ['Lc', 'Bd', "Bd'", "Bd''",]
        combined_labels = [f"{src}_{bd}" for src in data_src_label for bd in bd_labels]
        ems = ['npv', 'pv', 'soil', 'shade']
        frac_src = ["EMIT", "SLPIT"]
        combined_frac_labels = [f"{em}_{src}" for src in frac_src for em in ems]

        df_results = pd.concat(results, axis=0, ignore_index=True)
        df_results.columns = ['plot', 'group', 'em_library', 'index_src', 'index'] + combined_labels + combined_frac_labels
        df_results.to_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'), index=False)

    def slpit_figure(self):
        df_results = pd.read_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'))
        df_results = df_results.replace(-9999, np.nan)
        df_results['Plot_num'] = df_results['plot'].str.split('-').str[1].str.strip().astype(int)
        #df_results = df_results[df_results['Plot_num'] > 2]

        global_min = df_results.soil_SLPIT.min()
        global_max = df_results.soil_SLPIT.max()

        col_map = {
            0: 'Iron Oxides',
            1: 'Clays/Carbonates'}

        for em_lib in ['local', 'global']:
            for data_type in ['SLPIT', 'EMIT']:

                # # create figure
                fig = plt.figure(figsize=(self.fig_width, self.fig_height))
                ncols = 2
                nrows = 2
                gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.25, width_ratios=[1] * ncols,
                                       height_ratios=[1] * nrows)

                # loop through figure columns
                for row in range(nrows):
                    for col in range(ncols):
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_ylim(0, 0.3)
                        ax.set_xlim(0, 0.3)
                        ax.set_aspect('auto')

                        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                        if col == 0:
                            df_select = df_results[(df_results['group'] == 'g1') & (df_results['index_src'] == 'Contact')].copy()
                            if row == 0:
                                ax.set_title(col_map[col], fontsize=self.title_fontsize)

                        if col == 1:
                            df_select = df_results[(df_results['group'] == 'g2') & (df_results['index_src'] == 'Contact')].copy()
                            if row == 0:
                                ax.set_title(col_map[col], fontsize=self.title_fontsize)

                        # filter out low plots of soil fraction
                        df_select_val = df_select[(df_select['em_library'] == em_lib) & (df_select[f'soil_{data_type}'] >= 0.60)].copy()

                        if row == 0:
                            x = df_select_val["Contact_Bd"]
                            y = df_select_val[f"{data_type}_Bd"]
                            ax.set_ylabel(f"{data_type}_Bd", fontsize=self.axis_label_fontsize)
                            ax.set_xlabel("Contact_Bd", fontsize=self.axis_label_fontsize)

                        else:
                            # if soil fractions are more than 0.9; do not use correction

                            x = df_select_val["Contact_Bd"]
                            y = df_select_val[f"{data_type}_Bd'"]
                            ax.set_ylabel(f"{data_type}_Bd'", fontsize=self.axis_label_fontsize)
                            ax.set_xlabel("Contact_Bd", fontsize=self.axis_label_fontsize)

                        # Create a mask to filter out rows where either x or y is NaN
                        mask = ~np.isnan(x) & ~np.isnan(y)

                        # plot fractional cover values
                        m, b = np.polyfit(x[mask], y[mask], 1)
                        one_line = np.linspace(0, 1, 101)

                        # plot 1 to 1 line
                        ax.plot(one_line, one_line, color='black')
                        ax.plot(one_line, m * one_line + b, color='red')

                        # Extract values
                        soil_values = df_select_val['soil_SLPIT'].values
                        plot_names = df_select_val['plot'].values

                        # Normalize to range [0, 1]
                        soil_norm = (soil_values[mask] - global_min) / (global_max - global_min)

                        scatter = ax.scatter(x[mask], y[mask], marker='^', edgecolor='black', label='SLPIT point', zorder=10, s=150,
                                   c=soil_norm, cmap='viridis', vmin=0,
                                             vmax=1)

                        # Plot the values as text labels
                        # for xi, yi, val, plot in zip(x[mask], y[mask], soil_values, plot_names):
                        #     ax.text(xi, yi + 0.01, f'{plot}-{val:.2f}', ha='center', va='center', fontsize=8, color='black')

                        ax.tick_params(axis='both', labelsize=self.legend_text)

                        # Add error metrics
                        rmse = mean_squared_error(x[mask], y[mask])
                        mae = mean_absolute_error(x[mask], y[mask])

                        r2 = r2_calculations(x[mask], y[mask])

                        txtstr = '\n'.join((
                            r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
                            r'R$^2$: %.2f' % (r2[0],),
                            r'n = ' + str(len(x[mask])),
                        ))

                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                        ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                                verticalalignment='top', bbox=props)
                        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
                        cbar.set_label('SLPIT Soil Fraction (%)', fontsize=self.axis_label_fontsize)
                        cbar.ax.tick_params(labelsize=self.legend_text)

                plt.savefig(os.path.join(self.fig_directory, f'field_data_regressions_{em_lib}_{data_type}.png'), format="png", dpi=400,
                        bbox_inches="tight")
                plt.clf()
                plt.close()


    def slpit_figure_combined(self):
        df_results = pd.read_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'))
        df_results = df_results.replace(-9999, np.nan)
        df_results['Plot_num'] = df_results['plot'].str.split('-').str[1].str.strip().astype(int)
        df_results = df_results[df_results['Plot_num'] > 2]
        # df_results = df_results[df_results['Plot_num'] != 37]
        # df_results = df_results[df_results['Plot_num'] != 35]
        # df_results = df_results[df_results['Plot_num'] != 34]
        # df_results = df_results[df_results['Plot_num'] != 29]
        # df_results = df_results[df_results['Plot_num'] != 28]

        global_min = df_results.soil_SLPIT.min()
        global_max = df_results.soil_SLPIT.max()

        col_map = {
            0: 'Combined',
            1: 'Clays/Carbonates'}

        for em_lib in ['local', 'global']:
            for data_type in ['SLPIT', 'EMIT']:

                # # create figure
                fig = plt.figure(figsize=(self.fig_width, self.fig_height))
                ncols = 1
                nrows = 2
                gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.25, width_ratios=[1] * ncols,
                                       height_ratios=[1] * nrows)

                # loop through figure columns
                for row in range(nrows):
                    for col in range(ncols):
                        ax = fig.add_subplot(gs[row, col])
                        ax.set_ylim(0, 0.3)
                        ax.set_xlim(0, 0.3)
                        ax.set_aspect('auto')

                        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                        df_select = df_results[(df_results['index_src'] == 'Contact')].copy()
                        #if col == 0:

                            #if row == 0:
                            #   ax.set_title(col_map[col], fontsize=self.title_fontsize)

                        if col == 1:
                            if row == 0:
                                ax.set_title(col_map[col], fontsize=self.title_fontsize)

                        # filter out low plots of soil fraction
                        df_select_val = df_select[(df_select['em_library'] == em_lib) & (df_select[f'soil_{data_type}'] >= 0.60)].copy()

                        if row == 0:
                            x = df_select_val["Contact_Bd"]
                            y = df_select_val[f"{data_type}_Bd"]
                            ax.set_ylabel(f"{data_type} Bd$_w$", fontsize=self.axis_label_fontsize)
                            #ax.set_xlabel("Contact Probe Bd$_w$", fontsize=self.axis_label_fontsize)

                        else:
                            x = df_select_val["Contact_Bd"]
                            y = df_select_val[f"{data_type}_Bd'"]
                            ax.set_ylabel(f"{data_type} Bd$_w$'", fontsize=self.axis_label_fontsize)
                            ax.set_xlabel("Contact Probe Bd$_w$", fontsize=self.axis_label_fontsize)

                        # if col != 0:
                        #     ax.set_yticklabels([])
                        #
                        # if row == 0:
                        #     ax.set_xticklabels([])

                        # Create a mask to filter out rows where either x or y is NaN
                        mask = ~np.isnan(x) & ~np.isnan(y)

                        # plot fractional cover values
                        m, b = np.polyfit(x[mask], y[mask], 1)
                        one_line = np.linspace(0, 1, 101)

                        # plot 1 to 1 line
                        ax.plot(one_line, one_line, color='black')
                        ax.plot(one_line, m * one_line + b, color='red')

                        # Extract values
                        soil_values = df_select_val['soil_SLPIT'].values
                        plot_names = df_select_val['plot'].values

                        # Normalize to range [0, 1]
                        soil_norm = (soil_values[mask] - global_min) / (global_max - global_min)

                        groups = df_select_val['group'].values[mask]
                        g1_mask = groups == 'g1'
                        g2_mask = groups == 'g2'

                        scatter = ax.scatter(x[mask], y[mask], marker='^', edgecolor='black', zorder=10, s=150,
                                   c=soil_norm, cmap='viridis', vmin=0,
                                             vmax=1)

                        ax.scatter(x[mask][g1_mask], y[mask][g1_mask], marker='^', edgecolor='black', zorder=10,
                                   s=150, c=soil_norm[g1_mask], cmap='viridis', vmin=0, vmax=1)

                        ax.scatter(x[mask][g2_mask], y[mask][g2_mask], marker='o', edgecolor='black', zorder=10,
                                   s=150, c=soil_norm[g2_mask], cmap='viridis', vmin=0, vmax=1)


                        # Plot the values as text labels
                        # for xi, yi, val, plot in zip(x[mask], y[mask], soil_values[mask], plot_names[mask]):
                        #     ax.text(xi, yi + 0.01, f'{plot}-{val:.2f}', ha='center', va='center', fontsize=8, color='black')

                        ax.tick_params(axis='both', labelsize=self.legend_text)

                        # Add error metrics
                        rmse = np.sqrt(np.mean((x[mask] - y[mask]) ** 2))
                        mae = mean_absolute_error(x[mask], y[mask])
                        r2 = r2_calculations(x[mask], y[mask])


                        txtstr = '\n'.join((
                            r'MAE(RMSE): %.3f(%.3f)' % (mae, rmse),
                            r'R$^2$: %.2f' % (r2[0],),
                            r'n = ' + str(len(x[mask])),
                        ))

                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                        ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=self.legend_text,
                                verticalalignment='top', bbox=props)
                        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
                        cbar.set_label('SLPIT Soil Fraction (%)', fontsize=self.axis_label_fontsize)
                        cbar.ax.tick_params(labelsize=self.legend_text)

                        from matplotlib.lines import Line2D

                        legend_element = Line2D(
                            [0], [0],
                            marker='o',
                            color='w',  # no line
                            label='Group 2 Minerals',
                            markerfacecolor='black',
                            markersize=6
                        )

                        legend_element_1 = Line2D(
                            [0], [0],
                            marker='^',
                            color='w',  # no line
                            label='Group 1 Minerals',
                            markerfacecolor='black',
                            markersize=6
                        )

                        ax.legend(handles=[legend_element_1,legend_element], loc='lower right')

                plt.savefig(os.path.join(self.fig_directory, f'field_data_regressions_{em_lib}_{data_type}_combined.png'), format="png", dpi=400,
                        bbox_inches="tight")
                plt.clf()
                plt.close()

    def fraction_soil_vs_bd(self):
        bd_band = {'g1': 0, 'g2': 2}
        minerals = {'g1': 1, 'g2': 3}

        create_directory(os.path.join(self.fig_directory, 'fraction_vs_bd'))

        for group in ['g1', 'g2']:
            create_directory(os.path.join(self.fig_directory, 'fraction_vs_bd', group))

            soil_fractions = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_fractions'))[:,:, 2]
            soil_index = envi_to_array(os.path.join(self.sim_spectra_directory, f'tetracorder_{group}_simulation_index'))[:,:, 2]
            sim_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_spectra_augmented_min'))[:, 0:21, :]
            sim_soils_sa_arrary = envi_to_array(os.path.join(self.sa_outputs, f'tetracorder_{group}_simulation_soils_augmented_min'))[:, 0:21, :]

            minerals_detected = np.unique(sim_soils_sa_arrary[:, :, minerals[group]])

            for mineral in minerals_detected:
                if int(mineral) in [0]:
                    pass
                else:
                    plt.figure(figsize=(12, 12))

                    mask = ((sim_soils_sa_arrary[:, :, minerals[group]] == mineral) & (sim_sa_arrary[:, :, minerals[group]] == mineral))

                    soil_used = np.unique(soil_index[mask])

                    for soil in soil_used:
                        group_sa = sim_sa_arrary[:, :, bd_band[group]]
                        group_soil = sim_soils_sa_arrary[:, :, bd_band[group]]

                        mask = ((sim_soils_sa_arrary[:, :, minerals[group]] == mineral) & (soil_index == soil))

                        plt.scatter(soil_fractions[mask], group_sa[mask], s=5, alpha=0.6, label="Simulated Spectra")
                        plt.scatter(soil_fractions[mask] + 0.02, group_soil[mask], s=5, alpha=0.6, label="Soil")
                        plt.xlabel('Fractions')
                        plt.ylabel('Band Depth')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(os.path.join(self.fig_directory, 'fraction_vs_bd', group, f'{int(mineral)}_{int(soil)}_fraction_vs_band_depth_{group}.png'), dpi=300)
                        plt.clf()
                        plt.close()


    def fraction_threshold(self):
        df_results = pd.read_csv(os.path.join(self.fig_directory, 'slpit_band_depths.csv'))
        df_results = df_results.replace(-9999, np.nan).dropna()

        # # create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 2
        nrows = 1
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.25, hspace=0.25, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        col_map = {
            0: 'Iron Oxides',
            1: 'Clays/Carbonates'}

        # loop through figure columns
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_aspect('auto')

                ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(self.sig_figs)}f'))

                if col == 0:
                    df_select = df_results[
                        (df_results['group'] == 'g1') & (df_results['index_src'] == 'Contact')].copy()

                    ax.set_title(col_map[col], fontsize=self.title_fontsize)

                if col == 1:
                    df_select = df_results[
                        (df_results['group'] == 'g2') & (df_results['index_src'] == 'Contact')].copy()

                    ax.set_title(col_map[col], fontsize=self.title_fontsize)

                # filter out low plots of soil fraction
                for em_lib in ['global', 'local']:
                    values = np.arange(0, 1.00, 0.05)
                    mae_slpit_bd, mae_slpit_bd_prime, mae_emit_bd, mae_emit_bd_prime = [], [], [], []
                    r2_slpit_bd, r2_slpit_bd_prime, r2_emit_bd, r2_emit_bd_prime = [], [], [], []

                    for val in values:
                        df_select_val = df_select[(df_select['em_library'] == em_lib) & (df_select['soil_SLPIT'] >= val) & (df_select['soil_EMIT'] >= val)].copy()

                        if df_select_val.empty:
                            mae_slpit_bd.append(np.nan)
                            r2_slpit_bd.append(np.nan)

                            mae_slpit_bd_prime.append(np.nan)
                            r2_slpit_bd_prime.append(np.nan)

                            mae_emit_bd.append(np.nan)
                            r2_emit_bd.append(np.nan)

                            mae_emit_bd_prime.append(np.nan)
                            r2_emit_bd_prime.append(np.nan)
                            continue

                        x = df_select_val["Contact_Bd"]

                        mae_slpit_bd.append(mean_absolute_error(x, df_select_val["SLPIT_Bd"]))
                        r2_slpit_bd.append(r2_calculations(x, df_select_val["SLPIT_Bd"])[0])

                        mae_slpit_bd_prime.append(mean_absolute_error(x, df_select_val["SLPIT_Bd'"]))
                        r2_slpit_bd_prime.append(r2_calculations(x, df_select_val["SLPIT_Bd'"])[0])

                        mae_emit_bd.append(mean_absolute_error(x, df_select_val["EMIT_Bd"]))
                        r2_emit_bd.append(r2_calculations(x, df_select_val["EMIT_Bd"])[0])

                        mae_emit_bd_prime.append(mean_absolute_error(x, df_select_val["EMIT_Bd'"]))
                        r2_emit_bd_prime.append(r2_calculations(x, df_select_val["EMIT_Bd'"])[0])

                    # plot error
                    # ax.plot(values, mae_slpit_bd, label="SLPIT_Bd", color='red')
                    # ax.plot(values, mae_slpit_bd_prime, label="SLPIT_Bd'", color='green')
                    # ax.plot(values, mae_emit_bd, label="EMIT_Bd", color='blue')
                    # ax.plot(values, mae_emit_bd_prime, label="EMIT_Bd'", color='orange')

                    # plot r2
                    if em_lib == 'local':
                        linestyle = 'solid'
                        lw = 2
                    else:
                        linestyle = '--'
                        lw=1

                    ax.plot(values, mae_slpit_bd, label=f"SLPIT_Bd - {em_lib}", color='red', linestyle=linestyle,
                            linewidth=lw)
                    ax.plot(values, mae_slpit_bd_prime, label=f"SLPIT_Bd' - {em_lib}", color='green',
                            linestyle=linestyle, linewidth=lw)
                    ax.plot(values, mae_emit_bd, label=f"EMIT_Bd - {em_lib}", color='blue', linestyle=linestyle,
                            linewidth=lw)
                    ax.plot(values, mae_emit_bd_prime, label=f"EMIT_Bd' - {em_lib}", color='orange',
                            linestyle=linestyle, linewidth=lw)

                # Combine and deduplicate legend labels
                handles_1, labels_1 = ax.get_legend_handles_labels()

                # Use dict to remove duplicates
                unique = dict(zip(labels_1, handles_1))

                if col == 1:
                   ax.legend(unique.values(), unique.keys(),loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=False)

                ax.set_ylabel("MAE")
                #ax.set_ylabel("R²")
                ax.set_xlabel('Fraction of soil')

        plt.savefig(os.path.join(self.fig_directory, f'field_thresholds.png'), format="png", dpi=400, bbox_inches="tight")
        plt.clf()
        plt.close()

    def mineral_test(self):
        df=self.get_mineral_reclassification(group='g1')

    def bd_prime_map(self):
        bd_map = {0: 3, 1:4}

        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ncols = 3
        nrows = 2
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.20, hspace=0.20, width_ratios=[1] * ncols,
                               height_ratios=[1] * nrows)

        # loop through figure columns
        for row, i in zip(range(nrows), ['g1', 'g2']):
            veg_correction = envi_to_array(
                os.path.join(self.veg_correction_dir, f'EMIT_L2A_RFL_001_20230831T152735_veg_correction_{i}'))

            bd_mask = veg_correction[:, :, 3] != -9999.
            bd_prime_mask = veg_correction[:, :, 4] != -9999.
            combined_mask = bd_mask & bd_prime_mask
            veg_correction[veg_correction == -9999] = np.nan

            global_min = min(np.nanmin(veg_correction[:, :, 3]), np.nanmin(veg_correction[:, :, 4]))
            global_max = max(np.nanmax(veg_correction[:, :, 3]), np.nanmax(veg_correction[:, :, 4]))

            for col in range(ncols):
                if col == 2:
                    import mpl_scatter_density
                    from matplotlib.colors import LinearSegmentedColormap

                    # "Viridis-like" colormap with white background
                    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
                        (0, '#ffffff'),
                        (1e-20, '#440053'),
                        (0.2, '#404388'),
                        (0.4, '#2a788e'),
                        (0.6, '#21a784'),
                        (0.8, '#78d151'),
                        (1, '#fde624'),
                    ], N=256)

                    ax = fig.add_subplot(gs[row, col], projection='scatter_density')

                    x = veg_correction[:, :, 3].flatten()
                    y = veg_correction[:, :, 4].flatten()
                    flat_mask = combined_mask.flatten()

                    x_valid = x[flat_mask]
                    y_valid = y[flat_mask]
                    density = ax.scatter_density(x_valid, y_valid, cmap=white_viridis)
                    ax.set_ylabel("Bd$_w$'")
                    ax.set_xlabel("Bd$_w$")
                    ax.set_ylim(0, 0.4)
                    ax.set_xlim(0, 0.4)
                    #fig.colorbar(density, label='Number of points per pixel')


                else:
                    ax = fig.add_subplot(gs[row, col])
                    array = veg_correction[:, :, bd_map[col]].copy()
                    array[~combined_mask] = np.nan
                    array_norm = (array - global_min) / (global_max - global_min)
                    cmap = plt.get_cmap('viridis', 10)
                    im = ax.imshow(array_norm, cmap=cmap, vmin=global_min, vmax=global_max)

                    if col == 0:
                        ax.set_title(f'{i.capitalize()} - Bd: 20230831T152735')
                    else:
                        ax.set_title(f"{i.capitalize()} - Bd': 20230831T152735")

                    cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both')
                    cbar.set_label('Band Depth')

                    if col == 0:
                        ax.set_ylabel("Latitude")

                    if row == 1:
                        ax.set_xlabel('Longitude')

        plt.savefig(os.path.join(self.fig_directory, f'bd_map.png'), format="png", dpi=400,
                    bbox_inches="tight")
        plt.clf()
        plt.close()

def run_figure_workflow(base_directory):
    ems = ['soil']
    major_axis_fontsize = 22
    minor_axis_fontsize = 20
    title_fontsize = 30
    axis_label_fontsize = 22
    fig_height = 11
    fig_width = 17
    linewidth = 3
    sig_figs = 2
    legend_text = 20

    tc = tetracorder_figures(base_directory=base_directory, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs, legend_text=legend_text)

    #tc.classification_rates()
    #tc.f1_score_matrix_detailed()
    #tc.fraction_error()
    #tc.mae_by_classification()
    #tc.band_depth_no_reclaimer()
    tc.spectral_residual_by_class()
