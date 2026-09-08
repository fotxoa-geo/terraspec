import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.envi import envi_to_array, load_band_names
import os
from matplotlib.ticker import FormatStrFormatter
from utils.spectra_utils import spectra
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
import subprocess
from utils.results_utils import r2_calculations
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ast
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, BoundaryNorm


def apply_mask(img, bad_mask):
    """Broadcasting helper to assign NaNs across 2D/3D arrays."""
    img_masked = img.copy().astype(np.float32)
    if img_masked.ndim == 3:
        img_masked[bad_mask, :] = np.nan
    else:
        img_masked[bad_mask] = np.nan
    return img_masked


def get_target_ids_for_dataset(scene_path_pattern, mineral_name):
    """
    Finds the reflectance_minerals file recursively, parses reclassification,
    and returns integer IDs for a target mineral string.
    """
    matches = glob(scene_path_pattern, recursive=True)
    if not matches:
        return []

    mineral_class_dict, _ = spectra.get_mineral_reclassification(
        path_to_tetracorder_minerals=matches[0]
    )

    mineral_to_ids = {}
    for mineral_id, min_list in mineral_class_dict.items():
        for name in min_list:
            mineral_to_ids.setdefault(name.lower().strip(), []).append(mineral_id)

    return mineral_to_ids.get(mineral_name.lower().strip(), [])


def plot_mineral_overlay(ax, img_array, group_idx, bd_idx, target_ids, title, is_bad_pixel, cmap, norm, levels=15):
    """
    Plots smooth colored contours for band depths strictly over valid positive detections.
    """
    bad_mask = is_bad_pixel.astype(bool)
    img_masked = apply_mask(img_array, bad_mask)

    band_a = img_masked[:, :, group_idx]
    band_b = img_masked[:, :, bd_idx]

    samples_a = band_a[~np.isnan(band_a)]
    samples_b = band_b[~np.isnan(band_b)]

    if len(samples_a) > 0 and np.all(np.mod(samples_a, 1) == 0):
        mineral_index_img = band_a
        bd_data = band_b.copy()
    elif len(samples_b) > 0 and np.all(np.mod(samples_b, 1) == 0):
        mineral_index_img = band_b
        bd_data = band_a.copy()
    else:
        mineral_index_img = band_a
        bd_data = band_b.copy()

    if target_ids:
        clean_index_img = np.nan_to_num(mineral_index_img, nan=-9999).astype(np.int32)
        target_ids_int = [int(i) for i in target_ids]
        mineral_detected_mask = np.isin(clean_index_img, target_ids_int)
    else:
        mineral_detected_mask = np.zeros(mineral_index_img.shape, dtype=bool)

    valid_detections_mask = mineral_detected_mask & ~bad_mask
    n_detections = int(np.sum(valid_detections_mask))

    bd_data_contour = np.where(valid_detections_mask, bd_data, np.nan)

    h, w = bd_data.shape
    x = np.arange(w)
    y = np.arange(h)

    cs = None
    if n_detections > 0:
        cs = ax.contourf(
            x, y, bd_data_contour,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='neither'
        )

    ax.invert_yaxis()
    ax.set_title(f"{title}", fontsize=9)

    # Crucial: Allow the image box to match the grid boundaries exactly
    ax.set_aspect('auto')

    return cs, valid_detections_mask, n_detections

def prep_emit_rgb(rgb_img, red_idx=43, green_idx=24, blue_idx=11, stretch_percentile=(2, 98)):
    """
    Selects RGB bands and applies a percentile stretch.
    Default band indices roughly correspond to:
    Red ~ 650nm, Green ~ 560nm, Blue ~ 470nm (adjust based on your dataset's band list).
    """
    # Extract specific wavelength bands if image is 3D (Height, Width, Bands)
    if rgb_img.ndim == 3 and rgb_img.shape[2] > 3:
        rgb_data = rgb_img[:, :, [red_idx, green_idx, blue_idx]].astype(np.float32)
    else:
        rgb_data = rgb_img.astype(np.float32)

    # Handle bad values / fill values (EMIT data often uses -9999 for background)
    rgb_data[rgb_data < 0] = np.nan

    # Calculate 2% and 98% percentiles across valid data for contrast stretch
    p_low, p_high = np.nanpercentile(rgb_data, stretch_percentile)

    # Clip values to percentile boundaries and scale to [0, 1] range for imshow
    rgb_stretched = np.clip((rgb_data - p_low) / (p_high - p_low + 1e-8), 0, 1)

    # Replace remaining NaNs (background) with 0 or 1 for rendering
    rgb_stretched = np.nan_to_num(rgb_stretched, nan=0.0)

    return rgb_stretched

def calc_clean_metrics(x, y):
    """Safely calculates R2 and MAE after masking NaN values out of paired arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[valid_mask], y[valid_mask]

    if len(x_clean) < 2:
        return np.nan, np.nan

    try:
        r2, _ = r2_calculations(x_clean, y_clean)
        mae = mean_absolute_error(x_clean, y_clean)
        return r2, mae
    except Exception:
        return np.nan, np.nan


def evaluate_single_class(y_true_raw, y_pred_raw, target_class):
    """Evaluates binary metrics and correct prediction fraction for a target class."""

    def contains_target(item, target):
        if isinstance(item, str):
            try:
                item = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                item = [item]
        return 1 if target in item else 0

    y_true_binary = [contains_target(row, target_class) for row in y_true_raw]
    y_pred_binary = [contains_target(row, target_class) for row in y_pred_raw]

    # True Positives (correctly predicted presence)
    tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)

    # Total True Instances in Ground Truth
    n_true = sum(y_true_binary)

    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    return {
        'target_class': target_class,
        'f1_score': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'tp': tp,
        'n_true': n_true,
        'fraction_str': f"{tp}/{n_true}"  # Formatted string "4/7"
    }


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
        target_classes_dict = {'g1': sorted(['hematite', 'goethite']), 'g2': sorted(['kaolinite', 'illite', 'calcite', 'dolomite', 'montmorillonite', 'illite+muscovite'])}

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
                                                                  f'tetracorder_{group}_simulation_spectra_global_rock_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_rock_tc_aug_min'))[:, :21, group_dict[group]].astype(int)


            # aggregated confusion matrix
            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(path_to_tetracorder_minerals=os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                                                                            f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                                                                            f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]
            flat_truth = truth_array.flatten()
            flat_fractions = soil_fractions.flatten()

            target_classes = target_classes_dict[group]
            rows = 3  # One row per simulation iteration
            cols = len(target_classes)  # One col per target class
            fig, axes = plt.subplots(rows, cols, figsize=(8.5, 6.5), sharex=True, constrained_layout=True)

            colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']  # TP, FP, FN, TN
            labels = ['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)']

            simulations = [simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id,
                           veg_extracted_rock_mineral_id]

            # --- 2. Loop through Rows (Simulations) ---
            for row_idx, sim in enumerate(simulations):

                # Flatten your arrays for processing
                flat_sim = sim.flatten()

                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                y_pred_bin = mlb.transform(y_pred_labels)
                classes = list(mlb.classes_)

                # Unique Fraction Values
                unique_fractions = np.unique(flat_fractions)
                unique_fractions.sort()

                for col_idx, class_name in enumerate(target_classes):
                    ax = axes[row_idx, col_idx]

                    # Security check: skip if the target class wasn't found in the data at all
                    if class_name not in classes:
                        ax.text(0.5, 0.5, f"Missing:\n{class_name}", ha='center', va='center')
                        continue

                    class_idx = classes.index(class_name)
                    tp_v, fp_v, fn_v, tn_v = [], [], [], []

                    for val in unique_fractions:
                        mask = (flat_fractions == val)
                        yt, yp = y_true_bin[mask, class_idx], y_pred_bin[mask, class_idx]

                        tp_v.append(np.sum((yt == 1) & (yp == 1)))
                        fp_v.append(np.sum((yt == 0) & (yp == 1)))
                        fn_v.append(np.sum((yt == 1) & (yp == 0)))
                        tn_v.append(np.sum((yt == 0) & (yp == 0)))

                    bottom = np.zeros(len(unique_fractions))

                    for data, color, label in zip([tp_v, fp_v, fn_v, tn_v], colors, labels):
                        ax.bar(unique_fractions, data, bottom=bottom, color=color, label=label,
                               width=0.05)  # narrower width for float spacing
                        bottom += data

                    if row_idx == 0:
                        ax.set_title(f"{class_name.capitalize()}", fontsize=12)

                    if group == 'g1':
                        if col_idx == 0:
                            ax.set_ylim(0, 8000)  # Zoomed-in range for column 2
                        else:
                            ax.set_ylim(0, 55000)  # Original range for column 1

                    # Set your limits to match the fraction range (0 to 1)
                    ax.set_xlim(-0.05, 1.05)
                    ax.tick_params(axis='y', labelsize=8)

                    if col_idx == 0:
                        if row_idx == 0:
                            ax.set_ylabel(f"{r'$\rho$'}\nCount", fontsize=10)
                        elif row_idx == 1:
                            ax.set_ylabel(r"$\hat{\rho}_{vf}$" + "\nCount", fontsize=10)
                        else:
                            ax.set_ylabel(r"$\hat{\rho}_{vf}$'" + "\nCount", fontsize=10)

                    if group == 'g1':
                        if col_idx == 0:
                            # d controls the size of the diagonal lines
                            d = .015
                            kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, lw=1)

                            # Draw two diagonal lines at the top-left and top-right of the subplot frame
                            # Top-left break marks
                            ax.plot((-d, +d), (.9 - d, .9 + d), **kwargs)  # First diagonal line
                            ax.plot((-d, +d), (.9 - d - 0.03, .9 + d - 0.03), **kwargs)  # Second parallel line

                            ticks = list(ax.get_yticks())
                            # Generate standard labels for everything below the top tick
                            tick_labels = [f'{int(val * 1e-3)}K' if val > 0 else '0' for val in ticks]
                            # Explicitly force the very last (top) label to be 60K
                            tick_labels[-1] = '55K'

                            ax.yaxis.set_major_locator(MultipleLocator(5000))
                            ax.yaxis.set_minor_locator(MultipleLocator(1000))

                            ax.set_yticks(ticks)
                            ax.set_yticklabels(tick_labels)
                        else:
                            ax.yaxis.set_major_locator(MultipleLocator(10000))
                            ax.yaxis.set_minor_locator(MultipleLocator(5000))
                            ticks = list(ax.get_yticks())
                            # Generate standard labels for everything below the top tick
                            tick_labels = [f'{int(val * 1e-3)}K' if val > 0 else '0' for val in ticks]
                            tick_labels[-1] = '55K'
                            ax.set_yticklabels(tick_labels)


                    if group == 'g2':
                        if col_idx in [1, 2, 0]:
                            if col_idx == 0:
                                ax.set_ylim(0, 14000)
                                ax.yaxis.set_major_locator(MultipleLocator(2000))
                                ax.yaxis.set_minor_locator(MultipleLocator(1000))
                            elif col_idx == 1:
                                ax.set_ylim(0, 8000)
                                ax.yaxis.set_major_locator(MultipleLocator(1000))
                                ax.yaxis.set_minor_locator(MultipleLocator(500))
                            else:
                                ax.set_ylim(0, 1000)
                                ax.yaxis.set_major_locator(MultipleLocator(100))
                                ax.yaxis.set_minor_locator(MultipleLocator(50))

                            ymin, ymax = ax.get_ylim()

                            # --- 5. Custom Label Formatter ---
                            def dynamic_formatter(val, pos, ymax=ymax):
                                if np.isclose(val, ymax):
                                    return '55K'
                                elif val >= 1000:
                                    return f'{int(val * 1e-3)}K'
                                elif val > 0:
                                    return f'{int(val)}'
                                else:
                                    return '0'

                            ax.yaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

                            # Explicitly turn on minor tick mark visibility
                            ax.tick_params(axis='y', which='both', left=True, labelsize=8)

                            # --- 6. Position Diagonals Exactly Over the Highest Minor Tick ---
                            if col_idx == 0:
                                highest_minor_tick = ymax - 1000
                            elif col_idx == 1:
                                highest_minor_tick = ymax - 500
                            else:
                                highest_minor_tick = ymax - 50

                            break_height = highest_minor_tick / ymax

                            # Draw the custom double slash break lines
                            d = .015
                            kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, lw=1.5)
                            ax.plot((-d, +d), (break_height - d, break_height + d), **kwargs)
                            ax.plot((-d, +d), (break_height - d - 0.03, break_height + d - 0.03), **kwargs)

                            # Secure baseline locking
                            ax.set_ylim(bottom=0, top=ymax)

                        else:
                            ax.set_ylim(0, 55000)
                            ax.yaxis.set_major_locator(MultipleLocator(10000))
                            ax.yaxis.set_minor_locator(MultipleLocator(5000))
                            ticks = list(ax.get_yticks())
                            # Generate standard labels for everything below the top tick
                            tick_labels = [f'{int(val * 1e-3)}K' if val > 0 else '0' for val in ticks]
                            tick_labels[-1] = '55K'
                            ax.set_yticklabels(tick_labels)

                    # 3. Apply the locators to ALL subplots (since sharex=True relies on a unified scale)
                    ax.xaxis.set_major_locator(MultipleLocator(0.25))
                    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

                    # Format the major ticks to show 2 decimal places (0.00, 0.25, etc.)
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                    # 4. Only display labels and axis titles on the bottom row
                    if row_idx == rows - 1:
                        if group == 'g1':
                            ax.tick_params(axis='x', labelsize=8)
                        else:
                            ax.tick_params(axis='x', labelrotation=45 ,labelsize=8)

                        ax.set_xlabel('Fractional Cover\n(Soil)', fontsize=10)

            # Grab handles from the last active axis
            handles, labels = ax.get_legend_handles_labels()

            # Place the single legend at the bottom center
            fig.legend(handles, labels,
                       loc='lower center',
                       bbox_to_anchor=(0.5, -0.08),  # Anchored right above the figure bottom margin
                       ncol=4,  # 2x2 layout is perfect for a 6.5" width
                       fontsize=10,
                       frameon=True,
                       facecolor='white',
                       edgecolor='gray')

            # Save the final 3x2 figure
            plt.savefig(os.path.join(self.fig_directory, f"combined_{group}_identification_rate.png"),
                        bbox_inches='tight')
            plt.clf()
            plt.close()


    def f1_score_matrix_detailed(self):
        group_dict = {'g1': 1, 'g2': 3}
        target_classes_dict = {
            'g1': sorted(['hematite', 'goethite']),
            'g2': sorted(['kaolinite', 'illite', 'calcite', 'dolomite', 'montmorillonite', 'illite+muscovite'])
        }

        # Setup a unified 3x3 figure canvas
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(11, 10), constrained_layout=True)

        # Flatten axes array for simple sequential indexing (0 to 8)
        axes_flat = axes.flatten()

        # Visual configuration variables
        sim_labels = [f"{r'$\rho$'}", r"$\hat{\rho}_{vf}$", r"$\hat{\rho}_{vf}$'"]

        sim_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        plot_idx = 0  # Global tracker to index subplots across both groups

        for group in ['g1', 'g2']:
            soil_fractions = envi_to_array(
                os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[
                :, :, 2]
            soil_fractions = np.round(soil_fractions, 2)

            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[
                :, -1, group_dict[group]].astype(int)

            simulated_mineral_id_from_tetracorder = \
            envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                       f'tetracorder_{group}_simulation_spectra_global_lib',
                                       f'tetracorder_{group}_simulation_spectra_aug_min'))[
                :, :, group_dict[group]].astype(int)

            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[
                :, :, group_dict[group]].astype(int)

            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_global_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_global_rock_tc_aug_min'))[
                :, :21, group_dict[group]].astype(int)

            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(
                path_to_tetracorder_minerals=os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                          f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]
            flat_truth = truth_array.flatten()
            flat_fractions = soil_fractions.flatten()

            target_classes = target_classes_dict[group]
            simulations = [simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id,
                           veg_extracted_rock_mineral_id]

            unique_fractions = np.unique(flat_fractions)
            unique_fractions.sort()

            # Step 1: Pre-calculate F1 metrics for all simulation pipelines in this group
            group_f1_data = {class_name: {i: [] for i in range(len(simulations))} for class_name in target_classes}

            for sim_idx, sim in enumerate(simulations):
                flat_sim = sim.flatten()
                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                y_pred_bin = mlb.transform(y_pred_labels)
                classes = list(mlb.classes_)

                for class_name in target_classes:
                    if class_name not in classes:
                        group_f1_data[class_name][sim_idx] = None
                        continue

                    class_idx = classes.index(class_name)
                    for val in unique_fractions:
                        mask = (flat_fractions == val)
                        yt, yp = y_true_bin[mask, class_idx], y_pred_bin[mask, class_idx]

                        tp = np.sum((yt == 1) & (yp == 1))
                        fp = np.sum((yt == 0) & (yp == 1))
                        fn = np.sum((yt == 1) & (yp == 0))

                        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                        group_f1_data[class_name][sim_idx].append(f1)

            # Step 2: Plot the calculated mineral datasets sequentially into the grid
            for class_name in target_classes:
                ax = axes_flat[plot_idx]

                # Draw the lines for each simulation
                for sim_idx in range(len(simulations)):
                    scores = group_f1_data[class_name][sim_idx]
                    if scores is not None:
                        ax.plot(unique_fractions, scores,
                                color=sim_colors[sim_idx],
                                linewidth=1.5,
                                alpha=0.85,
                                label=sim_labels[sim_idx])
                    else:
                        ax.text(0.5, 0.5, f"Missing Data:\n{class_name}", ha='center', va='center')

                # Subplot Customization & Formatting
                ax.set_title(f"{class_name.capitalize()}", fontsize=12, fontweight='semibold')
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(-0.05, 1.05)
                ax.tick_params(axis='both', labelsize=8)

                # High density tick-marks
                ax.yaxis.set_major_locator(MultipleLocator(0.20))
                ax.yaxis.set_minor_locator(MultipleLocator(0.10))
                ax.xaxis.set_major_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                # Only label the vertical axis on leftmost subplots (columns 0, 3, 6)
                if plot_idx % 3 == 0:
                    ax.set_ylabel("F1 Score", fontsize=10)
                else:
                    ax.set_yticks([])

                # Only label horizontal axis on bottom positions or when wrapping up a group
                if plot_idx >= 5:
                    ax.set_xlabel('Soil Fractional Cover', fontsize=10)

                plot_idx += 1

        # --- Step 3: Handle the remaining 9th empty subplot (index 8) ---
        extra_ax = axes_flat[-1]
        extra_ax.axis('off')  # Completely hide the empty grid boundaries

        # Insert the unified legend neatly inside the vacant 9th grid spot
        handles, labels = axes_flat[0].get_legend_handles_labels()
        extra_ax.legend(handles, labels,
                        loc='center',
                        fontsize=10,
                        frameon=True,
                        facecolor='white',
                        edgecolor='gray',
                        title="Mixed Reflectances",
                        title_fontsize=11)

        # Save out the combined grid image asset
        plt.savefig(os.path.join(self.fig_directory, "unified_3x3_mineral_f1_scores.png"),
                    bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()

    def band_depth_mae(self):
        group_dict = {'g1': 1, 'g2': 3}
        bd_dict = {'g1': 0, 'g2': 2}

        target_classes_dict = {
            'g1': sorted(['hematite', 'goethite']),
            'g2': sorted(['kaolinite', 'illite', 'calcite', 'dolomite', 'montmorillonite', 'illite+muscovite'])
        }

        # Setup a unified 3x3 figure canvas
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(11, 10), constrained_layout=True)

        # Flatten axes array for simple sequential indexing (0 to 8)
        axes_flat = axes.flatten()

        # Visual configuration variables
        sim_labels = [f"{r'$\rho$'}", r"$\hat{\rho}_{vf}$ ($a\ priori$)",  r"$\hat{\rho}_{vf}$' ($a\ priori$)", f"RECLAIMER (posterior)", f"RECLAIMER' (posterior)", r'$\rho_k$ (known fractions)']
        sim_colors = ['black', 'magenta', 'orange', 'red', 'blue', 'green']

        plot_idx = 0  # Global tracker to index subplots across both groups

        for group in ['g1', 'g2']:
            soil_fractions = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}', f'tetracorder_{group}_simulation_fractions'))[:, :, 2]
            soil_fractions = np.round(soil_fractions, 2)

            # this is detection data
            soil_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, group_dict[group]].astype(int)

            simulated_mineral_id_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                               f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                               f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, group_dict[group]].astype(int)

            veg_extracted_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            veg_extracted_rock_mineral_id = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_global_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_global_rock_tc_aug_min'))[:, :, group_dict[group]].astype(int)

            # load band depth data
            soil_mineral_bd_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                                          f'tetracorder_{group}_simulation_spectra_aug_min'))[:, -1, bd_dict[group]].astype(float)

            simulated_mineral_bd_from_tetracorder = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                       f'tetracorder_{group}_simulation_spectra_global_lib',
                                       f'tetracorder_{group}_simulation_spectra_aug_min'))[:, :, bd_dict[group]].astype(float)

            veg_extracted_mineral_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                  f'tetracorder_{group}_simulation_spectra_global_lib_veg_ext',
                                                                  f'ext_veg_tetracorder_{group}_simulation_spectra_global_lib_tc_aug_min'))[:, :, bd_dict[group]].astype(float)

            veg_extracted_rock_mineral_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'tetracorder_{group}_simulation_spectra_global_rock_veg_ext',
                                                                       f'ext_veg_tetracorder_{group}_simulation_spectra_global_rock_tc_aug_min'))[:, :, bd_dict[group]].astype(float)
            # load post mae corrections
            post_veg_extracted_mineral_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                 f'RECLAIMER_{group}_global_lib_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, :, 2].astype(float)
            post_veg_extracted_mineral_bd[post_veg_extracted_mineral_bd == -9999 ] = np.nan
            post_veg_extracted_mineral_bd_soil = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                 f'RECLAIMER_{group}_global_lib_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, -1, 1].astype(float)
            post_veg_extracted_mineral_bd_soil[post_veg_extracted_mineral_bd_soil == -9999] = np.nan

            # load rock band depths
            post_veg_extracted_rock_mineral_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'RECLAIMER_{group}_global_rock_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, :, 2].astype(float)
            post_veg_extracted_rock_mineral_bd[post_veg_extracted_rock_mineral_bd == -9999 ] = np.nan

            post_veg_extracted_rock_mineral_bd_soil = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                            f'RECLAIMER_{group}_global_rock_tetracorder_{group}_simulation_spectra_normalization_brightness__fractional_cover'))[:, -1, 1].astype(float)
            post_veg_extracted_rock_mineral_bd_soil[post_veg_extracted_rock_mineral_bd_soil == -9999] = np.nan

            # load best case scenario
            known_bd = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'RECLAIMER_{group}_tetracorder_{group}_simulation_fractions'))[:, :, 2].astype(float)
            known_bd[known_bd == -9999] = np.nan

            known_bd_soil = envi_to_array(os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                                       f'RECLAIMER_{group}_tetracorder_{group}_simulation_fractions'))[:, -1, 1].astype(float)
            known_bd_soil[known_bd_soil == -9999] = np.nan

            # load mineral classifications
            mineral_class, df_minerals_sim = spectra.get_mineral_reclassification(
                path_to_tetracorder_minerals=os.path.join(self.synthetic_rfls, f'tetracorder_{group}',
                                                          f'tetracorder_{group}_simulation_spectra_global_lib',
                                                          f'tetracorder_{group}_simulation_spectra_aug_minerals'))

            # _________ This is for simulation ids _____________
            truth_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(int) * -9999
            truth_array[:] = soil_mineral_id_from_tetracorder[:, np.newaxis]
            truth_array[:, 0] = simulated_mineral_id_from_tetracorder[:, 0]
            flat_truth = truth_array.flatten()
            flat_fractions = soil_fractions.flatten()

            # ________ Load true band depths _________________
            truth_bd_array = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(float) * -9999
            truth_bd_array[:] = soil_mineral_bd_from_tetracorder[:, np.newaxis]
            truth_bd_array[:, 0] = simulated_mineral_bd_from_tetracorder[:, 0]
            flat_truth_bd = truth_bd_array.flatten()

            truth_bd_array_post = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(float) * -9999
            truth_bd_array_post[:] = post_veg_extracted_mineral_bd_soil[:, np.newaxis]
            truth_bd_array_post[:, 0] = simulated_mineral_bd_from_tetracorder[:, 0]
            flat_truth_bd_post = truth_bd_array_post.flatten()

            truth_bd_array_post_rock = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(float) * -9999
            truth_bd_array_post_rock[:] = post_veg_extracted_rock_mineral_bd_soil[:, np.newaxis]
            truth_bd_array_post_rock[:, 0] = simulated_mineral_bd_from_tetracorder[:, 0]
            flat_truth_bd_post_rock = truth_bd_array_post_rock.flatten()

            truth_bd_array_known = np.ones((soil_fractions.shape[0], soil_fractions.shape[1])).astype(float) * -9999
            truth_bd_array_known[:] = known_bd_soil[:, np.newaxis]
            truth_bd_array_known[:, 0] = simulated_mineral_bd_from_tetracorder[:, 0]
            flat_truth_bd_known = truth_bd_array_known.flatten()

            target_classes = target_classes_dict[group]
            simulations = [simulated_mineral_id_from_tetracorder, veg_extracted_mineral_id,
                           veg_extracted_rock_mineral_id, veg_extracted_mineral_id,
                           veg_extracted_rock_mineral_id, simulated_mineral_id_from_tetracorder]

            band_depths_simulations = [simulated_mineral_bd_from_tetracorder, veg_extracted_mineral_bd,
                                       veg_extracted_rock_mineral_bd, post_veg_extracted_mineral_bd,
                                       post_veg_extracted_rock_mineral_bd, known_bd]

            unique_fractions = np.unique(flat_fractions)
            unique_fractions.sort()

            group_mae_data = {class_name: {i: [] for i in range(len(simulations))} for class_name in target_classes}

            for sim_idx, sim in enumerate(simulations):
                flat_sim = sim.flatten()
                flat_sim_bd = band_depths_simulations[sim_idx].flatten()

                y_true_labels = [mineral_class.get(i, ['other']) for i in flat_truth]
                y_pred_labels = [mineral_class.get(i, ['other']) for i in flat_sim]

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true_labels)
                classes = list(mlb.classes_)

                for class_name in target_classes:
                    if class_name not in classes:
                        group_mae_data[class_name][sim_idx] = None
                        continue

                    class_idx = classes.index(class_name)
                    for val in unique_fractions:
                        # Isolate observations belonging to the current fraction value bin
                        fraction_mask = (flat_fractions == val)

                        # Target specific mineral index
                        yt = y_true_bin[:, class_idx]

                        # Intersect fraction bin with ground truth presence of this specific mineral
                        mae_mask = fraction_mask & (yt == 1)

                        if np.sum(mae_mask) == 0:
                            mae = 0.0  # Safe handling if target mineral isn't present in this fraction slice
                        else:

                            if sim_idx in [0,1,2]:
                                actuals = flat_truth_bd[mae_mask]
                            elif sim_idx in [3]:
                                actuals = flat_truth_bd_post[mae_mask]
                            elif sim_idx in [4]:
                                actuals = flat_truth_bd_post_rock[mae_mask]
                            else:
                               actuals = flat_truth_bd_known[mae_mask]

                            predictions = flat_sim_bd[mae_mask]

                            # Calculate absolute differences (will contain NaN if either actual or prediction is NaN)
                            abs_diffs = np.abs(actuals - predictions)

                            # Extract only the non-NaN values
                            valid_diffs = abs_diffs[~np.isnan(abs_diffs)]

                            if len(valid_diffs) == 0:
                                mae = 0.0  # Fallback if the entire masked region consisted of NaN values
                            else:
                                mae = np.mean(valid_diffs)

                        group_mae_data[class_name][sim_idx].append(mae)

            # Step 2: Plot the calculated mineral datasets sequentially into the grid
            for class_name in target_classes:
                ax = axes_flat[plot_idx]
                max_mae_val = 0.01  # baseline reference tracking boundary limit

                # Draw the lines for each simulation
                for sim_idx in range(len(simulations)):
                    scores = group_mae_data[class_name][sim_idx]
                    if scores is not None:
                        ax.plot(unique_fractions, scores,
                                color=sim_colors[sim_idx],
                                linewidth=1.5,
                                alpha=0.85,
                                label=sim_labels[sim_idx])
                        max_mae_val = max(max_mae_val, np.max(scores))
                    else:
                        ax.text(0.5, 0.5, f"Missing Data:\n{class_name}", ha='center', va='center')

                # Subplot Customization & Formatting
                ax.set_title(f"{class_name.capitalize()}", fontsize=10, fontweight='semibold')
                ax.set_xlim(0.05, 1.05)

                # Dynamic scaling wrapper to optimize chart vertical distribution spacing
                ax.set_ylim(0, 0.12)
                ax.tick_params(axis='both', labelsize=8)



                # High density tick-marks configuration
                ax.xaxis.set_major_locator(MultipleLocator(0.25))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.yaxis.set_major_locator(MultipleLocator(0.02))
                ax.yaxis.set_minor_locator(MultipleLocator(0.01))

                # Only label the vertical axis on leftmost subplots (columns 0, 3, 6)
                if plot_idx % 3 == 0:
                    ax.set_ylabel("Band Depth MAE", fontsize=9)
                else:
                    ax.set_yticks([])

                # Only label horizontal axis on bottom positions or when wrapping up a group
                if plot_idx >= 5:
                    ax.set_xlabel('Soil Fractional Cover', fontsize=9)

                plot_idx += 1

            # --- Step 3: Handle the remaining 9th empty subplot (index 8) ---
        extra_ax = axes_flat[-1]
        extra_ax.axis('off')  # Completely hide the empty grid boundaries

        # Insert the unified legend neatly inside the vacant 9th grid spot
        handles, labels = axes_flat[0].get_legend_handles_labels()
        extra_ax.legend(handles, labels,
                        loc='center',
                        fontsize=10,
                        frameon=True,
                        facecolor='white',
                        edgecolor='gray',
                        title="Mixed Reflectances",
                        title_fontsize=11)

        # Save out the combined grid image asset
        plt.savefig(os.path.join(self.fig_directory, "unified_3x3_mineral_mae_scores.png"),
                    bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()

    def field_table(self):
        group_dict = {'g1': 1, 'g2': 3}
        bd_dict = {'g1': 0, 'g2': 2}

        tc_contact_probe = sorted(list(glob(os.path.join(self.output_directory, 'field', '**', '*_EMS_emit_augmented_min'), recursive=True)))

        rows = []

        for group in ['g1', 'g2']:

            for cp in tc_contact_probe:

                # these are the base for truth
                plot_num = os.path.basename(cp).split('_')[0]
                try:
                    # this is contact probe data
                    cp_mineral_class, cp_df_minerals_sim = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'tc_contact',
                                                                  f'{plot_num}_EMS_emit_augmented_minerals'))
                    cp_mineral = envi_to_array(cp)[0, 0, group_dict[group]]
                    cp_bd =  envi_to_array(cp)[0, 0, bd_dict[group]]

                    cp_class = cp_mineral_class.get(cp_mineral, ['other'])

                    # uncorrected slpit
                    slpit_uncorrected_mineral_class, slpit_unc_df_minerals_sim = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_unc',
                                                                  f'{plot_num}_SLPIT_emit_augmented_minerals'))
                    slpit_unc_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_unc', f'{plot_num}_SLPIT_emit_augmented_min'))[0, 0, group_dict[group]]
                    slpit_unc_bd = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_unc', f'{plot_num}_SLPIT_emit_augmented_min'))[0, 0, bd_dict[group]]
                    slpit_unc_class = slpit_uncorrected_mineral_class.get(slpit_unc_mineral, ['other'])

                    # # corrected slpit - pre tetracorder ; global library
                    slpit_corrected_global_mineral_class, slpit_corrected_df_minerals = spectra.get_mineral_reclassification(
                    path_to_tetracorder_minerals = os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_global',
                                                                f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_lib_tc_minerals'))

                    slpit_corrected_global_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_global', f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_lib_tc_min'))[0, 0, group_dict[group]]
                    slpit_corrected_global_bd =  envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_global', f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_lib_tc_min'))[0, 0, bd_dict[group]]
                    slpit_corrected_global_class = slpit_corrected_global_mineral_class.get(slpit_corrected_global_mineral, ['other'])

                    # corrected slpit - rho s ; global library
                    slpit_rho_s_global_mineral_class,  df_slpit_rho_s_global_mineral_class = spectra.get_mineral_reclassification(
                         path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_global',
                                                                   f'recon_rho_{plot_num}_SLPIT_emit_augmented_global_lib_minerals'))

                    slpit_rho_s_global_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_global', f'recon_rho_{plot_num}_SLPIT_emit_augmented_global_lib_min'))[0, 0, group_dict[group]]
                    slpit_rho_s_global_bd = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_global', f'recon_rho_{plot_num}_SLPIT_emit_augmented_global_lib_min'))[0, 0, bd_dict[group]]
                    slpit_rho_s_corrected_global_class = slpit_rho_s_global_mineral_class.get(slpit_rho_s_global_mineral, ['other'])

                    # reclaimr slpit - post tetracorder; global library
                    base_call = (f'python ./tetracorder/reclaimer.py '
                                 f'-out_dir {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_global')} '
                                 f'-tc_out {os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_global', f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_lib_tc_min.hdr')} '
                                 f'-rfl {os.path.join(self.output_directory, 'field', plot_num, f'{plot_num}_SLPIT_emit_augmented.hdr')} '
                                 f'-um_out {os.path.join(self.output_directory, 'field', plot_num, 'emc2_SLPIT', f'global_lib_{plot_num}_SLPIT_emit_fractional_cover.hdr')} -g_num {int(group[-1:])} '
                                 f'-rho_gv {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_global', f'extracted_{plot_num}_SLPIT_emit_augmented_pv_global_lib_signal.hdr')} '
                                 f'-rho_npv {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_global' ,f'extracted_{plot_num}_SLPIT_emit_augmented_npv_global_lib_signal.hdr')}')
                    subprocess.call(base_call, shell=True)
                    reclaimer_slpit_global_lib = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_global', f'RECLAIMER_{group}_global_lib_{plot_num}_SLPIT_emit_fractional_cover'))[0, 0, :]

                    # # corrected slpit - pre tetracorder ; rock library
                    slpit_corrected_rock_mineral_class, df_slpit_corrected_rock_mineral_class = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_rock',
                                                                  f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_rock_tc_minerals'))

                    slpit_corrected_rock_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_rock', f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_rock_tc_min'))[0, 0, group_dict[group]]
                    slpit_corrected_rock_bd = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_rock', f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_rock_tc_min'))[0, 0, bd_dict[group]]
                    slpit_corrected_rock_class = slpit_corrected_rock_mineral_class.get(slpit_corrected_rock_mineral, ['other'])

                    # # corrected slpit - rho s ; rock library
                    slpit_rho_s_rock_mineral_class, df_slpit_rho_s_rock_mineral_class = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_rock',
                                                                  f'recon_rho_{plot_num}_SLPIT_emit_augmented_global_rock_minerals'))

                    slpit_rho_s_rock_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_rock', f'recon_rho_{plot_num}_SLPIT_emit_augmented_global_rock_min'))[0, 0, group_dict[group]]
                    slpit_rho_s_rock_bd = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_rock', f'recon_rho_{plot_num}_SLPIT_emit_augmented_global_rock_min'))[0, 0, bd_dict[group]]
                    slpit_rho_s_corrected_mineral_class = slpit_rho_s_rock_mineral_class.get(slpit_rho_s_rock_mineral, ['other'])

                    # reclaimr slpit - post tetracorder; rock library
                    base_call = (
                        f'python ./tetracorder/reclaimer.py '
                        f'-out_dir {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_rock')} '
                        f'-tc_out {os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_rock', f'ext_veg_{plot_num}_SLPIT_emit_augmented_global_rock_tc_min.hdr')} '
                        f'-rfl {os.path.join(self.output_directory, 'field', plot_num, f'{plot_num}_SLPIT_emit_augmented.hdr')} '
                        f'-um_out {os.path.join(self.output_directory, 'field', plot_num, 'emc2_SLPIT', f'global_rock_{plot_num}_SLPIT_emit_fractional_cover.hdr')} -g_num {int(group[-1:])} '
                        f'-rho_gv {os.path.join(self.output_directory, 'field', plot_num,'SLPIT_rock' , f'extracted_{plot_num}_SLPIT_emit_augmented_pv_global_rock_signal.hdr')} '
                        f'-rho_npv {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_rock', f'extracted_{plot_num}_SLPIT_emit_augmented_npv_global_rock_signal.hdr')}')

                    subprocess.call(base_call, shell=True)
                    reclaimer_slpit_global_rock = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_rock', f'RECLAIMER_{group}_global_rock_{plot_num}_SLPIT_emit_fractional_cover'))[0, 0, :]

                    if int(plot_num[-3:]) not in [14, 15, 16]:
                        # # corrected slpit - pre tetracorder ; local library
                        slpit_corrected_local_mineral_class, df_slpit_corrected_local_mineral_class = spectra.get_mineral_reclassification(
                            path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_lcl',
                                                                      f'ext_veg_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_emi_minerals'))

                        slpit_corrected_local_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_lcl', f'ext_veg_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_emi_min'))[0, 0, group_dict[group]]
                        slpit_corrected_local_bd = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_lcl', f'ext_veg_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_emi_min'))[0, 0, bd_dict[group]]
                        slpit_corrected_local_class = slpit_corrected_local_mineral_class.get(slpit_corrected_local_mineral, ['other'])

                        # # corrected slpit - rho s ; local library
                        slpit_rho_s_rock_local_class, df_slpit_rho_s_rock_local_class = spectra.get_mineral_reclassification(
                            path_to_tetracorder_minerals=os.path.join(self.output_directory, f'field', plot_num, 'SLPIT_lcl',
                                                                      f'recon_rho_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_e_minerals'))

                        slpit_rho_s_local_mineral = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_lcl', f'recon_rho_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_e_min'))[0, 0, group_dict[group]]
                        slpit_rho_s_local_bd = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_lcl', f'recon_rho_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_e_min'))[0, 0, bd_dict[group]]
                        slpit_rho_s_corrected_local_class = slpit_rho_s_rock_local_class.get(slpit_rho_s_local_mineral, ['other'])

                        # reclaimr slpit - post tetracorder; local library
                        base_call = (
                            f'python ./tetracorder/reclaimer.py '
                            f'-out_dir {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_lcl')} '
                            f'-tc_out {os.path.join(self.output_directory, 'field', plot_num, f'SLPIT_lcl', f'ext_veg_{plot_num}_SLPIT_emit_augmented_{plot_num}_EMS_emi_min.hdr')} '
                            f'-rfl {os.path.join(self.output_directory, 'field', plot_num, f'{plot_num}_SLPIT_emit_augmented.hdr')} '
                            f'-um_out {os.path.join(self.output_directory, 'field', plot_num, 'emc2_SLPIT', f'local_{plot_num}_SLPIT_emit_fractional_cover.hdr')} -g_num {int(group[-1:])} '
                            f'-rho_gv {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_lcl', f'extracted_{plot_num}_SLPIT_emit_augmented_PV_{plot_num}_EMS_emit_signal.hdr')} '
                            f'-rho_npv {os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_lcl', f'extracted_{plot_num}_SLPIT_emit_augmented_NPV_{plot_num}_EMS_emit_signal.hdr')}')
                        subprocess.call(base_call, shell=True)
                        reclaimer_slpit_local = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, 'SLPIT_lcl',
                                                                           f'RECLAIMER_{group}_local_{plot_num}_SLPIT_emit_fractional_cover'))[0, 0, :]
                    else:
                        print(f"Skipping local instance for plot: {plot_num}")
                        slpit_corrected_local_bd = -9999
                        slpit_corrected_local_class = -9999
                        slpit_rho_s_local_bd = -9999
                        slpit_rho_s_corrected_local_class = -9999
                        reclaimer_slpit_local = np.array([-9999, -9999])

                    # uncorrected EMIT
                    emit_uncorrected_mineral_class, emit_unc_df_minerals_sim = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_unc', f'*_EXT_augmented_minerals'))[0])

                    emit_unc_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_unc', f'*_EXT_augmented_min'))[0])[0, 0, group_dict[group]]
                    emit_unc_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_unc', f'*_EXT_augmented_min'))[0])[0, 0, bd_dict[group]]
                    emit_unc_class = emit_uncorrected_mineral_class.get(emit_unc_mineral, ['other'])

                    # # corrected emit - pre tetracorder ; global library
                    emit_corrected_global_mineral_class, emit_corrected_df_minerals = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_global',
                                                                       f'*_EXT_augmented_global__minerals'))[0])

                    emit_corrected_global_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_global', f'*_EXT_augmented_global__min'))[0])[0, 0, group_dict[group]]
                    emit_corrected_global_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_global', f'*_EXT_augmented_global__min'))[0])[0, 0, bd_dict[group]]
                    emit_corrected_global_class = emit_corrected_global_mineral_class.get(emit_corrected_global_mineral, ['other'])

                    # # corrected emit - rho s ; global library
                    emit_rho_s_global_mineral_class, df_emit_rho_s_global_mineral_class = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_global',
                                                                       f'*_EXT_augmented_globa_minerals'))[0])

                    emit_rho_s_global_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_global', f'*_EXT_augmented_globa_min'))[0])[0, 0, group_dict[group]]
                    emit_rho_s_global_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_global', f'*_EXT_augmented_globa_min'))[0])[0, 0, bd_dict[group]]
                    emit_rho_s_corrected_global_class = emit_rho_s_global_mineral_class.get(emit_rho_s_global_mineral, ['other'])

                    # reclaimr emit - post tetracorder; global library
                    base_call = (f'python ./tetracorder/reclaimer.py '
                                 f'-out_dir {os.path.join(self.output_directory, 'field', plot_num, 'RFL_global')} '
                                 f'-tc_out {glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_global', f'*ext_veg_{plot_num}_RFL_*EXT_augmented_global__min.hdr'))[0]} '
                                 f'-rfl {glob(os.path.join(self.output_directory, 'field', plot_num, f'*_EXT_augmented.hdr'))[0]} '
                                 f'-um_out {glob(os.path.join(self.output_directory, 'field', plot_num, 'emc2_RFL', f'*global_lib_{plot_num}_RFL*_EXT_fractional_cover.hdr'))[0]} -g_num {int(group[-1:])} '
                                 f'-rho_gv {glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_global', f'*extracted_{plot_num}_RFL_*_EXT_augmented_pv_global_lib_signal.hdr'))[0]} '
                                 f'-rho_npv {glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_global', f'*extracted_{plot_num}_RFL_*_EXT_augmented_npv_global_lib_signal.hdr'))[0]}')

                    subprocess.call(base_call, shell=True)
                    reclaimer_emit_global_lib = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_global',
                                                                            f'*RECLAIMER_{group}_global_lib_{plot_num}_RFL_*_EXT_fractional_cover'))[0])[0, 0, :]

                    # # corrected emit - pre tetracorder ; rock library
                    emit_corrected_rock_mineral_class, df_emit_corrected_rock_mineral_class = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_rock',
                                          f'*_EXT_augmented_global__minerals'))[0])

                    emit_corrected_rock_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_rock', f'*_EXT_augmented_global__min'))[0])[0, 0, group_dict[group]]
                    emit_corrected_rock_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_global', f'*_EXT_augmented_global__min'))[0])[0, 0, bd_dict[group]]
                    emit_corrected_rock_class = emit_corrected_rock_mineral_class.get(emit_corrected_rock_mineral, ['other'])

                    # # corrected emit - rho s ; rock library
                    emit_rho_s_rock_mineral_class, df_emit_rho_s_rock_mineral_class = spectra.get_mineral_reclassification(
                        path_to_tetracorder_minerals=glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_rock',
                                          f'*_EXT_augmented_globa_minerals'))[0])

                    emit_rho_s_rock_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_rock', f'*_EXT_augmented_globa_min'))[0])[0, 0, group_dict[group]]
                    emit_rho_s_rock_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_rock', f'*_EXT_augmented_globa_min'))[0])[0, 0, bd_dict[group]]
                    emit_rho_s_corrected_rock_class = emit_rho_s_rock_mineral_class.get(emit_rho_s_rock_mineral, ['other'])

                    # reclaimr emit - post tetracorder; rock library
                    base_call = (f'python ./tetracorder/reclaimer.py '
                                 f'-out_dir {os.path.join(self.output_directory, 'field', plot_num, 'RFL_rock')} '
                                 f'-tc_out {glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_rock', f'*ext_veg_{plot_num}_RFL_*EXT_augmented_global__min.hdr'))[0]} '
                                 f'-rfl {glob(os.path.join(self.output_directory, 'field', plot_num, f'*_EXT_augmented.hdr'))[0]} '
                                 f'-um_out {glob(os.path.join(self.output_directory, 'field', plot_num, 'emc2_RFL', f'*global_rock_{plot_num}_RFL*_EXT_fractional_cover.hdr'))[0]} -g_num {int(group[-1:])} '
                                 f'-rho_gv {glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_rock', f'*extracted_{plot_num}_RFL_*_EXT_augmented_pv_global_rock_signal.hdr'))[0]} '
                                 f'-rho_npv {glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_rock', f'*extracted_{plot_num}_RFL_*_EXT_augmented_npv_global_rock_signal.hdr'))[0]}')

                    subprocess.call(base_call, shell=True)
                    reclaimer_emit_rock_lib = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_rock',f'*RECLAIMER_{group}_global_rock_{plot_num}_RFL_*_EXT_fractional_cover'))[0])[0, 0, :]

                    if int(plot_num[-3:]) not in [14, 15, 16]:
                        # corrected emit - pre tetracorder ; local library
                        emit_corrected_local_mineral_class, df_emit_corrected_local_mineral_class = spectra.get_mineral_reclassification(
                            path_to_tetracorder_minerals= glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_lcl',
                                              f'*_EXT_augmented_Spectra_minerals'))[0])

                        emit_corrected_local_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_lcl', f'*_EXT_augmented_Spectra_min'))[0])[0, 0, group_dict[group]]
                        emit_corrected_local_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_lcl', f'*_EXT_augmented_Spectra_min'))[0])[0, 0, bd_dict[group]]
                        emit_corrected_local_class = emit_corrected_local_mineral_class.get(emit_corrected_local_mineral, ['other'])

                        # # corrected emit - rho s ; local library
                        emit_rho_s_local_mineral_class, df_emit_rho_s_local_mineral_class = spectra.get_mineral_reclassification(
                            path_to_tetracorder_minerals= glob(os.path.join(self.output_directory, f'field', plot_num, 'RFL_lcl',
                                              f'*_EXT_augmented_Spect_minerals'))[0])

                        emit_rho_s_local_mineral = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_lcl', f'*_EXT_augmented_Spect_min'))[0])[0, 0, group_dict[group]]
                        emit_rho_s_local_bd = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_lcl', f'*_EXT_augmented_Spect_min'))[0])[0, 0, bd_dict[group]]
                        emit_rho_s_corrected_local_class = emit_rho_s_local_mineral_class.get(emit_rho_s_local_mineral, ['other'])

                        # reclaimr emit - post tetracorder; local library
                        base_call = (
                            f'python ./tetracorder/reclaimer.py '
                            f'-out_dir {os.path.join(self.output_directory, 'field', plot_num, 'RFL_lcl')} '
                            f'-tc_out {glob(os.path.join(self.output_directory, 'field', plot_num, f'RFL_lcl', f'*ext_veg_{plot_num}_RFL_*_EXT_augmented_Spectra_min.hdr'))[0]} '
                            f'-rfl {glob(os.path.join(self.output_directory, 'field', plot_num, f'*{plot_num}_RFL_*_EXT_augmented.hdr'))[0]} '
                            f'-um_out {glob(os.path.join(self.output_directory, 'field', plot_num, 'emc2_RFL', f'*local_{plot_num}_RFL_*_EXT_fractional_cover.hdr'))[0]} -g_num {int(group[-1:])} '
                            f'-rho_gv {glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_lcl', f'*extracted_{plot_num}_RFL_*_EXT_augmented_PV_{plot_num}_EMS_emit_signal.hdr'))[0]} '
                            f'-rho_npv {glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_lcl', f'*extracted_{plot_num}_RFL_*_EXT_augmented_NPV_{plot_num}_EMS_emit_signal.hdr'))[0]}')
                        subprocess.call(base_call, shell=True)
                        reclaimer_emit_local = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, 'RFL_lcl',
                                                   f'*RECLAIMER_{group}_local_{plot_num}_RFL_*_EXT_fractional_cover'))[0])[0, 0, :]

                    else:
                        print(f"Skipping local instance for plot: {plot_num}")
                        emit_corrected_local_bd = -9999
                        emit_corrected_local_class = -9999
                        emit_rho_s_local_bd = -9999
                        emit_rho_s_corrected_local_class = -9999
                        reclaimer_emit_local = np.array([-9999, -9999])


                    # get fractional cover data
                    slpit_local_fractional_cover = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, 'emc2_SLPIT', f'local_{plot_num}_SLPIT_emit_fractional_cover'))[0, 0, :]
                    slpit_global_fractional_cover = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, 'emc2_SLPIT', f'global_lib_{plot_num}_SLPIT_emit_fractional_cover'))[0, 0, :]
                    slpit_rock_fractional_cover = envi_to_array(os.path.join(self.output_directory, 'field', plot_num, 'emc2_SLPIT', f'global_rock_{plot_num}_SLPIT_emit_fractional_cover'))[0, 0, :]

                    emit_local_fractional_cover = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, 'emc2_RFL', f'*local_{plot_num}_RFL_*_EXT_fractional_cover'))[0])[0, 0, :]
                    emit_global_fractional_cover = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, 'emc2_RFL', f'*global_lib_{plot_num}_RFL_*_EXT_fractional_cover'))[0])[0, 0, :]
                    emit_rock_fractional_cover = envi_to_array(glob(os.path.join(self.output_directory, 'field', plot_num, 'emc2_RFL', f'*global_rock_{plot_num}_RFL_*_EXT_fractional_cover'))[0])[0, 0, :]

                    row = ([plot_num, group,
                            cp_bd, cp_class,
                            slpit_unc_bd, slpit_unc_class,
                            slpit_corrected_global_bd, slpit_corrected_global_class,
                            slpit_rho_s_global_bd, slpit_rho_s_corrected_global_class]
                           + reclaimer_slpit_global_lib.tolist() + [slpit_corrected_rock_bd, slpit_corrected_rock_class, slpit_rho_s_rock_bd, slpit_rho_s_corrected_mineral_class]
                           + reclaimer_slpit_global_rock.tolist() + [slpit_corrected_local_bd, slpit_corrected_local_class, slpit_rho_s_local_bd, slpit_rho_s_corrected_local_class] +
                           reclaimer_slpit_local.tolist() + [emit_unc_bd, emit_unc_class, emit_corrected_global_bd, emit_corrected_global_class, emit_rho_s_global_bd, emit_rho_s_corrected_global_class] +
                           reclaimer_emit_global_lib.tolist() + [emit_corrected_rock_bd, emit_corrected_rock_class, emit_rho_s_rock_bd, emit_rho_s_corrected_rock_class] +
                           reclaimer_emit_rock_lib.tolist() + [emit_corrected_local_bd, emit_corrected_local_class, emit_rho_s_local_bd, emit_rho_s_corrected_local_class] +
                           reclaimer_emit_local.tolist() + slpit_local_fractional_cover.tolist() + slpit_global_fractional_cover.tolist() + slpit_rock_fractional_cover.tolist() +
                           emit_local_fractional_cover.tolist() + emit_global_fractional_cover.tolist() + emit_rock_fractional_cover.tolist() )

                    rows.append(row)

                except Exception as e:
                    print(f'{plot_num} {group} failed to load!')
                    raise e

        column_names = ['plot_num', 'group',
                        'cp_bd', 'cp_class',
                        'slpit_unc_bd', 'slpit_unc_class',
                        'slpit_global_bd', 'slpit_global_class',
                        'slpit_rho_s_global_bd',  'slpit_rho_s_global_class',
                        'reclaimer_slpit_global_bd', 'reclaimer_slpit_global_bd_prime',
                        'slpit_rock_bd', 'slpit_rock_class',
                        'slpit_rho_s_rock_bd', 'slpit_rho_s_rock_class',
                        'reclaimer_slpit_rock_bd', 'reclaimer_slpit_rock_bd_prime',
                        'slpit_local_bd', 'slpit_local_class',
                        'slpit_rho_s_local_bd', 'slpit_rho_s_local_class',
                        'reclaimer_slpit_local_bd', 'reclaimer_slpit_local_bd_prime',
                        'emit_unc_bd', 'emit_unc_class',
                        'emit_global_bd', 'emit_global_class',
                        'emit_rho_s_global_bd', 'emit_rho_s_global_class',
                        'reclaimer_emit_global_bd', 'reclaimer_emit_global_bd_prime',
                        'emit_rock_bd', 'emit_rock_class',
                        'emit_rho_s_rock_bd', 'emit_rho_s_rock_class',
                        'reclaimer_emit_rock_bd', 'reclaimer_emit_rock_bd_prime',
                        'emit_local_bd', 'emit_local_class',
                        'emit_rho_s_local_bd', 'emit_rho_s_local_class',
                        'reclaimer_emit_local_bd', 'reclaimer_emit_local_bd_prime',
                        'slpit_local_npv', 'slpit_local_pv', 'slpit_local_soil', 'slpit_local_shade',
                        'slpit_global_npv', 'slpit_global_pv', 'slpit_global_soil', 'slpit_global_shade',
                        'slpit_rock_npv', 'slpit_rock_pv', 'slpit_rock_soil', 'slpit_rock_shade',
                        'emit_local_npv', 'emit_local_pv', 'emit_local_soil', 'emit_local_shade',
                        'emit_global_npv', 'emit_global_pv', 'emit_global_soil', 'emit_global_shade',
                        'emit_rock_npv', 'emit_rock_pv', 'emit_rock_soil', 'emit_rock_shade',
                        ]

        df_rows = pd.DataFrame(rows)
        df_rows.columns = column_names
        df_rows.to_csv(os.path.join(self.fig_directory, 'field_results.csv'), index=False)


    def field_results(self):
        target_classes_dict = {'g1': sorted(['hematite',]),
                               'g2': sorted(['kaolinite', 'calcite', 'montmorillonite', 'illite+muscovite'])}

        df_field = pd.read_csv(os.path.join(self.fig_directory, 'field_results.csv'))
        df_field = df_field.replace([-9999, '-9999', -9999.0], np.nan)

        analysis_type = ['global', 'rock']
        markers = ['s', 'o', '^']

        bounds = np.arange(0.0, 1.1, 0.10)

        # Option A: Get 10 discrete colors from 'viridis'
        cmap = cm.get_cmap('viridis', len(bounds) - 1)

        for plot_type in ['slpit', 'emit']:
            # Set global publication styling
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.size'] = 8

            fig, axes = plt.subplots(4, 5, figsize=(9, 5.75))

            col_idx = 0

            for group in ['g1', 'g2']:
                df_select = df_field[df_field['group'] == group]
                target_classes = target_classes_dict[group]

                for target_class in target_classes:
                    df_target = df_select[df_select['cp_class'].astype(str).str.contains(target_class, regex=False)]

                    if df_target.empty:
                        col_idx += 1
                        continue

                    # Set tick locators
                    major_locator = ticker.MultipleLocator(0.10)
                    minor_locator = ticker.MultipleLocator(0.05)

                    # Add shared 1:1 reference line & common axis limits across all rows for this column
                    for r in range(4):
                        ax = axes[r, col_idx]
                        ax.plot([0, 0.5], [0, 0.5], color='gray', linestyle='--', linewidth=0.8,
                                alpha=0.7, zorder=1)
                        ax.set_xlim(0, 0.5)
                        ax.set_ylim(0, 0.5)

                        # Apply major (0.10) and minor (0.05) tick spacing
                        ax.xaxis.set_major_locator(major_locator)
                        ax.xaxis.set_minor_locator(minor_locator)
                        ax.yaxis.set_major_locator(major_locator)
                        ax.yaxis.set_minor_locator(minor_locator)

                        # Ensure minor ticks are visible visually
                        ax.tick_params(which='minor', length=2, color='gray', labelsize=6)
                        ax.tick_params(which='major', length=4, labelsize=6)
                        ax.set_aspect('equal', adjustable='box')

                        #ax.spines['top'].set_visible(False)
                        #ax.spines['right'].set_visible(False)

                    legend_kwargs = dict(
                        loc='upper right',
                        fontsize=4,  # Smaller font to avoid overlapping data points
                        frameon=True,
                        facecolor='white',
                        edgecolor='grey',  # Borderless legend looks cleaner in small boxes
                        framealpha=0.7,
                        #handletextpad=0.1,
                        #borderpad=0.2,
                        #labelspacing=0.2
                    )

                    # -------------------------------------------------------------
                    # ROW 0: Uncorrected Data
                    # -------------------------------------------------------------
                    ax_row0 = axes[0, col_idx]
                    ax_row0.set_title(target_class.capitalize(), fontsize=10, fontweight='bold')

                    if col_idx == 0:
                        ax_row0.set_ylabel(f"{plot_type.upper()}\n(Uncorrected)", fontsize=8)
                    else:
                        ax_row0.set_yticklabels([])

                    ax_row0.set_xticklabels([])
                    metrics = evaluate_single_class(
                        y_true_raw=df_target['cp_class'],
                        y_pred_raw=df_target[f'{plot_type}_unc_class'],
                        target_class=target_class
                    )

                    df_tp = df_target[
                        df_target[f'{plot_type}_unc_class'].astype(str).str.contains(target_class, regex=False)]

                    if not df_tp.empty:
                        r2, mae = calc_clean_metrics(df_tp['cp_bd'].values, df_tp[f'{plot_type}_unc_bd'].values)
                        frac = metrics['fraction_str']  # e.g., "4/7"
                        ## lbl = f"(Global) F1: {metrics['f1_score']:.2f} ({frac})\nR²: {r2:.2f}\nMAE: {mae:.2f}"
                        lbl = f"(U) R²: {r2:.2f} | MAE: {mae:.2f}"
                        soil_fraction = df_tp[f'{plot_type}_global_soil']
                        ax_row0.scatter(df_tp['cp_bd'], df_tp[f'{plot_type}_unc_bd'],
                                        label=lbl, c=soil_fraction, cmap=cmap,
                                vmin=0, vmax=1, marker='o', s=8, alpha=0.8, zorder=2
                            )

                        ax_row0.legend(**legend_kwargs)
                    # -------------------------------------------------------------
                    # ROW 1: Corrected Data (Pre-Tetracorder); pvfs
                    # -------------------------------------------------------------
                    ax_row1 = axes[2, col_idx]
                    if col_idx == 0:
                        ax_row1.set_ylabel(f"{plot_type.upper()}\n(ρ$_vfs$)", fontsize=8)
                    else:
                        ax_row1.set_yticklabels([])
                    ax_row1.set_xticklabels([])
                    for _i, i in enumerate(analysis_type):
                        metrics = evaluate_single_class(
                            y_true_raw=df_target['cp_class'],
                            y_pred_raw=df_target[f'{plot_type}_{i}_class'],
                            target_class=target_class
                        )

                        df_tp_corr = df_target[df_target[f'{plot_type}_{i}_class'].astype(str).str.contains(target_class, regex=False)]

                        if not df_tp_corr.empty:
                            r2, mae = calc_clean_metrics(df_tp_corr['cp_bd'].values,
                                                         df_tp_corr[f'{plot_type}_{i}_bd'].values)
                            frac = metrics['fraction_str']  # e.g., "4/7"
                            #lbl = f"{i}: F1: {metrics['f1_score']:.2f} ({frac})\nR²: {r2:.2f}\nMAE: {mae:.2f}"

                            if i == 'rock':
                                lbl_type = "U'"
                            else:
                                lbl_type = 'U'

                            lbl = f"({lbl_type}) R²: {r2:.2f} | MAE: {mae:.2f}"
                            soil_fraction = df_tp_corr[f'{plot_type}_{i}_soil'].values
                            ax_row1.scatter(
                                df_tp_corr['cp_bd'], df_tp_corr[f'{plot_type}_{i}_bd'],
                                label=lbl, c=soil_fraction, cmap=cmap,
                                vmin=0, vmax=1, marker=markers[_i], s=8, alpha=0.8, zorder=2
                            )

                    ax_row1.legend(**legend_kwargs)
                    # -------------------------------------------------------------
                    # ROW 2: Corrected Data (rho_s) ; this pvf
                    # -------------------------------------------------------------
                    ax_row2 = axes[1, col_idx]
                    if col_idx == 0:
                        ax_row2.set_ylabel(f"{plot_type.upper()}\n(ρ$_vf$)", fontsize=8)
                    else:
                        ax_row2.set_yticklabels([])

                    ax_row2.set_xticklabels([])
                    for _i, i in enumerate(analysis_type):
                        metrics = evaluate_single_class(
                            y_true_raw=df_target['cp_class'],
                            y_pred_raw=df_target[f'{plot_type}_rho_s_{i}_class'],
                            target_class=target_class
                        )

                        df_tp_corr = df_target[df_target[f'{plot_type}_{i}_class'].astype(str).str.contains(target_class, regex=False)]

                        if not df_tp_corr.empty:
                            r2, mae = calc_clean_metrics(df_tp_corr['cp_bd'].values,
                                                         df_tp_corr[f'{plot_type}_rho_s_{i}_bd'].values)
                            frac = metrics['fraction_str']  # e.g., "4/7"
                            #lbl = (f"{i}: F1: {metrics['f1_score']:.2f} ({frac})\n" f"R²: {r2:.2f}\n" f"MAE: {mae:.2f}")
                            if i == 'rock':
                                lbl_type = "U'"
                            else:
                                lbl_type = 'U'

                            lbl = f"({lbl_type}) R²: {r2:.2f} | MAE: {mae:.2f}"
                            soil_fraction = df_tp_corr[f'{plot_type}_{i}_soil'].values
                            ax_row2.scatter(
                                df_tp_corr['cp_bd'], df_tp_corr[f'{plot_type}_rho_s_{i}_bd'],
                                label=lbl, c=soil_fraction, cmap=cmap,
                                vmin=0, vmax=1, marker=markers[_i], s=8, alpha=0.8, zorder=2
                            )

                    ax_row2.legend(**legend_kwargs)
                    # -------------------------------------------------------------
                    # ROW 3: Corrected Data (RECLAIMER post-Tetracorder)
                    # -------------------------------------------------------------
                    ax_row3 = axes[3, col_idx]
                    if col_idx == 0:
                        ax_row3.set_ylabel(f"{plot_type.upper()}\n(RECLAIMER)", fontsize=8)
                    else:
                        ax_row3.set_yticklabels([])

                    for _i, i in enumerate(analysis_type):
                        metrics = evaluate_single_class(
                            y_true_raw=df_target['cp_class'],
                            y_pred_raw=df_target[f'{plot_type}_{i}_class'],
                            target_class=target_class
                        )

                        df_tp_corr = df_target[
                            df_target[f'{plot_type}_{i}_class'].astype(str).str.contains(target_class, regex=False)]

                        if not df_tp_corr.empty:
                            r2, mae = calc_clean_metrics(df_tp_corr['cp_bd'].values,
                                                         df_tp_corr[f'reclaimer_{plot_type}_{i}_bd_prime'].values)
                            frac = metrics['fraction_str']  # e.g., "4/7"
                            #lbl = f"{i}: F1: {metrics['f1_score']:.2f} ({frac})\nR²: {r2:.2f}\nMAE: {mae:.2f}"
                            if i == 'rock':
                                lbl_type = "U'"
                            else:
                                lbl_type = 'U'

                            lbl = f"({lbl_type}) R²: {r2:.2f} | MAE: {mae:.2f}"

                            soil_fraction = df_tp_corr[f'{plot_type}_{i}_soil'].values
                            mappable = ax_row3.scatter(
                                df_tp_corr['cp_bd'], df_tp_corr[f'reclaimer_{plot_type}_{i}_bd_prime'],
                                label=lbl, c=soil_fraction, cmap=cmap,
                                vmin=0, vmax=1, marker=markers[_i], s=8, alpha=0.8, zorder=2
                            )

                    ax_row3.set_xlabel("Contact Probe (Bd)", fontsize=8, labelpad=4)
                    ax_row3.legend(**legend_kwargs)

                    col_idx += 1

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(mappable, cax=cbar_ax, ticks=bounds)
            cbar.set_label('Soil Fraction', fontsize=8)
            cbar.ax.tick_params(labelsize=6)

            # Adjust layout so the figure margins leave room for the colorbar
            fig.subplots_adjust(right=0.90)

            plt.savefig(os.path.join(self.fig_directory, f'{plot_type}_field_regression.png'), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

    def scene_figures(self):
        scence_directories = sorted(list(glob(os.path.join(self.output_directory, 'scenes', '**'))))
        target_classes_dict = {
            'g1': sorted(['hematite', 'goethite']),
            'g2': sorted(['kaolinite', 'illite', 'calcite', 'dolomite', 'montmorillonite', 'illite+muscovite'])
        }
        group_dict = {'g1': 1, 'g2': 3}
        bd_dict = {'g1': 0, 'g2': 2}
        wvls, fwhm = spectra.load_wavelengths(sensor='emit')

        dataset_types = [
            {
                'type': 'global',
                'suffix': 'global',
                'frc_key': 'glb',
                'title_suffix': '(U)',
                'datasets': [
                    {"pattern": "recon_rho_*_min", "title": r"$\hat{\rho}_{vf}$", 'index_pattern': '*reflect*minerals*',
                     'type': 'global'},
                    {"pattern": "ext_veg_*_min", "title": r"$\hat{\rho}_{vfs}$", 'index_pattern': '*reflect*minerals*',
                     'type': 'global'},
                ]
            },
            {
                'type': 'rock',
                'suffix': 'rock',
                'frc_key': 'rock',
                'title_suffix': "(U')",
                'datasets': [
                    {"pattern": "recon_rho_*_min", "title": r"$\hat{\rho}'_{vf}$",
                     'index_pattern': '*reflect*minerals*', 'type': 'rock'},
                    {"pattern": "ext_veg_*_min", "title": r"$\hat{\rho}'_{vfs}$", 'index_pattern': '*reflect*minerals*',
                     'type': 'rock'},
                ]
            }
        ]

        for scene_directory in scence_directories:
            date = os.path.basename(scene_directory)
            if date == 'outlogs':
                continue

            # 1. Load Bad Pixel Mask
            mask_file = sorted(list(glob(os.path.join(self.slpit_output_directory, '..', 'gis', 'emit-data',
                                                      'products', date, 'L2A', f'EMIT_L2A_*_mask'))))[-1]
            mask = envi_to_array(mask_file)[:, :, -1]
            is_bad_pixel = (mask == 1)

            # 2. Load Raw Image Arrays
            unc_img = envi_to_array(glob(os.path.join(scene_directory, 'tc_unc', '*_min'))[0])
            rgb_file = glob(os.path.join(self.slpit_output_directory, '..', 'gis', 'emit-data',
                                         'products', date, 'L2A', f'EMIT_L2A_*_reflectance'))[0]
            rgb_img = envi_to_array(rgb_file)

            frc_dict = {
                'glb': envi_to_array(glob(os.path.join(scene_directory, 'emc2', f'*global_lib_*_fractional_cover'))[0]),
                'rock': envi_to_array(
                    glob(os.path.join(scene_directory, 'emc2', f'*global_rock_*_fractional_cover'))[0])
            }

            # 3. Apply Mask Across RGB and FRC Arrays
            rgb_img_masked = apply_mask(rgb_img, is_bad_pixel)
            rgb_display = prep_emit_rgb(rgb_img_masked)
            rgb_display[is_bad_pixel] = 1.0

            frc_plots = {}
            for k, arr in frc_dict.items():
                plot_arr = arr[:, :, :3].copy()
                plot_arr[is_bad_pixel] = 1.0
                frc_plots[k] = plot_arr

            # 4. Load Mineral Classification for Uncorrected Data
            mineral_class_unc, _ = spectra.get_mineral_reclassification(
                path_to_tetracorder_minerals=glob(os.path.join(scene_directory, '**', '*_reflectance_minerals'))[0]
            )

            mineral_to_ids = {}
            for mineral_id, min_list in mineral_class_unc.items():
                for name in min_list:
                    clean_name = name.lower().strip()
                    mineral_to_ids.setdefault(clean_name, []).append(mineral_id)

            # 5. Iterate Over Mineral Groups & Target Classes
            for group in ['g1', 'g2']:
                minerals_to_map = target_classes_dict[group]

                for mineral in minerals_to_map:
                    target_ids = mineral_to_ids.get(mineral.lower().strip())

                    if not target_ids:
                        print(f"Skipping {mineral}: No detections found in scene {date}.")
                        continue

                    current_cmap = plt.cm.viridis.copy()
                    current_cmap.set_bad(color='white')

                    for cfg in dataset_types:
                        run_datasets = cfg['datasets']

                        # --- COMPUTE GLOBAL VMIN/VMAX ---
                        all_bd_values = []

                        unc_bd = unc_img[:, :, bd_dict[group]]
                        valid_unc_bd = unc_bd[~is_bad_pixel & ~np.isnan(unc_bd) & (unc_bd > 0)]
                        if len(valid_unc_bd) > 0:
                            all_bd_values.append(valid_unc_bd)

                        for ds in run_datasets:
                            f_match = glob(os.path.join(scene_directory, f'tc_{ds["type"]}', ds["pattern"]))
                            if f_match:
                                raw_arr = envi_to_array(f_match[0])
                                band_a = raw_arr[:, :, group_dict[group]]
                                band_b = raw_arr[:, :, bd_dict[group]]

                                samples_a = band_a[~np.isnan(band_a)]
                                if len(samples_a) > 0 and np.all(np.mod(samples_a, 1) == 0):
                                    bd_img = band_b
                                else:
                                    bd_img = band_a

                                valid_ds_bd = bd_img[~is_bad_pixel & ~np.isnan(bd_img) & (bd_img > 0)]
                                if len(valid_ds_bd) > 0:
                                    all_bd_values.append(valid_ds_bd)

                        if len(all_bd_values) > 0:
                            concat_vals = np.concatenate(all_bd_values)
                            vmin = float(np.nanmin(concat_vals))
                            vmax = float(np.nanmax(concat_vals))
                        else:
                            vmin, vmax = 0.0, 1.0

                        shared_norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

                        # --- CONSTRUCT 2x4 DASHBOARD WITH INCREASED HORIZONTAL SPACING ---
                        fig = plt.figure(figsize=(15.5, 6.5))
                        gs = GridSpec(nrows=2, ncols=4, figure=fig, hspace=0.28, wspace=0.38)

                        active_cs = None

                        # ==========================================
                        # TOP ROW: RGB & FRACTIONAL COVER (Cols 1 & 2)
                        # ==========================================
                        ax_rgb = fig.add_subplot(gs[0, 1])
                        ax_rgb.imshow(rgb_display, aspect='auto')
                        ax_rgb.set_box_aspect(1)
                        ax_rgb.set_title(f"EMIT RGB Overview: {date}", fontsize=8, fontweight='bold', pad=4)

                        ax_frc = fig.add_subplot(gs[0, 2])
                        ax_frc.imshow(frc_plots[cfg['frc_key']], aspect='auto')
                        ax_frc.set_box_aspect(1)
                        ax_frc.set_title(f"Fractional Cover {cfg['title_suffix']}", fontsize=8, pad=4)

                        # ==========================================
                        # BOTTOM ROW: UNCORRECTED, CORRECTED, SPECTRA
                        # ==========================================
                        # 1. Uncorrected Scene (Col 0)
                        ax_unc = fig.add_subplot(gs[1, 0])

                        cs_unc, unc_detection_mask, unc_detects = plot_mineral_overlay(
                            ax=ax_unc,
                            img_array=unc_img,
                            group_idx=group_dict[group],
                            bd_idx=bd_dict[group],
                            target_ids=target_ids,
                            title=f"Uncorrected: {mineral.capitalize()}",
                            is_bad_pixel=is_bad_pixel,
                            cmap=current_cmap,
                            norm=shared_norm,
                            levels=15,
                        )
                        ax_unc.set_aspect('auto')
                        ax_unc.set_box_aspect(1)
                        ax_unc.set_title(f"Uncorrected: {mineral.capitalize()}", fontsize=8, pad=4)

                        if cs_unc is not None:
                            active_cs = cs_unc

                        non_corrected_rfl = (
                            rgb_img[unc_detection_mask, :]
                            if np.any(unc_detection_mask)
                            else None
                        )

                        # 2. Corrected Scenes (Cols 1 & 2)
                        ds_spectrum_data = []

                        for idx, ds in enumerate(run_datasets):
                            mineral_reclass_pattern = os.path.join(scene_directory, f'tc_{ds["type"]}', '**',
                                                                   ds["index_pattern"])
                            target_ids_mineral = get_target_ids_for_dataset(mineral_reclass_pattern, mineral)
                            file_match = glob(os.path.join(scene_directory, f'tc_{ds["type"]}', ds["pattern"]))

                            ax_sub_map = fig.add_subplot(gs[1, idx + 1])

                            if file_match:
                                raw_array = envi_to_array(file_match[0])
                                cs_sub, valid_detection_mask, n_detects = plot_mineral_overlay(
                                    ax=ax_sub_map,
                                    img_array=raw_array,
                                    group_idx=group_dict[group],
                                    bd_idx=bd_dict[group],
                                    target_ids=target_ids_mineral,
                                    title=f"{ds['title']}",
                                    is_bad_pixel=is_bad_pixel,
                                    cmap=current_cmap,
                                    norm=shared_norm,
                                    levels=15,
                                )
                                if cs_sub is not None:
                                    active_cs = cs_sub

                                if np.any(valid_detection_mask):
                                    ds_spectrum_data.append({
                                        "title": ds["title"],
                                        "mask": valid_detection_mask,
                                        "n": n_detects
                                    })
                            else:
                                ax_sub_map.text(0.5, 0.5, "File Missing", ha="center", va="center", fontsize=6)

                            ax_sub_map.set_aspect('auto')
                            ax_sub_map.set_box_aspect(1)
                            ax_sub_map.set_title(f"{ds['title']}", fontsize=7.5, pad=3)

                        # Colorbar attached to the Uncorrected Map with tighter padding & specific label positioning
                        divider = make_axes_locatable(ax_sub_map)
                        cax = divider.append_axes("right", size="4%", pad=0.03)

                        if active_cs is not None:
                            cbar = fig.colorbar(
                                active_cs,
                                cax=cax,
                                orientation='vertical',
                                ticks=np.linspace(vmin, vmax, 5)
                            )
                        else:
                            sm = plt.cm.ScalarMappable(norm=shared_norm, cmap=current_cmap)
                            sm.set_array([])
                            cbar = fig.colorbar(
                                sm,
                                cax=cax,
                                orientation='vertical',
                                ticks=np.linspace(vmin, vmax, 5)
                            )

                        cbar.set_label("Band Depth", fontsize=6.5, labelpad=2)
                        cbar.ax.tick_params(labelsize=5.5, pad=2)
                        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

                        # 3. Reflectance Spectrum Subplot (Col 3)
                        ax_spec = fig.add_subplot(gs[1, 3])
                        line_colors = ['#1f77b4', '#2ca02c']

                        if non_corrected_rfl is not None and len(non_corrected_rfl) > 0:
                            non_mean_rfl = np.nanmean(non_corrected_rfl, axis=0)
                            non_mean_rfl[non_mean_rfl < 0] = np.nan
                            non_std_rfl = np.nanstd(non_corrected_rfl, axis=0)

                            ax_spec.plot(
                                wvls,
                                non_mean_rfl,
                                color="red",
                                linewidth=1.0,
                                linestyle="--",
                                label=f"Uncorrected (n={unc_detects})",
                            )
                            ax_spec.fill_between(
                                wvls,
                                non_mean_rfl - non_std_rfl,
                                non_mean_rfl + non_std_rfl,
                                color="red",
                                alpha=0.15,
                            )

                        for idx, spec_info in enumerate(ds_spectrum_data):
                            if idx == 0:
                                rfl_corrected_array = envi_to_array(glob(os.path.join(self.output_directory, 'scenes', date, f'tc_{cfg['type']}', f'*ext_veg_EMIT_L2A_RFL_*_tc'))[0])
                            else:
                                rfl_corrected_array = envi_to_array(sorted(glob(os.path.join(self.output_directory, 'scenes', date, f'tc_{cfg['type']}', f'*recon_rho_EMIT_L2A_RFL_*_reflectance_global_*')))[0])

                            detected_rfl = rfl_corrected_array[spec_info["mask"], :]
                            mean_rfl = np.nanmean(detected_rfl, axis=0)
                            mean_rfl[mean_rfl < 0] = np.nan
                            std_rfl = np.nanstd(detected_rfl, axis=0)

                            color = line_colors[idx % len(line_colors)]
                            ax_spec.plot(
                                wvls,
                                mean_rfl,
                                color=color,
                                linewidth=1.0,
                                label=f"{spec_info['title']} (n={spec_info['n']})",
                            )
                            ax_spec.fill_between(
                                wvls,
                                mean_rfl - std_rfl,
                                mean_rfl + std_rfl,
                                color=color,
                                alpha=0.15,
                            )

                        ax_spec.set_xlabel("Wavelength (nm)", fontsize=7, labelpad=3)
                        ax_spec.set_ylabel("Reflectance (%)", fontsize=7, labelpad=3)
                        ax_spec.set_ylim(0, 1)
                        ax_spec.grid(True, linestyle="--", alpha=0.4)
                        ax_spec.tick_params(axis="both", labelsize=6, pad=2)

                        ax_spec.legend(
                            loc="upper right",
                            fontsize=5.0,
                            ncol=1,
                            frameon=True,
                            handlelength=1.0,
                            borderpad=0.2,
                        )

                        # --- CLEANUP AXES TICKS & SPINES ---
                        for ax in fig.get_axes():
                            if ax != ax_spec and ax != cbar.ax:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                for spine in ax.spines.values():
                                    spine.set_visible(True)
                                    spine.set_color('black')
                                    spine.set_linewidth(0.8)

                        plt.savefig(
                            os.path.join(self.fig_directory, 'scenes', f'{mineral}_{cfg["suffix"]}_{date}.png'),
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1
                        )

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
    #tc.band_depth_mae()
    #tc.field_table()
    #tc.field_results()
    tc.scene_figures()