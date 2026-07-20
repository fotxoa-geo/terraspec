import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.envi import envi_to_array, load_band_names
import os
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
import spectral.io.envi as envi
from emit_utils.file_checks import envi_header
from utils.spectra_utils import spectra
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.preprocessing import MultiLabelBinarizer



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
                        ax.set_title(f"{class_name.capitalize()}", fontsize=10)

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
                            ax.set_ylabel(f"{r'$\rho$'}\nCount", fontsize=8)
                        elif row_idx == 1:
                            ax.set_ylabel(f"{r'$\rho$'}'\nCount", fontsize=8)
                        else:
                            ax.set_ylabel(f"{r'$\rho$'}''\nCount", fontsize=8)

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

                        ax.set_xlabel('Soil Fractional Cover', fontsize=8)

            # Grab handles from the last active axis
            handles, labels = ax.get_legend_handles_labels()

            # Place the single legend at the bottom center
            fig.legend(handles, labels,
                       loc='lower center',
                       bbox_to_anchor=(0.5, -0.08),  # Anchored right above the figure bottom margin
                       ncol=4,  # 2x2 layout is perfect for a 6.5" width
                       fontsize=8,
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
        sim_labels = [f"{r'$\rho$'}", f"{r'$\rho$'}'", f"{r'$\rho$'}''"]

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
                ax.set_title(f"{class_name.capitalize()}", fontsize=10, fontweight='semibold')
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
                    ax.set_ylabel("F1 Score", fontsize=9)
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
        sim_labels = [f"{r'$\rho$'} (Uncorrected)", f"{r'$\rho$'}' (Corrected before TC run)", f"{r'$\rho$'}'' (Corrected before TC run)", f"{r'post-$\rho$'}' (Corrected after TC Run)", f"{r'post-$\rho$'}'' (Corrected after TC Run)", f"{r'$\rho$'}''' (best case scenario; known fractions)"]
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

    tc.classification_rates()
    tc.f1_score_matrix_detailed()
    tc.band_depth_mae()
