import os
from utils.create_tree import create_directory
import numpy as np
from utils.envi import envi_to_array
from spectral.io import envi
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from utils.results_utils import r2_calculations


class figures:
    def __init__(self, base_directory: str, aoi: str, sensor: str):

        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, 'output')
        self.fig_directory = os.path.join(base_directory, "figures")
        self.gis_directory = os.path.join(base_directory, "gis")
        self.time_series_directory = os.path.join(self.output_directory, 'time_series')

        create_directory(os.path.join(base_directory, "figures"))

        # instrument to indicate wavelengths in output folder
        self.instrument = sensor
        self.aoi = aoi

    def chronos(self):
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

        FREQ = '30D'


        for _em, em in enumerate(['npv', 'pv', 'soil']):
            fractional_cover = os.path.join(self.time_series_directory,
                                         f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_time_series')

            meta = envi.read_envi_header(f'{fractional_cover}.hdr')

            frac_array = envi_to_array(fractional_cover)

            num_rows, num_cols = frac_array.shape[0], frac_array.shape[1]

            random_row = np.random.randint(0, num_rows)
            random_col = np.random.randint(0, num_cols)

            pixel_time_series = frac_array[random_row, random_col, :]
            pixel_time_series[pixel_time_series == -9999. ] = np.nan
            non_nan_mask = ~np.isnan(pixel_time_series)
            filtered_series = pixel_time_series[non_nan_mask]

            time_points = meta['band names']

            dates = [datetime.strptime(d, '%Y%m%dT%H%M%S') for d in time_points]
            filtered_dates = np.array(dates)[non_nan_mask]

            fire_date = datetime(2024, 7, 4)  # Example: June 15, 2023

            # 1. Create Initial DataFrames
            full_df = pd.DataFrame({
                "timestamp": filtered_dates,
                "target": filtered_series
            })
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

            # 2. Resample to fix irregular dates
            # We resample the entire series first to ensure a continuous grid
            full_df = full_df.set_index("timestamp")['target'].resample(FREQ).mean().to_frame()

            full_df["target"] = full_df["target"].interpolate(method="linear")
            full_df = full_df.reset_index()
            full_df["id"] = f"{em}_pixel"  # Add the string ID back AFTER the math is done

            # 3. Split into Context and Future based on the fire_date
            fire_date = pd.to_datetime("2024-07-04")
            context_df = full_df[full_df["timestamp"] < fire_date].copy()
            test_df = full_df[full_df["timestamp"] >= fire_date].copy()

            # 4. Prepare the input for Chronos (drop target from future)
            future_df_input = test_df.drop(columns=["target"]).copy()

            # 5. Run Prediction
            # Ensure we don't predict more than what our future_df_input allows
            pred_df = pipeline.predict_df(
                context_df,
                future_df=future_df_input,
                prediction_length=len(future_df_input),
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            # 6. Visualization
            plt.figure(figsize=(14, 6))

            # Set indices for plotting
            ts_context = context_df.set_index("timestamp")["target"]
            ts_ground_truth = test_df.set_index("timestamp")["target"]
            ts_pred = pred_df.set_index("timestamp")

            # Plotting
            ts_context.tail(100).plot(label="Pre-Fire Context", color="xkcd:azure", lw=2)
            ts_ground_truth.plot(label="Actual Post-Fire (Ground Truth)", color="xkcd:grass green", lw=2)

            # Chronos uses strings/floats for quantile columns in the result
            ts_pred["0.5"].plot(label="Forecast (No-Fire Scenario)", color="xkcd:violet", ls="--")

            plt.fill_between(
                ts_pred.index,
                ts_pred["0.1"],
                ts_pred["0.9"],
                alpha=0.2,
                label="80% Prediction Interval",
                color="xkcd:light lavender",
            )

            plt.axvline(x=fire_date, color='red', ls=':', label='Fire Event', lw=2)
            plt.title(f"Impact Analysis for Endmember: {em.upper()}", fontsize=14)
            plt.ylabel("Fractional Cover")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            #plt.show()
            plt.clf()
            plt.close()


    def pixel_example_ts(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        plt.subplots_adjust(hspace=0.3)

        for _em, em in enumerate(['npv', 'pv', 'soil']):
            fractional_cover = os.path.join(self.time_series_directory,
                                         f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_time_series')
            meta = envi.read_envi_header(f'{fractional_cover}.hdr')

            frac_array = envi_to_array(fractional_cover)

            num_rows, num_cols = frac_array.shape[0], frac_array.shape[1]

            random_row = np.random.randint(0, num_rows)
            random_col = np.random.randint(0, num_cols)

            pixel_time_series = frac_array[random_row, random_col, :]
            pixel_time_series[pixel_time_series == -9999. ] = np.nan
            non_nan_mask = ~np.isnan(pixel_time_series)
            filtered_series = pixel_time_series[non_nan_mask]

            time_points = meta['band names']

            dates = [datetime.strptime(d, '%Y%m%dT%H%M%S') for d in time_points]
            filtered_dates = np.array(dates)[non_nan_mask]

            ax = axes[_em]
            ax.plot(filtered_dates, filtered_series, marker='o', markersize=4, linestyle='-', label=em.upper())

            fire_date = datetime(2024, 7, 4)  # Example: June 15, 2023
            pre_fire_mask = filtered_dates < fire_date
            post_fire_mask = filtered_dates >= fire_date

            pre_fc = filtered_series[pre_fire_mask]
            post_fc = filtered_series[post_fire_mask]

            baseline_median = np.median(pre_fc)
            baseline_std = np.std(pre_fc)

            stress_threshold = baseline_median - baseline_std
            failure_threshold = baseline_median * 0.90

            stress_idx = np.where(pre_fc < stress_threshold)[0]
            failure_idx = np.where(pre_fc < failure_threshold)[0]

            if len(stress_idx) > 0 and len(failure_idx) > 0:
                # Resistance Duration = time between first stress and first failure
                res_dur = failure_idx[0] - stress_idx[0]

                # Resistance Magnitude = total deviation during that "straining" period
                res_mag = np.sum(pre_fc[stress_idx[0]:failure_idx[0]] - baseline_median)
            else:
                res_dur, res_mag = 0, 0

            post_dev = post_fc - baseline_median

            # How many years was it significantly low after the fire?
            sig_post_mask = post_fc < stress_threshold

            post_dur = np.sum(sig_post_mask)  # Duration (years)
            post_mag = np.sum(post_dev[sig_post_mask])  # Magnitude (sum of depth)

            # Add the Fire Line
            ax.axvline(x=fire_date, color='green', linestyle='-', linewidth=2, label='Lakefire')


            ax.axhline(baseline_median, color='black', label='Median')
            ax.axhline(stress_threshold, color='orange', linestyle='--', label='Stress (1-std)')
            ax.axhline(failure_threshold, color='red', linestyle='--', label='Failure (10%)')

            # Shading the impact post-fire
            ax.fill_between(filtered_dates[post_fire_mask], baseline_median, post_fc,
                            where=(post_fc < stress_threshold), color='red', alpha=0.3)

            ax.set_ylabel(f'Fractional Cover')
            ax.set_title(f'Time Series: {em.upper()}')
            ax.grid(True, linestyle='--', alpha=0.7)

            #
            ax.legend(loc='upper right')

        plt.gcf().autofmt_xdate()  # Tilts dates for readability
        plt.xlabel('Date (UTC)')

        # Save the figure
        plt.savefig(os.path.join(self.fig_directory, 'endmember_time_series.png'), bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()

    def sam_vs_z_score(self):
        import rasterio
        from rasterio import features
        from scipy.stats import gaussian_kde

        fig, axes = plt.subplots(3, 3, figsize=(10, 12), sharex=True, constrained_layout=True)

        gdf = gpd.read_file(os.path.join('objects', 'lake_fire_approx_perimeter_v080124.kmz'))

        em_key = {0: 'npv', 1: 'pv', 2: 'soil'}

        for row in range(3):
            for col in range(3):

                em = em_key[col]

                fractional_cover_scores = envi_to_array(os.path.join(self.time_series_directory,
                                                f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_scores'))
                ax = axes[row, col]


                ax.set_xlim(-5, 5)

                ax.set_box_aspect(1)

                with rasterio.open(os.path.join(self.time_series_directory,
                                                f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_scores')) as src:
                    out_shape = src.shape  # (rows, cols)
                    transform = src.transform  # Affine transform
                    raster_crs = src.crs

                mask = features.rasterize(
                    shapes=gdf.geometry,
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,  # Value for pixels outside the shapes
                    default_value=1,  # Value for pixels inside the shapes
                    all_touched=True,
                    dtype='uint8'
                )

                z_flat = fractional_cover_scores[:,:,0].flatten()
                sam_flat = fractional_cover_scores[:,:,2].flatten()
                mask_flat = mask.flatten()

                if row == 0:
                    valid_indices = (mask_flat == 1) & (z_flat != -9999.0) & (sam_flat != -9999.0)
                    row_label = 'Burned Areas'


                if row == 1:
                    valid_indices = (mask_flat != 1) & (z_flat != -9999.0) & (sam_flat != -9999.0)
                    row_label = 'Non-Burned Areas'

                if row == 2:
                    valid_indices = (z_flat != -9999.0) & (sam_flat != -9999.0)
                    row_label = 'All Pixels'
                    ax.set_xlabel(f'Z-Score')


                z_scores_filtered = z_flat[valid_indices]
                sam_vals_filtered = sam_flat[valid_indices]

                xy = np.vstack([z_scores_filtered, sam_vals_filtered])
                density = gaussian_kde(xy)(xy)
                d_min = density.min()
                d_max = density.max()
                norm_density = (density - d_min) / (d_max - d_min)
                idx_sort = norm_density.argsort()
                x_plot, y_plot, c_plot = z_scores_filtered[idx_sort], sam_vals_filtered[idx_sort], norm_density[idx_sort]
                sc = ax.scatter(x_plot, y_plot, c=c_plot, s=2, cmap='viridis', alpha=0.6, vmin=0, vmax=1)

                r2, bias = r2_calculations(z_scores_filtered, sam_vals_filtered)
                txtstr = f'R$^2$: {r2:.2f}'
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=7, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

                ax.set_title(f'{em.upper()}')
                if col == 0:
                    ax.set_ylabel(f'Spectral Angle\n({row_label})')

        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, location='right')
        cbar.set_label('Normalized Point Density')

        plt.savefig(os.path.join(self.fig_directory, 'sam_vs_z-score.png'), bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()


def run_figures(base_directory, sensor, aoi):
    base_directory = base_directory
    fig = figures(base_directory=base_directory, sensor=sensor, aoi=aoi)
    fig.pixel_example_ts()
    fig.chronos()
    fig.sam_vs_z_score()


