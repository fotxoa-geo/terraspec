import os
from utils.create_tree import create_directory
from datetime import datetime
from utils.results_utils import r2_calculations
from glob import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons
from scipy.stats import gaussian_kde
from spectral.io import envi
from utils.envi import envi_to_array
from utils.spectra_utils import spectra


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

        plt.savefig(os.path.join(self.fig_directory, f'sam_vs_z-score_{self.instrument}.png'), bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()

    def fractional_cover_distributions(self):
        enmap_images = sorted(glob(os.path.join(self.output_directory, 'EnMAP_to_EMIT_emc2', '*_fractional_cover')))
        enmap_timestamps = sorted(list(set([os.path.basename(item).split('_')[5].lower() for item in enmap_images])))
        enmap_days = set([t[:8] for t in enmap_timestamps])

        mask_files = sorted(list(glob(os.path.join(self.gis_directory, 'emit-data', 'aoi', 'sedgwick_boundary_approx', 'EXT', '*_MASK_*_EXT'))))
        for _em, em in enumerate(['npv', 'pv', 'soil']):
            rows = 0
            cols = 0

            mask_files_update = []

            for _i, i in enumerate(mask_files):
                try:
                    mask_array = envi_to_array(i)
                    rows = max(rows, mask_array.shape[0])
                    cols = max(cols, mask_array.shape[1])
                    mask_files_update.append(i)

                except:
                    print(f'{i} could not be read!')
                    continue

            print(f"The largest dimensions are: {rows} x {cols}")

            band_names = []

            for _i, i in enumerate(mask_files_update):
                overpass = i.split('_')[-2]
                date = overpass[:8]
                band_names.append(date)

            emit_overpass_dates = sorted(list(set(band_names)))

            for i in emit_overpass_dates:
                emit_dt = datetime.strptime(i, "%Y%m%d")
                closest_enmap_day = min(
                    enmap_days,
                    key=lambda x: abs(datetime.strptime(x, "%Y%m%d") - emit_dt)
                )

                closest_dt = datetime.strptime(closest_enmap_day, "%Y%m%d")
                day_difference = abs((emit_dt - closest_dt).days)

                if day_difference > 5:
                    continue

                emit_fractional_cover = glob(os.path.join(self.output_directory, 'emc2', f'*_{i}*_fractional_cover'))
                enmap_fractional_cover = sorted(glob(os.path.join(self.output_directory, 'EnMAP_to_EMIT_emc2', f'*_{closest_enmap_day}*_fractional_cover')))

                fraction_grid = np.ones((rows, cols, 2)) * -9999. # first band is emit, second is enmap

                for x in emit_fractional_cover:
                    mask_datetime = os.path.basename(x).split('_')[4]
                    current_array = envi_to_array(x)[:, :, _em]
                    mask_array = envi_to_array(glob(os.path.join(self.gis_directory, 'emit-data', 'aoi', 'sedgwick_boundary_approx', 'EXT', f'*_MASK_{mask_datetime}_EXT'))[0])

                    current_array[mask_array[:, :, -1] == 1] = np.nan
                    r, c = current_array.shape
                    grid_slice = fraction_grid[:r, :c, 0]
                    valid_mask = ~np.isnan(current_array)
                    grid_slice[valid_mask] = current_array[valid_mask]

                for x in enmap_fractional_cover:
                    current_array = envi_to_array(x)[:, :, _em]
                    current_array[current_array == -9999.] = np.nan
                    r, c = current_array.shape
                    grid_slice = fraction_grid[:r, :c, 1]
                    valid_mask = ~np.isnan(current_array)
                    grid_slice[valid_mask] = current_array[valid_mask]

                emit_vals = fraction_grid[:, :, 0].flatten()
                enmap_vals = fraction_grid[:, :, 1].flatten()

                # 2. Filter out both the -9999. background and NaN values for each band
                emit_valid = emit_vals[(emit_vals != -9999.) & (~np.isnan(emit_vals))]
                enmap_valid = enmap_vals[(enmap_vals != -9999.) & (~np.isnan(enmap_vals))]

                # 3. Plot the overlapping histograms
                plt.figure(figsize=(10, 6))
                plt.hist(emit_valid, bins=50, alpha=0.5, label=f'EMIT {i} - {em}', color='blue', edgecolor='k')
                plt.hist(enmap_valid, bins=50, alpha=0.5, label=f'EnMAP {closest_enmap_day} - {em}', color='orange', edgecolor='k')

                plt.xlabel('Fractional Cover Value')
                plt.ylabel('Pixel Count')
                plt.title('Distribution Comparison: EMIT vs EnMAP Fractional Cover')
                plt.legend(loc='upper right')
                plt.grid(axis='y', alpha=0.3)

                plt.show()
                plt.clf()
                plt.close()



    def start_plot(self):
        target_path = "/Users/fochoa/PycharmProjects/terraspec/terraspec_output/fire/output/EMIT_to_ENMAP_RFL/sedgwick_boundary_approx_RFL_20240804T181605_EXT"

        dashboard = InteractiveMatplotlibDashboard(
            time_series_directory=self.time_series_directory,
            fig_directory=self.fig_directory,
            aoi=self.aoi,
            instrument=self.instrument,
            rgb_img_path=target_path,
            r_band_idx=42,  # Sliced target index for Red
            g_band_idx=24,  # Sliced target index for Green
            b_band_idx=9,  # Sliced target index for Blue,
            fire_date_str = "20240705"
        )


class InteractiveMatplotlibDashboard:
    def __init__(self, time_series_directory, fig_directory, aoi, instrument, rgb_img_path, r_band_idx, g_band_idx,
                 b_band_idx, fire_date_str):
        self.time_series_directory = time_series_directory
        self.fig_directory = fig_directory
        self.aoi = aoi
        self.instrument = instrument

        self.rgb_img_path = rgb_img_path
        self.r_idx = r_band_idx
        self.g_idx = g_band_idx
        self.b_idx = b_band_idx

        # Parse fire timestamp
        self.fire_date = datetime.strptime(fire_date_str, '%Y%m%d')

        self.gdf = gpd.read_file(os.path.join('objects', 'lake_fire_approx_perimeter_v080124.kmz'))
        self.em_key = {0: 'npv', 1: 'pv', 2: 'soil'}

        self.scores_data = {}
        self.time_series_data = {}
        self.spatial_mask = None
        self.raw_band_names = []
        self.load_datasets()

        # Initialize selected coordinates at image center
        self.selected_row = self.scores_data['npv'].shape[0] // 2
        self.selected_col = self.scores_data['npv'].shape[1] // 2

        self.current_region = 'Burned Areas'
        self.scatter_mappings = {em: None for em in self.em_key.values()}
        self.scatter_highlights = []
        self.visibility_states = {'NPV': True, 'PV': True, 'SOIL': True}

        # Build Canvas Grid Layout (3 Rows x 4 Columns)
        self.fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 0.9])

        # Row 0: Scatters
        self.ax_scatter_0 = self.fig.add_subplot(gs[0, 0])
        self.ax_scatter_1 = self.fig.add_subplot(gs[0, 1])
        self.ax_scatter_2 = self.fig.add_subplot(gs[0, 2])
        self.scatter_axes = [self.ax_scatter_0, self.ax_scatter_1, self.ax_scatter_2]

        # Row 1: Map and Fractional Time Series
        self.ax_map = self.fig.add_subplot(gs[1, 0])
        self.ax_ts = self.fig.add_subplot(gs[1, 1:3])

        # Row 2: Reflectance Spectra Panel
        self.ax_rfl = self.fig.add_subplot(gs[2, 0:3])

        # Far-Right Control Deck Widgets
        self.ax_radio = plt.axes([0.78, 0.70, 0.14, 0.12], facecolor='wheat')
        self.radio = RadioButtons(self.ax_radio, ('Burned Areas', 'Non-Burned Areas', 'All Pixels'))
        self.radio.on_clicked(self.on_radio_toggle)

        self.ax_check = plt.axes([0.78, 0.52, 0.14, 0.12], facecolor='lightgray')
        self.checkbox = CheckButtons(self.ax_check, ('NPV', 'PV', 'SOIL'), (True, True, True))
        self.checkbox.on_clicked(self.on_checkbox_toggle)

        self.generate_hyperspectral_rgb()
        self.update_plots()

        # Connect interaction event engine
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        plt.show()

    def load_datasets(self):
        """Preloads arrays and tracks dates from tagged ENVI files."""
        ts_pattern = os.path.join(self.time_series_directory, f'*_{self.instrument}_merged_time_series')
        self.ts_files = glob.glob(ts_pattern)

        for em_idx, em in self.em_key.items():
            score_path = os.path.join(self.time_series_directory,
                                      f'{os.path.basename(self.aoi).split(".")[0]}_{em}_{self.instrument}_scores')
            self.scores_data[em] = envi_to_array(score_path)

            if self.spatial_mask is None:
                with rasterio.open(score_path) as src:
                    out_shape, transform = src.shape, src.transform
                self.spatial_mask = features.rasterize(
                    shapes=self.gdf.geometry, out_shape=out_shape, transform=transform,
                    fill=0, default_value=1, all_touched=True, dtype='uint8'
                )

            em_ts_file = [f for f in self.ts_files if f"_{em}_" in os.path.basename(f)]
            if em_ts_file:
                meta = envi.read_envi_header(f'{em_ts_file[0]}.hdr')
                self.raw_band_names = meta['band names']
                self.time_series_data[em] = envi_to_array(em_ts_file[0])

    def generate_hyperspectral_rgb(self):
        """Loads specific bands, screens out-of-bounds values, and stretches contrast."""
        if not os.path.exists(self.rgb_img_path):
            raise FileNotFoundError(f"The specified RGB path does not exist: {self.rgb_img_path}")

        img_cube = envi_to_array(self.rgb_img_path)
        r_band = img_cube[:, :, self.r_idx].astype(float)
        g_band = img_cube[:, :, self.g_idx].astype(float)
        b_band = img_cube[:, :, self.b_idx].astype(float)

        def stretch_channel(band, nodata=-9999.0):
            band[band == nodata] = np.nan
            band[(band < 0.0) | (band > 1.0)] = np.nan

            valid_mask = ~np.isnan(band)
            if not np.any(valid_mask):
                return np.zeros_like(band)

            valid_pixels = band[valid_mask]
            p2, p98 = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)

            if p98 == p2:
                stretched = np.zeros_like(band)
            else:
                stretched = (band - p2) / (p98 - p2)

            stretched = np.clip(stretched, 0.0, 1.0)
            stretched[~valid_mask] = 0.0
            return stretched

        self.rgb_background = np.dstack([stretch_channel(r_band), stretch_channel(g_band), stretch_channel(b_band)])

    def fetch_filtered_scatter_arrays(self, em):
        """Isolates data masks based on active radio UI configuration."""
        scores = self.scores_data[em]
        z_flat = scores[:, :, 0].flatten()
        sam_flat = scores[:, :, 2].flatten()
        mask_flat = self.spatial_mask.flatten()

        rows, cols = scores.shape[0], scores.shape[1]
        r_grid, c_grid = np.mgrid[0:rows, 0:cols]
        r_flat, c_flat = r_grid.flatten(), c_grid.flatten()

        if self.current_region == 'Burned Areas':
            v_idx = (mask_flat == 1) & (z_flat != -9999.0) & (sam_flat != -9999.0)
        elif self.current_region == 'Non-Burned Areas':
            v_idx = (mask_flat != 1) & (z_flat != -9999.0) & (sam_flat != -9999.0)
        else:
            v_idx = (z_flat != -9999.0) & (sam_flat != -9999.0)

        z_f, sam_f, r_f, c_f = z_flat[v_idx], sam_flat[v_idx], r_flat[v_idx], c_flat[v_idx]
        if len(z_f) > 15000:
            sub = np.random.choice(len(z_f), 15000, replace=False)
            return z_f[sub], sam_f[sub], r_f[sub], c_f[sub]
        return z_f, sam_f, r_f, c_f

    def update_plots(self):
        """Redraws scatter plot structures when properties adjust."""
        for idx, ax in enumerate(self.scatter_axes):
            ax.clear()
            em = self.em_key[idx]
            z_vals, sam_vals, r_vals, c_vals = self.fetch_filtered_scatter_arrays(em)
            self.scatter_mappings[em] = {'z': z_vals, 'sam': sam_vals, 'r': r_vals, 'c': c_vals}

            if len(z_vals) > 1:
                xy = np.vstack([z_vals, sam_vals])
                density = gaussian_kde(xy)(xy)
                c_map = (density - density.min()) / (density.max() - density.min())
                idx_sort = c_map.argsort()
                z_p, sam_p, c_p = z_vals[idx_sort], sam_vals[idx_sort], c_map[idx_sort]
            else:
                z_p, sam_p, c_p = z_vals, sam_vals, 'blue'

            ax.scatter(z_p, sam_p, c=c_p, s=3, cmap='viridis', alpha=0.6)

            if len(z_vals) > 1:
                r2, _ = r2_calculations(z_vals, sam_vals)
                ax.text(0.05, 0.95, f'R$^2$: {r2:.2f}', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            ax.set_xlim(-5, 5)
            ax.set_box_aspect(1)
            ax.set_title(f'{em.upper()}')
            ax.set_xlabel('Z-Score')
            if idx == 0:
                ax.set_ylabel(f'Spectral Angle\n({self.current_region})')

        self.draw_spatial_temporal_and_highlights()

    def helper_load_rfl_cube(self, raw_date_string, sensor_type):
        """Helper to parse raw filenames based on user file mapping rules,

        with integrated robust wavelength recovery.
        """
        date_segment = raw_date_string.split('_')[0].upper()

        try:
            if "emit" in sensor_type:
                hdr_path = os.path.join('terraspec_output', 'fire', 'output', 'EMIT_to_ENMAP_RFL',
                                        f'sedgwick_boundary_approx_RFL_{date_segment}_EXT.hdr')
                img_path = hdr_path.replace('.hdr', '')
            else:
                search_glob = os.path.join('terraspec_output', 'fire', 'output', 'EnMAP_to_Spatial_EMIT_RFL',
                                           f'*_{date_segment}Z_*')
                matching_files = sorted(list(glob.glob(search_glob)))
                if not matching_files:
                    return None, None
                img_path = matching_files[0]
                hdr_path = img_path + '.hdr' if not img_path.endswith('.hdr') else img_path
                if img_path.endswith('.hdr'):
                    img_path = img_path.replace('.hdr', '')

            header_data = envi.read_envi_header(hdr_path)
            wavelengths = []

            if 'wavelength' in header_data:
                wavelengths = header_data['wavelength']
            elif 'Wavelength' in header_data:
                wavelengths = header_data['Wavelength']
            elif 'band names' in header_data and not "emit" in sensor_type:
                try:
                    test_vals = [float(w.split()[0]) for w in header_data['band names']]
                    if len(test_vals) > 0 and 300 < np.max(test_vals) < 3000:
                        wavelengths = test_vals
                except ValueError:
                    pass

            array_cube = envi_to_array(img_path).astype(float)

            if len(wavelengths) > 0:
                wavelengths = np.array([float(w) for w in wavelengths])
                if np.max(wavelengths) < 10.0:
                    wavelengths = wavelengths * 1000.0
            else:
                wavelengths = np.array([])

            return array_cube, wavelengths
        except Exception as e:
            print(f"Error accessing reflectance asset for {date_segment} ({sensor_type}): {e}")
            return None, None

    def draw_spatial_temporal_and_highlights(self):
        """Redraws map coordinates, overlays calculated peaks, and charts reflectance curves

        with explicit 1D water vapor band masking.
        """
        # 1. Map View Update
        self.ax_map.clear()
        self.ax_map.imshow(self.rgb_background)
        self.ax_map.plot(self.selected_col, self.selected_row, color='cyan', marker='+', markersize=14,
                         markeredgewidth=2)
        self.ax_map.set_title(f"Context Location ({self.selected_row}, {self.selected_col})")
        self.ax_map.axis('off')

        # 2. Scatter Highlights Update
        for hl in self.scatter_highlights:
            try:
                hl.remove()
            except ValueError:
                pass
        self.scatter_highlights.clear()

        for idx, ax in enumerate(self.scatter_axes):
            em = self.em_key[idx]
            pixel_z = self.scores_data[em][self.selected_row, self.selected_col, 0]
            pixel_sam = self.scores_data[em][self.selected_row, self.selected_col, 2]

            if pixel_z != -9999.0 and pixel_sam != -9999.0:
                hl_plot = ax.plot(pixel_z, pixel_sam, marker='o', color='red', markersize=8, markeredgecolor='white',
                                  markeredgewidth=1.5)
                self.scatter_highlights.append(hl_plot[0])

        # 3. Time Series Profile Update
        self.ax_ts.clear()
        colors = {'npv': 'red', 'pv': 'green', 'soil': 'orange'}
        self.ax_ts.axvline(x=self.fire_date, color='black', linestyle='--', linewidth=2, label='Fire Event')

        max_pre_record = {"val": -1.0, "raw_string": None, "sensor": None}
        max_post_record = {"val": -1.0, "raw_string": None, "sensor": None}

        for em in self.em_key.values():
            if not self.visibility_states[em.upper()]:
                continue

            if em in self.time_series_data:
                profile = self.time_series_data[em][self.selected_row, self.selected_col, :]

                clean_data = []
                for raw_name, val in zip(self.raw_band_names, profile):
                    if val != -9999.0 and not np.isnan(val):
                        dt_obj = datetime.strptime(raw_name.split('_')[0].split('t')[0], '%Y%m%d')
                        sensor = raw_name.split('_')[1].lower()
                        clean_data.append((dt_obj, val, sensor, raw_name))

                if not clean_data: continue
                clean_data.sort(key=lambda x: x[0])

                x_dates = [item[0] for item in clean_data]
                y_vals = [item[1] for item in clean_data]

                self.ax_ts.plot(x_dates, y_vals, color=colors[em], linewidth=1.5, alpha=0.3)
                for item in clean_data:
                    m_shape = 'o' if 'emit' in item[2] else 's'
                    self.ax_ts.plot(item[0], item[1], marker=m_shape, color=colors[em], markersize=5, alpha=0.85)

                pre_fire_pts = [item for item in clean_data if item[0] < self.fire_date]
                post_fire_pts = [item for item in clean_data if item[0] >= self.fire_date]

                if pre_fire_pts:
                    max_pre = max(pre_fire_pts, key=lambda x: x[1])
                    self.ax_ts.plot(max_pre[0], max_pre[1], marker='^', color='darkblue', markersize=10,
                                    linestyle='None', label=f'{em.upper()} Pre-Max')
                    if max_pre[1] > max_pre_record["val"]:
                        max_pre_record = {"val": max_pre[1], "raw_string": max_pre[3], "sensor": max_pre[2]}

                if post_fire_pts:
                    max_post = max(post_fire_pts, key=lambda x: x[1])
                    self.ax_ts.plot(max_post[0], max_post[1], marker='v', color='darkred', markersize=10,
                                    linestyle='None', label=f'{em.upper()} Post-Max')
                    if max_post[1] > max_post_record["val"]:
                        max_post_record = {"val": max_post[1], "raw_string": max_post[3], "sensor": max_post[2]}

        self.ax_ts.set_title("Temporal Profiler Index Trend")
        self.ax_ts.grid(True, linestyle='--', alpha=0.5)
        handles, labels = self.ax_ts.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_ts.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        self.ax_ts.tick_params(axis='x', labelrotation=25)

        # 4. Reflectance Spectrum Plotting (Row 2) with Explicit 1D Array Masking
        self.ax_rfl.clear()
        bad_regions = [[350, 400], [900, 1000], [1310, 1490], [1725, 2050], [2450, 2500]]

        # Plot Pre-Fire Spectra
        if max_pre_record["raw_string"] is not None:
            pre_cube, pre_wvl = self.helper_load_rfl_cube(max_pre_record["raw_string"], max_pre_record["sensor"])
            if pre_cube is not None:
                y_spectrum = pre_cube[self.selected_row, self.selected_col, :].copy()

                # Filter out raw background/sensor errors first (-9999.0)
                clean_mask = (y_spectrum != -9999.0) & (~np.isnan(y_spectrum))

                if np.any(clean_mask) and len(pre_wvl) == len(y_spectrum):
                    # Gather the mask defining true good vs. bad atmospheric regions
                    good_bands_mask = spectra.get_good_bands_mask(pre_wvl, wavelength_pairs=bad_regions)

                    x_axis = pre_wvl[clean_mask]
                    y_axis = y_spectrum[clean_mask]
                    is_good = good_bands_mask[clean_mask]

                    lbl = f"Pre-Fire Max ({max_pre_record['raw_string'].split('_')[0]}, {max_pre_record['sensor'].upper()})"

                    # A. Plot the FULL continuous baseline as translucent/faded first
                    self.ax_rfl.plot(x_axis, y_axis, color='navy', linewidth=1.5, alpha=0.25, linestyle=':')

                    # B. Mask out the bad bands from a secondary layer to overplot ONLY the solid good regions
                    y_good_only = y_axis.copy()
                    y_good_only[~is_good] = np.nan  # Matplotlib lifts pen on NaN, leaving only solid clean segments
                    self.ax_rfl.plot(x_axis, y_good_only, color='navy', linewidth=2.5, alpha=1.0, label=lbl)

        # Plot Post-Fire Spectra
        if max_post_record["raw_string"] is not None:
            post_cube, post_wvl = self.helper_load_rfl_cube(max_post_record["raw_string"], max_post_record["sensor"])
            if post_cube is not None:
                y_spectrum = post_cube[self.selected_row, self.selected_col, :].copy()

                clean_mask = (y_spectrum != -9999.0) & (~np.isnan(y_spectrum))

                if np.any(clean_mask) and len(post_wvl) == len(y_spectrum):
                    good_bands_mask = spectra.get_good_bands_mask(post_wvl, wavelength_pairs=bad_regions)

                    x_axis = post_wvl[clean_mask]
                    y_axis = y_spectrum[clean_mask]
                    is_good = good_bands_mask[clean_mask]

                    lbl = f"Post-Fire Max ({max_post_record['raw_string'].split('_')[0]}, {max_post_record['sensor'].upper()})"

                    # A. Plot full continuous baseline as translucent/faded first
                    self.ax_rfl.plot(x_axis, y_axis, color='crimson', linewidth=1.5, alpha=0.25, linestyle=':')

                    # B. Overplot ONLY the solid good regions
                    y_good_only = y_axis.copy()
                    y_good_only[~is_good] = np.nan
                    self.ax_rfl.plot(x_axis, y_good_only, color='crimson', linewidth=2.5, alpha=1.0, label=lbl)

        # Retain strict formatting bounds
        self.ax_rfl.set_ylim(0, 1)
        self.ax_rfl.set_title("Hyperspectral Reflectance Profile at Selected Coordinate")
        self.ax_rfl.set_xlabel("Wavelength (nm)" if 'pre_wvl' in locals() and len(pre_wvl) > 0 else "Band Index Number")
        self.ax_rfl.set_ylabel("Reflectance Value")
        self.ax_rfl.grid(True, linestyle=':', alpha=0.6)
        self.ax_rfl.legend(loc='upper right', fontsize=9)

        self.fig.canvas.draw_idle()

    def on_radio_toggle(self, label):
        self.current_region = label
        self.update_plots()

    def on_checkbox_toggle(self, label):
        self.visibility_states[label] = not self.visibility_states[label]
        self.draw_spatial_temporal_and_highlights()

    def on_canvas_click(self, event):
        if event.inaxes is None or event.inaxes in [self.ax_radio, self.ax_check, self.ax_ts, self.ax_rfl]:
            return

        if event.inaxes == self.ax_map:
            clicked_col = int(round(event.xdata))
            clicked_row = int(round(event.ydata))
            max_rows, max_cols = self.scores_data['npv'].shape[0], self.scores_data['npv'].shape[1]
            if 0 <= clicked_row < max_rows and 0 <= clicked_col < max_cols:
                self.selected_row = clicked_row
                self.selected_col = clicked_col
                self.draw_spatial_temporal_and_highlights()
            return

        clicked_em = None
        for idx, ax in enumerate(self.scatter_axes):
            if event.inaxes == ax:
                clicked_em = self.em_key[idx]
                break

        if clicked_em and self.scatter_mappings[clicked_em] is not None:
            mapping = self.scatter_mappings[clicked_em]
            distances = np.sqrt((mapping['z'] - event.xdata) ** 2 + (mapping['sam'] - event.ydata) ** 2)
            if len(distances) == 0: return

            nearest_point_idx = np.argmin(distances)
            self.selected_row = int(mapping['r'][nearest_point_idx])
            self.selected_col = int(mapping['c'][nearest_point_idx])
            self.draw_spatial_temporal_and_highlights()

def run_figures(base_directory, sensor, aoi):
    base_directory = base_directory
    fig = figures(base_directory=base_directory, sensor=sensor, aoi=aoi)
    fig.fractional_cover_distributions()
    #fig.pixel_example_ts()
    #fig.start_plot()
