import time
import pandas as pd
from glob import glob
from p_tqdm import p_map
from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
#from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from utils.spectra_utils import spectra
from utils.create_tree import create_directory
#from pypdf import PdfMerger
from utils.envi import envi_to_array
from datetime import datetime
import geopandas as gpd
from utils.results_utils import r2_calculations, load_data
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

# quadrat groupings
quad_phenophase_key = {'early leaf out': 'pv',
                       'early senescence': 'npv',
                       'flowers': 'pv',
                       'full leaf out': 'pv',
                       'full senescence': 'npv',
                       'last year senescence': 'npv',
                       'seeds': 'npv',
                       'yellow flower': 'pv'}


class figures:
    def __init__(self, base_directory: str, sensor: str, major_axis_fontsize, minor_axis_fontsize, title_fontsize,
                 axis_label_fontsize, fig_height, fig_width, linewidth, sig_figs, legend_text):

        self.base_directory = base_directory
        self.figure_directory = os.path.join(base_directory, 'figures')
        self.output_directory = os.path.join(base_directory, 'output')
        create_directory(self.figure_directory)

        # load wavelengths
        self.wvls, self.fwhm = spectra.load_wavelengths(sensor='aviris_ng')
        self.exclude = ['.hdr', '.csv', '.ini', '.xml']

        self.asd_wvls = spectra.load_asd_wavelenghts()
        self.good_asd_bands = spectra.get_good_bands_mask(self.asd_wvls, wavelength_pairs=None)
        self.asd_wvls[~self.good_asd_bands] = np.nan

        # ems
        self.ems = ['NPV', 'GV', 'Soil']

        self.ems_short = ['npv', 'GV', 'soil']

        # figure fonts, font size, etc
        self.legend_text = legend_text
        self.major_axis_fontsize = major_axis_fontsize
        self.minor_axis_fontsize = minor_axis_fontsize
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.fig_height = fig_height
        self.fig_width = fig_width
        self.linewidth = linewidth
        self.sig_figs = sig_figs
        self.cmap_kw = 'copper'
        self.axes_limits = {
            'ymin': 0,
            'ymax': 1,
            'xmin': 0,
            'xmax': 1}

        # load emit slpit
        terraspec_base = os.path.dirname(base_directory)
        self.slpit_figures = os.path.join(terraspec_base, 'slpit', 'figures')

        self.col_map = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

        self.col_map_wp = {
            0: 'npv',
            1: 'pv',
            2: 'soil'}

    def quad_cover(self):
        # load quadrat tallies
        df_quad = pd.read_csv(os.path.join('objects', 'SHIFT_vegetation_quadrat_tallies.csv'))
        #df_quad = df_quad.dropna(subset=['Phenophase'])
        for i in sorted(list(df_quad.Species_or_type.unique())):
            print(i)

        df_quad['Phenophase'] = df_quad['Phenophase'].replace(np.nan, 'na')
        df_quad['Phenophase'] = df_quad['Phenophase'].str.strip()
        df_quad['Phenophase'] = df_quad['Phenophase'].apply(str.lower)

        df_quad['Species_or_type'] = df_quad['Species_or_type'].apply(str.lower)
        df_quad['Species_or_type'] = df_quad['Species_or_type'].str.strip()
        df_quad['cover'] = ''

        df_quad['Date'] = pd.to_datetime(df_quad['Date'], format='%Y-%m-%d')

        # df coords with data and dates
        df_coords = pd.read_csv('gis/shift_plot_coordinates.csv')

        df_quads_to_merge = []
        for specie in sorted(list(df_quad.Species_or_type.unique())):
            df_quad_select = df_quad.loc[df_quad['Species_or_type'] == specie].copy()

            if specie in ['soil', 'rock', 'npv']:
                if specie == 'rock':
                    df_quad_select['cover'] = 'soil'
                else:
                    df_quad_select['cover'] = specie

            elif specie in ['water']:
                continue
            else:
                df_quad_select['cover'] = df_quad_select['Phenophase'].replace(quad_phenophase_key)
            df_quads_to_merge.append(df_quad_select)

        df_quad_cover = pd.concat(df_quads_to_merge, ignore_index=True)

        df_agg_rows = []
        for _plot, plot in enumerate(sorted(list(df_quad_cover['Plot_name'].unique()))):
            df_plot = df_quad_cover.loc[df_quad_cover['Plot_name'] == plot].copy()
            df_meta = df_coords.loc[df_coords['Plot Name'] == plot]

            if df_meta.empty:
                pass
            else:
                plot_date = df_meta['Date'].values[0]

                df_plot = df_plot.loc[df_plot['Date'] == plot_date].copy()
                df_plot = df_plot.drop(columns=['Date'])

                df_agg = df_plot.groupby(['cover']).sum().reset_index()

                df_agg['frac_cover'] = df_agg['Count'] / df_agg['Count'].sum()

                row = [plot, plot_date, ]
                for cover in sorted(list(df_agg.cover.unique())):
                    frac = df_agg.loc[df_agg['cover'] == cover, 'frac_cover'].iloc[0]
                    row.append((cover, frac))

                df_agg_rows.append(row)

        # aggregate all rows
        df_to_concat = []
        for row in df_agg_rows:
            plot = row[0]
            date = row[1]
            frac_covers = row[2:]
            cols = ['plot', 'date']
            values = [plot, date]
            for cover in frac_covers:
                cols.append(cover[0])
                values.append(cover[1])
            df = pd.DataFrame(values).T
            df.columns = cols
            df_to_concat.append(df)

        df_concat = pd.concat(df_to_concat)
        df_concat = df_concat.fillna(0)
        df_concat.to_csv(os.path.join(self.output_directory, 'quad_cover.csv'), index=False)

    def load_frac_data(self):
        #skip = ['SRA-000_SPRING', 'SRB-047_SPRING', 'SRB-004_FALL', 'SRB-050_FALL', 'SRB-200_FALL', 'SRA-056_SPRING',
        #        'DPA-004_FALL', 'DPB-027_SPRING', 'SRA-008_FALL', 'SRB-026_SPRING'] # excluding shrubland plots

        skip = ['SRA-000_SPRING', 'SRB-004_FALL', 'SRB-200_FALL', # these are bad plots
                 'DPA-004_FALL', 'DPB-027_SPRING', 'SRA-008_FALL', 'SRB-026_SPRING', 'SRA-056_SPRING']  # excluding shrubland plots

        df_all = pd.read_csv(os.path.join(self.figure_directory, 'shift_fraction_output.csv'))
        df_all = df_all[~df_all['plot'].isin(skip)]

        df_wonderpole = pd.read_excel(os.path.join('objects', 'wonderpole.xlsx'))
        df_wonderpole["plot_name"] = df_wonderpole["plot_name"].str.strip()
        df_wonderpole["season"] = df_wonderpole["season"].str.strip()
        df_wonderpole['plot'] = df_wonderpole['plot_name'].astype(str) + '_' + df_wonderpole['season'].astype(str)
        df_wonderpole = df_wonderpole.dropna()
        df_wonderpole = df_wonderpole[~df_wonderpole['plot'].isin(skip)]
        df_wonderpole = df_wonderpole.sort_values('plot')

        df_quad = pd.read_csv(os.path.join(self.output_directory, 'quad_cover.csv'))
        df_quad['date'] = pd.to_datetime(df_quad['date'], format='%m/%d/%Y')
        df_quad['season'] = df_quad['date'].apply(lambda x: 'FALL' if x.month == 9 else 'SPRING')
        df_quad['plot'] = df_quad["plot"].astype(str) + '_' + df_quad["season"].astype(str)
        df_quad = df_quad[~df_quad['plot'].isin(skip)]
        df_quad = df_quad.sort_values('plot')

        return df_all, df_wonderpole, df_quad


    def table_1(self):
        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)

        #  create table outline
        ncols = 4
        nrows = 4

        df_select_shift = df_all[(df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)].copy()
        for em in ['npv', 'pv', 'soil']:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16), sharex=True, sharey=True)
            fig.suptitle(f'Fraction Comparison: {em.upper()}', fontsize=16, fontweight='bold')

            for row in range(nrows):
                raw_string = [f'{em} MAD(RMSD; R2; Bias):']

                for col in range(ncols):
                    norm_x = 'brightness' if col in [0, 1] else 'none'
                    ax = axes[row, col]
                    if row == 0:
                        df_x = df_select_shift[
                            (df_select_shift['instrument'] == 'SLPIT') &
                            (df_select_shift['unmix_mode'] == 'emc2') &
                            (df_select_shift['normalization'] == norm_x)
                            ]
                        x = df_x[em].tolist()
                        row_label = f"SLPIT emc2"
                    elif row == 1:
                        df_x = df_select_shift[
                            (df_select_shift['instrument'] == 'SLPIT') &
                            (df_select_shift['unmix_mode'] == 'mesma') &
                            (df_select_shift['normalization'] == norm_x)
                            ]
                        x = df_x[em].tolist()
                        row_label = f"SLPIT mesma"
                    elif row == 2:
                        x = (df_wonderpole[em] / 100).tolist()
                        row_label = "Wonderpole"
                    elif row == 3:
                        x = df_quad[em].tolist()
                        row_label = "Quad"
                    col_map = {
                        0: ('emc2', 'brightness'),
                        1: ('mesma', 'brightness'),
                        2: ('emc2', 'none'),
                        3: ('mesma', 'none')
                    }

                    mode_y, norm_y = col_map[col]
                    df_y = df_select_shift[
                        (df_select_shift['instrument'] == 'RFL') &
                        (df_select_shift['unmix_mode'] == mode_y) &
                        (df_select_shift['normalization'] == norm_y)
                        ]
                    y = df_y[em].tolist()
                    col_label = f"AVIRIS {mode_y} ({norm_y})"

                    rmse = root_mean_squared_error(x, y)
                    mae = mean_absolute_error(x, y)
                    r2, bias = r2_calculations(x, y)

                    raw_string.append(f'{mae:.2f}({rmse:.2f}; {r2:.2f}; {bias:.2f}),')

                    #ax.scatter(x, y, alpha=0.5, s=10, edgecolors='none')

                    # Add 1:1 line
                    #full_range = np.array([0, 1])
                    #ax.plot(full_range, full_range, color='black', linestyle='--', alpha=0.6, label='1:1')

                    # Add Regression Line
                    if len(x) > 1:
                        # Calculate regression
                        slope, intercept, r_val, p_val, std_err = linregress(x, y)
                        error = np.array(x) - np.array(y)

                        bin_width = 0.02
                        low_edge = np.floor(error.min() / bin_width) * bin_width
                        high_edge = np.ceil(error.max() / bin_width) * bin_width
                        bins_list = np.arange(low_edge, high_edge + bin_width, bin_width)

                        ax.hist(error, bins=bins_list, color='skyblue', edgecolor='black', density=True)
                        # Calculate Y values for the absolute edges of the plot (0 and 1)
                        #y_reg_full = slope * full_range + intercept
                        #ax.plot(full_range, y_reg_full, color='red', linestyle='-', linewidth=1.5, label='Regression')

                    # Add text metrics to the plot
                    stats_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n$R^2$: {r2:.2f}\nBias: {bias:.2f}\nSlope: {slope:.2f}"
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                    # Labels for outer edges
                    if row == 3: ax.set_xlabel(f"X: {col_label}")

                    ax.set_ylabel(f"Y: {row_label}")

                    #ax.set_xlim(0, 1)
                    #ax.set_ylim(0, 1)
                    ax.set_xlim(-0.5, 0.5)
                    ax.grid(True, linestyle=':', alpha=0.6)

                print(" ".join(raw_string))

            plt.savefig(os.path.join(self.figure_directory, f'table_1_error_distribution_{em}.png'))
            plt.clf()
            plt.close()
            print()
            print()



    def table_2(self):

        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)

        #  create table outline
        ncols = 4
        nrows = 2

        df_select_shift = df_all[(df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)].copy()
        for em in ['npv', 'pv', 'soil']:
            for row in range(nrows):
                raw_string = [f'{em} MAD(RMSD; R2; Bias):']
                for col in range(ncols):
                    norm_x = 'brightness' if col in [0, 1] else 'none'

                    if row == 0:
                        df_x = df_select_shift[
                            (df_select_shift['instrument'] == 'RFL') &
                            (df_select_shift['unmix_mode'] == 'emc2') &
                            (df_select_shift['normalization'] == norm_x)
                            ]
                        x = df_x[em].tolist()

                    elif row == 1:
                        df_x = df_select_shift[
                            (df_select_shift['instrument'] == 'SLPIT') &
                            (df_select_shift['unmix_mode'] == 'emc2') &
                            (df_select_shift['normalization'] == norm_x)
                            ]
                        x = df_x[em].tolist()

                    col_map = {
                        0: ('mesma', 'brightness', 'RFL'),
                        1: ('mesma', 'brightness', 'SLPIT'),
                        2: ('mesma', 'none', 'RFL'),
                        3: ('mesma', 'none', 'SLPIT')
                    }

                    mode_y, norm_y, instrument = col_map[col]
                    df_y = df_select_shift[
                        (df_select_shift['instrument'] == instrument) &
                        (df_select_shift['unmix_mode'] == mode_y) &
                        (df_select_shift['normalization'] == norm_y)
                        ]
                    y = df_y[em].tolist()

                    rmse = root_mean_squared_error(x, y)
                    mae = mean_absolute_error(x, y)
                    r2, bias = r2_calculations(x, y)

                    raw_string.append(f'{mae:.2f}({rmse:.2f}; {r2:.2f}; {bias:.2f}),')

                print(" ".join(raw_string))
            print()

    def table_3(self):

        df_all, df_wonderpole, df_quad = figures.load_frac_data(self)

        #  create table outline
        ncols = 5
        nrows = 2

        df_select_shift = df_all[(df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)].copy()
        for em in ['npv', 'pv', 'soil']:
            for row in range(nrows):
                raw_string = [f'{em} MAD(RMSD; R2; Bias):']
                for col in range(ncols):

                    if row == 0:
                        x = (df_wonderpole[em] / 100).tolist()

                    elif row == 1:
                        x = df_quad[em].tolist()

                    col_map = {
                        0: ('emc2', 'brightness', 'SLPIT'),
                        1: ('mesma', 'brightness', 'SLPIT'),
                        2: ('emc2', 'none', 'SLPIT'),
                        3: ('mesma', 'none', 'SLPIT'),
                    }

                    if col not in [4]:
                        mode_y, norm_y, instrument = col_map[col]
                        df_y = df_select_shift[
                            (df_select_shift['instrument'] == instrument) &
                            (df_select_shift['unmix_mode'] == mode_y) &
                            (df_select_shift['normalization'] == norm_y)
                            ]
                        y = df_y[em].tolist()
                    else:
                        y = df_quad[em].tolist()

                    rmse = root_mean_squared_error(x, y)
                    mae = mean_absolute_error(x, y)
                    r2, bias = r2_calculations(x, y)

                    raw_string.append(f'{mae:.2f}({rmse:.2f}; {r2:.2f}; {bias:.2f}),')

                print(" ".join(raw_string))
            print()


    def methods_diagram(self):
        fig, axes = plt.subplots(3, 3, figsize=(8, 6.5), layout='constrained')

        (ax_slpit, ax_rfl_plot, ax_slpit_photo), (ax_wp, ax_wp_grid, ax_wp_photo), (ax_quadrat, ax_quadrat_grid, ax_quad_photo) = axes

        # Formatting helper to keep them square
        for ax in [ax_slpit, ax_wp, ax_quadrat]:
            ax.set_aspect('equal', adjustable='box')

        # ------SLPIT Diagram--------
        ax_slpit.set_title('SLPIT Diagram', fontweight='bold', fontsize=10)
        grid_limit = 8
        x_transects = [2, 6]
        y_points = np.arange(0, 8.01, 0.33)
        sensor_offset = 0.25
        radius = 0.07

        ax_slpit.axvline(grid_limit / 2, color='black', linestyle='--', alpha=0.5)
        ax_slpit.axhline(grid_limit / 2, color='black', linestyle='--', alpha=0.5)

        for x in x_transects:
            for y in y_points:
                circle = plt.Circle((x + sensor_offset, y), radius,
                                    facecolor=(1, 0, 0, 0.2), edgecolor='red', linewidth=0.5)
                ax_slpit.add_patch(circle)

        circle_proxy = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                     markersize=10, markerfacecolor=(1, 0, 0, 0.2),
                                     markeredgecolor='red', label='ASD GIFOV')
        ax_slpit.legend(handles=[circle_proxy], fontsize=8) # loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, borderaxespad=0)

        ax_rfl_plot.set_box_aspect(1)
        ax_rfl_plot.set_title('SLPIT Reflectance', fontweight='bold', fontsize=10)
        ax_rfl_plot.set_xlabel('Wavelength (nm)', fontsize=8)
        ax_rfl_plot.set_ylabel('Reflectance (%)', fontsize=8)
        ax_rfl_plot.set_xlim(300, 2550)
        ax_rfl_plot.xaxis.set_major_locator(MultipleLocator(500))
        ax_rfl_plot.xaxis.set_minor_locator(MultipleLocator(100))
        ax_rfl_plot.set_ylim(0, 1)
        ax_rfl_plot.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_rfl_plot.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax_rfl_plot.tick_params(axis='both', which='major', labelsize=8)

        slpit_rfl = envi_to_array(os.path.join(self.output_directory, 'spectral_transects', 'DPB-020_SPRING', 'RFL',
                                               f'DPB-020_SPRING_SLPIT_asd'))
        slpit_rfl[slpit_rfl == -9999] = np.nan
        y_mean = np.nanmean(slpit_rfl, axis=(0, 1))
        y_std = np.nanstd(slpit_rfl, axis=(0, 1))


        ax_rfl_plot.plot(self.asd_wvls, y_mean, label=f"SLPIT mean", linewidth=2, color='red')
        ax_rfl_plot.fill_between(self.asd_wvls, y_mean - y_std * 1, y_mean + y_std * 1,
                                 color='red', alpha=0.2, label=f"1σ", linewidth=2)
        ax_rfl_plot.legend(fontsize=8)


        #  ------Wonderpole Diagram--------
        inset_gap = 0.05
        square_size = (grid_limit / 2) - (2 * inset_gap)

        ax_wp.set_title('Wonderpole Diagram', fontweight='bold', fontsize=10)
        ax_wp.set_xlabel('8 m')
        ax_wp.set_ylabel('8 m')

        ax_wp.axvline(grid_limit / 2, color='black', linestyle='--', alpha=0.5, zorder=1)
        ax_wp.axhline(grid_limit / 2, color='black', linestyle='--', alpha=0.5, zorder=1)

        quad_data = [
            (4 + inset_gap, 4 + inset_gap, square_size, square_size, 'blue', 'NW Photo'),
            (0 + inset_gap, 4 + inset_gap, square_size, square_size, 'green', 'NE Photo'),
            (0 + inset_gap, 0 + inset_gap, square_size, square_size, 'orange', 'SE Photo'),
            (4 + inset_gap, 0 + inset_gap, square_size, square_size, 'red', 'SW Photo')
        ]

        for x_start, y_start, w, h, color, label in quad_data:
            rect = Rectangle((x_start, y_start), w, h,
                             facecolor=color,
                             edgecolor=color,
                             linewidth=1,
                             alpha=0.3,  # Fill alpha as requested
                             zorder=2)
            ax_wp.add_patch(rect)

            text_x = x_start + (square_size / 2)
            text_y = y_start + (square_size / 2)

            ax_wp.text(text_x, text_y, label,
                       color='black',
                       fontweight='bold',
                       fontsize=9,
                       horizontalalignment='center',
                       verticalalignment='center',
                       zorder=3)

        #  ------Wonderpole Data--------
        ax_wp_grid.set_title('Wonderpole Photos', fontweight='bold', fontsize=10)

        ax_wp_grid.set_xlim(0, 8)
        ax_wp_grid.set_ylim(0, 8)

        photo_dir = os.path.join('objects', 'shift_photos')
        photo_basepath = f'20220324_DPB-020_wp'

        photo_configs = [
            {'path': os.path.join(photo_dir, f'{photo_basepath}_NE_1.jpg'), 'pos': (0, 1), 'color': 'green', 'label': 'Q1'},
            {'path': os.path.join(photo_dir, f'{photo_basepath}_NW_1.jpg'), 'pos': (1, 1), 'color': 'blue', 'label': 'Q2'},
            {'path': os.path.join(photo_dir, f'{photo_basepath}_SE_1.jpg'), 'pos': (0, 0), 'color': 'orange', 'label': 'Q3'},
            {'path': os.path.join(photo_dir, f'{photo_basepath}_SW_1.jpg'), 'pos': (1, 0), 'color': 'red', 'label': 'Q4'}
        ]

        buffer = 0.01
        sq_size = 1.0 - (2 * buffer)

        for p in photo_configs:
            x_base, y_base = p['pos']

            # Calculate the 'Safe Zone' for this specific photo
            xmin = x_base + buffer
            ymin = y_base + buffer
            ext = [xmin, xmin + sq_size, ymin, ymin + sq_size]

            img_data = mpimg.imread(p['path'])
            # Rotate 90 degrees for vertical orientation

            # --- 3. Plot the Image ---
            ax_wp_grid.imshow(img_data, extent=ext, aspect='auto', zorder=1)

            # --- 4. Add the Border ---
            # Because xmin/ymin are buffered, the blue line will never touch the green line
            rect = Rectangle((xmin, ymin), sq_size, sq_size,
                             edgecolor=p['color'],
                             facecolor='none',
                             linewidth=4,
                             zorder=2)
            ax_wp_grid.add_patch(rect)

        # 3. Plot the Main Quadrant Division Lines (to keep it consistent with the diagram)
        ax_wp_grid.set_xlim(0, 2)
        ax_wp_grid.set_ylim(0, 2)
        ax_wp_grid.axis('off')

        # -------Quadrats----------
        ax_quadrat.set_title('Quadrat Diagram', fontweight='bold', fontsize=10)

        # 1. Grid Divisions
        ax_quadrat.axvline(4, color='black', linestyle='--', alpha=0.5, zorder=1)
        ax_quadrat.axhline(4, color='black', linestyle='--', alpha=0.5, zorder=1)

        # 2. Quadrat Dimensions
        q_width = 0.5
        q_height = 1

        # 3. Define Centers for each 4x4 quadrant
        quad_centers = [(2, 6, 'green', 'NE Quadrat'), (6, 6, 'blue', 'NW Quadrat'),
                        (2, 2, 'orange', 'SE Quadrat'), (6, 2, 'red', 'SW Quadrat')]

        for cx, cy, color, label in quad_centers:
            # Calculate bottom-left corner from center
            x_start = cx - (q_width / 2)
            y_start = cy - (q_height / 2)

            # Add the Rectangle
            rect = Rectangle((x_start, y_start), q_width, q_height,
                             facecolor=color, edgecolor=color,
                             linewidth=1.5, alpha=0.4, zorder=2)
            ax_quadrat.add_patch(rect)

            # Add a small label above/inside the quadrat
            ax_quadrat.text(cx, cy + 0.625, label, ha='center', va='bottom',
                            fontsize=8, fontweight='bold')


        #  ------Quadrat Data--------
        ax_quadrat_grid.set_title('Quadrat Photos', fontweight='bold', fontsize=10)

        photo_dir = os.path.join('objects', 'shift_photos')
        photo_basepath = f'20220324_DPB-020_quad'

        photo_configs = [
            {'path': os.path.join(photo_dir, f'{photo_basepath}_NE.jpg'), 'pos': (0, 1), 'color': 'green', 'label': 'Q1'},
            {'path': os.path.join(photo_dir, f'{photo_basepath}_NW.jpg'), 'pos': (1, 1), 'color': 'blue', 'label': 'Q2'},
            {'path': os.path.join(photo_dir, f'{photo_basepath}_SE.jpg'), 'pos': (0, 0), 'color': 'orange', 'label': 'Q3'},
            {'path': os.path.join(photo_dir, f'{photo_basepath}_SW.jpg'), 'pos': (1, 0), 'color': 'red', 'label': 'Q4'}]

        buffer = 0.01
        sq_size = 1.0 - (2 * buffer)

        for p in photo_configs:
            x_base, y_base = p['pos']

            xmin = x_base + buffer
            ymin = y_base + buffer
            ext = [xmin, xmin + sq_size, ymin, ymin + sq_size]

            img_data = mpimg.imread(p['path'])
            ax_quadrat_grid.imshow(img_data, extent=ext, aspect='auto', zorder=1)
            rect = Rectangle((xmin, ymin), sq_size, sq_size,
                             edgecolor=p['color'],
                             facecolor='none',
                             linewidth=4,
                             zorder=2)
            ax_quadrat_grid.add_patch(rect)

        ax_quadrat_grid.set_xlim(0, 2)
        ax_quadrat_grid.set_ylim(0, 2)
        ax_quadrat_grid.axis('off')

        for ax in [ax_slpit, ax_wp, ax_quadrat]:
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('8 m')
            ax.set_ylabel('8 m')

            ax.annotate('N', xy=(-0.05, 0.98), xytext=(-0.05, 0.775),
                           arrowprops=dict(facecolor='black', width=1, headwidth=5.5),
                           ha='center', va='center', fontsize=10, color='black',
                           xycoords='axes fraction', annotation_clip=False)

        landscape_paths = [os.path.join(photo_dir, 'SLPIT.jpg'),
                           os.path.join(photo_dir, 'WP.jpg'),
                           os.path.join(photo_dir, 'quadrats.jpg')]

        title = {0: "SLPIT", 1: "Wonderpole", 2:"Quadrat"}

        landscape_axes = [ax_slpit_photo, ax_wp_photo, ax_quad_photo]
        for i, ax in enumerate(landscape_axes):
            ax.set_title(f'{title[i]}', fontweight='bold', fontsize=10)
            img = mpimg.imread(landscape_paths[i])
            ax.imshow(img, zorder=1, aspect='equal')
            ax.axis('off')

        for ax in [ax_wp_grid, ax_quadrat_grid]:
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_aspect('equal', adjustable='box')

        plt.savefig(os.path.join(self.figure_directory, 'methods_diagram.png'), dpi=600)
        plt.clf()
        plt.close()


def run_figures(base_directory):
    base_directory = base_directory
    sensor = 'aviris_ng'
    major_axis_fontsize = 22
    minor_axis_fontsize = 20
    title_fontsize = 30
    axis_label_fontsize = 26
    fig_height = 10
    fig_width = 17
    linewidth = 3
    sig_figs = 2
    legend_text = 20

    fig = figures(base_directory=base_directory, sensor=sensor, major_axis_fontsize=major_axis_fontsize,
                        minor_axis_fontsize=minor_axis_fontsize, title_fontsize=title_fontsize,
                        axis_label_fontsize=axis_label_fontsize, fig_height=fig_height, fig_width=fig_width,
                        linewidth=linewidth, sig_figs=sig_figs, legend_text=legend_text)

    fig.quad_cover()
    fig.table_1()
    fig.table_2()
    fig.table_3()
    fig.methods_diagram()
