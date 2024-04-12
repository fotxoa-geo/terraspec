import time
import pandas as pd

import asdreader
from download import create_directory
from glob import glob
from p_tqdm import p_map
from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from osgeo import gdal
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pypdf import PdfMerger
from utils.envi import envi_to_array
from sklearn.linear_model import LinearRegression
from datetime import datetime
import geopandas as gpd
import rasterio
from matplotlib.animation import ArtistAnimation, FuncAnimation
from PIL import Image


# bad wavelength regions
bad_wv_regions = [[0, 440], [1310, 1490], [1770, 2050], [2440, 2880]]

# quadrat groupings
quad_phenophase_key = {'early leaf out': 'pv',
                       'early senescence': 'npv',
                       'flowers': 'pv',
                       'full leaf out': 'pv',
                       'full senescence': 'npv',
                       'last year senescence': 'npv',
                       'seeds': 'npv',
                       'yellow flower': 'pv'}


def fraction_file_info(fraction_file):
    name = os.path.basename(fraction_file)
    unmix_mode = os.path.basename(os.path.dirname(fraction_file))

    library_mode = name.split("_")[0].split('-')[0]
    instrument = name.split("-")[1]
    plot = f"{name.split('_')[0].split('-')[2]}-{name.split('_')[0].split('-')[3]}_{name.split('_')[1]}"

    num_cmb_em = name.split("_")[5]
    num_mc = name.split("_")[8]
    normalization = name.split("_")[10]

    ds = gdal.Open(fraction_file, gdal.GA_ReadOnly)
    array = ds.ReadAsArray().transpose((1, 2, 0))

    mean_fractions = []
    for _band, band in enumerate(range(0, array.shape[2])):
            mean_fractions.append(np.mean(array[:, :, _band]))

    return [instrument, unmix_mode, plot, library_mode, int(num_cmb_em), int(num_mc), normalization] + mean_fractions


def r2_calculations(x_vals, y_vals):
    X = np.array(x_vals)
    y = np.array(y_vals)
    X = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X)
    r2 = r2_score(y_vals, y_pred)

    return np.round(r2,2)


def get_good_bands_mask(wavelengths, wavelength_pairs):
    wavelengths = np.array(wavelengths)
    if wavelength_pairs is None:
        wavelength_pairs = bad_wv_regions
    good_bands = np.ones(len(wavelengths)).astype(bool)

    for wvp in wavelength_pairs:
        wvl_diff = wavelengths - wvp[0]
        wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
        lower_index = np.nanargmin(wvl_diff)

        wvl_diff = wvp[1] - wavelengths
        wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
        upper_index = np.nanargmin(wvl_diff)
        good_bands[lower_index:upper_index + 1] = False
    return good_bands


def load_wavelengths(wavelength_file: str):
    wl = np.loadtxt(wavelength_file, usecols=1)
    fwhm = np.loadtxt(wavelength_file, usecols=2)
    if np.all(wl < 100):
        wl *= 1000
        fwhm *= 1000
    return wl, fwhm


def plot_asd(row, out_directory, starting_col):
    # Load asd data
    data = row[1].to_frame().transpose().reset_index(drop=True)

    asd_refl = data.iloc[0][starting_col:]
    asd_wl = np.array(data.columns[starting_col:]).astype(float)
    good_bands = get_good_bands_mask(asd_wl, bad_wv_regions)
    asd_wl[~good_bands] = np.nan

    plot_name = data.iloc[0]['plot_name']
    line_num = data.iloc[0]['line_num']
    file_num = data.iloc[0]['file_num']

    try:
        level_1 = data.iloc[0]['level_1']
    except:
        level_1 = 'transect'

    fname = plot_name + level_1 + "_" + str(line_num) + "_" + str(file_num)
    outfname = os.path.join(out_directory, os.path.basename(fname) + '.png')


def plot_em_asd(row, out_directory, starting_col):
    # Load asd data
    data = row[1].to_frame().transpose().reset_index(drop=True)
    plt.ylim(0, 1)
    plt.xlim(325, 2525)

    asd_refl = data.iloc[0][starting_col:]
    asd_wl = np.array(data.columns[starting_col:]).astype(float)
    good_bands = get_good_bands_mask(asd_wl, bad_wv_regions)
    #asd_wl[~good_bands] = np.nan

    plot_name = data.iloc[0]['plot_name']
    line_num = data.iloc[0]['line_num']
    file_num = data.iloc[0]['file_num']
    level_1 = data.iloc[0]['level_1']

    plt.plot(asd_wl, asd_refl, label=f"EM: {level_1} ({line_num}:{file_num})")
    plt.legend()

    fname = plot_name + level_1 + "_" + str(line_num) + "_" + str(file_num)
    outfname = os.path.join(out_directory, os.path.basename(fname) + '.png')
    plt.savefig(outfname)
    plt.clf()
    plt.close()


class plots:
    def __init__(self, base_directory: str, wavelength_file:str):

        self.base_directory = base_directory
        self.figure_directory = os.path.join(base_directory, 'figures')
        self.output_directory = os.path.join(base_directory, 'output')
        create_directory(self.figure_directory)
        self.instrument = os.path.basename(wavelength_file).split("_")[0]

        # load wavelengths
        self.wvls, self.fwhm = load_wavelengths(wavelength_file=wavelength_file)
        create_directory(os.path.join(self.base_directory, 'data', 'spectral_transects', 'asd_plots'))
        self.asd_plot_outputs = os.path.join(self.base_directory, 'data', 'spectral_transects', 'asd_plots')

        self.exclude = ['.hdr', '.csv', '.ini', '.xml']
        # ems
        self.ems = ['npv', 'pv', 'soil']

    def plot_asd(self):
        df_ems = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-asd.csv'))
        df_trans = pd.read_csv(os.path.join(self.output_directory, 'all-transect-asd.csv'))

        for plot in sorted(df_ems.plot_name.unique()):
            create_directory(os.path.join(self.asd_plot_outputs, plot))
            df_select_ems = df_ems[df_ems['plot_name'] == plot].copy()
            df_select_trans = df_trans[df_trans['plot_name'] == plot].copy()

            # process the endmembers
            p_map(partial(plot_asd, out_directory=os.path.join(self.asd_plot_outputs, plot), starting_col=11), df_select_ems.iterrows(),
                  **{
                      "desc": "\t\t plotting transect asd files: " + plot + " ...",
                      "ncols": 150})

            # process the transect spectra
            p_map(partial(plot_asd, out_directory=os.path.join(self.asd_plot_outputs, plot), starting_col=9), df_select_trans.iterrows(),
                  **{
                      "desc": "\t\t plotting transect asd files: " + plot + " ...",
                      "ncols": 150})

    def quad_cover(self):
        # load quadrat tallies
        df_quad = pd.read_csv(os.path.join(self.base_directory, 'field', 'quadrat_tallies.csv'))
        df_quad['phenophase'] = df_quad['phenophase'].apply(str.lower)
        df_quad['cover_species'] = df_quad['cover_species'].apply(str.lower)
        df_quad['cover'] = ''

        # df coords with data and dates
        df_coords = pd.read_csv('plot_coordinates.csv')

        df_quads_to_merge = []
        for specie in sorted(list(df_quad.cover_species.unique())):
            df_quad_select = df_quad.loc[df_quad['cover_species'] == specie].copy()

            if specie in ['soil', 'rock', 'npv']:
                if specie == 'rock':
                    df_quad_select['cover'] = 'soil'
                else:
                    df_quad_select['cover'] = specie

            elif specie in ['water']:
                continue
            else:
                df_quad_select['cover'] = df_quad_select['phenophase'].replace(quad_phenophase_key)

            df_quads_to_merge.append(df_quad_select)

        df_quad_cover = pd.concat(df_quads_to_merge, ignore_index=True)
        print(df_quad_cover)

        df_agg_rows = []
        for _plot, plot in enumerate(sorted(list(df_quad_cover['plot_name'].unique()))):
            df_plot = df_quad_cover.loc[df_quad_cover['plot_name'] == plot].copy()
            df_meta = df_coords.loc[df_coords['Plot Name'] == plot]

            if df_meta.empty:
                pass
            else:
                plot_date = df_meta['Date'].values[0]
                df_plot = df_plot.loc[df_plot['date'] == df_meta['Date'].values[0]].copy()
                df_plot = df_plot.drop(columns=['date'])

                df_agg = df_plot.groupby(['cover']).sum().reset_index()
                df_agg['frac_cover'] = df_agg['count']/df_agg['count'].sum()

                row = [plot, plot_date,]
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

    def plot_summary(self):
        # spectral data
        create_directory(os.path.join(self.figure_directory, 'plot_stats'))
        transect_data = pd.read_csv(os.path.join(self.output_directory, 'all-transect-asd.csv'))
        em_data = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-aviris-ng.csv'))

        # load asd wavelengths
        asd_wavelengths = np.array(transect_data.columns[9:]).astype(float)
        good_bands = get_good_bands_mask(asd_wavelengths, bad_wv_regions)
        asd_wavelengths[~good_bands] = np.nan

        # load instrument wavelengths
        ins_wvls = np.array(em_data.columns[10:]).astype(float)
        good_ins_band = get_good_bands_mask(ins_wvls, bad_wv_regions)
        ins_wvls[~good_ins_band] = np.nan

        # plot summary - merged
        merger = PdfMerger()

        skip = []

        # fractional cover
        for plot in sorted(list(transect_data.plot_name.unique())):

            plot = plot.upper()
            fig = plt.figure(figsize=(14, 8))

            fig.suptitle(f"SHIFT: {plot}")
            gs1 = gridspec.GridSpec(2, 2)
            gs1.update(left=0.05, right=0.49, wspace=0.05, hspace=0.1)
            ax1 = plt.subplot(gs1[:-1, 0])
            ax2 = plt.subplot(gs1[:-1, 1])
            ax3 = plt.subplot(gs1[-1, :])

            gs2 = gridspec.GridSpec(4, 4)
            gs2.update(left=0.53, right=0.98, hspace=0.1, wspace=0.05)

            ax4 = plt.subplot(gs2[0, :])
            ax5 = plt.subplot(gs2[1, :])
            ax6 = plt.subplot(gs2[2, :])
            ax7 = plt.subplot(gs2[3, :])

            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            site = plot.split("-")[0] + '-' + plot.split("-")[1]
            season = plot.split("-")[-1]
            df_coords = pd.read_csv('plot_coordinates.csv')
            df_coords['Date'] = pd.to_datetime(df_coords['Date'])
            date = df_coords.loc[(df_coords['Plot Name'] == site) & (df_coords['Season'] == season.upper()), 'Date'].iloc[0]
            date = datetime.strptime(str(date).split(' ')[0], "%Y-%m-%d")

            em_data = pd.read_csv(os.path.join(self.output_directory, 'spectral_transects', 'endmembers',
                                               f'{site}_{season.upper()}_original_{self.instrument}.csv'))
            for ax in axes:
                if ax == ax1:
                    ax.set_title('Plot Map')
                    long = df_coords.loc[(df_coords['Plot Name'] == site) & (df_coords['Season'] == season.upper()), 'longitude'].iloc[0]
                    lat = df_coords.loc[(df_coords['Plot Name'] == site) & (df_coords['Season'] == season.upper()), 'latitude'].iloc[0]

                    if site[:2] == 'DP':
                        lleft_lat = 34.4086695
                        uright_lat = 34.7477056
                        lleft_lon = -120.6851337
                        uright_lon = -120.0943748
                    else:
                        lleft_lat = 34.5747161
                        uright_lat = 34.7916992
                        lleft_lon = -120.2198138
                        uright_lon = -119.8417282

                    m = Basemap(projection='merc', llcrnrlat=lleft_lat, urcrnrlat=uright_lat,
                                llcrnrlon=lleft_lon, urcrnrlon=uright_lon, ax=ax1, epsg=4326)
                    m.arcgisimage(service='World_Imagery', xpixels=1000, ypixels=1000, dpi=300, verbose=True)
                    ax.scatter(float(long), float(lat), color='red', s=12)

                if ax == ax2:

                    flip = ['SRA-019-SPRING', 'DPA-004-FALL', 'DPB-005-FALL', 'SRB-010-FALL', 'SRB-045-FALL', 'SRB-050-FALL',
                            'SRB-004-FALL', 'SRA-056-FALL', 'SRA-007-FALL', 'SRA-008-FALL']
                    flip_90 = ['SRA-034-SPRING', 'SRB-026-SPRING', 'SRB-046-FALL', 'SRB-010-FALL']

                    ax.set_title('Landscape\nPicture')
                    pics = glob(os.path.join(self.base_directory, 'field', 'pictures', f'*{site}*'))

                    if not pics:
                        ax.text(0.5, 0.5, 'No Plot Picture Available', transform=ax.transAxes, fontsize=8,
                                verticalalignment='top')
                    else:
                        img = mpimg.imread(pics[0])
                        if plot in flip:
                            ax.imshow(img, origin='lower')
                        elif plot in flip_90:
                            flipped_img = np.flip(np.transpose(img, axes=(1, 0, 2)), axis=0)
                            ax.imshow(flipped_img, origin='lower')
                        else:
                            ax.imshow(img)
                    ax.axis('off')

                if ax == ax7:
                    ax.set_title('Mineralogy')

                if ax == ax4:
                    df_spectra = em_data[em_data['level_1'] == 'npv'].copy()
                    specific_string = 'UNK-'
                    df_spectra['species'] = df_spectra['species'].fillna(specific_string)
                    df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
                    num_species = len(sorted(list(df_spectra.species.unique())))
                    colors = np.random.rand(num_species, 3)

                    for _species, species in enumerate(sorted(list(df_spectra.species.unique()))):
                        df_species = df_spectra[df_spectra['species'] == species].copy()
                        em_spectra = df_species.iloc[:, 10:].to_numpy()

                        for _row, row in enumerate(em_spectra):
                            ax.plot(ins_wvls, row, color=colors[_species])

                        common_name = df_species_key.loc[df_species_key['key_value'] == species, ['label']].values[0][0]
                        ax.plot(ins_wvls, row, c=colors[_species], label=common_name)

                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"NPV (n = {str(df_spectra.shape[0])})", fontsize=12)
                    ax.legend(prop={'size': 6})

                if ax == ax5:
                    df_spectra = em_data[em_data['level_1'] == 'pv'].copy()
                    specific_string = 'UNK-'
                    df_spectra['species'] = df_spectra['species'].fillna(specific_string)
                    df_species_key = pd.read_csv(os.path.join('utils', 'species_santabarbara_ca.csv'))
                    num_species = len(sorted(list(df_spectra.species.unique())))
                    colors = np.random.rand(num_species, 3)

                    for _species, species in enumerate(sorted(list(df_spectra.species.unique()))):
                        df_species = df_spectra[df_spectra['species'] == species].copy()
                        em_spectra = df_species.iloc[:, 10:].to_numpy()

                        for _row, row in enumerate(em_spectra):
                            ax.plot(ins_wvls, row, color=colors[_species])

                        common_name = df_species_key.loc[df_species_key['key_value'] == species, ['label']].values[0][0]
                        ax.plot(ins_wvls, row, c=colors[_species], label=common_name)

                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"PV (n = {str(df_spectra.shape[0])})", fontsize=12)
                    ax.legend(prop={'size': 6})

                if ax == ax6:
                    df_spectra = em_data[em_data['level_1'] == 'soil'].copy()
                    em_spectra = df_spectra.iloc[:, 10:].to_numpy()
                    for _row, row in enumerate(em_spectra):
                        ax.plot(ins_wvls, row, color='blue')
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.text(385, 0.85, f"Soil (n = {str(df_spectra.shape[0])})", fontsize=12)

                if ax == ax3:
                    # plot average spectra
                    df_spectra = transect_data[transect_data['plot_name'] == site + '-' + season.upper()].copy()
                    transect_spectra = df_spectra.iloc[:, 9:].to_numpy()

                    refl_data = sorted(glob(os.path.join(self.base_directory, 'gis', 'shift-data-clip', f"*{plot}_*")))
                    exclude = ['.hdr', '.aux', '.xml']

                    for i in refl_data:
                        if os.path.splitext(i)[1] in exclude:
                            pass
                        else:
                            img_date = datetime.strptime(i.split("_")[-1], "ang%Y%m%dt%H%M%S")
                            refl = envi_to_array(i)
                            refl = np.mean(refl, axis=(0, 1))
                            img_diff = img_date - date
                            img_diff = img_diff.days

                            if np.absolute(img_diff) > 10:
                                pass
                            else:
                                ax.plot(ins_wvls, refl, label=f"{img_date}: {img_diff}")

                    # plot average of SLPIT
                    ax.set_title(f'Field Sample Date: {date}')
                    ax.plot(asd_wavelengths, np.mean(transect_spectra, axis=0), color='black', label=f"ASD Mean Refl")
                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylim(0, 1)
                    ax.set_xlim(320, 2550)
                    ax.legend(fontsize='8')

            plt.savefig(os.path.join(self.figure_directory, 'plot_stats', plot + '.pdf'), format="pdf", dpi=300)
            plt.clf()
            plt.close()
            merger.append(os.path.join(self.figure_directory, 'plot_stats', plot + '.pdf'))

        # write pdf
        merger.write(os.path.join(self.figure_directory, 'plot_stats', 'plot_summary.pdf'))
        merger.close()

    def regression(self):
        # load plots
        df_coords = pd.read_csv('plot_coordinates.csv')

        skip = ['SRB-050_FALL', 'SRB-200_FALL']  #'SRB-047_SPRING', 'SRA-000_SPRING', 'SRB-004_FALL']
        #skip = []

        df_frac_row = []
        for plot in df_coords.iterrows():
            row = plot[1]
            date = row['Date']
            plot_name = row['Plot Name']
            season = row['Season']
            site = plot_name[:2]

            if f"{plot_name}_{season}" in skip:
                continue

            # load slpit file - sma
            sma_local_slpit = sorted(glob(os.path.join(self.output_directory, 'sma', '*emit-asd-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]
            sma_local_aviris = sorted(glob(os.path.join(self.output_directory, 'sma', '*emit-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]

            # load slpit file - local
            mesma_local_slpit = sorted(glob(os.path.join(self.output_directory, 'mesma', '*emit-asd-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]
            mesma_local_aviris = sorted(glob(os.path.join(self.output_directory, 'mesma', '*emit-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]

            # load slpit file - global
            sma_global_slpit = sorted(glob(os.path.join(self.output_directory, 'sma', '*global-asd-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]
            sma_global_aviris = sorted(glob(os.path.join(self.output_directory, 'sma','*global-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]

            # load slpit file - global
            mesma_global_slpit = sorted(glob(os.path.join(self.output_directory, 'mesma', '*global-asd-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]
            mesma_global_aviris = sorted(glob(os.path.join(self.output_directory, 'mesma', '*global-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover')))[0]

            # load slpit file - sma uncertainty
            sma_local_slpit_uncertainty = sorted(glob(os.path.join(self.output_directory, 'sma', '*emit-asd-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]
            sma_local_aviris_uncertainty = sorted(glob(os.path.join(self.output_directory, 'sma', '*emit-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]

            # load slpit file - local uncertainty
            mesma_local_slpit_uncertainty = sorted(glob(os.path.join(self.output_directory, 'mesma', '*emit-asd-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]
            mesma_local_aviris_uncertainty = sorted(glob(os.path.join(self.output_directory, 'mesma', '*emit-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]

            # load slpit file - global uncertainty
            sma_global_slpit_uncertainty = sorted(glob(os.path.join(self.output_directory, 'sma', '*global-asd-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]
            sma_global_aviris_uncertainty = sorted(glob(os.path.join(self.output_directory, 'sma', '*global-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]

            # load slpit file - global uncertainty
            mesma_global_slpit_uncertainty = sorted(glob(os.path.join(self.output_directory, 'mesma', '*global-asd-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]
            mesma_global_aviris_uncertainty = sorted(glob(os.path.join(self.output_directory, 'mesma','*global-aviris-' + plot_name + '_' + season.upper() + '*fractional_cover_uncertainty')))[0]


            # read data in arrays
            sma_local_slpit_array = envi_to_array(sma_local_slpit)
            sma_local_aviris_array = envi_to_array(sma_local_aviris)
            mesma_local_slpit_array = envi_to_array(mesma_local_slpit)
            mesma_local_aviris_array = envi_to_array(mesma_local_aviris)
            sma_global_slpit_array = envi_to_array(sma_global_slpit)
            sma_global_aviris_array = envi_to_array(sma_global_aviris)
            mesma_global_slpit_array = envi_to_array(mesma_global_slpit)
            mesma_global_aviris_array = envi_to_array(mesma_global_aviris)

            sma_local_slpit_array_uncertainty = envi_to_array(sma_local_slpit_uncertainty)
            sma_local_aviris_array_uncertainty = envi_to_array(sma_local_aviris_uncertainty)
            mesma_local_slpit_array_uncertainty = envi_to_array(mesma_local_slpit_uncertainty)
            mesma_local_aviris_array_uncertainty = envi_to_array(mesma_local_aviris_uncertainty)
            sma_global_slpit_array_uncertainty = envi_to_array(sma_global_slpit_uncertainty)
            sma_global_aviris_array_uncertainty = envi_to_array(sma_global_aviris_uncertainty)
            mesma_global_slpit_array_uncertainty = envi_to_array(mesma_global_slpit_uncertainty)
            mesma_global_aviris_array_uncertainty = envi_to_array(mesma_global_aviris_uncertainty)

            all_arrays = [(sma_local_slpit_array, sma_local_slpit_array_uncertainty),
                          (sma_local_aviris_array, sma_local_aviris_array_uncertainty),
                          (mesma_local_slpit_array, mesma_local_slpit_array_uncertainty),
                          (mesma_local_aviris_array, mesma_local_aviris_array_uncertainty),
                          (sma_global_slpit_array, sma_global_slpit_array_uncertainty),
                          (sma_global_aviris_array,sma_global_aviris_array_uncertainty),
                          (mesma_global_slpit_array, mesma_global_slpit_array_uncertainty),
                          (mesma_global_aviris_array, mesma_global_aviris_array_uncertainty)]

            position_array_index = {0: 'sma_local_slpit',
                                    1: 'sma_local_aviris',
                                    2: 'mesma_local_slpit',
                                    3: 'mesma_local_aviris',
                                    4: 'sma_global_slpit',
                                    5: 'sma_global_aviris',
                                    6: 'mesma_global_slpit',
                                    7: 'mesma_global_aviris'}

            df_row = [date, plot_name, season]
            df_cols = []
            for _array, array in enumerate(all_arrays):
                for _em, em in enumerate(self.ems):
                    array[0][array[0] == -9999] = np.nan
                    array[1][array[1] == -9999] = np.nan

                    mean_frac_cover = np.nanmean(array[0][:, :, _em])
                    uncertainty = np.nanmean(array[1][:, :, _em])

                    col_name_frac = f"{position_array_index[_array]}_{em}"
                    col_name_uncer = f"{position_array_index[_array]}_{em}_uncer"
                    df_row.append(mean_frac_cover)
                    df_row.append(uncertainty)
                    df_cols.append(col_name_frac)
                    df_cols.append(col_name_uncer)

            df_frac_row.append(df_row)

        df_frac_merge = pd.DataFrame(df_frac_row)
        df_frac_merge.columns = ['date', 'plot_name', 'season'] + df_cols

        df_frac_merge['date'] = pd.to_datetime(df_frac_merge['date'])
        df_frac_merge['date'] = df_frac_merge['date'].dt.strftime('%Y-%m-%d')
        df_frac_merge.to_csv(os.path.join(self.output_directory, 'fraction_outputs.csv'), index=False)

        # # load quad cover
        df_quad = pd.read_csv(os.path.join(self.output_directory, 'quad_cover.csv'))
        df_quad['date'] = pd.to_datetime(df_quad['date'])
        df_quad['date'] = df_quad['date'].dt.strftime('%Y-%m-%d')
        df_merge_no_fall = df_frac_merge.merge(df_quad, how='left', left_on=['plot_name', 'date'],  right_on=['plot', 'date'])
        df_merge_no_fall.to_csv(os.path.join(self.output_directory, 'fraction_outputs_quadrats.csv'), index=False)
        #df_merge_no_fall = df_merge_no_fall.loc[df_merge_no_fall['season'] != 'SPRING'].copy()

        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        ncols = 3
        nrows = 6
        cmap = plt.get_cmap('viridis')
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        position_rows = {0: ('sma_local_slpit', 'sma_local_aviris'),
                         1: ('mesma_local_slpit', 'mesma_local_aviris'),
                         2: ('sma_global_slpit', 'sma_global_aviris'),
                         3: ('mesma_global_slpit', 'mesma_global_aviris'),
                         4: 'sma_global_slpit',
                         5: 'mesma_global_slpit'}

        for row in range(nrows):
            # if row in [0, 1, 2]:
            #     continue
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_xlabel('SLPIT')

                if row > 3:
                    ax.set_ylabel("Quadrats")
                else:
                    ax.set_ylabel("AVIRIS-NG Fractions")

                # plot npv
                if col == 0:
                    ax.set_title(f"NPV: {position_rows[row][0]}")
                    if row < 4:
                        x = df_merge_no_fall[f"{position_rows[row][0]}_npv"]
                        x_u = df_merge_no_fall[f"{position_rows[row][0]}_npv_uncer"]
                        y = df_merge_no_fall[f"{position_rows[row][1]}_npv"]
                        y_u = df_merge_no_fall[f"{position_rows[row][1]}_npv_uncer"]
                    else:
                        x = df_merge_no_fall[f"{position_rows[row]}_npv"]
                        x_u = df_merge_no_fall[f"{position_rows[row]}_npv_uncer"]
                        y = df_merge_no_fall[f"npv"]
                        y_u = [0] * len(x_u)

                # plot gv
                elif col == 1:
                    ax.set_title(f"GV: {position_rows[row][0]}")
                    if row < 4:
                        x = df_merge_no_fall[f"{position_rows[row][0]}_pv"]
                        x_u = df_merge_no_fall[f"{position_rows[row][0]}_pv_uncer"]
                        y = df_merge_no_fall[f"{position_rows[row][1]}_pv"]
                        y_u = df_merge_no_fall[f"{position_rows[row][1]}_pv_uncer"]
                    else:
                        x = df_merge_no_fall[f"{position_rows[row]}_pv"]
                        x_u = df_merge_no_fall[f"{position_rows[row]}_pv_uncer"]
                        y = df_merge_no_fall[f"pv"]
                        y_u = [0] * len(x_u)

                # plot soil
                else:
                    ax.set_title(f"Soil: {position_rows[row][0]}")
                    if row < 4:
                        x = df_merge_no_fall[f"{position_rows[row][0]}_soil"]
                        x_u = df_merge_no_fall[f"{position_rows[row][0]}_soil_uncer"]
                        y = df_merge_no_fall[f"{position_rows[row][1]}_soil"]
                        y_u = df_merge_no_fall[f"{position_rows[row][1]}_soil_uncer"]
                    else:
                        x = df_merge_no_fall[f"{position_rows[row]}_soil"]
                        x_u = df_merge_no_fall[f"{position_rows[row]}_soil_uncer"]
                        y = df_merge_no_fall[f"soil"]
                        y_u = [0] * len(x_u)

                if np.isnan(y).any():
                    nan_mask = ~np.isnan(y)
                    y = y[nan_mask]
                    x = x[nan_mask]
                    x_u = x_u[nan_mask]
                    y_u = np.array(y_u)
                    y_u = y_u[nan_mask]

                    c = x
                else:
                    c = df_merge_no_fall[f"{position_rows[0][1]}_pv"]

                # Add labels to each point
                count = 0
                for xi, yi,xu,yu, label in zip(x, y, x_u, y_u, df_merge_no_fall['plot_name']):
                    #if label in ['SPEC-026', 'SPEC-022', 'SPEC-003']:
                        #ax.errorbar(xi, yi, yerr=yu, xerr=xu, fmt='o')

                    plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
                    count += 1

                ax.scatter(x,y)
                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', markersize=4, zorder=1)
                scatter = ax.scatter(x, y, c=c, cmap=cmap, edgecolor='black')

                if col == 2:
                    fig.colorbar(scatter, label='GV Cover')

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)

                # plot 1 to 1 line
                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red')

                txtstr = '\n'.join((
                    r'MAE(RMSE): %.2f(%.2f)' % (mae,rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x))))

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', bbox=props)
                #ax.set_aspect(1 / ax.get_data_ratio())

        plt.savefig(os.path.join(self.figure_directory, 'aviris_regression.png'), format='png', dpi=300, bbox_inches="tight")
        plt.show()

    def plot_ems(self):
        em_data = pd.read_csv(os.path.join(self.output_directory, 'all-endmembers-aviris-ng.csv'))

        for plot in em_data['plot_name'].unique():
            create_directory(os.path.join(self.base_directory, 'data', plot))
            df_select = em_data[em_data['plot_name'] == plot].copy()

            p_map(partial(plot_em_asd, out_directory=os.path.join(self.base_directory, 'data', plot),
                          starting_col=10), df_select.iterrows(),
                  **{
                      "desc": "\t\t\t plot: " + plot + " ...",
                      "ncols": 150})

        # merge em df into one
        df_ems_dp = pd.read_csv(os.path.join(self.output_directory, 'spectral_endmembers', 'DPA-9999_FALL_aviris-ng_em.csv'))
        df_ems_sr = pd.read_csv(os.path.join(self.output_directory, 'spectral_endmembers', 'SRA-9999_FALL_aviris-ng_em.csv'))
        df_em_merge = pd.concat([df_ems_sr.reset_index(drop=True), df_ems_dp.reset_index(drop=True)],
                                ignore_index=True)
        df_em_merge = df_em_merge.sort_values("level_1")

        for plot in df_em_merge['plot_name'].unique():
            create_directory(os.path.join(self.base_directory, 'data', plot))
            df_select = df_em_merge[df_em_merge['plot_name'] == plot].copy()

            p_map(partial(plot_em_asd, out_directory=os.path.join(self.base_directory, 'data', plot),
                          starting_col=10), df_select.iterrows(),
                  **{"desc": "\t\t\t plot: " + plot + " ...", "ncols": 150})


    def fraction_time_cubes(self):
        images = glob(os.path.join(self.output_directory, 'mesma', '*mosaic-*'))
        exclude = ['.hdr', '.csv', '.ini', '.xml']

        #fraction_grid = np.zeros(, len())

        array_shapes = []
        for reflectance_file in images:
            if os.path.splitext(reflectance_file)[1] in exclude or os.path.basename(reflectance_file).split("_")[-1] == 'uncertainty' \
                    or os.path.basename(reflectance_file).split("_")[-1] == 'fractions':
                continue

            array = envi_to_array(reflectance_file)
            array_shapes.append(array.shape)

    def frac_sync_emit(self):
        fraction_files_sma = sorted(glob(os.path.join(self.output_directory, 'sma', '*fractional_cover')))
        fraction_files_mesma = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover')))
        all_files = fraction_files_sma + fraction_files_mesma
        exclude = 'mosaic'
        filtered_file_names = [file_name for file_name in all_files if exclude not in file_name]

        results = p_map(fraction_file_info, filtered_file_names, **{"desc": "\t\t retrieving mean fractional cover: ...", "ncols": 150})
        df_all = pd.DataFrame(results)
        df_all.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization', 'npv', 'pv', 'soil', 'shade']
        df_all.to_csv(os.path.join(self.figure_directory, 'fraction_output_shift.csv'), index=False)

        # uncertainty
        fraction_files_sma_uncer = sorted(glob(os.path.join(self.output_directory, 'sma', '*fractional_cover_uncertainty')))
        fraction_files_mesma_uncer = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover_uncertainty')))
        all_files_uncer = fraction_files_sma_uncer + fraction_files_mesma_uncer
        exclude = 'mosaic'
        filtered_file_names = [file_name for file_name in all_files_uncer if exclude not in file_name]

        results = p_map(fraction_file_info, filtered_file_names, **{
            "desc": "\t\t retrieving mean fractional cover uncertainty: ...",
            "ncols": 150})
        df_all = pd.DataFrame(results)
        df_all.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization',
                          'npv', 'pv', 'soil', 'shade']
        df_all.to_csv(os.path.join(self.figure_directory, 'fraction_output_shift_uncertainty.csv'), index=False)

    def animations(self, boundary=None):
        images = glob(os.path.join(self.output_directory, 'mesma', '*mosaic-*'))

        exclude = ['.hdr', '.csv', '.ini', '.xml']

        shapefile_path = os.path.join('gis', 'sedgewick_boundary_utm.shp')
        shape_data = gpd.read_file(shapefile_path)

        frames = []
        fig, ax = plt.subplots()
        title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')
        ax.axis('off')

        cmap = plt.get_cmap('viridis')

        def update(frame):
            reflectance_file = images[frame]

            if os.path.splitext(reflectance_file)[1] in exclude or os.path.basename(reflectance_file).split("_")[-1] == 'uncertainty' \
                    or os.path.basename(reflectance_file).split("_")[-1] == 'fractions':
                pass

            date = os.path.basename(reflectance_file).split("_")[0].split('-')[1]
            img_original = os.path.join(self.base_directory, 'airborne', 'SHIFT', 'jean_original', f"{date}")

            with rasterio.open(img_original) as envi_data_original:
                transform = envi_data_original.transform

            with rasterio.open(reflectance_file) as envi_data:
                soil = envi_data.read(3)
                pv = envi_data.read(2)
                npv = envi_data.read(1)

            rgb_image = np.array([npv, pv, soil]).transpose((1, 2, 0))
            extent = [transform.c, transform.c + transform.a * rgb_image.shape[1],
                      transform.f + transform.e * rgb_image.shape[0], transform.f]

            ax.set_xlim([transform.c, transform.c + transform.a * rgb_image.shape[1]])
            ax.set_ylim([transform.f + transform.e * rgb_image.shape[0], transform.f])
            shape_data.boundary.plot(ax=ax, color='black', facecolor='none')

            title.set_text(f"Sedgewick Reserve: {date}")
            alpha_channel = np.ones(rgb_image.shape[:2])
            alpha_channel[np.isnan(npv)] = 0

            cur_fig = ax.imshow(np.ma.masked_invalid(rgb_image), extent=extent, cmap=cmap, aspect='auto', alpha=alpha_channel)

            return cur_fig,

        ani = FuncAnimation(fig, update, frames=len(images), interval=100, blit=True, repeat_delay=1000)
        ani.save(os.path.join(self.figure_directory, 'sedgewick_animation.gif'))

    def plot_rmse(self):
        # load fraction plots
        df_emit = pd.read_csv(os.path.join(self.figure_directory, 'fraction_output_emit.csv'))
        df_emit['campaign'] = 'emit'
        df_aviris = pd.read_csv(os.path.join(self.figure_directory, 'fraction_output_shift.csv'))
        df_aviris['campaign'] = 'shift'
        df_all = pd.concat([df_emit, df_aviris], ignore_index=True)

        # load all uncertainty files
        df_emit_uncer = pd.read_csv(os.path.join(self.figure_directory, 'uncertainty_output_emit.csv'))
        df_emit_uncer['campaign'] = 'emit'
        df_aviris_uncer = pd.read_csv(os.path.join(self.figure_directory, 'fraction_output_shift_uncertainty.csv'))
        df_aviris_uncer['campaign'] = 'shift'
        df_all_uncer = pd.concat([df_emit_uncer, df_aviris_uncer], ignore_index=True)

        # load cpu report for shift data
        df_cpu_aviris = pd.read_csv(os.path.join(self.figure_directory, 'computing_performance_report.csv'))
        df_cpu_emit = pd.read_csv(os.path.join(self.figure_directory, 'computing_performance_report_emit.csv'))

        #  create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        ncols = 3
        nrows = 4
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        # # loop through figure columns
        for row in range(nrows):
            if row == 0:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'emit')].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'shift')].copy()
                df_uncer_emit = df_all_uncer[(df_all_uncer['unmix_mode'] == 'sma') & (df_all_uncer['lib_mode'] == 'local')& (df_all_uncer['campaign'] == 'emit')].copy()
                df_uncer_shift = df_all_uncer[(df_all_uncer['unmix_mode'] == 'sma') & (df_all_uncer['lib_mode'] == 'local') & (df_all_uncer['campaign'] == 'shift')].copy()
                df_performance = df_cpu_aviris[(df_cpu_aviris['library'] == 'local') & (df_cpu_aviris['mode'] == '"sma"')].copy()
                df_performance_emit = df_cpu_emit[(df_cpu_emit['library'] == 'local') & (df_cpu_emit['mode'] == '"sma"')].copy()

            if row == 2:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'emit')].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'sma') & (df_all['lib_mode'] == 'global') & (df_all['campaign'] == 'shift')].copy()
                df_uncer_emit = df_all_uncer[(df_all_uncer['unmix_mode'] == 'sma') & (df_all_uncer['lib_mode'] == 'global') & (df_all_uncer['campaign'] == 'emit')].copy()
                df_uncer_shift = df_all_uncer[(df_all_uncer['unmix_mode'] == 'sma') & (df_all_uncer['lib_mode'] == 'global') & (df_all_uncer['campaign'] == 'shift')].copy()
                df_performance = df_cpu_aviris[(df_cpu_aviris['library'] == 'global') & (df_cpu_aviris['mode'] == '"sma"')].copy()
                df_performance_emit = df_cpu_emit[(df_cpu_emit['library'] == 'global') & (df_cpu_emit['mode'] == '"sma"')].copy()
            if row == 1:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['num_mc'] == 25) & (df_all['campaign'] == 'emit')].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'local') & (df_all['campaign'] == 'shift')].copy()
                df_uncer_emit = df_all_uncer[(df_all_uncer['unmix_mode'] == 'mesma') & (df_all_uncer['lib_mode'] == 'local') & (df_all_uncer['num_mc'] == 25) & (df_all_uncer['campaign'] == 'emit')].copy()
                df_uncer_shift = df_all_uncer[(df_all_uncer['unmix_mode'] == 'mesma') & (df_all_uncer['lib_mode'] == 'local') & (df_all_uncer['campaign'] == 'shift')].copy()
                df_performance = df_cpu_aviris[(df_cpu_aviris['library'] == 'local') & (df_cpu_aviris['mode'] == '"mesma"')].copy()
                df_performance_emit = df_cpu_emit[(df_cpu_emit['library'] == 'local') & (df_cpu_emit['mode'] == '"mesma"')].copy()
            if row == 3:
                df_select_emit = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25) & (df_all['campaign'] == 'emit')].copy()
                df_select_shift = df_all[(df_all['unmix_mode'] == 'mesma') & (df_all['lib_mode'] == 'global') & (df_all['num_mc'] == 25)& (df_all['campaign'] == 'shift')].copy()
                df_uncer_emit = df_all_uncer[(df_all_uncer['unmix_mode'] == 'mesma') & (df_all_uncer['lib_mode'] == 'global') & (df_all_uncer['num_mc'] == 25) & (df_all_uncer['campaign'] == 'emit')].copy()
                df_uncer_shift = df_all_uncer[(df_all_uncer['unmix_mode'] == 'mesma') & (df_all_uncer['lib_mode'] == 'global') & (df_all_uncer['campaign'] == 'shift')].copy()
                df_performance = df_cpu_aviris[(df_cpu_aviris['library'] == 'global') & (df_cpu_aviris['mode'] == '"mesma"')].copy()
                df_performance_emit = df_cpu_emit[(df_cpu_emit['library'] == 'global') & (df_cpu_emit['mode'] == '"mesma"')].copy()

            aviris_performance = df_performance['spectra_per_s'].mean()
            emit_performance = df_performance_emit['spectra_per_s'].mean()

            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_xlabel('SLPIT')
                ax.set_ylabel("Spaceborne/Ground Fractions")
                ax.grid('on', linestyle='--')

                mode = list(df_select_emit['unmix_mode'].unique())[0]
                lib_mode = list(df_select_emit['lib_mode'].unique())[0]

                ax.set_title(f'{self.ems[col]} - {mode} - {lib_mode} Library')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # plot 1 to 1 line
                one_line = np.linspace(0, 1, 101)
                ax.plot(one_line, one_line, color='red')

                # emit variables
                df_x_emit = df_select_emit[(df_select_emit['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y_emit = df_select_emit[(df_select_emit['instrument'] == 'emit')].copy().reset_index(drop=True)
                df_x_emit_uncer = df_uncer_emit[(df_uncer_emit['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y_emit_uncer = df_uncer_emit[(df_uncer_emit['instrument'] == 'emit')].copy().reset_index(drop=True)

                # aviris variables
                df_x_shift = df_select_shift[(df_select_shift['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y_shift = df_select_shift[(df_select_shift['instrument'] == 'aviris')].copy().reset_index(drop=True)
                df_x_shift_uncer = df_uncer_shift[(df_uncer_shift['instrument'] == 'asd')].copy().reset_index(drop=True)
                df_y_shift_uncer = df_uncer_shift[(df_uncer_shift['instrument'] == 'aviris')].copy().reset_index(drop=True)

                # plot fractional cover values
                x_emit = df_x_emit[self.ems[col]]
                y_emit = df_y_emit[self.ems[col]]

                x_shift = df_x_shift[self.ems[col]]
                y_shift = df_y_shift[self.ems[col]]

                x_emit_uncer = df_x_emit_uncer[self.ems[col]]
                y_emit_uncer = df_y_emit_uncer[self.ems[col]]

                x_shift_uncer = df_x_shift_uncer[self.ems[col]]
                y_shift_uncer = df_y_shift_uncer[self.ems[col]]

                x = list(x_emit.values) + list(x_shift.values)
                y = list(y_emit.values) + list(y_shift.values)
                x_u = list(x_emit_uncer.values) + list(x_shift_uncer.values)
                y_u = list(y_emit_uncer.values) + list(y_shift_uncer.values)

                ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='none', markersize=4, zorder=1)
                emit_scatter = ax.scatter(x_emit,y_emit, marker='s', edgecolor='black', label='EMIT')
                shift_scatter = ax.scatter(x_shift,y_shift, marker='^', edgecolor='black', label='AVIRIS$_{ng}$')

                # Add labels to each point
                # for xi, yi,xu,yu, label in zip(x, y, x_u, y_u, df_x['plot']):
                #     ax.errorbar(xi, yi, yerr=yu, xerr=xu, fmt='o')
                #     plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

                if col == 2:
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                # Add error metrics
                rmse = mean_squared_error(x, y, squared=False)
                mae = mean_absolute_error(x, y)
                r2 = r2_calculations(x, y)
                m, b = np.polyfit(x, y, 1)



                txtstr = '\n'.join((
                     r'MAE(RMSE): %.2f(%.2f)' % (mae,rmse),
                    r'R$^2$: %.2f' % (r2,),
                    r'n = ' + str(len(x)),
                    r'EMIT(AVIRIS) Spectra/s :  %.2f(%.2f)' % (emit_performance,aviris_performance)
                ))

                # plot predicted line
                ax.plot(one_line, m*one_line + b, color='black')

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
                ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(self.figure_directory, 'regression_combined.png'), format="png", dpi=300, bbox_inches="tight")# load all fraction files

    def mesma_cmbs(self):
        df_cpu = pd.read_csv(os.path.join(self.figure_directory, 'computing_performance_report.csv'))
        df_cpu = df_cpu[df_cpu['mode'] == '"mesma"'].copy()

        fraction_files = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover')))
        fraction_files = sorted([file for file in fraction_files if not any(file.endswith(ext) for ext in self.exclude)])

        uncertainty_files = sorted(glob(os.path.join(self.output_directory, 'mesma', '*fractional_cover_uncertainty')))
        uncertainty_files = sorted([file for file in uncertainty_files if not any(file.endswith(ext) for ext in self.exclude)])

        if os.path.isfile(os.path.join(self.figure_directory, 'mesma_fraction_outputs.csv')) and os.path.isfile(os.path.join(self.figure_directory, 'mesma_uncertainty_outputs.csv')):
            df_all = pd.read_csv(os.path.join(self.figure_directory, 'mesma_fraction_outputs.csv'))
            df_uncertainty = pd.read_csv(os.path.join(self.figure_directory, 'mesma_uncertainty_outputs.csv'))

        else:
            results = p_map(fraction_file_info, fraction_files,
                            **{"desc": "\t\t retrieving mean fractional cover: ...", "ncols": 150})
            df_all = pd.DataFrame(results)
            df_all.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization',
                              'npv', 'pv', 'soil', 'shade']
            df_all.to_csv(os.path.join(self.figure_directory, 'mesma_fraction_outputs.csv'), index=False)

            results_uncertainty = p_map(fraction_file_info, uncertainty_files,
                                        **{"desc": "\t\t retrieving mean fractional cover uncertainty: ...", "ncols": 150})
            df_uncertainty = pd.DataFrame(results_uncertainty)
            df_uncertainty.columns = ['instrument', 'unmix_mode', 'plot', 'lib_mode', 'num_cmb_em', 'num_mc', 'normalization',
                              'npv', 'pv', 'soil', 'shade']
            df_uncertainty.to_csv(os.path.join(self.figure_directory, 'mesma_uncertainty_outputs.csv'), index=False)

        emit_log = []
        global_log = []
        shift_log = []
        performance_time_global = []
        performance_time_local = []
        performance_time_emit = []
        emit_log_uncertainty = []
        global_log_uncertainty = []
        shift_log_uncertainty = []

        for lib_mode in sorted((df_cpu.library.unique())):
            for cmbs in sorted(list(df_cpu.max_combinations.unique())):
                df_frac_cmbs = df_all[(df_all['num_cmb_em'] == cmbs) & (df_all['lib_mode'] == lib_mode)]
                df_cpu_cmbs = df_cpu[(df_cpu['max_combinations'] == cmbs) & (df_cpu['library'] == lib_mode)]
                df_uncer_cmbs = df_uncertainty[(df_uncertainty['num_cmb_em'] == cmbs) & (df_uncertainty['lib_mode'] == lib_mode)]

                cpu_avg = df_cpu_cmbs['total_time'].mean()
                ins_x, ins_y =sorted(list(df_frac_cmbs.instrument.unique()))

                error = []
                uncer = []

                for em in self.ems:
                    df_x = df_frac_cmbs[(df_frac_cmbs['instrument'] == ins_x)]
                    df_y = df_frac_cmbs[(df_frac_cmbs['instrument'] == ins_y)]
                    x = df_x[em]
                    y = df_y[em]

                    df_x_u = df_uncer_cmbs[(df_uncer_cmbs['instrument'] == ins_x)]
                    df_y_u = df_uncer_cmbs[(df_uncer_cmbs['instrument'] == ins_y)]
                    x_u = df_x_u[em]
                    y_u = df_y_u[em]

                    avg_uncer_x = np.mean(x_u)
                    avg_uncer_y = np.mean(y_u)

                    uncer.append(avg_uncer_x)
                    uncer.append(avg_uncer_y)

                    mae = mean_absolute_error(x, y)
                    error.append(mae)

                if lib_mode == 'emit':
                    emit_log.append(error)
                    performance_time_emit.append(cpu_avg)
                    emit_log_uncertainty.append(uncer)
                elif lib_mode == 'local':
                    shift_log.append(error)
                    performance_time_local.append(cpu_avg)
                    shift_log_uncertainty.append(uncer)
                else:
                    global_log.append(error)
                    performance_time_global.append(cpu_avg)
                    global_log_uncertainty.append(uncer)

        # stack error results
        global_log_array = np.vstack(global_log)
        emit_log_array = np.vstack(emit_log)
        shift_log_array = np.vstack(shift_log)

        # stack uncertainty
        global_uncertainty_array = np.vstack(global_log_uncertainty)
        emit_uncertainty_array = np.vstack(emit_log_uncertainty)
        shift_uncertainty_array = np.vstack(shift_log_uncertainty)


        # create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        ncols = 3
        nrows = 1
        gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.025, hspace=0.0001, figure=fig)

        x_vals = sorted(list(df_cpu.max_combinations.unique()))
        uncer_indices = {0: [0,1], 1: [2,3], 2: [4,5]}

        for col in range(ncols):
            ax = fig.add_subplot(gs[col])

            ax.set_title(f'{self.ems[col]}')
            ax.set_xlabel('MESMA Models/Combinations')
            ax.set_ylabel("MAE")
            ax.grid('on', linestyle='--')

            line2 = ax.errorbar(x_vals, global_log_array[:, col], yerr=global_uncertainty_array[:, uncer_indices[col][1]], capsize=3, color='blue', fmt='', label=f"{ins_y} Uncertainty", linestyle='none')
            line3 = ax.errorbar(x_vals, global_log_array[:, col], yerr=global_uncertainty_array[:, uncer_indices[col][0]], capsize=5, color='green', fmt='', label=f"{ins_x} Uncertainty", linestyle='none')
            line1, = ax.plot(x_vals, global_log_array[:, col], label='EMIT Global Lib (4D Convex Hull)', color='red')

            ax.set_ylim(-0.25, 0.5)
            ax.set_xlim(0, 105)
            ax.legend()
            #ax.set_aspect(1 / ax.get_data_ratio())

            ax2 = ax.twinx()
            line4, = ax2.plot(x_vals, performance_time_global, label='s', color='red', linestyle='dashed', linewidth=0.5)
            ax2.set_ylabel("Total time (s)")
            #ax2.set_ylim(0, 1.5)

            handles = [line1, line2, line3, line4]
            labels = [h.get_label() for h in handles]
            ax.legend(handles=handles, labels=labels, loc='upper right')

        plt.savefig(os.path.join(self.figure_directory, 'mesma_model_opt.png'), format="png", dpi=300, bbox_inches="tight")