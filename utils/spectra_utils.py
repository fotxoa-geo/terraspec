import numpy as np
import isofit.core.common as isc
import os
import itertools
from sklearn.decomposition import PCA
import pandas as pd
import time
from p_tqdm import p_map
from utils.envi import get_meta, save_envi
from utils import asdreader
from glob import glob
import geopandas as gpd
import matplotlib.pyplot as plt


def get_dd_coords(coord):
    dd_mm = float(str(coord).split(".")[0][-2:] + "." + str(coord).split(".")[1])/60
    dd_dd = float(str(coord).split(".")[0][:-2])
    dd = dd_dd + dd_mm
    return dd

# bad wavelength regions
bad_wv_regions = [[0, 440], [1310, 1490], [1770, 2050], [2440, 2880]]


def gps_asd(latitude_ddmm, longitude_ddmm, file):
    try:
        dd_lat = get_dd_coords(latitude_ddmm)
        dd_long = get_dd_coords(longitude_ddmm) * -1  # used to correct longitude

    except:
        gdf = gpd.read_file('gis/Observation.shp')
        gdf['longitude'] = gdf['geometry'].x
        gdf['latitude'] = gdf['geometry'].y
        plot_name = os.path.basename(os.path.dirname(file)).replace('Spectral', 'SPEC')
        df = gdf.drop(columns='geometry')

        long = df.loc[(df['Name'] == plot_name), 'longitude'].iloc[0]
        lat = df.loc[(df['Name'] == plot_name), 'latitude'].iloc[0]

        dd_lat = lat
        dd_long = long

    return dd_lat, dd_long

class spectra:
    "spectra class allows for different calls for instrument and asd wavelengths"
    def __init__(self):
        print("")

    @classmethod
    def load_wavelengths(cls, sensor: str):
        wavelength_file = os.path.join('utils', 'wavelengths', sensor + '_wavelengths.txt')
        wl = np.loadtxt(wavelength_file, usecols=1)
        fwhm = np.loadtxt(wavelength_file, usecols=2)
        if np.all(wl < 100):
            wl *= 1000
            fwhm *= 1000
        return wl, fwhm

    @classmethod
    def get_good_bands_mask(cls, wavelengths, wavelength_pairs:None):
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

    @classmethod
    def convolve(cls, df_row, wvl, fwhm, asd_wvl, spectra_starting_col):
        # Convolve spectra
        refl_convolve = isc.resample_spectrum(x=df_row[1].values[spectra_starting_col:], wl=asd_wvl, wl2=wvl, fwhm2=fwhm,
                                              fill=False)
        return refl_convolve

    @classmethod
    def convolve_asdfile(cls, asd_file, wvl, fwhm):
        # Load asd data
        data = asdreader.reader(asd_file)
        asd_wl = data.wavelengths
        asd_refl = np.round(data.reflectance, 4)

        # Convolve spectra
        refl_convolve = isc.resample_spectrum(x=asd_refl, wl=asd_wl, wl2=wvl, fwhm2=fwhm, fill=False)

        return refl_convolve

    @classmethod
    def load_asd_wavelenghts(cls):
        wavelengths_asd = np.linspace(350, 2500, 2151).tolist()

        return wavelengths_asd

    @classmethod
    def load_global_library(cls, output_directory):
        df = pd.read_csv(os.path.join(output_directory, 'convolved', 'geofilter_convolved.csv'))
        return df

    @classmethod
    def latin_hypercubes(cls, points, get_quadrants_index=False):
        ndims = points.shape[1]
        existing_quadrants = list(itertools.product([-1, 1], repeat=ndims))
        quadrants_dict = dict(zip(existing_quadrants, range(len(existing_quadrants))))
        sign_points = np.sign(points)
        quadrants_idx = np.apply_along_axis(lambda x: quadrants_dict[(tuple(x.astype(int)))], 1, sign_points)

        if get_quadrants_index:
            return quadrants_idx
        points_split_into_quadrants = []

        for i in set(quadrants_idx):
            points_split_into_quadrants.append(points[quadrants_idx == i])

        return points_split_into_quadrants

    @classmethod
    def pca_analysis(cls, df, spectra_starting_col:int):

        # target values
        df_select = df.loc[(df['level_1'] == 'soil')].copy()
        metadata = df_select.iloc[:, :spectra_starting_col].reset_index(drop=True)

        # Separating out the features
        x = df_select.iloc[:, spectra_starting_col:].values

        # # PCA analysis for chosen em
        pca = PCA(n_components=x.shape[1])
        df_pca = pd.DataFrame(pca.fit_transform(x))
        df_pca = pd.concat([metadata, df_pca], axis=1)

        return df_pca

    @classmethod
    def increment_synthetic_reflectance(cls, data, em, em_fraction, seed, wvls, spectra_start):
        np.random.seed(seed)

        # calculate the fractions
        if em == 'soil':
            soil_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            npv_frac = pv_frac = remaining_fraction/2

        if em == 'npv':
            npv_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            soil_frac = pv_frac = remaining_fraction / 2

        if em == 'pv':
            pv_frac = em_fraction
            remaining_fraction = 1 - em_fraction
            npv_frac = soil_frac = remaining_fraction / 2

        fractions = [npv_frac, pv_frac, soil_frac]

        # crate grid to store reflectance
        col_spectra = np.zeros((data.shape[0], len(wvls)))

        # create grid to store index
        col_index = np.zeros((data.shape[0], 3)).astype(int)

        for _row, row in enumerate(data):
            col_index[_row, :] = list(map(int, [row[0][0], row[1][0], row[2][0]]))
            col_spectra[_row, :] = (row[0][spectra_start:].astype(dtype=float) * npv_frac) + \
                                      (row[1][spectra_start:].astype(dtype=float) * pv_frac) + \
                                      (row[2][spectra_start:].astype(dtype=float) * soil_frac)

        return fractions, col_index, col_spectra


    @classmethod
    def synthetic_reflectance(cls, data):
        row_spectra = []
        row_fractions = []
        row_index = [data[2][0, 0], data[2][1, 0], data[2][2, 0]]

        for seed in data[1]:
            np.random.seed(seed)
            fractions = np.random.dirichlet(np.ones(3))

            spectra = np.array(data[0][0]).astype(dtype=float) * fractions[0] + \
                      np.array(data[0][1]).astype(dtype=float) * fractions[1] + \
                      np.array(data[0][2]).astype(dtype=float) * fractions[2]
            row_spectra.append(spectra)
            row_fractions.append(fractions)

        return row_spectra, row_fractions, row_index

    @classmethod
    def increment_reflectance(cls, class_names: list, simulation_table: str, level: str, spectral_bundles:int,
                              increment_size:float, output_directory: str, wvls, name: str, spectra_starting_col:int,
                              endmember:str):
        ts = time.time()
        # define seed for random sampling of spectral bundles
        np.random.seed(13)

        df = simulation_table
        df = df.reset_index(drop=True)
        df.insert(0, 'index', df.index)
        class_lists = []

        for em in class_names:
            df_select = df.loc[df[level] == em].copy()
            df_select = df_select.values.tolist()
            class_lists.append(df_select)

        all_combinations = list(itertools.product(*class_lists))

        if len(all_combinations) < spectral_bundles:
            index = np.random.choice(len(all_combinations), replace=False, size=len(all_combinations))
        else:
            index = np.random.choice(len(all_combinations), replace=False, size=spectral_bundles)

        cols = int(1 / increment_size) + 1
        spectra_all = [all_combinations[i] for i in index]
        fraction_grid = np.zeros((len(index), cols, len(class_names)))
        spectra_grid = np.zeros((len(index), cols, len(wvls)))
        index_grid = np.zeros((len(index), cols, len(class_names)))

        # spectra array - combinations x # of classes (each col is an em) x wavelengths
        spec_array = np.array(spectra_all)

        for _col, col in enumerate(range(0, cols)):
            col_frac = np.round(col * increment_size, 2)
            col_fractions, col_index, col_spectra = spectra.increment_synthetic_reflectance(data=spec_array, em=endmember,
                                                                                            wvls=wvls,
                                                                                            em_fraction=col_frac,
                                                                                            seed=_col, spectra_start=spectra_starting_col)

            fraction_grid[:, _col, :] = col_fractions
            spectra_grid[:, _col, :] = col_spectra
            index_grid[:, _col, :] = col_index

        # save the datasets
        refl_meta = get_meta(lines=len(index), samples=cols, bands=wvls, wvls=True)
        index_meta = get_meta(lines=len(index), samples=cols, bands=class_names, wvls=False)
        fraction_meta = get_meta(lines=len(index), samples=cols, bands=class_names, wvls=False)

        # save index, spectra, fraction grid
        output_files = [os.path.join(output_directory, name + '_index.hdr'),
                        os.path.join(output_directory, name + '_spectra.hdr'),
                        os.path.join(output_directory, name + '_fractions.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta]
        grids = [index_grid, spectra_grid, fraction_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})
        del index_grid, spectra_grid, fraction_grid

    @classmethod
    def generate_reflectance(cls, class_names: list, simulation_table: str, level: str, spectral_bundles:int, cols:int,
                             output_directory: str, wvls, name: str, spectra_starting_col:int):

        ts = time.time()
        # define seed for random sampling of spectral bundles
        np.random.seed(13)

        df = simulation_table
        df = df.reset_index(drop=True)
        df.insert(0, 'index', df.index)
        class_lists = []

        for em in class_names:
            df_select = df.loc[df[level] == em].copy()
            df_select = df_select.values.tolist()
            class_lists.append(df_select)

        all_combinations = list(itertools.product(*class_lists))

        if len(all_combinations) < spectral_bundles:
            index = np.random.choice(len(all_combinations), replace=False, size=len(all_combinations))
        else:
            index = np.random.choice(len(all_combinations), replace=False, size=spectral_bundles)

        spectra_all = [all_combinations[i] for i in index]
        fraction_grid = np.zeros((len(index), cols, len(class_names)))
        spectra_grid = np.zeros((len(index), cols, len(wvls)))
        index_grid = np.zeros((len(index), cols, len(class_names)))

        # spectra array - combinations x # of classes (each col is an em) x wavelengths
        spec_array = np.array(spectra_all)

        # Process row in parallel
        seeds = list(range(0, len(index) * cols))
        seeds_array = np.asarray(seeds)
        seeds_array = seeds_array.reshape(len(index), cols)

        # parallel spectra processes ; # we are using +1 since we added an index identifier
        process_spectra = p_map(spectra.synthetic_reflectance,
                                [(spec_array[_row, :, spectra_starting_col + 1:], seeds_array[_row, :], spec_array[_row, :, :4]) for _row, row
                                 in enumerate(spectra_grid)], **{"desc": "\t\t generating fractions...", "ncols": 150})

        # Populate results row by row
        for _row, row in enumerate(process_spectra):
            for _col, (refl, frac) in enumerate(zip(row[0], row[1])):
                spectra_grid[_row, _col, :] = refl
                fraction_grid[_row, _col, :] = frac
                index_grid[_row, _col, :] = np.array(row[2])

        # save the datasets
        refl_meta = get_meta(lines=len(index), samples=cols, bands=wvls, wvls=True)
        index_meta = get_meta(lines=len(index), samples=cols, bands=class_names, wvls=False)
        fraction_meta = get_meta(lines=len(index), samples=cols, bands=class_names, wvls=False)

        # save index, spectra, fraction grid
        output_files = [os.path.join(output_directory, name + '_index.hdr'),
                        os.path.join(output_directory, name + '_spectra.hdr'),
                        os.path.join(output_directory, name + '_fractions.hdr')]

        meta_docs = [index_meta, refl_meta, fraction_meta]
        grids = [index_grid, spectra_grid, fraction_grid]

        p_map(save_envi, output_files, meta_docs, grids, **{"desc": "\t\t saving envi files...", "ncols": 150})
        del index_grid, spectra_grid, fraction_grid

    @classmethod
    def simulate_reflectance(cls, df_sim, df_unmix, dimensions, sim_libraries_output, mode, level, spectral_bundles, cols,
                             output_directory, wvls, spectra_starting_col:int):
        """
        @param df_sim: Simulation csv format
        @param df_unmix: Unmixing library
        @param dimensions: dimensions used in convex hull or PCA
        @param sim_libraries_output: output to save csv for simulation
        @param mode: latin hypercube, convex hull, or geographic
        @param level: column having spectral em classification
        @param spectral_bundles: spectral bundles with
        @param cols: number of columns to use in output
        @param output_directory: directory to save outputs
        @param wvls: instrument wavelengths
        @param spectra_starting_col: spectral starting column of dataframe
        @return: none
        """
    
        # check for duplicates from dataframes using actual wavelengths
        df_sim_array = df_sim.iloc[:, spectra_starting_col:].to_numpy()
        df_unmix_array = df_unmix.iloc[:, spectra_starting_col:].to_numpy()

        # check for duplicates again
        dup_check = list((df_unmix_array[None, :] == df_sim_array[:, None]).all(-1).any(0))
        if dup_check.count(True) > 0:
            raise Exception(
                "The simulation found duplicates in both the simulation and unmixing library at: " + str(dimensions))

        df_sim.to_csv(
            os.path.join(sim_libraries_output, mode + '__n_dims_' + str(dimensions) + '_simulation_library.csv'),
            index=False)

        # # create the reflectance file
        spectra.generate_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim, level=level,
                         spectral_bundles=spectral_bundles, cols=cols, output_directory=output_directory,
                         wvls=wvls, name=mode + '__n_dims_' + str(dimensions), spectra_starting_col=spectra_starting_col)

    @classmethod
    def get_reflectance_endmember(cls, df_row, plot_directory:str, team_name_key:str):
        file_num = df_row[0]
        em_classification = df_row[1]
        species = df_row[2]
        plot_name = os.path.basename(plot_directory)
        file_name = os.path.join(plot_directory, team_name_key + "_" + f"{file_num:05d}.asd")
        asd = asdreader.reader(file_name)
        asd_refl = asd.reflectance
        asd_gps = asd.get_gps()
        latitude_ddmm, longitude_ddmm, elevation, utc_time = asd_gps[0], asd_gps[1], asd_gps[2], asd_gps[3]
        utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])

        dd_lat, dd_long = gps_asd(latitude_ddmm=latitude_ddmm, longitude_ddmm=longitude_ddmm, file=file_name)

        return [plot_name, file_name, em_classification, species, dd_long, dd_lat, elevation, utc_time] + list(asd_refl)

    @classmethod
    def get_reflectance_transect(cls, file, plot_directory:str, team_name_key:str):
        plot_name = os.path.basename(plot_directory)
        asd = asdreader.reader(file)
        asd_refl = asd.reflectance
        asd_gps = asd.get_gps()
        latitude_ddmm, longitude_ddmm, elevation, utc_time = asd_gps[0], asd_gps[1], asd_gps[2], asd_gps[3]

        if int(utc_time[0]) + int(utc_time[1]) + int(utc_time[2]) == 0:
            file_time = asd.get_save_time()
            utc_time = str(file_time[2]) + ":" + str(file_time[1]) + ":" + str(file_time[0])
        else:
            utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])

        file_num = int(os.path.basename(file).split(".")[0].split("_")[1])

        dd_lat, dd_long = gps_asd(latitude_ddmm=latitude_ddmm, longitude_ddmm=longitude_ddmm, file=file)

        return [plot_name, file, file_num, dd_long, dd_lat, elevation, utc_time] + list(asd_refl)

    @classmethod
    def first_derivative(cls, df_row, spectral_starting_col, wvls):
        spectral_sample = np.array(df_row[1].values[spectral_starting_col:])

        first_derivative = []
        for _i, i in enumerate(wvls):
            # last position is a duplicate of previous ?
            if _i == len(wvls) - 1:
                first_derivative.append(first_derivative[-1])
            else:
                ds = spectral_sample[_i + 1] - spectral_sample[_i]
                dx = wvls[_i + 1] - wvls[_i]
                first_derivative.append(ds/dx)

        return first_derivative

    @classmethod
    def get_all_ems(cls,output_directory: str, instrument: str):
        spectral_endmembers = glob(os.path.join(output_directory, 'spectral_endmembers', '*' + instrument + ".csv"))
        emit_transect_endmembers = glob(os.path.join(output_directory, 'spectral_transects', 'endmembers', '*' + instrument + ".csv"))
        all_ems = spectral_endmembers + emit_transect_endmembers

        return all_ems

    @classmethod
    def df_to_shapefile(cls,df, base_directory: str, out_name):
        df_shp = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        df_shp.to_file(os.path.join(base_directory, "gis", out_name + '.shp'), driver='ESRI Shapefile')

    @classmethod
    def save_df_em(cls, df, output, instrument):
        df = df.sort_values('level_1')
        df.to_csv(output.replace(" ", "") + '-' + instrument + '.csv', index=False)

    @classmethod
    def df_to_envi(cls, df, spectral_starting_column:int, wvls, output_raster):

        df_array = df.iloc[:, spectral_starting_column:].to_numpy()
        spectra_grid = np.zeros((df_array.shape[0], 1, len(wvls)))

        # fill spectral data
        for _row, row in enumerate(df_array):
            spectra_grid[_row, 0, :] = row

        # save the spectra
        print('\t\t\tcreating reflectance file...', sep=' ', end='', flush=True)
        meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls,
                                wvls=True)
        save_envi(output_raster, meta_spectra, spectra_grid)

    @classmethod
    def plot_asd_file(cls, asd_file, out_directory):
        # Load asd data
        data = asdreader.reader(asd_file)
        asd_wl = data.wavelengths
        try:
            outfname = os.path.join(out_directory, os.path.basename(asd_file) + '.png')
            if os.path.isfile(outfname):
                pass

            else:
                asd_refl = data.reflectance

                plt.plot(asd_wl, asd_refl, label=os.path.basename(asd_file))
                plt.legend()
                plt.ylabel("Reflectance (%)")
                plt.xlabel("Wavelenghts (nm)")
                plt.ylim([0, 1.1])

                plt.savefig(outfname, bbox_inches='tight')
                plt.clf()
                plt.close()

        except:
            print(asd_file, out_directory)



