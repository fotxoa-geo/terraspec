import os
import pandas as pd
import itertools
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from p_tqdm import p_map
from utils.spectra_utils import spectra
from utils.text_guide import cursor_print
from utils.create_tree import create_directory


def test_train_split(df, test_split=0.9):
    dfs = []

    for em in df.level_1.unique():
        if em != 'soil':
            df_select = df.loc[(df['level_1'] == em)].copy()
            unmixing_spectra = df_select.sample(n=int(np.ceil(df_select.shape[0] * (1 - test_split))), random_state=13)
            dfs.append(unmixing_spectra)
        else:
            pass

    unmix_df = pd.concat(dfs)

    return unmix_df


def get_library_outputs(output_directory):

    em_lib_output = os.path.join(output_directory, 'endmember_libraries')
    sim_lib_output = os.path.join(output_directory, 'simulation_libraries')
    create_directory(em_lib_output)
    create_directory(sim_lib_output)

    return em_lib_output, sim_lib_output


def build_hypercubes(dimensions: int, max_dimension: int, spectra_starting_col:int, output_directory:str):
    df = spectra.load_global_library(output_directory=output_directory)

    # get output paths
    em_libraries_output, sim_libraries_output = get_library_outputs(output_directory=output_directory)

    # pc analysis for soil library
    print()
    cursor_print("loading latin hypercube ... d = " + str(dimensions))
    samples_from_cube = int((2 ** max_dimension) / (2 ** dimensions))
    print()

    # use test train split on pv and npv for unmmix library
    unmix_npv_pv = test_train_split(df)

    # run soil PCA and get hypercubes/quadrants
    pc_components = spectra.pca_analysis(df, spectra_starting_col=spectra_starting_col)
    pc_array = np.asarray(pc_components)

    # here the columns represent the dimensions
    cubes = spectra.latin_hypercubes(points=pc_array[:, spectra_starting_col: spectra_starting_col + dimensions],
                                     get_quadrants_index=True)

    # append cube designation to rows
    df_pc_with_index = pd.concat([pc_components, pd.DataFrame(cubes)], axis=1)
    df_pc_with_index.columns = [*df_pc_with_index.columns[:-1], 'cube']

    # list to store soil spectra dfs from random sample in latin hypercubes
    soil_dfs = []

    # sample from each cube
    for _cube, cube in enumerate(df_pc_with_index.cube.unique()):
        df_select = df_pc_with_index.loc[(df_pc_with_index['cube'] == cube)].copy()
        soil_df = df_select.sample(n=samples_from_cube, random_state=13)
        soil_df = pd.merge(df, soil_df, left_on=list(df.columns[:spectra_starting_col]),
                           right_on=list(df.columns[:spectra_starting_col]), how='inner').sort_values("fname").iloc[:, :df.shape[1]]
        soil_dfs.append(soil_df)

    # merge the em dataframes
    df_unmix = pd.concat([unmix_npv_pv, pd.concat(soil_dfs, axis=0)], axis=0).sort_values("level_1")

    # save the dataframes to a csv - unmixing library
    df_unmix.to_csv(
        os.path.join(em_libraries_output, 'latin_hypercube__n_dims_' + str(dimensions) + '_unmix_library.csv'),
        index=False)

    df_sim = pd.concat([df, df_unmix]).drop_duplicates(keep=False).sort_values("level_1")

    spectral_bundles, cols, level, wvls = get_sim_parameters()

    # save unmix library as envi
    spectra.df_to_envi(df=df_unmix, spectral_starting_column=spectra_starting_col, wvls=wvls,
                       output_raster=os.path.join(em_libraries_output,
                                                  'latin_hypercube__n_dims_' + str(dimensions) + '_unmix_library.hdr'))

    # save simulation library as envi
    spectra.df_to_envi(df=df_sim, spectral_starting_column=spectra_starting_col, wvls=wvls,
                       output_raster=os.path.join(sim_libraries_output, 'latin_hypercube__n_dims_' + str(dimensions) + '_simulation_library.hdr'))

    # simulate the reflectance
    spectra.simulate_reflectance(df_sim=df_sim, df_unmix=df_unmix, dimensions=dimensions,
                                 sim_libraries_output=sim_libraries_output, mode='latin_hypercube', level=level,
                                 spectral_bundles=spectral_bundles, cols=cols, output_directory=output_directory,
                                 wvls=wvls, spectra_starting_col=spectra_starting_col)


def build_hull(dimensions: int, output_directory:str,  spectra_starting_col:int, normalize=False):
    df = spectra.load_global_library(output_directory=output_directory)
    df_soil = df.loc[(df['level_1'] == 'soil')].copy().reset_index(drop=True)

    if normalize:
        df_only = df.iloc[:, spectra_starting_col:]
        df_array = df_only.to_numpy()
        norm = p_map(spectra.vector_normalize_spectrum, df_array,
                                     **{"desc": f"\t\t\tnormalizing spectrum: CH d = {dimensions}...", "ncols": 150})
        df_norm = pd.DataFrame(norm)
        df_norm.columns = df_only.columns
        df_norm = pd.concat([df.iloc[:, :spectra_starting_col].reset_index(drop=True), df_norm], axis=1)
    else:
        pass

    # # pc analysis for em library
    cursor_print(f"loading convex hull... d = {dimensions}")
    print()

    # use test train split on pv and npv for unmmix library
    unmix_npv_pv = test_train_split(df)

    # get output paths
    em_libraries_output, sim_libraries_output = get_library_outputs(output_directory=output_directory)

    # run soil PCA
    if normalize:
        pc_components = spectra.pca_analysis(df_norm, spectra_starting_col=spectra_starting_col)
        pc_array = np.asarray(pc_components)[:, spectra_starting_col: spectra_starting_col + dimensions]

    else:
        pc_components = spectra.pca_analysis(df, spectra_starting_col=spectra_starting_col)
        pc_array = np.asarray(pc_components)[:, spectra_starting_col: spectra_starting_col + dimensions]

    # get the convex hull of n-dimensions
    ch = ConvexHull(pc_array)

    # Get the indices of the hull points
    hull_indices = ch.vertices

    # hull merged df with metadata
    df_ch = df_soil.iloc[hull_indices]

    # merge the em dataframes
    df_unmix = pd.concat([unmix_npv_pv, df_ch], axis=0).sort_values("level_1")

    # save the dataframes to a csv - unmixing library
    df_unmix.to_csv(os.path.join(em_libraries_output, 'convex_hull__n_dims_' + str(dimensions) + '_unmix_library.csv'), index=False)

    df_sim = pd.concat([df, df_unmix]).drop_duplicates(keep=False).sort_values("level_1")

    # get simulation parameters
    spectral_bundles, cols, level, wvls = get_sim_parameters()

    # save unmix library as envi
    spectra.df_to_envi(df=df_unmix, spectral_starting_column=spectra_starting_col, wvls=wvls,
                       output_raster=os.path.join(em_libraries_output, 'convex_hull__n_dims_' + str(dimensions) + '_unmix_library.hdr'))

    # save simulation library as envi
    spectra.df_to_envi(df=df_sim, spectral_starting_column=spectra_starting_col, wvls=wvls,
                       output_raster=os.path.join(sim_libraries_output, 'convex_hull__n_dims_' + str(dimensions) + '_simulation_library.hdr'))

    # # simulate the reflectance
    spectra.simulate_reflectance(df_sim=df_sim, df_unmix=df_unmix, dimensions=dimensions,
                                 sim_libraries_output=sim_libraries_output, mode='convex_hull', level=level,
                                 spectral_bundles=spectral_bundles, cols=cols, output_directory=output_directory,
                                 wvls=wvls, spectra_starting_col=spectra_starting_col)


def get_sim_parameters():
    cols = 1
    level = 'level_1'
    spectral_bundles = 1000
    emit_wvls, emit_fwhm = spectra.load_wavelengths(sensor='emit')

    return spectral_bundles, cols, level, emit_wvls


def run_build_reflectance(output_directory):
    num_dimensions = [2, 3, 4, 5, 6]  # dimensions to use for convex hull and latin hypercubes
    #num_dimensions = [4]  # dimensions to use for convex hull and latin hypercubes
    max_dimension = max(num_dimensions)
    spectral_starting_col = 7

    # build convex hulls and latin hypercubes across different dimensional space
    for i in num_dimensions:
        #build_hypercubes(dimensions=i, max_dimension=max_dimension, spectra_starting_col=spectral_starting_col,
        #                 output_directory=output_directory)
        build_hull(dimensions=i, spectra_starting_col=spectral_starting_col, output_directory=output_directory, normalize=True)

    #build_geographic(dimensions=4, output_directory=output_directory, spectra_starting_col=spectral_starting_col)
