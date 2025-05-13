import os
import isofit
import numpy as np
import pandas as pd
from utils.spectra_utils import spectra
from utils.envi import augment_envi
from p_tqdm import p_map
from functools import partial
from glob import glob
from utils import asdreader

def round_column_name(name):
    try:
        return f'{float(name):.4f}'
    except:
        return name

def main():
    # convolve soil library
    df_soil_bo = pd.read_csv(r'G:\My Drive\Level_4_bo\interpolate.csv')
    df_gv_pv = pd.read_excel(r'G:\My Drive\Level_4_bo\gv_npv.xlsx')

    # # convolve wavelengths to user specified instrument
    wvls, fwhm = spectra.load_wavelengths(sensor='emit')
    asd_wvls = spectra.load_asd_wavelenghts()

    soils_convolve = p_map(partial(spectra.convolve, wvl=wvls, fwhm=fwhm, asd_wvl=asd_wvls, spectra_starting_col=1),
                      [row for row in df_soil_bo.iterrows()], **{"desc": "\t\t\tconvulsing soil " +  "...", "ncols": 150})

    df = pd.concat([df_soil_bo.iloc[:, 0], pd.DataFrame(soils_convolve)], axis=1)
    df.columns = ['fname'] + [str(x) for x in wvls]
    df.insert(1, 'latitude', 'unk')
    df.insert(1, 'longitude', 'unk')
    df.insert(0, 'level_3', 'soil')
    df.insert(0, 'level_2', 'soil')
    df.insert(0, 'level_1', 'soil')
    df.insert(0, 'dataset', 'bo')

    df_gv_pv.columns = df.columns

    # create sim library
    df_sim = pd.concat([df_gv_pv, df])
    df_sim.to_csv(r'G:\My Drive\Level_4_bo\simulation_lib.csv', index=False)

    # create reflectance
    output_directory = r'G:\My Drive\Level_4_bo\\'

    # create the reflectance file
    # spectra.generate_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim,
    #                              level='level_1', spectral_bundles=50000, cols=1, output_directory=output_directory,
    #                              wvls=wvls, name='bo_level4', spectra_starting_col=7)


    ## create the increment reflectance with 5 %
    spectra.increment_reflectance(class_names=sorted(list(df_sim.level_1.unique())), simulation_table=df_sim, level='level_1',
                                  spectral_bundles=50000, increment_size=0.05, output_directory=output_directory,
                                  wvls=wvls, name='bo_level4_increment', spectra_starting_col=8, endmember='soil')


    # create reflectance files
    # all_asd_files = sorted(glob(os.path.join(r'G:\My Drive\Geog - 185', '*.asd')))
    #
    # all_reflectances = []
    # for i in all_asd_files:
    #     asd = asdreader.reader(i)
    #     asd_refl = asd.reflectance
    #     all_reflectances.append(asd_refl)
    #
    # df = pd.DataFrame(all_reflectances)
    # df.columns = spectra.load_asd_wavelenghts()
    # df.insert(0, "transect_position", 0)
    # print(df)
    # df.to_csv(r'G:\My Drive\Geog - 185\\geog_185.csv', index=False)
    #
    # spectra.df_to_envi(df=df, spectral_starting_column=1, wvls=spectra.load_asd_wavelenghts(), output_raster=os.path.join(r'G:\My Drive\Geog - 185', 'spectra.hdr'))

    # process data for Mike F
    # folders = glob(r'C:\Users\spect\Desktop\Wind_Tunnel_2023\**')
    # asd_wvls = spectra.load_asd_wavelenghts()
    # for folder in folders:
    #     folder_name = os.path.basename(folder)
    #     if folder_name == 'augmented':
    #         continue
    #     all_asd_files = sorted(glob(os.path.join(folder, '*.asd')))
    #
    #     if all_asd_files:
    #         all_reflectances = []
    #         for i in all_asd_files:
    #
    #             asd = asdreader.reader(i)
    #
    #             try:
    #                 asd_refl = asd.reflectance
    #             except:
    #                 print(i, ' does not exist')
    #             all_reflectances.append(asd_refl)
    #
    #         df = pd.DataFrame(all_reflectances)
    #         df.columns = spectra.load_asd_wavelenghts()
    #         convolve = p_map(partial(spectra.convolve, wvl=wvls, fwhm=fwhm, asd_wvl=asd_wvls, spectra_starting_col=0),
    #                          [row for row in df.iterrows()], **{"desc": "\t\t\tconvulsing soil " + "...", "ncols": 150})
    #         df_convolve = pd.DataFrame(convolve)
    #
    #         spectra.df_to_envi(df=df_convolve, spectral_starting_column=0, wvls=wvls,
    #                            output_raster=os.path.join(r'C:\Users\spect\Desktop\Wind_Tunnel_2023\\', folder_name + '_spectra.hdr'))
    #
    #         augment_envi(file=os.path.join(r'C:\Users\spect\Desktop\Wind_Tunnel_2023\\', folder_name + '_spectra', ), wvls=wvls,
    #                      out_raster=os.path.join(r'C:\Users\spect\Desktop\Wind_Tunnel_2023\augmented\\', folder_name + '_spectra.hdr' ))
    #     else:
    #         pass

if __name__ == '__main__':
    main()