import argparse
import os
from osgeo import gdal
from utils.envi import save_envi, get_meta, envi_to_array
import numpy as np
from utils.spectra_utils import spectra
from glob import glob

def augment_envi(file, out_envi_file, wvls, vertical_average=False, em_index_min=None, em_index_max=None, bad_bands=None):

    ds = gdal.Open(file, gdal.GA_ReadOnly)
    ds_array = envi_to_array(file)
    ds_array[ds_array == -9999.] = np.nan  # this sets -9999 to no data

    # set bad bands
    if bad_bands is not None:
        ds_array[:, :, bad_bands] = np.nan

    # output raster sizes
    if ds.RasterYSize == 3:  # this is for the 3x3 windows
        spectra_grid = np.ones((100, 100, len(wvls))) * -9999
    else:
        spectra_grid = np.ones((ds.RasterYSize, 100, len(wvls))) * -9999

    if not vertical_average:
        for _row, row in enumerate(ds_array):
            for _col, col in enumerate(row):
                spectra_grid[_row, _col, :] = ds_array[_row, _col, :]
    else:
        if em_index_min is None:  # this averages all spectral readings from slpit into tone
            spectra_grid[0, 0, :] = np.nanmean(ds_array, axis=(0, 1))
        else:
            spectra_grid[0, 0, :] = np.nanmean(ds_array[em_index_min:em_index_max + 1, :, :], axis=(0, 1))

    meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls,
                            wvls=True)
    meta_spectra['data ignore value'] = -9999
    save_envi(out_envi_file, meta_spectra, spectra_grid)
    print(f'saved augmented envi file to {out_envi_file}')

def deaugment_envi(file, augmented_file, out_envi_file):
    ds = gdal.Open(file, gdal.GA_ReadOnly)
    ds_array = envi_to_array(file)
    
    # load augmented data
    ds_augmented = gdal.Open(augmented_file, gdal.GA_ReadOnly)
    ds_augmented_array = envi_to_array(augmented_file)
    
    # spectra grid of original file 
    spectra_grid = np.ones((ds_array.shape[0], ds_array.shape[1], ds_augmented_array.shape[2])) * -9999
    
    # transfer data from augmented to original size
    for _row, row in enumerate(ds_array):
        for _col, col in enumerate(row):
            spectra_grid[_row, _col, :] = ds_augmented_array[_row, _col, :]
    
    # save data
    meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=list(range(spectra_grid.shape[2])), wvls=False)
    meta_spectra['data ignore value'] = -9999
    save_envi(out_envi_file, meta_spectra, spectra_grid)
    print(f'saved augmented envi file to {out_envi_file}')


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('out_directory', type=str, help="Specify output destination")
    parser.add_argument('--augment', action='store_true', help="augment data")
    parser.add_argument('--deaugment', action='store_true', help="deaugment data")
    parser.add_argument('--sensor', type=str, help="sensor", default='emit')
    args = parser.parse_args()

    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)
    
    augmented_envi_file = os.path.join(args.out_directory, f'{os.path.basename(args.reflectance_image)}_augmented.hdr')

    if args.augment:
        augment_envi(file=args.reflectance_image, out_envi_file=augmented_envi_file, wvls=wvls, vertical_average=False,
                     em_index_min=None, em_index_max=None, bad_bands=None)

    if args.deaugment:
        
        for i in ["_augmented", "_augmented_min", "_augmented_minunc"]:
            augmented_file = os.path.join(args.out_directory, f'{os.path.basename(args.reflectance_image)}{i}')
            deaugment_envi(file=args.reflectance_image, augmented_file=augmented_file, out_envi_file=f'{augmented_file}.hdr')

if __name__ == '__main__':
    main()
