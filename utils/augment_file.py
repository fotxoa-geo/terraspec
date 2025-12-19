import argparse
import os
from osgeo import gdal
from utils.envi import save_envi, get_meta, envi_to_array
import numpy as np
from utils.spectra_utils import spectra

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

def deaugment_envi():
    print('hiii')

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('out_directory', type=str, help="Specify output destination")
    parser.add_argument('--augment', type='store_true', help="augment data")
    parser.add_argument('--deaugment', type='store_true', help="deaugment data")
    parser.add_argument('--sensor', type='str', help="sensor", default='emit')
    args = parser.parse_args()

    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    if args.augment:
        out_envi_file = os.path.join(args.out_directory, f'{os.path.basename(args.reflectance_image)}_augmented.hdr')
        augment_envi(file=args.reflectance_image, out_envi_file=out_envi_file, wvls=wvls, vertical_average=False,
                     em_index_min=None, em_index_max=None, bad_bands=None)

    if args.deaugment:
        out_envi_file = os.path.join(args.out_directory, f'{os.path.basename(args.reflectance_image)}.hdr')
        print(out_envi_file)
        deaugment_envi()

if __name__ == '__main__':
    main()