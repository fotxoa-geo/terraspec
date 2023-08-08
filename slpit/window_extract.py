import argparse
from osgeo import gdal
import math
from utils.envi import get_meta, save_envi
from utils.spectra_utils import spectra
import os
from spectral.io import envi
import pandas as pd
import geopandas as gp


def get_index(x: float, y: float, ox: float, oy: float, pw: float, ph: float) -> tuple:
    """
    Gets the row (i) and column (j) indices in an NumPy 2D array for a given
    pair of coordinates.

    Parameters
    ----------
    x : float
        x (longitude) coordinate
    y : float
        y (latitude) coordinate
    ox : float
        Raster x origin (minimum x coordinate)
    oy : float
        Raster y origin (maximum y coordinate)
    pw : float
        Raster pixel width
    ph : float
        Raster pixel height

    Returns
    -------
    Two-element tuple with the column and row indices.

    Notes
    -----
    This function is based on: https://gis.stackexchange.com/a/92015/86131.

    Both x and y coordinates must be within the raster boundaries. Otherwise,
    the index will not correspond to the actual values or will be out of
    bounds.
    """
    # make sure pixel height is positive

    i = int(math.floor((y - oy) / ph))
    j = int(math.floor((x - ox) / pw))

    return i, j


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('-w_size', '--window_size', type=int, help="Specify window size", default=3)
    parser.add_argument('-shp', '--shapefile', type=str, help="Specify point shapefile")
    parser.add_argument('-pad', '--padding', type=int, help="Specify padding", default=1)
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    parser.add_argument('-sns', '--sensor', help="sensor", default='emit')
    args = parser.parse_args()

    ds = gdal.Open(args.reflectance_image, gdal.GA_ReadOnly)
    ox, pw, xskew, oy, yskew, ph = ds.GetGeoTransform()
    nd_value = ds.GetRasterBand(1).GetNoDataValue()
    arr = envi.open(args.rfl_img + '.hdr').open_memmap(interlave='bip')

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # load shapefile
    df = pd.DataFrame(gp.read_file(args.shapefile))

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[4]
    acquisition_type = os.path.basename(args.reflectance_image).split("_")[2]

    image_width = ds.RasterXSize
    image_height = ds.RasterYSize

    for index, row in df.iterrows():
        plot = row['Name']
        lon = row['geometry'].x
        lat = row['geometry'].y

        # check if lon lat is within image
        pixel_x = int((lon - ox) / pw)
        pixel_y = int((lat - oy) / ph)

        if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
            # get index
            row, col = get_index(lon, lat, ox, oy, pw, ph)

            # get pixel value and its neighboors
            window = ds.ReadAsArray(row - args.padding, col - args.padding, args.padding *2 +1, args.padding *2 +1)
            window[window == -0.1] = -9999.0

            if window.size == 0:
                print('empty array')
                print(args.output_name)
            elif window[0, 0, 0] == 0:
                print("no data values?")
                print(args.output_name)
            else:
                # make array an envi array for unmixing
                meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)
                output_name = os.path.join(args.output_directory, plot + '_' + acquisition_type + '_' + date_acquisition + ".hdr")
                save_envi(output_name, meta, window, ds, ul=[ox + (col - args.padding) * pw, oy + (row - args.padding) * ph])

                print('successfully saved ', output_name)

        else:
            print('plot is not within image')


if __name__ == '__main__':
    main()