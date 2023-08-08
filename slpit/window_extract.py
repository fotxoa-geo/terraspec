import argparse
import os
from osgeo import gdal
import math
import sys
import numpy as np 

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, '..')
sys.path.append(utils_path)


from utils.envi import get_meta, save_envi
from utils.spectra_utils import spectra
import os
from spectral.io import envi
import pandas as pd
import geopandas as gp


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

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # load shapefile
    df = pd.DataFrame(gp.read_file(args.shapefile))
    df = df.sort_values('Name')

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[4]
    acquisition_type = os.path.basename(args.reflectance_image).split("_")[2]

    image_width = ds.RasterXSize
    image_height = ds.RasterYSize

    for index, row in df.iterrows():
        plot = row['Name']
        lon = row['geometry'].x
        lat = row['geometry'].y

        # check if lon lat is within image with index
        pixel_x = int((lon - ox) / pw)
        pixel_y = int((lat - oy) / ph)

        if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:

            # get pixel value and its neighboors
            window = ds.ReadAsArray(pixel_x, pixel_y, args.padding *2 +1, args.padding *2 +1)#.transpose((1,2,0))

            if np.any(window == 0) and acquisition_type != 'MASK':
                print(plot, " has at least one pixel of fill values")
            elif np.all(window == 0) and acquisition_type != 'MASK':
                print(plot, " has all fill values")
            else:
                # make array an envi array for unmixing
                window[window == -0.1] = -9999.0
                meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)
                output_name = os.path.join(args.output_directory, plot.replace(" ", "") + '_' + acquisition_type + '_' + date_acquisition + ".hdr")
                save_envi(output_name, meta, window.transpose((1,2,0)), ds, ul=[ox + (pixel_x - args.padding) * pw, oy + (pixel_y - args.padding) * ph])

                print(plot, 'successfully saved ', output_name)

        else:
            print(plot, ' is not within image', os.path.basename(args.reflectance_image))


if __name__ == '__main__':
    main()
