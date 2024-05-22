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
import netCDF4 as nc

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('-nc_file', '--netcdf_file', type=str, help='.nc file')
    parser.add_argument('-w_size', '--window_size', type=int, help="Specify window size", default=3)
    parser.add_argument('-shp', '--shapefile', type=str, help="Specify point shapefile")
    parser.add_argument('-pad', '--padding', type=int, help="Specify padding", default=1)
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    parser.add_argument('-sns', '--sensor', help="sensor", default='emit')
    args = parser.parse_args()

    # load glts and spatial info from nc file
    emit_nc = nc.Dataset(args.netcdf_file, 'r', format='NETCDF4')
    gt = np.array(emit_nc.__dict__["geotransform"])
    proj_string = f'{{ {emit_nc.__dict__["spatial_ref"]} }}'

    lon_array = np.array(emit_nc.groups['location']['lon'])
    lat_array = np.array(emit_nc.groups['location']['lat'])
    glt_x =  np.array(emit_nc.groups['location']['glt_x'])
    glt_y = np.array(emit_nc.groups['location']['glt_y'])

    # load rfl img w/out spatial ref
    ds = gdal.Open(args.reflectance_image, gdal.GA_ReadOnly)
    image_width = ds.RasterXSize
    image_height = ds.RasterYSize

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # load shapefile
    df = pd.DataFrame(gp.read_file(args.shapefile))
    df = df.sort_values('Name')

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[4]
    acquisition_type = os.path.basename(args.reflectance_image).split("_")[2]

    # calculation for sphere - Harversine Distance
    r = 6378.137 # earths radius in km from wgs 84
    lat_rad_array = np.radians(lat_array)
    lon_rad_array = np.radians(lon_array)

    for index, row in df.iterrows():
        plot = row['Name']
        lon = row['geometry'].x
        lat = row['geometry'].y


        # check if lon lat is within bounds
        if np.min(lon_array) <= lon <= np.max(lon_array) and np.min(lat_array) <= lat <= np.max(lat_array):

            # gps points in radians
            lon_rad = np.radians(lon)
            lat_rad = np.radians(lat)

            # difference in coords
            dlat = lat_rad_array - lat_rad
            dlon = lon_rad_array - lon_rad
            # harvesine formula
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(lat_rad_array) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = r * c # in km
            #distance = r * np.sqrt(dlat**2 + dlon**2)

            print(f"min distance found: {np.min(distance)}")

            row_index, col_index = np.unravel_index(np.argmin(distance), distance.shape)

            print(f"value at index: {distance[row_index, col_index]}")

            print(f"index: {row_index, col_index}, plot: {plot}, coords: {lon, lat}, shape: {distance.shape}")

            arr = ds.ReadAsArray().transpose((1,2,0))
            window = arr[row_index - 1: row_index + 2, col_index - 1: col_index + 2, :]

            # get pixel value and its neighboors
            try:
                window = ds.ReadAsArray(float(col_index) - args.padding, float(row_index) - args.padding, args.padding *2 +1, args.padding *2 +1).transpose((1,2,0))

                #if np.any(window == 0) and acquisition_type != 'MASK':
                #    print(plot, " has at least one pixel of fill values")

                #elif np.all(window == 0) and acquisition_type != 'MASK':
                #    print(plot, " has all fill values")

                if window.shape != (3,3, window.shape[2]):
                    print(f"{plot} does not have enough coverage!")
                else:
                    # make array an envi array for unmixing
                    window[window == -0.01] = -9999.0
                    meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)
                    meta['coordinate system string'] = proj_string

                    # map info with updated ul coordinates
                    lons_in_window = lon_array[row_index -1: row_index + 2, col_index-1: col_index + 2]
                    lats_in_window = lat_array[row_index -1: row_index + 2, col_index-1: col_index + 2]

                    upper_left_longitude = lons_in_window[0, 0]
                    upper_left_latitude = lats_in_window[0, 0]

                    rows, cols = lons_in_window.shape
                    for i in range(rows):
                        for j in range(cols):

                            # Update the upper leftmost longitude if current value is smaller
                            if lons_in_window[i, j] < upper_left_longitude:
                                upper_left_longitude = lons_in_window[i, j]

                            # Update the upper leftmost latitude if current value is greater
                            if lats_in_window[i, j] > upper_left_latitude:
                                upper_left_latitude = lats_in_window[i, j]


                    meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {upper_left_longitude}, {upper_left_latitude}, {gt[1]}, {gt[5]*-1},WGS-84}}'

                    output_name = os.path.join(args.output_directory, f'{plot.replace(" ", "")}_{acquisition_type}_{date_acquisition}.hdr')
                    save_envi(output_name, meta, window, ds)

                    print(plot, 'successfully saved ', output_name)

            except:
                print(f"{plot} could not open!")

        else:
            print(plot, ' is not within image', os.path.basename(args.reflectance_image))


if __name__ == '__main__':
    main()
