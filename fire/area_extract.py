import argparse
from osgeo import gdal
import os
import numpy as np
from utils.envi import get_meta, save_envi, envi_to_array
from utils.spectra_utils import spectra
from utils.create_tree import create_directory
from spectral.io import envi
import pandas as pd
import geopandas as gp
import netCDF4 as nc

def harvesine_distance(dlat, dlon, lat_rad, lat_rad_array):
    # harvesine formula
    r = 6378.137  # earths radius in km from wgs 84
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(lat_rad_array) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = r * c  # in km

    return distance

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('-nc_file', '--netcdf_file', type=str, help='.nc file')
    parser.add_argument('-aoi', '--area_of_interest', type=str, help="Specify point shapefile")
    parser.add_argument('-pad', '--padding', type=int, help="Specify padding", default=1)
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    parser.add_argument('-sns', '--sensor', help="sensor", default='emit')
    parser.add_argument('-del_nc', '--delete_nc_file', help='Delete nc file', action='store_true')
    args = parser.parse_args()

    # load glts and spatial info from nc file
    emit_nc = nc.Dataset(args.netcdf_file, 'r', format='NETCDF4')
    gt = np.array(emit_nc.__dict__["geotransform"])
    proj_string = f'{{ {emit_nc.__dict__["spatial_ref"]} }}'

    lon_array = np.array(emit_nc.groups['location']['lon'])
    lat_array = np.array(emit_nc.groups['location']['lat'])

    # load rfl img w/out spatial ref
    ds = gdal.Open(args.reflectance_image, gdal.GA_ReadOnly)
    image_width = ds.RasterXSize
    image_height = ds.RasterYSize

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # load aoi
    df = gp.read_file(args.area_of_interest)
    lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat = df.total_bounds

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[4]
    acquisition_type = os.path.basename(args.reflectance_image).split("_")[2]

    # calculation for sphere - Harversine Distance
    lat_rad_array = np.radians(lat_array)
    lon_rad_array = np.radians(lon_array)

    print(f'loading.... {date_acquisition}')

    # difference in coords for lower left
    difference_lower_lat = lat_rad_array - np.radians(lower_left_lat)
    difference_lower_lon = lon_rad_array - np.radians(lower_left_lon)

    # difference in coors for upper right
    difference_upper_lat = lat_rad_array - np.radians(upper_right_lat)
    difference_upper_lon = lon_rad_array - np.radians(upper_right_lon)

    lower_left_distance = harvesine_distance(difference_lower_lat, difference_lower_lon, np.radians(lower_left_lat), lat_rad_array)
    upper_right_distance = harvesine_distance(difference_upper_lat, difference_upper_lon, np.radians(upper_right_lat), lat_rad_array)

    print(f"\t lower left min distance found: {np.min(lower_left_distance)}")
    print(f"\t upper right min distance found: {np.min(upper_right_distance)}")

    left_row_index, left_col_index = np.unravel_index(np.argmin(lower_left_distance), lower_left_distance.shape)
    right_row_index, right_col_index = np.unravel_index(np.argmin(upper_right_distance), upper_right_distance.shape)

    print(f"\t distance value at right index: {lower_left_distance[left_row_index, left_col_index]}")
    print(f"\t distance value at left index: {upper_right_distance[right_row_index, right_col_index]}")

    arr = envi_to_array(args.reflectance_image)
    print(f'image size: {arr.shape}')
    window = arr[right_row_index : left_row_index + 1, left_col_index : right_col_index + 2, :]

    try:
        # make array an envi array for unmixing
        window[window == -0.01] = -9999.0
        print(acquisition_type)
        if acquisition_type in ['MASK']:
            meta = get_meta(lines=window.shape[0], samples=window.shape[1],
                            bands=list(range(window.shape[2])), wvls=False)
        else:
            meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)

        meta['coordinate system string'] = proj_string

        # map info with updated ul coordinates
        lons_in_window = lon_array[right_row_index : left_row_index + 1, left_col_index : right_col_index + 2]
        lats_in_window = lat_array[right_row_index : left_row_index + 1, left_col_index : right_col_index + 2]

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

        meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {upper_left_longitude}, {upper_left_latitude}, {gt[1]}, {gt[5] * -1}, WGS-84}}'

        # create output directory
        create_directory(os.path.join(args.output_directory, f'{date_acquisition}'))
        create_directory(os.path.join(args.output_directory, f'{date_acquisition}', 'EXT'))
        plot_output = os.path.join(args.output_directory, f'{date_acquisition}', 'EXT')

        output_name = os.path.join(plot_output,f'{os.path.basename(args.area_of_interest).split('.')[0]}_{acquisition_type}_{date_acquisition}_EXT.hdr')
        save_envi(output_name, meta, window, ds)

        print(f"\t {date_acquisition} successfully saved: {output_name}")

    except:
        print(f" could not exctract! : {date_acquisition}")

    if args.delete_nc_file:
        if os.path.exists(args.netcdf_file):
            os.remove(args.netcdf_file)
            print(f"{args.netcdf_file} deleted successfully.")
        else:
            print(f"{args.netcdf_file} does not exist.")


if __name__ == '__main__':
    main()
