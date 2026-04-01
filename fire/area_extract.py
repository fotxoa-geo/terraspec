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
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance orthorectified image')
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
        
    # load rfl img w/out spatial ref
    ds = gdal.Open(args.reflectance_image, gdal.GA_ReadOnly)
    ox, pw, xskew, oy, yskew, ph = ds.GetGeoTransform()

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # load aoi
    df = gp.read_file(args.area_of_interest)
    lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat = df.total_bounds
    
    # get bounding box - complete
    corners = [
    (lower_left_lat, lower_left_lon),    # LL
    (upper_right_lat, upper_right_lon),  # UR
    (lower_left_lat, upper_right_lon),   # LR
    (upper_right_lat, lower_left_lon)]   # UL
    
    row_indices = []
    col_indices = []
   
    for c_lat, c_lon in corners:
        col = int(np.floor((c_lon - ox) / pw))
        row = int(np.floor((c_lat - oy) / ph))
        
        row_indices.append(row)
        col_indices.append(col)

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[4]
    acquisition_type = os.path.basename(args.reflectance_image).split("_")[2]
    
    print(f'loading.... {date_acquisition}')
    row_start, row_end = min(row_indices), max(row_indices)
    col_start, col_end = min(col_indices), max(col_indices)
    
    print(f"\t row start, row end: {row_start, row_end}")
    print(f"\t col_start, col_end: {col_start, col_end}")

    arr = envi_to_array(args.reflectance_image)
    print(f'image size: {arr.shape}')
    window = arr[row_start: row_end + 1, col_start : col_end + 1, :]

    # make array an envi array for unmixing
    window[window == -0.01] = -9999.0
    window[window == 0.] = -9999.0

    if acquisition_type in ['MASK']:
        meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=list(range(window.shape[2])), wvls=False)
    else:
        meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)

    meta['coordinate system string'] = proj_string

    # map info with updated ul coordinatesi
    meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {lower_left_lon}, {upper_right_lat}, {gt[1]}, {gt[5] * -1}, WGS-84}}'

    # create output directory
    create_directory(os.path.join(args.output_directory, 'EXT'))
    out_dest = os.path.join(args.output_directory, 'EXT')
    output_name = os.path.join(out_dest, f'{os.path.basename(args.area_of_interest).split(".")[0]}_{acquisition_type}_{date_acquisition}_EXT.hdr')
    save_envi(output_name, meta, window, ds)

    print(f"\t {date_acquisition} successfully saved: {output_name}")
    
    if args.delete_nc_file:
        if os.path.exists(args.netcdf_file):
            os.remove(args.netcdf_file)
            print(f"{args.netcdf_file} deleted successfully.")
        else:
            print(f"{args.netcdf_file} does not exist.")


if __name__ == '__main__':
    main()
