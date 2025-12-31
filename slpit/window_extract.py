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
    parser.add_argument('-del_nc', '--delete_nc_file', help='Delete nc file', action='store_true')
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

        print(f'loading.... {plot}')
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
            
            print(f"\t min distance found: {np.min(distance)}")
            
            row_index, col_index = np.unravel_index(np.argmin(distance), distance.shape)

            print(f"\t value at index: {distance[row_index, col_index]}")

            print(f"\t index: {row_index, col_index}, plot: {plot}, coords: {lon, lat}, shape: {distance.shape}")

            arr = envi_to_array(args.reflectance_image)
            print(f'image size: {arr.shape}')
            window = arr[row_index - 1: row_index + 2, col_index - 1: col_index + 2, :]

            # get pixel value and its neighboors
            try:
                if window.shape != (3,3, window.shape[2]):
                    print(f"\t {plot} does not have enough coverage!")
                else:
                    # make array an envi array for unmixing
                    window[window == -0.01] = -9999.0
                    print(acquisition_type)
                    if acquisition_type in ['MASK']:
                        meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=list(range(window.shape[2])), wvls=False)
                    else:
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

                    meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {upper_left_longitude}, {upper_left_latitude}, {gt[1]}, {gt[5]*-1}, WGS-84}}'

                    #create output directory
                    create_directory(os.path.join(args.output_directory,  f'{plot.replace(" ", "").replace("SPEC", "Spectral")}'))
                    create_directory(os.path.join(args.output_directory,  f'{plot.replace(" ", "").replace("SPEC", "Spectral")}', 'EXT'))
                    plot_output = os.path.join(args.output_directory,  f'{plot.replace(" ", "").replace("SPEC", "Spectral")}', 'EXT')

                    output_name = os.path.join(plot_output, f'{plot.replace(" ", "").replace("SPEC", "Spectral")}_{acquisition_type}_{date_acquisition}_EXT.hdr')
                    save_envi(output_name, meta, window, ds)

                    print(f"\t {plot} successfully saved: {output_name}")
                
            except:
                raise
                print(f"\t {plot} could not open!")

        else:
            print(f"\t {plot} is not within image: {os.path.basename(args.reflectance_image)}")
    
    if args.delete_nc_file:
        if os.path.exists(args.netcdf_file):
            os.remove(args.netcdf_file)
            print(f"{args.netcdf_file} deleted successfully.")
        else:
            print(f"{args.netcdf_file} does not exist.")

if __name__ == '__main__':
    main()
