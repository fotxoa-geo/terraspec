import argparse
from osgeo import gdal, osr, ogr
import math
import sys
import numpy as np 
from utils.envi import get_meta
from utils.spectra_utils import spectra
import os
import pandas as pd
import geopandas as gp
import netCDF4 as nc
from pyproj import Proj
from utils.create_tree import create_directory
from spectral.io import envi
from emit_utils.file_checks import envi_header


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('-w_size', '--window_size', type=int, help="Specify window size", default=3)
    parser.add_argument('-shp', '--shapefile', type=str, help="Specify point shapefile")
    parser.add_argument('-pad', '--padding', type=int, help="Specify padding", default=1)
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    parser.add_argument('-sns', '--sensor', help="sensor", default='aviris_ng')
    parser.add_argument('-nc', '--nc_file', help='nc file', type=str)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing file')
    parser.add_argument('-del_nc', '--delete_nc_file', help='Delete nc file', action='store_true')
    args = parser.parse_args()

    aviris_nc = nc.Dataset(args.nc_file, 'r', format='NETCDF4')
    # get UTM information
    spatial_coords_string = aviris_nc['transverse_mercator'].spatial_ref
    srs = osr.SpatialReference()
    srs.ImportFromWkt(spatial_coords_string)
    zone_full = srs.GetUTMZone()
    zone = abs(zone_full)
    hemisphere = "North" if zone_full > 0 else "South"
    datum = srs.GetAttrValue("DATUM").replace("_", "-")
    units = srs.GetAttrValue("UNIT")

    # geo transform info
    gt_string = aviris_nc["transverse_mercator"].GeoTransform
    gt_string_list = gt_string.split()
    gt = np.array(gt_string_list).astype(np.float64)
    ox, pw, xskew, oy, yskew, ph = gt
    
    # conversion to UTM
    myProj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # load shapefile
    df = pd.DataFrame(gp.read_file(args.shapefile))
    df = df.sort_values('plot')

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[0]
    acquisition_type = os.path.basename(args.reflectance_image).split("_")[5]
    version = os.path.basename(args.reflectance_image).split("_")[1]
    ds = gdal.Open(args.reflectance_image, gdal.GA_ReadOnly)
    image_width = ds.RasterXSize
    image_height = ds.RasterYSize

    for index, row in df.iterrows():
        plot = row['plot']
        site = plot.split('-')[0]
        num = plot.split('_')[0].split('-')[1]
        season = row['season'] 
        plot = f'{site}-{num}_{season}'
        lon = row['geometry'].x
        lat = row['geometry'].y
        
        utm_easting, utm_northing = myProj(lon, lat)

        # check if lon lat is within image with index
        pixel_x = int(np.floor((utm_easting - ox) / pw))
        pixel_y = int(np.floor((utm_northing - oy) / ph))
        
        if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:

            # get pixel value and its neighboors
            window = ds.ReadAsArray(pixel_x-args.padding, pixel_y-args.padding, args.padding *2 +1, args.padding *2 +1)            
            
            # make array an envi array for unmixing
            window[window == 0] = -9999.0
            window =  window.transpose((1,2,0))
            meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)
            
            # create output directory
            create_directory(os.path.join(args.output_directory, plot))
            create_directory(os.path.join(args.output_directory, plot, 'EXT'))
            plot_output = os.path.join(args.output_directory, plot, 'EXT')

            output_name = os.path.join(plot_output, f'{plot}_{acquisition_type}_{date_acquisition}_{version}_EXT')
            upper_left_lon = ox + (pixel_x - args.padding) * pw
            upper_left_lat = oy + (pixel_y - args.padding) * ph

            meta['map info'] = f'{{UTM, 1.0, 1.0, {upper_left_lon}, {upper_left_lat}, {gt[1]}, {gt[5]*-1}, {zone}, {hemisphere}, {datum}, Units=meter}}'
            meta['coordinate system string'] = f'{{ {aviris_nc["transverse_mercator"].spatial_ref} }}'
            try:
                meta['wavelength'] = aviris_nc['reflectance']['wavelength'][:].astype(str).tolist()
                meta['fwhm'] = aviris_nc['reflectance']['fwhm'][:].astype(str).tolist()
                meta['band names'] = aviris_nc['reflectance']['wavelength'][:].astype(str).tolist()
            except:
                meta['wavelength'] = aviris_nc['uncertainty']['wavelength'][:].astype(str).tolist()
                meta['fwhm'] = aviris_nc['uncertainty']['fwhm'][:].astype(str).tolist()
                meta['band names'] = aviris_nc['uncertainty']['wavelength'][:].astype(str).tolist()

            meta['data ignore value'] = -9999
            
            envi_ds = envi.create_image(envi_header(output_name), meta, ext='', force=args.overwrite)
            mm = envi_ds.open_memmap(writable=True, interleave='bip')
            mm[...] = window

            print(f'{plot} successfully saved: {output_name}')

        else:
            print(f'{plot} is not within image!: {os.path.basename(args.reflectance_image)}')

    if args.delete_nc_file:
        if os.path.exists(args.nc_file):
            os.remove(args.nc_file)
            print(f"{args.nc_file} deleted successfully.")
        else:
            print(f"{args.nc_file} does not exist.")

if __name__ == '__main__':
    main()
