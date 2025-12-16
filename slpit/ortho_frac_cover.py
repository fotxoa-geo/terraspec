import argparse
import os
import netCDF4 as nc
import sys
import numpy as np
from osgeo import gdal,osr
from emit_utils.reformat import single_image_ortho
from utils.envi import get_meta, save_envi, envi_to_array, envi_tiff


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-frac_img', '--fraction_image', type=str, help='Reflectance image')
    parser.add_argument('-nc_file', '--netcdf_file', type=str, help='.nc file')
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    args = parser.parse_args()
    
    # load nc file
    nc_ds = nc.Dataset(args.netcdf_file, 'r', format='NETCDF4')
    gt = np.array(nc_ds.__dict__["geotransform"])
    
    # create glt
    glt = np.zeros(list(nc_ds.groups['location']['glt_x'].shape) + [2], dtype=np.int32)
    glt[...,0] = np.array(nc_ds.groups['location']['glt_x'])
    glt[...,1] = np.array(nc_ds.groups['location']['glt_y'])
    
    # load frac cover
    frac_array = envi_to_array(args.fraction_image) 

    # apply glt - returns ortho'd image for figures
    fraction_ortho_dat = single_image_ortho(frac_array, glt)
    meta = get_meta(lines=fraction_ortho_dat.shape[0], samples=fraction_ortho_dat.shape[1], bands=['npv', 'pv', 'soil', 'shade'], wvls=False)
    meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5]*-1},WGS-84}}'
    meta['coordinate system string'] = f'{{ {nc_ds.__dict__["spatial_ref"]} }}'
    output_name = os.path.join(args.output_directory, f'{os.path.basename(args.fraction_image)}.hdr')
    save_envi(output_name, meta, fraction_ortho_dat)

    # create a version for field ipads
    basename = os.path.basename(os.path.splitext(output_name)[0])
    driver = gdal.GetDriverByName("GTiff")
    out_ras = os.path.join(args.output_directory, f'ipad_{os.path.basename(args.fraction_image)}.tif')
    ds = gdal.Open(os.path.splitext(output_name)[0], gdal.GA_ReadOnly)
    prj = ds.GetProjection()

    ds_array = envi_to_array(os.path.splitext(output_name)[0])
    outRaster = driver.Create(out_ras, ds_array.shape[1], ds_array.shape[0], ds_array.shape[2], gdal.GDT_Byte)

    originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    for _b, b in enumerate(range(0, ds_array.shape[2])):
        outband = outRaster.GetRasterBand(_b + 1)
        band_select = ds_array[:, :, _b]
        band_select *= 255.0 / band_select.max()
        outband.WriteArray(band_select)

    # settings srs from input tif file.
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


    envi_tiff(os.path.join(args.output_directory, f'{os.path.basename(args.fraction_image)}'), args.output_directory)
    print(f'successfully saved {output_name}')


if __name__ == '__main__':
    main()
