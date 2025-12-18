import argparse
import os
import netCDF4 as nc
import numpy as np
from emit_utils.reformat import single_image_ortho
from utils.envi import get_meta, save_envi, envi_to_array
from osgeo import gdal,osr

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('-nc_file', '--netcdf_file', type=str, help='.nc file')
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    args = parser.parse_args()
    
    # load nc file
    nc_ds = nc.Dataset(args.netcdf_file, 'r', format='NETCDF4')
    gt = np.array(nc_ds.__dict__["geotransform"])
    
    # load wvls from nc file
    nc_wvls = np.array(nc_ds['sensor_band_parameters'].variables['wavelengths']).astype(str).tolist()
    nc_wvls = [float(x) for x in nc_wvls]

    # create glt
    glt = np.zeros(list(nc_ds.groups['location']['glt_x'].shape) + [2], dtype=np.int32)
    glt[...,0] = np.array(nc_ds.groups['location']['glt_x'])
    glt[...,1] = np.array(nc_ds.groups['location']['glt_y'])
    
    # load rgb
    rfl_array = envi_to_array(args.reflectance_image)
    load_order = {470.0304: 2, 574.20905: 1, 656.1857: 0}
    
    rgb_wvls_out = [656.1857,574.20905, 470.0304]

    # apply glt - returns ortho'd image 
    rfl_ortho_dat = single_image_ortho(rfl_array, glt)
    
    # create rgb subset - for figure purposes
    rgb_subset = np.zeros((rfl_ortho_dat.shape[0], rfl_ortho_dat.shape[1], len(rgb_wvls_out)))
    rgb_subset[rgb_subset == 0] = -9999
    
    for wvl in rgb_wvls_out:
        emit_wvl_index = nc_wvls.index(wvl)
        rgb_subset[:, :, load_order[wvl]] = rfl_ortho_dat[:, :, emit_wvl_index]

    meta = get_meta(lines=rgb_subset.shape[0], samples=rgb_subset.shape[1], bands=rgb_wvls_out, wvls=True)
    meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5]*-1},WGS-84}}'
    meta['coordinate system string'] = f'{{ {nc_ds.__dict__["spatial_ref"]} }}'
    
    output_name = os.path.join(args.output_directory, f'RGB_{os.path.basename(args.reflectance_image)}.hdr')
    save_envi(output_name, meta, rgb_subset)
    print(f'successfully saved {output_name}')

    # create rgb subset (8 bit tif) - for field ipads
    driver = gdal.GetDriverByName("GTiff")
    out_ras = os.path.join(args.output_directory, f'ipad_{os.path.basename(args.reflectance_image)}.tif')
    ds = gdal.Open(os.path.splitext(output_name)[0], gdal.GA_ReadOnly)
    prj = ds.GetProjection()

    ds_array = envi_to_array(os.path.splitext(output_name)[0])
    outRaster = driver.Create(out_ras, ds_array.shape[1], ds_array.shape[0], 4, gdal.GDT_Byte)

    originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    for _b, b in enumerate(range(0, ds_array.shape[2])):
        band_wvl = float(ds.GetRasterBand(_b + 1).GetDescription().split(" ")[0])

        if band_wvl in rgb_wvls_out:
            outband = outRaster.GetRasterBand(load_order[band_wvl] + 1)
            band_select = ds_array[:, :, _b]
            band_select *= 255.0 / band_select.max()
            outband.WriteArray(band_select)
            outband.SetNoDataValue(0)

        else:
            outband = outRaster.GetRasterBand(4)
            band_select = ds_array[:, :, _b]
            band_select *= 255.0 / band_select.max()
            band_select[band_select != 0] = 255
            outband.WriteArray(band_select)

    # settings srs from input tif file.
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

if __name__ == '__main__':
    main()
