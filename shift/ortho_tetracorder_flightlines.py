import argparse
import os
import netCDF4 as nc
import sys
import numpy as np
from osgeo import gdal, osr
from glob import glob
from spectral.io import envi

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-tc_dir', '--tetracorder_directory', type=str, help='Tetraocrder out directory')
    parser.add_argument('-nc_file', '--netcdf_file', type=str, help='.nc file')
    args = parser.parse_args()
    
    # load nc file
    nc_ds = nc.Dataset(args.netcdf_file, 'r', format='NETCDF4')
 
     # get UTM information
    spatial_coords_string = nc_ds['transverse_mercator'].spatial_ref
    srs = osr.SpatialReference()
    srs.ImportFromWkt(spatial_coords_string)
    zone_full = srs.GetUTMZone()
    zone = abs(zone_full)
    hemisphere = "North" if zone_full > 0 else "South"
    datum = srs.GetAttrValue("DATUM").replace("_", "-")
    units = srs.GetAttrValue("UNIT")

    # geo transform info
    gt_string = nc_ds["transverse_mercator"].GeoTransform
    gt_string_list = gt_string.split()
    gt = np.array(gt_string_list).astype(np.float64)
    ox, pw, xskew, oy, yskew, ph = gt
   
    hdr_files = glob(os.path.join(args.tetracorder_directory, '*.hdr'))
    
    for hdr_path in hdr_files:
        
        # image outputs
        meta = envi.read_envi_header(hdr_path)

        # apply glt - returns ortho'd image for figures
        meta['map info'] = f'{{UTM, 1.0, 1.0, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5]*-1}, {zone}, {hemisphere}, {datum}, Units=meter}}'
        meta['coordinate system string'] = f'{{ {nc_ds["transverse_mercator"].spatial_ref} }}' 
        
        envi.write_envi_header(hdr_path, meta)
        print(f"Updated: {os.path.basename(hdr_path)}")

if __name__ == '__main__':
    main()

