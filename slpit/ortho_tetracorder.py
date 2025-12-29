import argparse
import os
import netCDF4 as nc
import sys
import numpy as np
from osgeo import gdal, osr
from glob import glob
from emit_utils.reformat import single_image_ortho
from utils.envi import get_meta, save_envi, envi_to_array

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-tc_dir', '--tetracorder_directory', type=str, help='Tetraocrder out directory')
    parser.add_argument('-nc_file', '--netcdf_file', type=str, help='.nc file')
    args = parser.parse_args()
    
    # load nc file
    nc_ds = nc.Dataset(args.netcdf_file, 'r', format='NETCDF4')
    gt = np.array(nc_ds.__dict__["geotransform"])
 
    # create glt
    glt = np.zeros(list(nc_ds.groups['location']['glt_x'].shape) + [2], dtype=np.int32)
    glt[...,0] = np.array(nc_ds.groups['location']['glt_x'])
    glt[...,1] = np.array(nc_ds.groups['location']['glt_y'])
    
    hdr_files = glob(os.path.join(args.tetracorder_directory, '*.hdr'))
    
    for i in hdr_files:
        
        # image outputs
        img = os.path.splitext(i)[0]

        # load minerals
        mineral_array = envi_to_array(img)
    
        # apply glt - returns ortho'd image for figures
        mineral_ortho_dat = single_image_ortho(mineral_array, glt)
        
        meta = get_meta(lines=mineral_ortho_dat.shape[0], samples=mineral_ortho_dat.shape[1], bands=['Group 1 Band Depth', 'Group 1 Index', 'Group 2 Band Depth', 'Group 2 Index'], wvls=False)
        meta['map info'] = f'{{Geographic Lat/Lon, 1, 1, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5]*-1},WGS-84}}'
        meta['coordinate system string'] = f'{{ {nc_ds.__dict__["spatial_ref"]} }}'
        
        output_name = os.path.join(args.tetracorder_directory, i)
        save_envi(output_name, meta, mineral_ortho_dat)
        print(f'successfully saved {output_name}')

        # create a version for field ipads
        # will require re-classifications of g1 and g2 minerals


if __name__ == '__main__':
    main()
