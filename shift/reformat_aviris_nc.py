"""
A simple script to reformat AVIRIS (from SHIFT) netCDFs to alternate formats.

Original Author: Philip G. Brodrick
Modified by: Francisco Ochoa
"""
import argparse
import netCDF4
import numpy as np
from spectral.io import envi
from emit_utils.file_checks import envi_header
import os
from osgeo import osr
import matplotlib.pyplot as plt

envi_typemap = {
    'uint8': 1,
    'int16': 2,
    'int32': 3,
    'float32': 4,
    'float64': 5,
    'complex64': 6,
    'complex128': 9,
    'uint16': 12,
    'uint32': 13,
    'int64': 14,
    'uint64': 15
}

def main(rawargs=None):
    parser = argparse.ArgumentParser(description="Apply OE to a block of data.")
    parser.add_argument('input_netcdf', type=str, help='File to convert.')
    parser.add_argument('output_dir', type=str, help='Base directory for output ENVI files')
    parser.add_argument('-ot', '--output_type', type=str, default='ENVI', choices=['ENVI'], help='Output format')
    parser.add_argument('--interleave', type=str, default='BIL', choices=['BIL','BIP','BSQ'], help='Interleave of ENVI file to write')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing file')
    parser.add_argument('--orthorectify', action='store_true', help='Orthorectify data')
    args = parser.parse_args(rawargs)

    nc_ds = netCDF4.Dataset(args.input_netcdf, 'r', format='NETCDF4')

    if os.path.isdir(args.output_dir) is False:
        err_str = f'Output directory {args.output_dir} does not exist - please create or try again'
        raise AttributeError(err_str)

    if args.output_type == 'ENVI':
        group_names = list(nc_ds.groups.keys())

        for ds in group_names:
            output_name = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_netcdf))[0] + '_' + ds)
            print(output_name)
            if ds in ['aerosol_optical_thickness', 'water_vapor']:
                continue

            if os.path.isfile(output_name) and args.overwrite is False:
                err_str = f'File {output_name} already exists. Please use --overwrite to replace'
                raise AttributeError(err_str)

            nbands = 1

            if nc_ds[ds]['wavelength'].shape[0] > 2:
                nbands = nc_ds[ds]['wavelength'].shape[0]

            metadata = {
                'lines': nc_ds.dimensions['northing'].size,
                'samples': nc_ds.dimensions['easting'].size,
                'bands': nbands,
                'interleave': args.interleave,
                'header offset' : 0,
                'file type' : 'ENVI Standard',
                'data type' : envi_typemap[str(nc_ds[ds][ds].dtype)],
                'byte order' : 0
            }

            gt_string = nc_ds["transverse_mercator"].GeoTransform
            gt_string_list = gt_string.split()
            gt = np.array(gt_string_list).astype(np.float64)
            spatial_coords_string = nc_ds['transverse_mercator'].spatial_ref
            srs = osr.SpatialReference()
            srs.ImportFromWkt(spatial_coords_string)

            # Get the UTM Zone and Hemisphere
            # GetUTMZone() returns a positive int for North, negative for South
            zone_full = srs.GetUTMZone()
            zone = abs(zone_full)
            hemisphere = "North" if zone_full > 0 else "South"
            datum = srs.GetAttrValue("DATUM").replace("_", "-")
            units = srs.GetAttrValue("UNIT")

            metadata['map info'] = f'{{UTM, 1.0, 1.0, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5]*-1}, {zone}, {hemisphere}, {datum}, Units=meter}}'
            metadata['coordinate system string'] = f'{{ {nc_ds["transverse_mercator"].spatial_ref} }}'
            metadata['wavelength'] = nc_ds[ds]['wavelength'][:].astype(str).tolist()
            metadata['fwhm'] = nc_ds[ds]['fwhm'][:].astype(str).tolist()
            metadata['band names'] = nc_ds[ds]['wavelength'][:].astype(str).tolist()
            metadata['data ignore value'] = nc_ds[ds][ds]._FillValue


            dat = np.array(nc_ds[ds][ds]).astype(np.float32)
            dat = dat.transpose((1, 2, 0))

            envi_ds = envi.create_image(envi_header(output_name), metadata, ext='', force=args.overwrite)
            mm = envi_ds.open_memmap(writable=True, interleave='bip')
            mm[...] = np.array(dat)

            del mm, envi_ds


if __name__ == "__main__":
    main()
