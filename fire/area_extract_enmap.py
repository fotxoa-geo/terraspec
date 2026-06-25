import argparse
from osgeo import gdal, osr
import os
import numpy as np
from utils.create_tree import create_directory
from spectral.io import envi
import geopandas as gp

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

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance orthorectified image in utm')
    parser.add_argument('-aoi', '--area_of_interest', type=str, help="Specify shapefile")
    parser.add_argument('-pad', '--padding', type=int, help="Specify padding", default=1)
    parser.add_argument('-out', '--output_directory', type=str, help="Specify output destination")
    args = parser.parse_args()

    df = gp.read_file(args.area_of_interest)
    projection_aoi = df.crs.to_wkt()

    hdr = envi.open(f"{args.reflectance_image}.hdr")
    wavelengths = [float(x) for x in hdr.metadata['wavelength']]
    fwhm = [float(x) for x in hdr.metadata['fwhm']]
    no_data_value = float(hdr.metadata['data ignore value'])

    warp_options = gdal.WarpOptions(
        format='MEM',
        dstSRS=projection_aoi,
        resampleAlg='bilinear'
    )

    ds_mem = gdal.Warp('', args.reflectance_image, options=warp_options)
    ox, pw, xskew, oy, yskew, ph = ds_mem.GetGeoTransform()

    lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat = df.total_bounds

    print("--- Reprojection Complete ---")
    print(f"New Image Projection matches AOI: {df.crs.to_string()}")
    print(f"Image Origin (Top Left Lat/Lon): {oy}, {ox}")
    print(f"AOI Bounding Box: Min Lon: {lower_left_lon}, Min Lat: {lower_left_lat}")

    # # get bounding box - complete
    corners = [
        (lower_left_lat, lower_left_lon),  # LL
        (upper_right_lat, upper_right_lon),  # UR
        (lower_left_lat, upper_right_lon),  # LR
        (upper_right_lat, lower_left_lon)]  # UL

    row_indices = []
    col_indices = []

    for c_lat, c_lon in corners:
        col = int(np.floor((c_lon - ox) / pw))
        row = int(np.floor((c_lat - oy) / ph))

        row_indices.append(row)
        col_indices.append(col)

    # get image date
    date_acquisition = os.path.basename(args.reflectance_image).split("_")[5]
    print(date_acquisition)

    print(f'loading.... {date_acquisition}')
    row_start, row_end = min(row_indices), max(row_indices)
    col_start, col_end = min(col_indices), max(col_indices)

    print(f"\t row start, row end: {row_start, row_end}")
    print(f"\t col_start, col_end: {col_start, col_end}")

    arr = ds_mem.ReadAsArray()
    arr = np.transpose(arr, (1, 2, 0))

    print(f'image size: {arr.shape}')
    window = arr[row_start: row_end + 1, col_start: col_end + 1, :]

    # make array an envi array for unmixing
    window[window == no_data_value] = -9999.

    metadata = {'lines': window.shape[0],
                'samples': window.shape[1],
                'bands': window.shape[2],
                'wavelength': wavelengths,
                'fwhm': fwhm,
                'interleave': 'bil',
                'header offset': 0,
                'file type': 'ENVI Standard',
                'data type': envi_typemap[str(window.dtype)],
                'byte order': 0,
                'map_info': f"{{Geographic Lat/Lon, 1, 1, {lower_left_lon}, {upper_right_lat}, {pw}, {ph * -1}, WGS-84}}",
                'data ignore value': -9999.,
                'wavelength units': 'nm'}

    # create output directory
    create_directory(os.path.join(args.output_directory, 'EXT'))
    out_dest = os.path.join(args.output_directory, 'EXT')
    output_hdr = os.path.join(out_dest,
                               f'{os.path.basename(args.area_of_interest).split(".")[0]}_{os.path.basename(args.reflectance_image)}_EXT.hdr')

    envi.save_image(output_hdr, window,
                    metadata=metadata,
                    force=True,
                    interleave='bil', ext='')

    print(f"\t {date_acquisition} successfully saved: {output_hdr}")



if __name__ == '__main__':
    main()

