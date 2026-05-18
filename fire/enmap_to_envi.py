import os
import rasterio
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from spectral import envi
from pyproj import Proj

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
    parser = argparse.ArgumentParser(description='Run Vegetation Extraction')

    parser.add_argument('-out_dir', '--output_directory', type=str, help='Out directory')
    parser.add_argument('-rfl', '--reflectance_image', type=str, help='Reflectance image that was unmixed')
    parser.add_argument('-unmix_lib_csv', '--unmixing_library_csv', type=str, help='Unmixing library csv file')
    parser.add_argument('-scale', '--scale_factor', type=str, default='10000.0')
    args = parser.parse_args()

    # open image
    with rasterio.open(args.reflectance_image) as src:
        cube = src.read().astype(float)
        nodata = src.nodata
        if nodata is not None:
            cube[cube == nodata] = np.nan

        # Scale only the data values (Reflectance)
        cube /= float(args.scale_factor)

    # reformat image to Row, Column, Bands
    cube = cube.transpose(1, 2, 0)

    # load quality tiff
    quality_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-QL_QUALITY_CLASSES_COG.TIF'
    quality_path = os.path.join(os.path.dirname(args.reflectance_image), quality_file_name)

    with rasterio.open(quality_path) as qsrc:
        quality_cube = qsrc.read(1)

    # apply quality filder from cube to image
    cube[quality_cube != 1] = np.nan

    # load cloud tiff
    cloud_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-QL_QUALITY_CLOUD_COG.TIF'
    cloud_path = os.path.join(os.path.dirname(args.reflectance_image), cloud_file_name)

    with rasterio.open(cloud_path) as csrc:
        cloud_cube = csrc.read(1)

    # apply quality filder from cube to image
    cube[cloud_cube != 0] = np.nan

    # load cloud shadow
    cloud_shadow_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-QL_QUALITY_CLOUDSHADOW_COG.TIF'
    cloud_shadow_path = os.path.join(os.path.dirname(args.reflectance_image), cloud_shadow_file_name)

    with rasterio.open(cloud_shadow_path) as cssrc:
        cloud_shadow_cube = cssrc.read(1)

    # apply quality filder from cube to image
    cube[cloud_shadow_cube != 0] = np.nan

    # load snow mask
    snow_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-QL_QUALITY_SNOW_COG.TIF'
    snow_path = os.path.join(os.path.dirname(args.reflectance_image), snow_file_name)

    with rasterio.open(snow_path) as snow_src:
        snow_cube = snow_src.read(1)

    # apply quality filder from cube to image
    cube[snow_cube != 0] = np.nan

    # load cirrus mask
    cirrus_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-QL_QUALITY_CIRRUS_COG.TIF'
    cirrus_shadow_path = os.path.join(os.path.dirname(args.reflectance_image), cirrus_file_name)

    with rasterio.open(cirrus_shadow_path) as cirrus_src:
        cirrus_cube = cirrus_src.read(1)

    # apply quality filder from cube to image
    cube[cirrus_cube != 0] = np.nan

    # load haze mask
    haze_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-QL_QUALITY_HAZE_COG.TIF'
    haze_path = os.path.join(os.path.dirname(args.reflectance_image), haze_file_name)

    with rasterio.open(haze_path) as haze_src:
        haze_cube = haze_src.read(1)

    # apply quality filder from cube to image
    cube[haze_cube != 0] = np.nan

    # load metadata
    metadata_file_name = f'ENMAP01-____L2A-{os.path.basename(args.reflectance_image).split("-")[2]}-METADATA.XML'
    metadata_path = os.path.join(os.path.dirname(args.reflectance_image), metadata_file_name)
    tree = ET.parse(metadata_path)
    root = tree.getroot()

    no_data = root.find(".//backgroundValue").text
    pixel_size = root.find(".//pixelSize").text
    projection = root.find(".//projection").text

    hemisphere = projection.split("_")[2]
    utm_zone = projection.split("_")[1][-2:]

    ul_lat = float(root.find(".//spatialCoverageOfOrthoScene//point[frame='upper_left']/latitude").text)
    ul_lon = float(root.find(".//spatialCoverageOfOrthoScene//point[frame='upper_left']/longitude").text)

    wavelengths = []
    fwhms = []

    p = Proj(proj='utm', zone=utm_zone, ellps='WGS84', preserve_units=False)
    easting, northing = p(ul_lon, ul_lat)

    for band in root.findall(".//bandCharacterisation/bandID"):
        wl = band.find("wavelengthCenterOfBand").text
        fwhm = band.find("FWHMOfBand").text
        wavelengths.append(float(wl))
        fwhms.append(float(fwhm))

    metadata = {'lines': cube.shape[0],
                'samples': cube.shape[1],
                'bands': cube.shape[2],
                'wavelength' : wavelengths,
                'fwhm' : fwhms,
                'interleave': 'bsq',
                'header offset': 0,
                'file type': 'ENVI Standard',
                'data type': envi_typemap[str(cube.dtype)],
                'byte order': 0,
                'map_info': f"{{UTM, 1.0, 1.0, {easting}, {northing}, {pixel_size}, {pixel_size}, {utm_zone}, {hemisphere}, WGS-84, units=Meters}}",
                'data ignore value': float(no_data)}

    cube = np.nan_to_num(cube, nan=float(no_data))
    output_hdr = os.path.join(args.output_directory, f'{os.path.basename(args.reflectance_image).split(".")[0]}.hdr')
    envi.save_image(output_hdr, cube,
                    metadata=metadata,
                    force=True,
                    interleave='bsq', ext='')

    print(f"Successfully saved ENVI file to {output_hdr}")

if __name__ == '__main__':
    main()


