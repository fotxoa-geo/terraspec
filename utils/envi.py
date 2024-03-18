from spectral.io import envi
from osgeo import gdal, osr
import os
import numpy as np

def get_meta(lines: int, samples: int, bands, wvls: bool):
    if wvls:
        meta = {
            'lines': lines,
            'samples': samples,
            'bands': len(bands),
            'wavelength': bands,
            'interleave': 'bil',
            'data type': 4,
            'file_type': 'ENVI Standard',
            'byte order': 0,
            'header offset': 0,
            'wavelength units': 'nm'
        }
    else:
        meta = {
            'lines': lines,
            'samples': samples,
            'bands': len(bands),
            'interleave': 'bil',
            'data type': 4,
            'file_type': 'ENVI Standard',
            'byte order': 0,
            'header offset': 0
        }

    return meta


def save_envi(output_file, meta, grid, ds=None, ul=None):
    if ds is not None and ul is not None:
        driver = gdal.GetDriverByName("ENVI")
        trans = ds.GetGeoTransform()
        outRaster = driver.Create(os.path.splitext(output_file)[0], grid.shape[0], grid.shape[1], grid.shape[2],
                                  gdal.GDT_Float32)
        outRaster.SetGeoTransform([ul[0], trans[1], trans[2], ul[1], trans[4], trans[5]])
        outRaster.SetProjection(ds.GetProjection())
        # outRaster.SetMetadata(list(map(str, list(meta['wavelength']))), "wavelength")

        del outRaster
        outDataset = envi.open(output_file)
        header = envi.read_envi_header(output_file)

        # add wavelength info to file
        header['wavelength'] = list(meta['wavelength'])
        del header['band names']
        outDataset = envi.create_image(output_file, header, ext='', force=True)

    else:
        outDataset = envi.create_image(output_file, meta, ext='', force=True)

    mm = outDataset.open_memmap(interleave='bip', writable=True)
    mm[...] = grid
    del mm

def rgb_quicklook(envi_file, output_directory):
    basename = os.path.basename(envi_file).split("_")[4]
    driver = gdal.GetDriverByName("GTiff")
    out_ras = os.path.join(output_directory, 'emit_rfl_' + basename + '.tif')

    ds = gdal.Open(envi_file, gdal.GA_ReadOnly)
    prj = ds.GetProjection()

    ds_array = ds.ReadAsArray().transpose((1, 2, 0))
    outRaster = driver.Create(out_ras, ds_array.shape[1], ds_array.shape[0], 3, gdal.GDT_Byte)

    originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    rgb_wvls = [470.0304, 574.20905, 656.1857]
    load_order = {
        470.0304: 2,
        574.20905: 1,
        656.1857: 0}

    for _b, b in enumerate(range(0, ds_array.shape[2])):
        band_wvl = float(ds.GetRasterBand(_b + 1).GetDescription().split(" ")[0])

        if band_wvl in rgb_wvls:
            outband = outRaster.GetRasterBand(load_order[band_wvl] + 1)
            band_select = ds_array[:, :, _b]
            band_select *= 255.0 / band_select.max()
            outband.WriteArray(band_select)
            outband.SetNoDataValue(0)

    # # setteing srs from input tif file.
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def envi_tiff_rgb(envi_file, output_directory):
    basename = os.path.basename(envi_file).split("_")[4]
    driver = gdal.GetDriverByName("GTiff")
    out_ras = os.path.join(output_directory, 'emit_' + basename + '.tif')
    ds = gdal.Open(envi_file, gdal.GA_ReadOnly)
    prj = ds.GetProjection()

    ds_array = ds.ReadAsArray().transpose((1, 2, 0))
    outRaster = driver.Create(out_ras, ds_array.shape[1], ds_array.shape[0], 4, gdal.GDT_Byte)

    originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    rgb_wvls = [470.0304, 574.20905, 656.1857]
    load_order = {470.0304: 2, 574.20905: 1, 656.1857: 0}

    for _b, b in enumerate(range(0, ds_array.shape[2])):
        band_wvl = float(ds.GetRasterBand(_b + 1).GetDescription().split(" ")[0])

        if band_wvl in rgb_wvls:
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


def envi_tiff(envi_file, output_directory):
    basename = os.path.basename(envi_file)
    driver = gdal.GetDriverByName("GTiff")
    out_ras = os.path.join(output_directory, basename + '.tif')
    ds = gdal.Open(envi_file, gdal.GA_ReadOnly)
    prj = ds.GetProjection()

    ds_array = ds.ReadAsArray().transpose((1, 2, 0))
    outRaster = driver.Create(out_ras, ds_array.shape[1], ds_array.shape[0], 4, gdal.GDT_Byte)

    originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    for _b, b in enumerate(range(0, ds_array.shape[2])):
        outband = outRaster.GetRasterBand(_b + 1)
        band_select = ds_array[:, :, _b]
        band_select[band_select != 0] = 255
        outband.WriteArray(band_select)

    # settings srs from input tif file.
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def envi_to_array(envi_file):
    ds = gdal.Open(envi_file, gdal.GA_ReadOnly)
    array = ds.ReadAsArray().transpose((1, 2, 0))

    return array


def load_band_names(file):
    ds = gdal.Open(file, gdal.GA_ReadOnly)
    bands = {ds.GetRasterBand(i).GetDescription(): i for i in range(1, ds.RasterCount + 1)}

    return list(bands.keys())


def  augment_envi(file, wvls, out_raster, vertical_average=False, em_index=None):
    ds = gdal.Open(file, gdal.GA_ReadOnly)
    ds_array = envi_to_array(file)

    if ds.RasterYSize == 3: # this is for the EMIT 3x3 windows
        spectra_grid = np.zeros((100, 100, len(wvls)))
    else:
        spectra_grid = np.zeros((ds.RasterYSize, 100, len(wvls)))

    if not vertical_average:
        for _row, row in enumerate(ds_array):
            for _col, col in enumerate(row):
                spectra_grid[_row, _col, :] = ds_array[_row, _col, :]

    else:
        if em_index == None: # this averages all spectral readings from slpit into tone
            spectra_grid[0, 0, :] = np.mean(ds_array, axis=(0, 1))

        else:
            spectra_grid[0, 0, :] = np.mean(ds_array[em_index:, :, :], axis=(0, 1))

    meta_spectra = get_meta(lines=spectra_grid.shape[0], samples=spectra_grid.shape[1], bands=wvls,
                            wvls=True)
    save_envi(out_raster, meta_spectra, spectra_grid)

def read_metadata(hdr_file):
    metadata = {}
    with open(hdr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()
    return metadata


def tetracorder_rgb(envi_file, output_directory):
    basename = os.path.basename(envi_file)
    driver = gdal.GetDriverByName("GTiff")
    out_ras = os.path.join(output_directory, f"minerals_{basename}.tif")

    ds = gdal.Open(envi_file, gdal.GA_ReadOnly)
    prj = ds.GetProjection()
    ds_array = ds.ReadAsArray().transpose((1, 2, 0))

    outRaster = driver.Create(out_ras, ds_array.shape[1], ds_array.shape[0], 4, gdal.GDT_Byte)

    originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    band_names = load_band_names(f"{envi_file}")

    rgb_array = np.zeros((ds_array.shape[0], ds_array.shape[1], 3))

    oxides = ['Goethite', 'Hematite', 'Chlorite']
    clays = ['Illite+Muscovite', 'Kaolinite', 'Montmorillonite', 'Vermiculite']
    carbonates = ['Calcite', 'Dolomite', 'Gypsum']

    # geenrate rgb abundance array
    for _mineral_band, mineral_band in enumerate(band_names):

        if mineral_band in oxides:
            rgb_array[:, :, 0] += ds_array[:, :, _mineral_band]
        elif mineral_band in clays:
            rgb_array[:, :, 2] += ds_array[:, :, _mineral_band]
        elif mineral_band in carbonates:
            rgb_array[:, :, 1] += ds_array[:, :, _mineral_band]

    # set no data bands
    no_data_mask = np.all(rgb_array == np.array([0, 0, 0]), axis=-1)
    rgb_array[no_data_mask] = 0

    # save tiff
    for _b, b in enumerate(range(0, rgb_array.shape[2])): #-1 for alpha band
        outband = outRaster.GetRasterBand(_b + 1)
        band_select = rgb_array[:, :, _b]
        band_select *= 255.0 / band_select.max()
        outband.WriteArray(band_select)
        outband.SetNoDataValue(0)

    # set alpha band
    outband = outRaster.GetRasterBand(4)
    band_select = np.zeros((ds_array.shape[0], ds_array.shape[1]))
    band_select[~no_data_mask] = 255
    outband.WriteArray(band_select)

    # # setteing srs from input tif file.
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()