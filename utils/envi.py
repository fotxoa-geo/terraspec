from spectral.io import envi
from osgeo import gdal
import os


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
