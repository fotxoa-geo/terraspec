import argparse
from osgeo import gdal
import math
from utils.spectra_utils import get_meta, save_envi, spectra


def get_index(x: float, y: float, ox: float, oy: float, pw: float, ph: float) -> tuple:
    """
    Gets the row (i) and column (j) indices in an NumPy 2D array for a given
    pair of coordinates.

    Parameters
    ----------
    x : float
        x (longitude) coordinate
    y : float
        y (latitude) coordinate
    ox : float
        Raster x origin (minimum x coordinate)
    oy : float
        Raster y origin (maximum y coordinate)
    pw : float
        Raster pixel width
    ph : float
        Raster pixel height

    Returns
    -------
    Two-element tuple with the column and row indices.

    Notes
    -----
    This function is based on: https://gis.stackexchange.com/a/92015/86131.

    Both x and y coordinates must be within the raster boundaries. Otherwise,
    the index will not correspond to the actual values or will be out of
    bounds.
    """
    # make sure pixel height is positive

    i = int(math.floor((y - oy) / ph))
    j = int(math.floor((x - ox) / pw))

    return i, j


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Run vegetation workflow')
    parser.add_argument('-rfl_img', '--reflectance_image', type=str, help='Reflectance image')
    parser.add_argument('-w_size', '--window_size', type=int, help="Specify window size", default=3)
    parser.add_argument('-pad', '--padding', type=int, help="Specify padding", default=1)
    parser.add_argument('-lon', '--longitude_center', type=float, help="Specify longitude center")
    parser.add_argument('-lat', '--latitude_center', type=float, help="Specify latitude center")
    parser.add_argument('-out_name', '--output_name', type=str, help="Specify output desination and name")
    parser.add_argument('-sns', '--sensor', help="sensor", default='emit')
    args = parser.parse_args()

    ds = gdal.Open(args.reflectance_image, gdal.GA_ReadOnly)
    ox, pw, xskew, oy, yskew, ph = ds.GetGeoTransform()
    nd_value = ds.GetRasterBand(1).GetNoDataValue()
    arr = ds.ReadAsArray().transpose((1, 2, 0))

    # get wavelengths
    wvls, fwhm = spectra.load_wavelengths(sensor=args.sensor)

    # get index
    row, col = get_index(args.longitude_center, args.latitude_center, ox, oy, pw, ph)

    # get pixel value and its neighboors
    window = arr[row - args.padding: row + args.padding + 1, col - args.padding: col + args.padding + 1, :]
    window[window == -0.1] = -9999.0

    if window.size == 0:
        print('empty array')
        print(args.output_name)
    elif window[0, 0, 0] == 0:
        print("no data values?")
        print(args.output_name)
    else:
        # make array an envi array for unmixing
        meta = get_meta(lines=window.shape[0], samples=window.shape[1], bands=wvls, wvls=True)
        print(row, col, args.padding)
        print(ox, oy, pw, ph)
        save_envi(args.output_name, meta, window, ds,
                  ul=[ox + (col - args.padding) * pw, oy + (row - args.padding) * ph])

        print('successfully saved ', args.output_name)


if __name__ == '__main__':
    main()