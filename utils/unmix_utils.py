import os
from utils.create_tree import create_directory
from utils.text_guide import execute_call
from osgeo import gdal
from utils.envi import get_meta, save_envi

n_cores = '40'
level_arg = 'level_1'


def hypertrace_meta(reflectance_file):
    # get metadata from hypertrace outputs
    atmosphere = os.path.abspath(reflectance_file).split('/')[-7].split("__")[0].split("_")[1:]
    atmosphere = "-".join(atmosphere)
    altitude = os.path.abspath(reflectance_file).split('/')[-7].split("__")[1].split("_")[1]
    doy = os.path.abspath(reflectance_file).split('/')[-7].split("__")[2].split("_")[1]
    lat = os.path.abspath(reflectance_file).split('/')[-7].split("__")[3].split("_")[1]
    long = os.path.abspath(reflectance_file).split('/')[-7].split("__")[4].split("_")[1]

    # get geometry information from hypertrace output
    azimuth = os.path.abspath(reflectance_file).split('/')[-6].split("__")[0].split("_")[1]
    sensor_zenith = os.path.abspath(reflectance_file).split('/')[-6].split("__")[1].split("_")[1]
    time = os.path.abspath(reflectance_file).split('/')[-6].split("__")[2].split("_")[1]
    elev = os.path.abspath(reflectance_file).split('/')[-6].split("__")[3].split("_")[1]

    # get atmoshperic conditions from hypertrace output
    aod = os.path.abspath(reflectance_file).split('/')[-3].split("__")[0].split("_")[1]
    h2o = os.path.abspath(reflectance_file).split('/')[-3].split("__")[1].split("_")[1]

    basename = [atmosphere, altitude, doy, long, lat, azimuth, sensor_zenith, time, elev, aod, h2o]
    basename = "_".join(basename)

    return basename
def create_uncertainty(uncertainty_file: str, wvls):
    output = os.path.join(os.path.dirname(uncertainty_file), 'reflectance_uncertainty.hdr')
    if os.path.isfile(output):
        pass
    else:
        ds_uncertainty = gdal.Open(uncertainty_file, gdal.GA_ReadOnly)
        uncertainty_array = ds_uncertainty.ReadAsArray().transpose((1, 2, 0))
        uncertainty_array = uncertainty_array[:, :, :-2]

        uncertainty_meta = get_meta(lines=uncertainty_array.shape[0], samples=uncertainty_array.shape[1], bands=wvls, wvls=True)
        save_envi(output_file=output, meta=uncertainty_meta, grid=uncertainty_array)


def call_unmix(mode: str, reflectance_file: str, em_file: str, dry_run: bool, parameters: list, output_dest: str,
               scale: str, spectra_starting_column: str, uncertainty_file=None):

    create_directory(os.path.join(output_dest, mode))
    create_directory(os.path.join(output_dest, mode, 'outlogs'))
    outlog_name = os.path.join(output_dest, mode, 'outlogs', os.path.basename(reflectance_file) + '.out')

    if uncertainty_file == None:
        uncertainty_file = ""
    else:
        uncertainty_file = uncertainty_file

    # call the unmmix run
    base_call = f'julia -p {n_cores} ~/EMIT/SpectralUnmixing/unmix.jl {reflectance_file} {em_file} ' \
                f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column {spectra_starting_column} --refl_scale {scale} --reflectance_uncertainty_file {uncertainty_file} --write_complete_fractions 1' \
                f'{" ".join(parameters)} '

    execute_call(['sbatch', '-N', '1', '-c', n_cores, '--mem', "80G", '--output', outlog_name, '--wrap', f'{base_call}'],dry_run)


def call_hypertrace_unmix(mode: str, reflectance_file: str, em_file: str, dry_run: bool, parameters: list, output_dest: str,
                          scale: str, spectra_starting_column: str):

    # path to reflectance uncertainty file
    uncertainty_file = os.path.join(os.path.dirname(reflectance_file), "reflectance_uncertainty")

    base_call = f'julia -p {n_cores} ~/EMIT/SpectralUnmixing/unmix.jl {reflectance_file} {em_file} ' \
                f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column {spectra_starting_column} --refl_scale {scale} --reflectance_uncertainty_file {uncertainty_file} ' \
                f'--reflectance_uncertainty_file {uncertainty_file} ' \
                f'{" ".join(parameters)}'

    execute_call(['sbatch', '-N', "1", '-c', n_cores, '--mem', "180G", '--wrap', f'{base_call}'], dry_run)

