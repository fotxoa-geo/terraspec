import os
from utils.create_tree import create_directory
from utils.text_guide import execute_call

n_cores = '40'
level_arg = 'level_1'


def call_unmix(mode: str, reflectance_file: str, em_file: str, dry_run: bool, parameters: list, output_dest: str,
               scale: str, spectra_starting_column: str, uncertainty_file=None):

    create_directory(os.path.join(output_dest, mode))
    create_directory(os.path.join(output_dest, mode, 'outlogs'))
    outlog_name = os.path.join(output_dest, mode, 'outlogs', os.path.basename(output_dest) + '.out')

    if uncertainty_file is None:
        uncertainty_file = ""
    else:
        uncertainty_file = uncertainty_file

    # call the unmmix run
    base_call = f'julia -p {n_cores} ~/EMIT/SpectralUnmixing/unmix.jl {reflectance_file} {em_file} ' \
                f'{level_arg} {output_dest} --mode {mode} --spectral_starting_column {spectra_starting_column} --refl_scale {scale} --write_complete_fractions 1 --reflectance_uncertainty_file {uncertainty_file}' \
                f'{" ".join(parameters)} '

    execute_call(['sbatch', '-N', '1', '-c', n_cores, '--mem', "80G", '--output', outlog_name, '--wrap', f'{base_call}'],dry_run)

