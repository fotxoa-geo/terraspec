import os
import sys
import json
import itertools
import logging
from simulation.hypertrace.hypertrace import mkabs
from sys import platform
from simulation.run_unmix import execute_call
import subprocess
from utils.create_tree import create_directory


# this is a bug fix for isofit; it's changing the path
if "win" not in platform:
    from isofit.utils import surface_model

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;', ':')
    os.environ['PATH'] = path

level_arg = 'level_1'
n_cores = '40'


def build_surface(wavelength_file: str):
    # create output directory for surface file
    sensor = os.path.basename(wavelength_file).split("_")[0]
    outpath = os.path.join(f'simulation/hypertrace/hypertrace-data/priors/', sensor)
    create_directory(outpath)

    # build surface mat file
    surface_model(config_path=f'simulation/hypertrace/surface/surface_20221020.json',
                  wavelength_path=f'utils/wavelengths',
                  output_path=os.path.join(outpath, sensor + '.mat'))


def hypertrace_workflow(dry_run: bool, clean: bool, configfile: str):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    configfile = mkabs(configfile)
    logger.info("Using config file `%s`", configfile)

    with open(configfile) as f:
        config = json.load(f)

    wavelength_file = mkabs(config["wavelength_file"])
    reflectance_file = mkabs(config["reflectance_file"])
    if "libradtran_template_file" in config:
        raise Exception("`libradtran_template_file` is deprecated. Use `rtm_template_file` instead.")
    rtm_template_file = mkabs(config["rtm_template_file"])
    lutdir = mkabs(config["lutdir"])
    outdir = mkabs(config["outdir"])

    if clean and outdir.exists():
        import shutil

        shutil.rmtree(outdir)

    isofit_config = config["isofit"]
    hypertrace_config = config["hypertrace"]

    # Make RTM paths absolute
    vswir_settings = isofit_config["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]

    for key in ["lut_path", "template_file", "engine_base_dir"]:
        if key in vswir_settings:
            vswir_settings[key] = str(mkabs(vswir_settings[key]))

    # Create iterable config permutation object
    ht_iter = itertools.product(*hypertrace_config.values())
    logger.info("Starting Hypertrace workflow.")

    for ht in ht_iter:
        argd = dict()
        for key, value in zip(hypertrace_config.keys(), ht):
            argd[key] = value
        logger.info("Running config: %s", argd)

        atm_aod_h2o = json.dumps(json.dumps(argd["atm_aod_h2o"]))
        base_call = f'./hypertrace/hypertrace.py -wvls {wavelength_file} -refl {reflectance_file} -rtm {rtm_template_file} -lutd {lutdir} -od {outdir} ' \
                    f'-surface {argd["surface_file"]} -noise {argd["noisefile"]} -type {argd["atm_aod_h2o"][0]} ' \
                    f'-aod {str(argd["atm_aod_h2o"][1])} -h2o {str(argd["atm_aod_h2o"][2])} -time {str(argd["localtime"])} -configs {configfile}'

        job_name = "py-hyper:time:" + str(argd["localtime"]) + '-aod:' + str(argd["atm_aod_h2o"][1]) + '-h2o' + str(argd["atm_aod_h2o"][2])

        #execute_call(['sbatch', '-N', '1', '-c', n_cores, '--mem', '180G', '--job-name', job_name, '--wrap', f'python {base_call}'], dry_run)
            # execute_call(['srun', '-N', '1', '-c', n_cores, '--mem', '180G', '--job-name', job_name, '--pty', f'python {base_call}'], dry_run)
            # subprocess.call(['srun', '-N', '1', '-c', n_cores, '--mem', '180G', '--pty', f'python {base_call}'])
        subprocess.call([f'srun -N 1 -c 40 --mem 180G --pty  python {base_call}'], shell=True)  # this works
    logging.info("Workflow completed successfully.")


def run_hypertrace_workflow():
    build_surface(wavelength_file='./utils/wavelengths/emit_wavelengths.txt')
    hypertrace_workflow(dry_run=True, clean=False, configfile=os.path.join('simulation', 'hypertrace', 'veg-sim.json'))
