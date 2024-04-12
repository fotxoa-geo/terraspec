import os
import time

import matplotlib.pyplot as plt

from utils import asdreader
import struct
import numpy as np
import pandas as pd


def get_asd_binary(data):
    # unpack the binary asd file
    asdformat = '<3s 157s 18s b b b b l b l f f b b b b b H 128s 56s L hh H H f f f f h b 4b H H H b L HHHH f f f 5b'

    file_version, comment, save_time, parent_version, format_version, itime, dc_corrected, dc_time, \
        data_type, ref_time, ch1_wave, wave1_step, data_format, old_dc_count, old_ref_count, old_sample_count, \
        application, channels, app_data, gps_data, intergration_time, fo, dcc, calibration, instrument_num, \
        ymin, ymax, xmin, xmax, ip_numbits, xmode, flags1, flags2, flags3, flags4, dc_count, ref_count, \
        sample_count, instrument, cal_bulb_id, swir1_gain, swir2_gain, swir1_offset, swir2_offset, \
        splice1_wavelength, splice2_wavelength, smart_detector_type, \
        spare1, spare2, spare3, spare4, spare5 = struct.unpack_from(asdformat, data)

    return save_time, gps_data, file_version, format_version


def get_dd_coords(coord):
    dd_mm = float(str(coord).split(".")[0][-2:] + "." + str(coord).split(".")[1])/60
    dd_dd = float(str(coord).split(".")[0][:-2])
    dd = dd_dd + dd_mm
    return dd


def get_reflectance_transect(file, season, plot_directory: str):
    line_num = os.path.split(os.path.split(os.path.split(file)[0])[0])[1]
    plot_name = os.path.split(os.path.split(os.path.split(os.path.split(file)[0])[0])[0])[1]

    try:
        asd = asdreader.reader(file)
        try:
            asd_refl = asd.reflectance
            asd_gps = asd.get_gps()
            latitude_ddmm, longitude_ddmm, elevation, utc_time = asd_gps[0], asd_gps[1], asd_gps[2], asd_gps[3]

            if int(utc_time[0]) + int(utc_time[1]) + int(utc_time[2]) == 0:
                file_time = asd.get_save_time()
                utc_time = str(file_time[2]) + ":" + str(file_time[1]) + ":" + str(file_time[0])

            else:
                utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])

            file_num = int(os.path.basename(file).split(".")[0].split("_")[-1])

            try:
                dd_lat = get_dd_coords(latitude_ddmm)
                dd_long = get_dd_coords(longitude_ddmm) * -1  # used to correct longitude

            except:
                # get long lat from shift plots csv
                df_coords = pd.read_csv(os.path.join('gis', 'plot_coordinates.csv'))
                long = df_coords.loc[
                    (df_coords['Plot Name'] == plot_name) & (df_coords['Season'] == season.upper()), 'longitude'].iloc[
                    0]
                lat = df_coords.loc[
                    (df_coords['Plot Name'] == plot_name) & (df_coords['Season'] == season.upper()), 'latitude'].iloc[0]

                dd_lat = lat
                dd_long = long

            return [plot_name + '-' + season, file, line_num, file_num, dd_long, dd_lat, elevation, utc_time] + list(
                asd_refl)

        except:
            print(file)
            pass

    except:

        # read data on the old ASD files
        data = open(file, "rb").read()
        file_num = int(os.path.basename(file).split(".")[1])

        meta_data_asd = get_asd_binary(data)
        print(meta_data_asd[2])
        print(struct.iter_unpack('3s', data[:2]))

        # get gps data
        gps_binary = struct.unpack('=5d 2b 2b b b l 2b 2b b b', meta_data_asd[1])

        latitude_ddmm, longitude_ddmm, elevation, utc_time = gps_binary[2], gps_binary[3], gps_binary[4], (
        gps_binary[10], gps_binary[9], gps_binary[8])
        utc_time = str(utc_time[0]) + ":" + str(utc_time[1]) + ":" + str(utc_time[2])
        dd_lat = get_dd_coords(latitude_ddmm)
        dd_long = get_dd_coords(longitude_ddmm) * -1  # used to correct longitude

        # asd reflectance
        spectrum = data[484:]
        asd_refl = np.array(list(struct.iter_unpack('<f', spectrum)), dtype=float).flatten()
        asd_refl[:651] *= asd_refl[651] / asd_refl[650]
        plt.plot(asd_refl)
        plt.show()
        time.sleep(10000)

        return [plot_name + '-' + season, file, line_num, file_num, dd_long, dd_lat, elevation, utc_time] + list(
            asd_refl)


def get_gps_data_plots(str_line):
    try:
        str_data = str_line.split(",")
        lat = str_data[0].split(':')[1]
        long = str_data[1].split(':')[1]
        alt = str_data[2].split(':')[1]

        return lat, long, alt

    except:

        return '','',''