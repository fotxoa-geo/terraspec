"""""
|----------------------------------------------------------------------------------------------------------------------
| Date                : March 2022
| Copyright           : (C) 2022 by fotxoa-geo; The University of California - Los Angeles
| Email               : fochoa1@g.ucla.edu
| Acknowledgements    : Inspired by ajtag's asd reader (https://github.com/ajtag/asdreader)
| ----------------------------------------------------------------------------------------------------------------------
"""
import struct
import datetime
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

## constant defs
spectra_type = ('RAW', 'REF', 'RAD', 'NOUNITS', 'IRRAD', 'QI', 'TRANS', 'UNKNOWN', 'ABS')
data_type = ('FLOAT', 'INTEGER', 'DOUBLE', 'UNKNOWN')
instrument_type = ('UNKNOWN', 'PSII', 'LSVNIR', 'FSVNIR', 'FSFR', 'FSNIR', 'CHEM', 'FSFR_UNATTENDED',)
calibration_type = ('ABSOLUTE', 'BASE', 'LAMP', 'FIBER')


def parse_bstr(asd, offset):
    try:
        size = struct.unpack_from('<h', asd, offset)
        offset += struct.calcsize('<h')

        bstr_format = '<{}s'.format(size[0])
        bstr = struct.unpack_from(bstr_format, asd, offset)

        offset += struct.calcsize(bstr_format)
        return bstr[0], offset

    except:
        print("errorrrr")
        raise


def parse_constituants(asd, offset):
    name, offset = parse_bstr(asd, offset)
    passfail, offset = parse_bstr(asd, offset)
    fmt = 'd d d d d d d d d l d d'
    offset += struct.calcsize(fmt)
    return name, offset


def parse_gps(gps_field):
    gps_data = struct.unpack('=5d 2b 2b b b l 2b 2b b b', gps_field)
    latitude, longitude, elevation, time = gps_data[2], gps_data[3], gps_data[4], (gps_data[10], gps_data[9], gps_data[8])
    #print(latitude, longitude, elevation, time) # time data is in HH:MM:SS UTC(gps) time!!
    return [latitude, longitude, elevation, time]


def parse_time(save_time):
    time_info = namedtuple('time_info',
                           '''seconds minutes hour day month year_since_1900 day_of_week doy daylight_savings''')
    seconds, minutes, hour, day, month, year_since_1900, day_of_week, doy, daylight_savings = \
        struct.unpack_from('9h', save_time)
    time_tuple = time_info._make(
        (seconds, minutes, hour, day, month, year_since_1900, day_of_week, doy, daylight_savings))

    return time_tuple


def parse_spectra(asd, offset, channels):
    spec = np.array(struct.unpack_from('<{}d'.format(channels), asd, offset))
    offset += (channels * 8)
    return spec, offset


def parse_reference(asd, offset):
    reference_file_header = struct.unpack_from('<h q q'.format(), asd, offset)
    description, offset = parse_bstr(asd, offset + 18)
    return reference_file_header + (description,), offset


def parse_metadata(asd):
    asdformat = '<3s 157s 18s b b b b l b l f f b b b b b H 128s 56s L hh H H f f f f h b 4b H H H b L HHHH f f f 5b'
    asd_file_info = namedtuple('metadata', '''
         file_version comment save_time parent_version format_version itime 
         dc_corrected dc_time data_type ref_time ch1_wave wave1_step 
         data_format old_dc_count old_ref_count old_sample_count 
         application channels 
         app_data gps_data 
         intergration_time fo dcc calibration instrument_num
         ymin ymax xmin xmax
         ip_numbits xmode flags1 flags2 flags3 flags4
         dc_count ref_count sample_count instrument cal_bulb_id swir1_gain
         swir2_gain swir1_offset swir2_offset
         splice1_wavelength splice2_wavelength smart_detector_type
         spare1 spare2 spare3 spare4 spare5
         ''')

    file_version, comment, save_time, program_version, format_version, itime, dc_corr, dc_time, \
    data_type, ref_time, ch1_wave, wave1_step, data_format, old_dc_count, old_ref_count, old_sample_count, \
    application, channels, app_data, gps_data, intergration_time, fwd_op, dcc, calibration, instrument_num, \
    ymin, ymax, xmin, xmax, ip_numbits, xmode, flags1, flags2, flags3, flags4, dc_count, ref_count, \
    sample_count, instrument, cal_bulb_id, swir1_gain, swir2_gain, swir1_offset, swir2_offset, \
    splice1_wavelength, splice2_wavelength, smart_detector_type, \
    spare1, spare2, spare3, spare4, spare5 = struct.unpack_from(asdformat, asd)

    comment = comment.strip(b'\x00')

    save_time = parse_time(save_time)
    dc_time = datetime.datetime.fromtimestamp(dc_time)
    ref_time = datetime.datetime.fromtimestamp(ref_time)
    app_data = '' # used for runtime functions; can set to blank no format on asd files
    gps_data = parse_gps(gps_data)

    fi = asd_file_info._make(
        (file_version, comment, save_time, program_version, format_version, itime, dc_corr, dc_time, \
         data_type, ref_time, ch1_wave, wave1_step, data_format, old_dc_count, old_ref_count, old_sample_count, \
         application, channels, app_data, gps_data, intergration_time, fwd_op, dcc, calibration, instrument_num, \
         ymin, ymax, xmin, xmax, ip_numbits, xmode, flags1, flags2, flags3, flags4, dc_count, ref_count, \
         sample_count, instrument, cal_bulb_id, swir1_gain, swir2_gain, swir1_offset, swir2_offset, \
         splice1_wavelength, splice2_wavelength, smart_detector_type, \
         spare1, spare2, spare3, spare4, spare5))

    # 484 is the offset of the metadata data
    return fi, 484


def parse_classifier(asd, offset):
    offset += struct.calcsize('bb')

    title, offset = parse_bstr(asd, offset)
    subtitle, offset = parse_bstr(asd, offset)
    productname, offset = parse_bstr(asd, offset)
    vendor, offset = parse_bstr(asd, offset)
    lotnumber, offset = parse_bstr(asd, offset)
    sample, offset = parse_bstr(asd, offset)
    modelname, offset = parse_bstr(asd, offset)
    operator, offset = parse_bstr(asd, offset)
    datetime, offset = parse_bstr(asd, offset)
    instrument, offset = parse_bstr(asd, offset)
    serialnumber, offset = parse_bstr(asd, offset)
    displaymode, offset = parse_bstr(asd, offset)
    comments, offset = parse_bstr(asd, offset)
    units, offset = parse_bstr(asd, offset)
    filename, offset = parse_bstr(asd, offset)
    username, offset = parse_bstr(asd, offset)
    reserved1, offset = parse_bstr(asd, offset)
    reserved2, offset = parse_bstr(asd, offset)
    reserved3, offset = parse_bstr(asd, offset)
    reserved4, offset = parse_bstr(asd, offset)
    constituantCount, = struct.unpack_from('<h', asd, offset)
    offset += struct.calcsize('<h')
    for i in range(constituantCount):
        name, offset = parse_constituants(asd, offset)
        print(name)
    return '', offset


def normalise_spectrum(spec, wavelengths, metadata):
    res = spec.copy()

    splice1_index = np.where(wavelengths == int(metadata.splice1_wavelength))[0][0]
    splice2_index = np.where(wavelengths == int(metadata.splice2_wavelength))[0][0]
    res[:splice1_index] = spec[:splice1_index] / metadata.intergration_time
    res[splice1_index:splice2_index] = spec[
                                       splice1_index:splice2_index] * metadata.swir1_gain / 2048
    res[splice2_index:] = spec[splice2_index:] * metadata.swir1_gain / 2048

    # fix first slice of spectra
    res[:splice1_index + 1] *= res[splice1_index + 1] / res[splice1_index]
    return res


def parse_dependants(asd, offset):
    dependant_format = '< ?h'
    dependants = struct.unpack_from(dependant_format, asd, offset)
    offset += struct.calcsize(dependant_format)

    s, offset = parse_bstr(asd, offset)

    dependant_format = '< f'
    dependants = dependants + struct.unpack_from(dependant_format, asd, offset)
    offset += struct.calcsize(dependant_format)
    return dependants, offset


def parse_calibration_header(asd, offset):
    header_format = '<b'
    calibration_buffer_format = '<b 20s i h h'

    offset += 1
    buffer_count = struct.unpack_from(header_format, asd, offset)[0]
    offset += struct.calcsize(header_format)

    calibration_buffer = []
    for i in range(buffer_count):
        (cal_type, name, intergration_time, swir1gain, swir2gain) = struct.unpack_from(calibration_buffer_format, asd, offset)
        name = name.strip(b'\x00')

        calibration_buffer.append(((cal_type, name, intergration_time, swir1gain, swir2gain)))
        offset += struct.calcsize(calibration_buffer_format)

    return calibration_buffer, offset


class reader:
    def __init__(self, filename):
        # read in file to memory
        fh = open(filename, 'rb')
        self.asd = fh.read()
        fh.close()

        self.md, offset = parse_metadata(self.asd) # returns metadata tuple and metadata offset
        self.wavelengths = np.arange(self.md.ch1_wave, self.md.ch1_wave + self.md.channels * self.md.wave1_step,
                                     self.md.wave1_step) # returns wavelengths
        self.spec, offset = parse_spectra(self.asd, offset, self.md.channels) # returns spectra in DN and offset
        reference_header, offset = parse_reference(self.asd, offset)
        self.reference, offset = parse_spectra(self.asd, offset, self.md.channels)

        self.classifier, offset = parse_classifier(self.asd, offset)
        self.dependants, offset = parse_dependants(self.asd, offset)
        self.calibration_header, offset = parse_calibration_header(self.asd, offset)

        for hdr in self.calibration_header:  # Number of calibration buffers in the file.
            if calibration_type[hdr[0]] == 'BASE':
                self.calibration_base, offset = parse_spectra(self.asd, offset, self.md.channels)
            elif calibration_type[hdr[0]] == 'LAMP':
                self.calibration_lamp, offset = parse_spectra(self.asd, offset, self.md.channels)
            elif calibration_type[hdr[0]] == 'FIBER':
                self.calibration_fibre, offset = parse_spectra(self.asd, offset, self.md.channels)

        # plt.plot(self.wavelengths, self.spec, label='Measured Spectra')
        # plt.plot(self.wavelengths, self.reference, label='Reference Spectra')
        # plt.legend()
        # plt.show()

    def __getattr__(self, item):
        if item == 'reflectance':
            return self.get_reflectance()

        elif item == 'white_reference':
            return self.get_white_reference()

        elif item == 'ref':
            return self.reference

        elif item == 'gps':
            return self.get_gps()

    def get_reflectance(self):
        if spectra_type[self.md.data_type] == 'REF':
            res = normalise_spectrum(self.spec, self.wavelengths, self.md) / normalise_spectrum(self.reference, self.wavelengths, self.md)
        else:
            raise TypeError('spectral data contains {}. REF data is needed'.format(spectra_type[self.md.data_type]))
        return res

    def get_white_reference(self):
        return normalise_spectrum(self.reference, self.wavelengths, self.md)

    def get_gps(self):
        gps = self.md.gps_data
        return gps

    def get_save_time(self):
        save_time = self.md.save_time
        return save_time