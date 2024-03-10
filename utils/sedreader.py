import numpy as np
import re
from collections import namedtuple


def sed_gps(coord):
    updated_coord = coord.replace("Â°", "")
    minutes = float(updated_coord.split(" ")[1])/60
    degrees = float(updated_coord.split(" ")[0])
    dd = degrees + minutes
    return dd


def parse_metadata(sed_lines):
    sed_file_info = namedtuple('metadata', '''date longitude latitude altitude utc_time''')

    for i in range(27): # 27 is where the spectrum starts in sed files
        line_select = sed_lines[i]

        # date match
        date_match = re.search(r"Date: (\b\d{2}/\d{2}/\d{4}\b),(\b\d{2}/\d{2}/\d{4}\b)", line_select.rstrip())
        if date_match:
            date = date_match.group(1)

        # altitude match
        altitude_match = re.search(r"Altitude: (\d+\.\d+)m", line_select.rstrip())
        if altitude_match:
            altitude = altitude_match.group(1)

        # gps match
        longitude_match = re.search(r"Longitude: (\d+Â° \d+\.\d+)\'([WE])", line_select.rstrip()) # the Â° is a weird read
        if longitude_match:

            longitude = longitude_match.group(1)
            longitude = sed_gps(longitude)
            hemisphere = longitude_match.group(2)
            if hemisphere == 'W':
                longitude = longitude * -1

        latitude_match = re.search(r"Latitude: (\d+Â° \d+\.\d+)\'([NS])", line_select.rstrip())  # the Â° is a weird read
        if latitude_match:
            latitude = latitude_match.group(1)
            latitude = sed_gps(latitude)

            hemisphere = latitude_match.group(2)
            if hemisphere == 'S':
                latitude = latitude * -1

        # utc time
        if re.search(r"UTC Time: ", line_select.rstrip()):
            utc_time = line_select.split(" ")[2][:-3]
            hour = utc_time[:2]
            minute = utc_time[2:4]
            second = utc_time[4:]
            utc_time = f"{hour}:{minute}:{second}"


    file_info = sed_file_info._make((date, longitude, latitude, altitude, utc_time))

    return file_info


def load_spectral_info(sed_lines):
    spectral_data = np.loadtxt(sed_lines, delimiter='\t', skiprows=27)
    wavelengths = spectral_data[:, 0].ravel()
    spectrum = spectral_data[:,1].ravel()

    return wavelengths, spectrum


class reader:
    def __init__(self, filename):
        # read file to memory
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
        self.sed = lines
        file.close()

        self.metadata = parse_metadata(self.sed)
        self.wavelengths, self.spectrum = load_spectral_info(self.sed)

    def __getattr__(self, item):
        if item == 'reflectance':
            return self.get_reflectance()

        elif item == 'gps':
            return self.get_gps()

    def get_reflectance(self):
        spectrum = self.spectrum
        return spectrum

    def get_gps(self):
        longitude, latitude = self.metadata.longitude, self.metadata.latitude
        utc_time = self.metadata.utc_time
        elevation = self.metadata.altitude

        return longitude, latitude, utc_time, elevation