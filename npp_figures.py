import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from utils.spectra_utils import spectra

em_csv = r'G:\My Drive\terraspec\slpit\output\spectral_transects\endmembers-raw\Spectral-051-asd.csv'
df_em = pd.read_csv(em_csv)

hematite_spec = np.loadtxt(r'C:\Users\spect\Desktop\npp_figs\splib07a_Hematite_FE2602_BECKb_AREF.txt', skiprows=1)
hematite_wvls = np.loadtxt(r'C:\Users\spect\Desktop\npp_figs\splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt', skiprows=1) * 1000

df_gv = df_em.loc[df_em['level_1'] == 'PV']
df_spectra = df_gv.iloc[0, 11:].to_numpy()
df_wvls = df_gv.columns[11:]
df_wvls = np.array([float(item) for item in df_wvls])

good_bands_df = spectra.get_good_bands_mask(df_wvls, wavelength_pairs=None)
df_wvls[~good_bands_df] = np.nan

good_emit_usgs = spectra.get_good_bands_mask(hematite_wvls, wavelength_pairs=None)
hematite_wvls[~good_emit_usgs] = np.nan

# plot cholorophyll abs
chlorophyll_abs = [0.64, 0.66]
hematite_abs = [0.86, 0.9, 0.65]



fig = plt.figure(figsize=(6, 4))
plt.plot(df_wvls, df_spectra, label='GV', color='green')
plt.plot(hematite_wvls, hematite_spec, label='Hematite', color='red')

for x in chlorophyll_abs:
    plt.axvline(x=x * 1000, color='green', linestyle='--', linewidth=0.8)

for x in hematite_abs:
    plt.axvline(x=x * 1000, color='red', linestyle='--', linewidth=0.8)


plt.xlim(200, 1200)
plt.xticks(range(200, 1201, 200))
plt.ylim(0, 0.6)
plt.xlabel('Wavelengths (nm)')
plt.ylabel('Reflectance (%)')
plt.gca().xaxis.set_minor_locator(MultipleLocator(100))
plt.gca().tick_params(which='minor', length=4, color='b')
plt.legend()

plt.savefig(r'C:\Users\spect\Desktop\npp_figs\abs_overlapp.png', dpi = 300)
plt.clf()
plt.close()