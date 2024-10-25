from utils.envi import envi_to_array
from utils.spectra_utils import spectra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mixed_spectra = r'G:\My Drive\terraspec\tetracorder\output\simulated_spectra\tetracorder_soil_spectra'
fraction_file = r'G:\My Drive\terraspec\tetracorder\output\simulated_spectra\tetracorder_soil_fractions'
spectra_array = envi_to_array(mixed_spectra)
fractions = envi_to_array(fraction_file)

wvls = spectra.load_wavelengths(sensor='emit')

red = spectra_array[:,:, 40] # 656 nm
nir = spectra_array[:,:, 68]
ndvi = (nir-red)/(nir+red)

fig = plt.figure(figsize=(6, 4))
unique_x_val = np.unique(fractions[0, :, 2])
plt.violinplot([ndvi[:, i] for i in range(ndvi.shape[1])], showmeans=True)

plt.xlabel('% Soil Cover')
plt.ylabel('NDVI')
step = max(1, len(unique_x_val) // 10)
ticks = np.arange(1, len(unique_x_val) + 1, step)
labels = [f'{val*100:.0f}%' for val in unique_x_val[::step]]

plt.xticks(ticks=ticks, labels=labels)
plt.ylim(0, 1)

plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().tick_params(which='minor', length=4, color='b')

plt.savefig(r'C:\Users\spect\Desktop\npp_figs\ndvi_simulation.png', dpi = 300)
plt.clf()
plt.close()


[{'continuum': [0.44, 0.455, 0.6117, 0.6315], 'feature_type': 'DLw', 'rct/lct>': ['0.9', '1.1'], 'ct': ['[CTHRESH4]']},
{'continuum': [0.76, 0.795, 1.08, 1.11], 'feature_type': 'OLw', 'rct/lct>': ['0.4', '0.6'], 'ct': ['[CTHRESH4]']}]