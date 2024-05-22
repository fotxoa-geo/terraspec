import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from utils.spectra_utils import spectra
from scipy import signal

path = r'G:\My Drive\terraspec\slpit\output\spectral_transects\endmembers\Spectral-060-asd.csv'

data = np.array(pd.read_csv(path).iloc[45, 11:].values)

time = np.linspace(350, 2500, 2151)
vector = data  # Replace with your own vector
numerator, denominator, _ = signal.residue([1], [1, 0, 1])
# Evaluate the Laplace transformation
vector_transformed = signal.lfilter(numerator, denominator, vector)

# Plot the original vector and its Laplace transformation
plt.subplot(2, 1, 1)
plt.plot(time, vector)
plt.title('Original Vector')

plt.subplot(2, 1, 2)
plt.plot(time, vector_transformed)
plt.title('Laplace Transformation')

plt.tight_layout()
plt.show()