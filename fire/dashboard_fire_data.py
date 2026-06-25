import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from scipy.stats import gaussian_kde
from spectral.io import envi
from utils.envi import envi_to_array



# --- INSTANTIATION HOOK ---
dashboard = InteractiveMatplotlibDashboard(
    time_series_directory=self.time_series_directory,
    fig_directory=self.fig_directory,
    aoi=self.aoi,
    instrument=self.instrument
)