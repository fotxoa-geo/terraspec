import numpy as np
from glob import glob
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


bad_wv_regions = [[0, 440], [1310, 1490], [1770, 2050], [2440, 2880]]

def get_good_bands_mask(wavelengths, wavelength_pairs):
    wavelengths = np.array(wavelengths)
    if wavelength_pairs is None:
        wavelength_pairs = bad_wv_regions
    good_bands = np.ones(len(wavelengths)).astype(bool)

    for wvp in wavelength_pairs:
        wvl_diff = wavelengths - wvp[0]
        wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
        lower_index = np.nanargmin(wvl_diff)

        wvl_diff = wvp[1] - wavelengths
        wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
        upper_index = np.nanargmin(wvl_diff)
        good_bands[lower_index:upper_index + 1] = False
    return good_bands


# load meyer-okin
df_kalahari = pd.read_csv(r'G:\My Drive\terraspec\simulation\output\production\meyer-okin.csv')

df_kalahari = df_kalahari[df_kalahari['level_1'].isin(['npv', 'pv'])].copy()

# load ochoa
df_emit = pd.read_csv(r'G:\My Drive\terraspec\simulation\output\production\simulation_asd_data.csv')
df_emit = df_emit[df_emit['level_1'].isin(['npv', 'pv'])].copy()


# load brazil data
df_brz = pd.read_csv(r'G:\My Drive\terraspec\test\raw_data\\2012-leaf-reflectance-spectra-of-tropical-trees-in-tapajos-national-forest.csv')
df_brz.columns = df_brz.columns.astype(float).astype(str)
df_brz['dataset'] = 'brazil'
df_brz['level_1'] = 'pv'

csvs = glob(r'G:\My Drive\terraspec\test\raw_data\*.csv')


list_of_dfs = []

for i in csvs:
    basename = os.path.basename(i).split('.')[0]
    print(basename)
    df_current = pd.read_csv(i)
    df_current.columns = df_current.columns.astype(float).astype(str)
    df_current['dataset'] = basename
    df_current['level_1'] = 'pv'
    list_of_dfs.append(df_current)


# load field data
csv_paths = r'G:\My Drive\terraspec\slpit\output\spectral_transects\endmembers-raw\\'

csvs = sorted(glob(os.path.join(csv_paths, '*asd.csv')))
df_field_all = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
df_field_all['level_1'] = df_field_all['level_1'].str.lower()
df_field_all = df_field_all[df_field_all['level_1'].isin(['npv', 'pv'])].copy()
df_field_all['dataset'] = 'field_cmp'


def is_wavelength(col):
    try:
        return 300 <= float(col) <= 2500  # Adjust range as needed for wavelengths
    except ValueError:
        return False


dfs = [df_kalahari, df_emit, df_field_all, df_brz] + list_of_dfs

print(len(dfs))

# Step 1: Identify all unique columns across all DataFrames
all_columns = set().union(*[df.columns for df in dfs])

# Step 2: Identify common columns
common_columns = set(dfs[0].columns)
for df in dfs[1:]:
    common_columns &= set(df.columns)

# Step 3: Separate columns into non-wavelength and wavelength
non_wavelength_columns = [col for col in all_columns if not is_wavelength(col)]
wavelength_columns = [col for col in all_columns if is_wavelength(col)]

# Step 4: Sort wavelength columns numerically
wavelength_columns_sorted = sorted(wavelength_columns, key=lambda x: float(x))

# Step 5: Define the final column order
ordered_columns = non_wavelength_columns + wavelength_columns_sorted

# Step 6: Reindex each DataFrame to match the final ordered columns
dfs_aligned = [df.reindex(columns=ordered_columns) for df in dfs]

# Step 7: Concatenate vertically
df_all = pd.concat(dfs_aligned, ignore_index=True)
#df_all.to_csv(r'G:\My Drive\terraspec\test\all_asd.csv', index=False)

# run pca on gv data first
for i in ['pv', 'npv']:
    df_select = df_all[df_all['level_1'] == i].copy()
    df_select = df_select[~df_select.iloc[:, 24:].isna().any(axis=1)]

    df_array = df_select.iloc[:, 24:].to_numpy()
    wavelengths = np.array(df_select.columns[24:], dtype='f')
    good_bands = get_good_bands_mask(wavelengths, bad_wv_regions)
    wavelengths[~good_bands] = np.nan
    df_array[:, ~good_bands] = np.nan
    df_array = df_array[:, ~np.isnan(df_array).any(axis=0)]

    print(df_array.shape)
    print(np.min(df_array), np.max(df_array))
    df_array = np.where(df_array >= 1, df_array / 100, df_array)
    print(np.min(df_array), np.max(df_array))

    #df_array = df_array[~np.isnan(df_array).any(axis=1)]

    n_components = min(df_array.shape[0], df_array.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_array)

    pc_df = pd.DataFrame(principal_components,
                         columns=[f'PC{i + 1}' for i in range(principal_components.shape[1])],
                         index=df_select.index)

    df_select = pd.concat([df_select, pc_df], axis=1)

    # plot the results
    plt.figure(figsize=(20, 12))
    class_counts = df_select['dataset'].value_counts()

    for class_label in class_counts.index:
        subset = df_select[df_select['dataset'] == class_label]
        plt.scatter(subset['PC1'], subset['PC2'], label=class_label, alpha=0.6)

    # Find the index of the row with the max PC1 value (overall)
    max_pc1_index = df_select['PC1'].idxmax()
    # lost test

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'{i.upper()}')
    plt.legend(title='Dataset', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(r'G:\My Drive\terraspec\test\\' + i + '_augment.png')