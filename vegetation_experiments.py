import numpy as np
from glob import glob
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


dfs = [df_kalahari, df_emit, df_field_all, df_brz]

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
df_all.to_csv(r'G:\My Drive\terraspec\test\all_asd.csv', index=False)

# run pca on gv data first
for i in ['pv', 'npv']:
    df_select = df_all[df_all['level_1'] == i].copy()
    df_array = df_select.iloc[:, 18:].to_numpy()

    n_components = min(df_array.shape[0], df_array.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_array)

    pc_df = pd.DataFrame(principal_components,
                         columns=[f'PC{i + 1}' for i in range(principal_components.shape[1])],
                         index=df_select.index)

    df_select = pd.concat([df_select, pc_df], axis=1)

    # plot the results
    plt.figure(figsize=(8, 6))
    class_counts = df_select['dataset'].value_counts()

    for class_label in class_counts.index:
        subset = df_select[df_select['dataset'] == class_label]
        plt.scatter(subset['PC1'], subset['PC2'], label=class_label, alpha=0.6)

    # Find the index of the row with the max PC1 value (overall)
    max_pc1_index = df_select['PC1'].idxmax()

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'{i.upper()}')
    plt.legend(title='Dataset')
    plt.savefig(r'G:\My Drive\terraspec\test\\' + i + '_augment.png')