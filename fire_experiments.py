import numpy as np
from glob import glob
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# JPL samples
path = r'G:\My Drive\terraspec\slpit\output\spectral_transects\endmembers-raw\\Spectral-'
df_9999 = pd.read_csv(path + '9999-asd.csv')
df_9999.insert(0, 'dataset', 'JPL')
df_9999['level_1'] = 'ash'

df_94 = pd.read_csv(path + '094-asd.csv')
df_94.insert(0, 'dataset', 'JPL')
df_94['level_1'] = 'ash'
df_98 = pd.read_csv(path + '098-asd.csv')
df_98.insert(0, 'dataset', 'JPL')
df_98['level_1'] = 'ash'

# lake fire spectra
dfs_lf = []
for i in ['063' , '066', '069', '070', '071', '072', '074', '075', '078', '079']:
    df_lf = pd.read_csv(f'{path}{i}-asd.csv')
    df_lf.insert(0, 'dataset', 'Lake Fire')
    dfs_lf.append(df_lf)

#df_lake_fire['level_1'] = 'ash'


def normalize_column_names(df):
    df = df.copy()
    df.columns = [
        str(int(float(col))) if col.replace('.', '', 1).isdigit() else col
        for col in df.columns
    ]
    return df


dfs = [df_9999, df_94, df_98] + dfs_lf

df_list = [normalize_column_names(df) for df in dfs]
df_all = pd.concat(df_list, axis=0, ignore_index=True)
df_all.to_csv(r'G:\My Drive\terraspec\test\all_fire.csv', index=False)

# run pca on gv data first
for i in ['ash']:
    #df_select = df_all[df_all['level_1'] == i].copy()
    df_select = df_all
    print(df_select.level_1.unique())
    df_array = df_select.iloc[:, 13:].to_numpy()

    n_components = min(df_array.shape[0], df_array.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_array)

    pc_df = pd.DataFrame(principal_components,
                         columns=[f'PC{i + 1}' for i in range(principal_components.shape[1])],
                         index=df_select.index)

    df_select = pd.concat([df_select, pc_df], axis=1)

    # plot the results
    plt.figure(figsize=(8, 6))
    df_select_jpl = df_select[df_select['dataset'] == 'JPL']
    class_counts = df_select_jpl['dataset'].value_counts()

    for class_label in class_counts.index:
        subset = df_select_jpl[df_select_jpl['dataset'] == class_label]
        plt.scatter(subset['PC1'], subset['PC2'], label=class_label, alpha=0.6)

    # plot ems for LF spectra
    df_select_lf = df_select[df_select['dataset'] == 'Lake Fire']
    class_counts = df_select_lf['level_1'].value_counts()

    for class_label in class_counts.index:
        subset = df_select_lf[df_select_lf['level_1'] == class_label]
        plt.scatter(subset['PC1'], subset['PC2'], label=class_label, alpha=0.6)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'{i.upper()}')
    plt.legend(title='Dataset')
    plt.savefig(r'G:\My Drive\terraspec\test\\' + i + '_augment.png')