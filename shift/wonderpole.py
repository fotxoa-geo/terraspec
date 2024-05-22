import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.results_utils import r2_calculations
from matplotlib.ticker import FormatStrFormatter

skip = ['SRA-000-SPRING', 'SRB-047-SPRING', 'SRB-004-FALL', 'SRB-050-FALL', 'SRB-200-FALL', 'SRA-056-SPRING']

df_slpit = pd.read_csv(r'G:\My Drive\terraspec\shift\figures\shift_fraction_output.csv')
df_slpit = df_slpit[(df_slpit['lib_mode'] == 'local') & (df_slpit['normalization'] == 'brightness') & (df_slpit['unmix_mode'] == 'sma')].copy()
df_slpit = df_slpit[~df_slpit['plot'].isin(skip)]

df_wonderpole = pd.read_excel(r'G:\My Drive\terraspec\shift\figures\wonderpole.xlsx')
df_wonderpole['plot'] = df_wonderpole["plot_name"] + '-' + df_wonderpole["season"]
df_wonderpole = df_wonderpole.dropna()
df_wonderpole = df_wonderpole[~df_wonderpole['plot'].isin(skip)]

# # create figure
fig = plt.figure(figsize=(12, 12))
ncols = 3
nrows = 3
gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.25, width_ratios=[1] * ncols, height_ratios=[1] * nrows)

col_map = {
    0: 'npv',
    1: 'pv',
    2: 'soil'}


col_map_wp = {
    0: 'npv_sh',
    1: 'pv_sh',
    2: 'soil_sh'}



ems = ["NPV", "GV", "Soil"]

# loop through figure columns
for row in range(nrows):
    if row == 0:
        df_x = df_wonderpole
        df_y = df_slpit[(df_slpit['instrument'] == 'aviris')].copy()
    if row == 1:
        df_x = df_wonderpole
        df_y = df_slpit[(df_slpit['instrument'] == 'asd')].copy()

    if row == 2:
        df_x = df_slpit[(df_slpit['instrument'] == 'asd')].copy()
        df_y = df_slpit[(df_slpit['instrument'] == 'aviris')].copy()

    df_x = df_x.sort_values('plot')
    df_y = df_y.sort_values('plot')

    for col in range(ncols):
        ax = fig.add_subplot(gs[row, col])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        mode = list(df_slpit['unmix_mode'].unique())[0]
        lib_mode = list(df_slpit['lib_mode'].unique())[0]

        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str(2)}f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{str(2)}f'))

        # titles
        if row == 0:
            ax.set_title(ems[col], fontsize=26)

        # x axis
        if row == 0:
            ax.set_xlabel("Wonderpole", fontsize=18)

        if row == 1:
            ax.set_xlabel("Wonderpole", fontsize=18)

        if row == 2:
            ax.set_xlabel("SLPIT", fontsize=18)

        # y axis
        if row == 0 and col == 0:
            ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$\nAVIRIS', fontsize=18)

        if row == 1 and col == 0:
            ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$\nSLPIT', fontsize=18)

        if row == 2 and col == 0:
            ax.set_ylabel(mode.upper() + '$_{' + lib_mode + '}$\nAVIRIS', fontsize=18)

        # set ticks
        ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

        if col != 0:
            ax.set_yticklabels([])

        if row != 3:
            ax.set_yticklabels([''] + ax.get_yticklabels()[1:])
            ax.set_xticklabels([])

        # plot fractional cover values
        if row != 2:
            x = df_x[col_map_wp[col]]/100
        else:
            x = df_x[col_map[col]]

        y = df_y[col_map[col]]

        print(len(x), len(y))
        m, b = np.polyfit(x, y, 1)
        one_line = np.linspace(0, 1, 101)

        # plot 1 to 1 line
        ax.plot(one_line, one_line, color='red')
        ax.plot(one_line, m * one_line + b, color='black')
        #ax.errorbar(x, y, yerr=y_u, xerr=x_u, fmt='', linestyle='None', capsize=5)
        ax.scatter(x, y, marker='^', edgecolor='black', color='orange', label='AVIRIS$_{ng}$', zorder=10)

        # for i, label in enumerate(df_x['plot'].values):
        #     ax.text(x[i], y[i], label, fontsize=12, ha='center', va='bottom')

        # Add error metrics
        rmse = mean_squared_error(x, y, squared=False)
        mae = mean_absolute_error(x, y)
        r2 = r2_calculations(x, y)

        txtstr = '\n'.join((
            r'MAE(RMSE): %.2f(%.2f)' % (mae, rmse),
            r'R$^2$: %.2f' % (r2,),
            r'n = ' + str(len(x)),
            # r'CPU: %.2f' % (performance,),
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
        ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)


# fig.supylabel('AVIRIS$_{NG}$ Fractions', fontsize=self.axis_label_fontsize)
plt.savefig(r'G:\My Drive\terraspec\shift\figures\wonderpole_sh.png', format="png", dpi=400,
            bbox_inches="tight")
plt.clf()
plt.close()