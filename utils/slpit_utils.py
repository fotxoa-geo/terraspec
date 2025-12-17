import time
import os
import pandas as pd
from utils import asdreader, sedreader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class slpit:
    "ceratin utilities for split processing of asd data and arranging data"
    def __init__(self):
        print("")

    @classmethod
    def df_white_ref_table(cls, record):
        df_white_ref = pd.json_normalize(record['white_ref'])
        df_white_ref = df_white_ref.iloc[:, 14:]

        return df_white_ref

    @classmethod
    def df_em_table(cls, record):

        if not record["solo_slpit_toggle"]:
            df_transect_em = pd.json_normalize(record['emit_transect_endmembers'])
            df_transect_em = df_transect_em.iloc[:, 14:]

        else:
            df_transect_solo = pd.json_normalize(record['solo_slpit'])
            df_transect_solo = df_transect_solo.iloc[:, 14:]
            df_transect_solo.sort_values('start_em_range')

            df_em_rows = []
            for _row, row in df_transect_solo.iterrows():
                if not row['bad_em']:
                    bad_em = []
                else:
                    bad_em = pd.json_normalize(row['bad_em'])
                    bad_em = bad_em.iloc[:, 14:]
                    bad_em = bad_em['erroneous_endmembers'].tolist()

                for i in range(row['start_em_range'], row['end_em_range'] + 1):
                    if i in bad_em:
                        em_condition = 'bad'
                    else:
                        em_condition = ''

                    if row['em_classification'] == 'Soil':
                        species = ''
                    else:
                        species = 'UNK-'
                    df_em_rows.append([i, row["line_num"], row['em_classification'], species, '', '', '', em_condition])

                df_transect_em = pd.DataFrame(df_em_rows)
                df_transect_em.columns = ['asd_file_num', "transect_line_num", 'endmembers', "species", 'photo_toggle',
                                          'em_photo', 'notes', 'em_condition']

        return df_transect_em

    @classmethod
    def white_ref_correction(cls, spectra, time_s, white_reference_spectra_t1, white_reference_spectra_t2, time_1,
                             time_2):

        m = (white_reference_spectra_t2 - white_reference_spectra_t1) / int((time_2 - time_1).total_seconds())
        r = spectra / (white_reference_spectra_t1 + m * (int((time_s - time_1).total_seconds())))

        return r

    @classmethod
    def plot_asd_file(cls, asd_file, out_directory):
        # Load asd data
        data = asdreader.reader(asd_file)
        asd_wl = data.wavelengths

        try:
            outfname = os.path.join(out_directory, f'{os.path.basename(asd_file)}.png')
            if os.path.isfile(outfname):
                pass

            else:
                asd_refl = data.reflectance

                plt.plot(asd_wl, asd_refl, label=os.path.basename(asd_file))
                plt.legend()
                plt.ylabel("Reflectance (%)")
                plt.xlabel("Wavelenghts (nm)")
                plt.ylim([0, 1 * 1.05])
                plt.xlim([325, 2525])

                ax = plt.gca()

                # Major ticks every 100
                ax.xaxis.set_major_locator(MultipleLocator(500))
                # Minor ticks every 50
                ax.xaxis.set_minor_locator(MultipleLocator(100))

                # Major ticks every 0.10 - yaxis
                ax.yaxis.set_major_locator(MultipleLocator(0.10))
                # Minor ticks every 0.05 - yaxis
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))

                plt.savefig(outfname, bbox_inches='tight')
                plt.clf()
                plt.close()

        except:
            raise
            print(asd_file, out_directory)

    @classmethod
    def plot_sed_file(cls, sed_file, out_directory):
        # load sed data
        data = sedreader.reader(sed_file)
        sed_wvl = data.wavelengths

        try:
            outfname = os.path.join(out_directory, os.path.basename(sed_file) + '.png')
            if os.path.isfile(outfname):
                pass

            else:
                sed_refl = data.reflectance

                plt.plot(sed_wvl, sed_refl, label=os.path.basename(sed_file))
                plt.legend()
                plt.ylabel("Reflectance (%)")
                plt.xlabel("Wavelenghts (nm)")
                plt.ylim([0, 110])

                plt.savefig(outfname, bbox_inches='tight')
                plt.clf()
                plt.close()

        except:
            print(sed_file, out_directory)
