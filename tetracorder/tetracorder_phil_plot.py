def tetracorder_libraries(self):
    # TODO: Get these from....direct input?  Configuration file?
    MINERAL_FRACTION_FILES = [ \
        'calcite.group2.txt',
        'chlorite.group2.txt',
        'dolomite.group2.txt',
        'goethite-all-for-reference.group1.txt',
        'gypsum.group2.txt',
        'hematite-all-for-reference.group1.txt',
        'illite+muscovite.group2.txt',
        'kaolinite.group2.txt',
        'montmorillonite.group2.txt',
        'vermiculite.group2.txt',
    ]

    SPECTRAL_REFERENCE_LIBRARY = { \
        'splib06': os.path.join('utils', 'tetracorder', 's06emitd_envi'),
        'sprlb06': os.path.join('utils', 'tetracorder', 'r06emitd_envi'),
    }

    decoded_expert = tetracorder.decode_expert_system(os.path.join('utils', 'tetracorder', 'cmd.lib.setup.t5.27c1'),
                                                      log_file=None, log_level='INFO')
    mff = [os.path.join('utils', 'tetracorder', 'minerals', x) for x in MINERAL_FRACTION_FILES]
    mineral_fractions = tetracorder.read_mineral_fractions(mff)
    unique_file_names, fractions, scaling, library_names, records, reference_band_depths = unique_file_fractions(
        mineral_fractions, decoded_expert)

    df_matrix = pd.read_csv(os.path.join('utils', 'tetracorder', 'mineral_grouping_matrix_20230503.csv'))
    spectral_reference_library_files = SPECTRAL_REFERENCE_LIBRARY
    libraries = {}

    ind = 0

    # set up the figure
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[:, 2])

    transect_data = pd.read_csv(os.path.join(self.slpit_output, 'all-transect-emit.csv'))

    for plot in sorted(list(transect_data.plot_name.unique()), reverse=True):
        slpit_ems_records = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "") +
                                              '*emit_ems_augmented_min'))

        slpit_ems_spectra = glob(os.path.join(self.aug_directory, '*' + plot.replace(" ", "") +
                                              '*emit_ems_augmented'))
        #
        # slpit_transect_abundance = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "") +
        #                  '*transect_augmented_min'))
        #
        # emit_spectral_abundance = glob(os.path.join(self.sa_outputs, '*' + plot.replace(" ", "").replace('Spectral', 'SPEC') +
        #                  '*pixels_augmented_min'))

        g1_em_records = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 1]), 'Record'].iloc[0]
        g2_em_records = df_matrix.loc[df_matrix['Index'] == int(envi_to_array(slpit_ems_records[0])[0, 0, 3]), 'Record'].iloc[0]
        plot_spectra = envi_to_array(slpit_ems_spectra[0])[0, 0, :]

        for key, item in spectral_reference_library_files.items():
            ind += 1
            library = envi.open(envi_header(item), item)
            library_reflectance = library.spectra.copy()
            library_records = [int(q) for q in library.metadata['record']]

            hdr = envi.read_envi_header(envi_header(item))
            wavelengths = np.array([float(q) for q in hdr['wavelength']])

            if ';;;' in key:
                key = key.replace(';;;', ',')
                logging.debug(f'found comma replacement, now: {key}')

            libraries[key] = {
                'reflectance': library_reflectance,
                'library_records': library_records,
                'wavelengths': wavelengths}

            band_depths = np.zeros(fractions.shape[0])

            for _f, (frac, filename, library_name, record) in enumerate(
                    zip(fractions, unique_file_names, library_names.tolist(), records.tolist())):
                if library_name == key and record in [g1_em_records, g2_em_records]:
                    for cont_feat in decoded_expert[filename.split('.depth.gz')[0].replace('/', '\\')]['features']:
                        if np.all(np.array(cont_feat['continuum']) < 0.8):
                            cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(record), :],
                                                cont_feat['continuum'])
                            split_cont, wvls = cont_rem(wavelengths, plot_spectra, cont_feat['continuum'])
                            ax1.plot(wl, cont,
                                     label=f'{frac[0]} ||| {os.path.basename(unique_file_names[_f]).split(".depth.gz")[0]}')
                            ax1.plot(wl, split_cont,
                                     label=f'SLPIT {frac[0]} ||| {os.path.basename(unique_file_names[_f]).split(".depth.gz")[0]}')
                        if np.all(np.array(cont_feat['continuum']) > 0.7):
                            cont, wl = cont_rem(wavelengths, library_reflectance[library_records.index(record), :],
                                                cont_feat['continuum'])
                            ax2.plot(wl, cont, label=frac[0])

                    ax3.plot(wavelengths, library_reflectance[library_records.index(record), :], label=frac[0])
                    ax3.plot(wavelengths, plot_spectra, label='SLPIT')
                    ax3.legend()

        handles, labels = ax1.get_legend_handles_labels()
        order = np.argsort(labels)
        handles = np.array(handles)[order].tolist()
        labels = np.array(labels)[order].tolist()
        ax4.legend(handles, labels)
        ax4.axis('off')

        ax1.set_title('Continuum Removed R1')
        ax2.set_title('Continuum Removed R2')
        ax3.set_title('Full Spectrum')
        ax4.set_title('XRD Quantity || Mineral Name')

        plt.savefig(r'G:\My Drive\test\test_tc.png')
        time.sleep(1000)