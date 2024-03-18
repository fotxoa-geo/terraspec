from envi import tetracorder_rgb

min_bands = r'G:\My Drive\terraspec\tetracorder\output\spectral_abundance\emit_20230831t52735_abun_mineral'

output_directory = r'C:\Users\fotxo\OneDrive\Desktop\dataextract\\'

tetracorder_rgb(envi_file=min_bands, output_directory=output_directory)