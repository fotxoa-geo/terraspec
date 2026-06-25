import argparse
import os
import time
import numpy as np
from spectral.io import envi

envi_typemap = {
    'uint8': 1,
    'int16': 2,
    'int32': 3,
    'float32': 4,
    'float64': 5,
    'complex64': 6,
    'complex128': 9,
    'uint16': 12,
    'uint32': 13,
    'int64': 14,
    'uint64': 15
}

# this requires the ORT rfl scene - mostlly used to ortho results

def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='Orthorectify AVIRIS scene')
    parser.add_argument('-out_dir', '--out_directory', type=str, help='Specify out directory')
    parser.add_argument('-rfl', '--rfl_file', type=str, help='non-orthorectified envi file')
    parser.add_argument('-res_img', '--result_image', type=str, help='Specify result image')
    parser.add_argument('-glt', '--glt_file', type=str, help='glt envi file')
    args = parser.parse_args()

    # load glt metadata and array
    glt_img = envi.open(f'{args.glt_file}.hdr')
    glt_meta = glt_img.metadata
    glt_nodata = float(glt_meta['data ignore value'])
    glt_array = glt_img.load().astype(np.float32)

    # load rfl data
    res_img = envi.open(f'{args.result_image}.hdr')
    res_img_array = res_img.load().astype(np.float32)
    print(res_img_array.shape)
    # load rfl ort data
    rfl_img_ort = envi.open(f'{args.rfl_file}_ORT.hdr')
    rfl_meta_ort = rfl_img_ort.metadata

    data_grid = np.ones((int(glt_meta['lines']), int(glt_meta['samples']), 1), dtype=np.float32) * -9999

    for _r, r in enumerate(glt_array):
        for _c, c in enumerate(r):
            index = glt_array[_r, _c]

            if glt_nodata in index:
                pass
            else:
                index_row = index[1]
                index_col = index[0]

                try:
                    r_idx = np.absolute(int(index_row)) - 1
                    c_idx = np.absolute(int(index_col)) - 1

                    # Check if it falls completely within the valid image boundaries
                    data_grid[_r, _c, 0] = res_img_array[r_idx, c_idx]
                except:
                    print(index_row, index_col)
                    time.sleep(10000)

    # # get spatial information
    metadata = {'lines': data_grid.shape[0],
                'samples': data_grid.shape[1],
                'bands': 1,
                'interleave': 'bil',
                'header offset': 0,
                'file type': 'ENVI Standard',
                'data type': envi_typemap[str(data_grid.dtype)],
                'byte order': 0,
                'map info': glt_meta['map info'],
                'coordinate system string': rfl_meta_ort['coordinate system string'],
                'data ignore value': -9999}

    output_hdr = os.path.join(args.out_directory, f'{os.path.basename(args.result_image)}_ORT.hdr')
    envi.save_image(output_hdr, data_grid,
                        metadata=metadata, force=True,
                        interleave='bil', ext='')

    print(f"\t successfully saved: {output_hdr}")

if __name__ == '__main__':
    main()