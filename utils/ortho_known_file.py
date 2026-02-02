import argparse
import os
from spectral.io import envi

def main():
    #define arguments
    parser = argparse.ArgumentParser(description='Run orthorectification on envi scence')
    parser.add_argument('-non_ortho_img', '--non_orthorectified_image', type=str, help='Envi img non-ortho')
    parser.add_argument('-ortho_img', '--orthorectified_image', type=str, help='Envi img to rectify to')
    args = parser.parse_args()

    # load non-ortho 
    non_ortho_img = envi.open(f'{args.non_orthorectified_image}.hdr')
    meta_non_ortho = non_ortho_img.metadata

    # load ortho img
    ortho_img = envi.open(f'{args.orthorectified_image}.hdr')
    meta_ortho_img = ortho_img.metadata

    # update spatial info
    meta_non_ortho['map info'] = meta_ortho_img['map info']
    meta_non_ortho['coordinate system string'] = meta_ortho_img['coordinate system string']
    
    envi.write_envi_header(f'{args.non_orthorectified_image}.hdr', meta_non_ortho)


if __name__ == '__main__':
    main()
