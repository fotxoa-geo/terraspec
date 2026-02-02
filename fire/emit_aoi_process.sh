#!/bin/bash
START_TIME=$SECONDS

nc_file=$1
global_unmixing_library=$2
out_base=$3
aoi=$4

filebase_name=$(basename --suffix=".nc" "$nc_file")

# This seperates the nc file by delimter
IFS='_' read -r -a SPLIT_ARRAY <<< "$filebase_name"
SPLICE='.' read -r -a EXTENSION_ARRAY <<< "$filebase_name"

file_basename="${EXTENSION_ARRAY[0]}"
aoi_basename="${aoi##*/}" && aoi_basename="${aoi_basename%.*}"
product="${SPLIT_ARRAY[1]}"
fid="${SPLIT_ARRAY[4]}"
data_type="${SPLIT_ARRAY[2]}"

echo "Filebase name: ${filebase_name}"
echo "AOI basename: ${aoi_basename}"
echo "Prdocut: ${product}"
echo "FID: ${fid}"
echo "Data type: ${data_type}"

# setup directory for output nc file data; this is the root tree
nc_fid_directory=${out_base}/${fid}
echo "FID Directory: ${nc_fid_directory}"
mkdir ${nc_fid_directory} -p

# Run geoprocessing to create envi outputs image from nc file
nc_out_directory=${nc_fid_directory}/${product}/
mkdir ${nc_out_directory} -p

python ./emit-utils/emit_utils/reformat.py ${nc_file} ${nc_out_directory} --overwrite --orthorectify

# extract windows from images
ext_out_parent_dir=$(dirname "${out_base}")
ext_out_directory=${ext_out_parent_dir}/aoi
mkdir -p ${ext_out_directory}

# dictionary for data types
declare -A file_endings
file_endings=([RFL]="reflectance" [MASK]="mask" [RFLUNCERT]="reflectance_uncertainty")

# extract images
ext_out_dir_img=${ext_out_directory}/${aoi_basename}
mkdir -p ${ext_out_dir_img}
echo "Extraction outdir: ${ext_out_dir_img}"
img_name=${nc_out_directory}/${filebase_name}_${file_endings[$data_type]}
python ./fire/area_extract.py -rfl_img ${img_name} -nc_file ${nc_file}  -aoi ${aoi} -pad 1 -out ${ext_out_dir_img}
echo "Geoprocess complete!"

## Run unmixing on image with global library using Ochoa et al. (2025) EMC^2; this will not use uncertainty!
## Uncertainty runs will only be used with extracted SLPIT runs. Run Tetracorder as well.
if [ "${data_type}" = "RFL" ]; then
    aoi_img=${ext_out_dir_img}/EXT/${aoi_basename}_${data_type}_${fid}_EXT
    
    if [ -f ${aoi_img} ]; then
        echo "$aoi_img File exists."
    else
        echo "$aoi_img File does not exist."
    fi

    # unmixing code
    unmix_out_directory=${ext_out_dir_img}/emc2/
    mkdir -p ${unmix_out_directory}
    julia -p 15 ../SpectralUnmixing/unmix.jl ${aoi_img} ${global_unmixing_library} level_1 ${unmix_out_directory}/${filebase_name} --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 8

    # run Tetracorder
    tetracorder_out_directory=${ext_out_dir_img}/tetracorder/${fid}/
    if [ -d "$tetracorder_out_directory" ]; then
        echo "Directory '$tetracorder_out_directory' exists. Removing..."
        rm -rf "$tetracorder_out_directory"
    fi

    mkdir -p ${tetracorder_out_directory}
    echo "Created directory: $tetracorder_out_directory"
    tetracorder/tetracorder.sh ${aoi_img} ${tetracorder_out_directory}

    #ortho-rectify mineral outputs
    python utils/ortho_known_file.py -non_ortho_img ${tetracorder_out_directory}/${aoi_basename}_${data_type}_${fid}_EXT_min -ortho_img ${aoi_img}

    # push data to drive
    #/store/shared/rclone/bin/rclone copy ${ext_out_dir_img} cdrive:terraspec_output/slpit/gis/emit-data/products/${fid} -P --exclude "*.nc"

else
    echo "Reflectance data not detected. Skipping spectral processes!!"
fi


DURATION=$(( $SECONDS - $START_TIME ))
echo "processing time: $DURATION seconds."
