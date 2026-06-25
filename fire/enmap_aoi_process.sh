#!/bin/bash
START_TIME=$SECONDS

tif_file=$1
global_unmixing_library=$2
out_base=$3
aoi=$4

filebase_name=$(basename "$tif_file" ".TIF")

# This seperates the nc file by delimter
IFS='_' read -r -a SPLIT_ARRAY <<< "$filebase_name"

aoi_basename="${aoi##*/}" && aoi_basename="${aoi_basename%.*}"
fid="${SPLIT_ARRAY[5]}"
product="${SPLIT_ARRAY[4]}"

echo "Filebase name: ${filebase_name}"
echo "AOI basename: ${aoi_basename}"
echo "FID: ${fid}"
echo "Product: ${product}"

# setup directory for output nc file data; this is the root tree
fid_directory=${out_base}/${fid}
echo "FID Directory: ${fid_directory}"
mkdir ${fid_directory} -p


### Run geoprocessing to create envi outputs image from nc file
out_directory=${fid_directory}/${product}/
mkdir ${out_directory} -p

python ./fire/enmap_to_envi.py -out_dir ${out_directory} -rfl ${tif_file}

## extract windows from images
ext_out_parent_dir=$(dirname "${out_base}")
ext_out_directory=${ext_out_parent_dir}/aoi
mkdir -p ${ext_out_directory}

# extract images
ext_out_dir_img=${ext_out_directory}/${aoi_basename}
mkdir -p ${ext_out_dir_img}
echo "Extraction outdir: ${ext_out_dir_img}"
img_name=${out_directory}/${filebase_name}
python ./fire/area_extract_enmap.py -rfl_img ${img_name} -aoi ${aoi} -out ${ext_out_dir_img}

echo "Geoprocess complete!"

#### Run unmixing on image with global library using Ochoa et al. (2025) EMC^2
aoi_img=${ext_out_dir_img}/EXT/${aoi_basename}_${filebase_name}_EXT

if [ -f ${aoi_img} ]; then
    echo "$aoi_img File exists."

    # unmixing code
    unmix_out_directory=${ext_out_dir_img}/emc2/
    mkdir -p ${unmix_out_directory}
    julia -p 15 ../SpectralUnmixing/unmix.jl ${aoi_img} ${global_unmixing_library} level_1 ${unmix_out_directory}/${filebase_name} --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 8

    # push data to drive
    #/store/shared/rclone/bin/rclone copy ${ext_out_dir_img} cdrive:terraspec_output/slpit/gis/emit-data/products/${fid} -P --exclude "*.nc"

else
    echo "$aoi_img File does not exist."
fi

DURATION=$(( $SECONDS - $START_TIME ))
echo "processing time: $DURATION seconds."

