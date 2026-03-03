#!/bin/bash
START_TIME=$SECONDS

nc_file=$1
global_unmixing_library=$2
out_base=$3
sensor=$4

filebase_name=$(basename --suffix=".nc" "$nc_file")

# This seperates the nc file by delimter
IFS='_' read -r -a SPLIT_ARRAY <<< "$filebase_name"
SPLICE='.' read -r -a EXTENSION_ARRAY <<< "$filebase_name"

file_basename="${EXTENSION_ARRAY[0]}"
product="${SPLIT_ARRAY[2]}"
fid="${SPLIT_ARRAY[0]}"
data_type="${SPLIT_ARRAY[5]}"
version="${SPLIT_ARRAY[1]}"

echo ${filebase_name}
echo ${product}
echo ${fid}
echo ${data_type}

# setup directory for output nc file data; this is the root tree
nc_fid_directory=${out_base}/${fid}_${version}
echo "FID Directory: ${nc_fid_directory}"
mkdir ${nc_fid_directory} -p

# Run geoprocessing to create envi outputs image from nc file
nc_out_directory=${nc_fid_directory}/${product}/
mkdir ${nc_out_directory} -p

python ./shift/reformat_aviris_nc.py ${nc_file} ${nc_out_directory} --overwrite
    
# Run unmixing on image with global library using Ochoa et al. (2025) EMC^2; this will not use uncertainty!
# Uncertainty runs will only be used with extracted SLPIT runs. Run Tetracorder as well.
if [ "${data_type}" = "RFL" ]; then
    rfl_img=${nc_out_directory}/${filebase_name}_reflectance
    if [ -f ${rfl_img} ]; then
        echo "$rfl_img File exists."
    else
        echo "$rfl_img File does not exist."
    fi
    
    # unmixing code
    unmix_out_directory=${nc_fid_directory}/emc2/
    mkdir -p ${unmix_out_directory}
    julia -p 20 ../SpectralUnmixing/unmix.jl ${rfl_img} ${global_unmixing_library} level_1 ${unmix_out_directory}/${filebase_name} --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 8
    
    # run Tetracorder
    tetracorder_out_directory=${nc_fid_directory}/tetracorder/
    if [ -d "$tetracorder_out_directory" ]; then
        echo "Directory '$tetracorder_out_directory' exists. Removing..."
        rm -rf "$tetracorder_out_directory"
    fi
    
    mkdir -p ${tetracorder_out_directory}
    echo "Created directory: $tetracorder_out_directory"
    tetracorder/tetracorder.sh ${rfl_img} ${tetracorder_out_directory} $sensor --delete_tc_output
    
    #ortho-rectify mineral outputs
    python shift/ortho_tetracorder_flightlines.py -tc_dir ${tetracorder_out_directory} -nc_file ${nc_file}
    
    # push data to drive
    #/store/shared/rclone/bin/rclone copy ${nc_fid_directory} cdrive:terraspec_output/shift/gis/aviris_ng-data/products/${fid} -P --exclude "*.nc"
else
    echo "Reflectance data not detected. Skipping spectral processes!!"
fi   

# extract windows from images
ext_out_directory=terraspec_output/shift/output/spectral_transects/
mkdir -p ${ext_out_directory}
declare -A file_endings
file_endings=([RFL]="reflectance" [MASK]="mask" [UNC]="uncertainty")
img_name=${nc_out_directory}/${filebase_name}_${file_endings[$data_type]}
python ./shift/window_extract.py -rfl_img ${img_name} --nc_file ${nc_file} -w_size 3 -shp ./gis/shift_transects_centroid.geojson -pad 1 -out ${ext_out_directory} --overwrite -del_nc

echo "Geoprocess complete!"

DURATION=$(( $SECONDS - $START_TIME ))
echo "processing time: $DURATION seconds."
