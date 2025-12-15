#!/bin/sh
START_TIME=$SECONDS

nc_file=$1
global_unmixing_library=$2
out_base=$3
NORMALIZED_PATH_OUTBASE=$(echo "${out_base}" | tr '\\' '/')
NORMALIZED_GLOBAL_LIB_PATH=$(echo "${global_unmixing_library}" | tr '\\' '/')

filebase_name=$(basename --suffix=".nc" "$nc_file")

echo ${filebase_name}

# This seperates the nc file by delimter
IFS='_' read -r -a SPLIT_ARRAY <<< "$filebase_name"

product="${SPLIT_ARRAY[1]}"
fid="${SPLIT_ARRAY[4]}"
data_type="${SPLIT_ARRAY[2]}"

echo ${product}
echo ${fid}

# setup directory for output nc file data; this is the root tree
nc_fid_directory=${NORMALIZED_PATH_OUTBASE}/${fid}
echo ${nc_fid_directory}
mkdir ${nc_fid_directory} -p

# Run geoprocessing to create envi outputs image from nc file
nc_out_directory=${nc_fid_directory}/${product}/
mkdir ${nc_out_directory} -p

if [ "${product}" = "L1B" ]; then
    python ./emit-utils/emit_utils/reformat.py ${nc_file} ${nc_out_directory} "--orthorectify"
else
    python ./emit-utils/emit_utils/reformat.py ${nc_file} ${nc_out_directory}
fi

echo "Geoprocess complete!"

# Run unmixing on image with global library using Ochoa et al. (2025) EMC^2; this will not use uncertainty!
# Uncertainty runs will only be used with extracted SLPIT runs. Run Tetracorder as well.
if [ "${data_type}" = "RFL" ]; then

    rfl_img=${nc_out_directory}/${filebase_name}_reflectance
    
    # unmixing code
    unmix_out_directory=${nc_fid_directory}/emc2/
    mkdir -p ${unmix_out_directory}
    julia -p 40 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 ${unmix_out_directory}/${filebase_name} --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 8

    # run Tetracorder
    tetracorder_out_directory=${nc_fid_directory}/tetracorder/
    mkdir -p ${teracorder_out_directory}
    ./tetracorder/tetracorder.sh ${rfl_img} ${tetracorder_out_directory}

else
    echo "Reflectance data not detected. Skipping spectral processes!!"
fi

DURATION=$(( $SECONDS - $START_TIME ))
echo "processing time: $DURATION seconds."
