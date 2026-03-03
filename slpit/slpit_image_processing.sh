#!/bin/shG
START_TIME=$SECONDS
echo "################# Running unmixing  ################"
echo " "
# These are the variables
rfl_img=$1
unmixing_library_local=$2
out_base=$3
em_rfl_file=$4
sensor_rfl_ext_file=$5
sensor_rfl_unc=$6
unmixing_library_global=./terraspec_output/simulation/output/endmember_libraries/convex_hull__n_dims_4_sensor_emit_geofilter_True_unmix_library.csv
kalahari_unmixing_library=./terraspec_output/simulation/output/production/meyer-okin.csv

filebase_name=$(basename "$rfl_img")
ext_filebase_name=$(basename "$sensor_rfl_ext_file")
NORMALIZED_PATH_OUTBASE=$(echo "${out_base}" | tr '\\' '/')
NORMALIZED_GLOBAL_LIB_PATH=$(echo "${unmixing_library_global}" | tr '\\' '/')
NORMALIZED_LOCAL_LIB_PATH=$(echo "${unmixing_library_local}" | tr '\\' '/')
NORMALIZED_Kalahari_LIB_PATH=$(echo "${kalahari_unmixing_library}" | tr '\\' '/')

echo "basename: ${rfl_img}"
echo "You are currently in: $PWD"

# Seperate basename into components
IFS='_' read -r -a RFL_ARRAY <<< "$filebase_name"

# run unmixing call on data
emc_out_directory=${out_base}/emc2/
if [ -d "$emc_out_directory" ]; then
    echo "Directory '$emc_out_directory' exists. Removing..."
    rm -rf "$emc_out_directory"
fi

echo "Created emc2 dir: ${emc_out_directory}"
mkdir -p ${emc_out_directory}

# these are global unmix calls
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 "${emc_out_directory}/global_${filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 20 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 "${emc_out_directory}/global_${ext_filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 20 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 "${emc_out_directory}/global_${filebase_name}_normalization_none_" --mode sma --normalization none --num_endmember 20 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 "${emc_out_directory}/global_${ext_filebase_name}_normalization_none_" --mode sma --normalization none --num_endmember 20 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}

# these are local unmix calls
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_LOCAL_LIB_PATH} level_1 "${emc_out_directory}/local_${filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 20 --n_mc 25 --spectral_starting_col 12
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_LOCAL_LIB_PATH} level_1 "${emc_out_directory}/local_${ext_filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 20 --n_mc 25 --spectral_starting_col 12 --reflectance_uncertainty_file ${sensor_rfl_unc}
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_LOCAL_LIB_PATH} level_1 "${emc_out_directory}/local_${filebase_name}_normalization_none_" --mode sma --normalization none --num_endmember 20 --n_mc 25 --spectral_starting_col 12
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_LOCAL_LIB_PATH} level_1 "${emc_out_directory}/local_${ext_filebase_name}_normalization_none_" --mode sma --normalization none --num_endmember 20 --n_mc 25 --spectral_starting_col 12 --reflectance_uncertainty_file ${sensor_rfl_unc}

# these are kalahari lib
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_Kalahari_LIB_PATH} level_1 "${emc_out_directory}/kalahari_${filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 20 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_Kalahari_LIB_PATH} level_1 "${emc_out_directory}/kalahari_${ext_filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 20 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_Kalahari_LIB_PATH} level_1 "${emc_out_directory}/kalahari_${filebase_name}_normalization_none_" --mode sma --normalization none --num_endmember 20 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_Kalahari_LIB_PATH} level_1 "${emc_out_directory}/kalahari_${ext_filebase_name}_normalization_none_" --mode sma --normalization none --num_endmember 20 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}

mesma_out_directory=${out_base}/mesma/

if [ -d "$mesma_out_directory" ]; then
    echo "Directory '$mesma_out_directory' exists. Removing..."
    rm -rf "$mesma_out_directory"
fi

mkdir -p ${mesma_out_directory}
echo "Created mesma dir: ${mesma_out_directory}"

# these are global unmix calls
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 ${mesma_out_directory}/global_${filebase_name}_normalization_brightness_ --mode mesma --normalization brightness --max_combinations 100 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 ${mesma_out_directory}/global_${ext_filebase_name}_normalization_brightness_ --mode mesma --normalization brightness --max_combinations 100 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 ${mesma_out_directory}/global_${filebase_name}_normalization_none_ --mode mesma --normalization none --max_combinations 100 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 ${mesma_out_directory}/global_${ext_filebase_name}_normalization_none_ --mode mesma --normalization none --max_combinations 100 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}

# these are local unmix calls
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_LOCAL_LIB_PATH} level_1 ${mesma_out_directory}/local_${filebase_name}_normalization_brightness_ --mode mesma --normalization brightness --max_combinations 100 --n_mc 25 --spectral_starting_col 12
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_LOCAL_LIB_PATH} level_1 ${mesma_out_directory}/local_${ext_filebase_name}_normalization_brightness_ --mode mesma --normalization brightness --max_combinations 100 --n_mc 25 --spectral_starting_col 12 --reflectance_uncertainty_file ${sensor_rfl_unc}
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_LOCAL_LIB_PATH} level_1 ${mesma_out_directory}/local_${filebase_name}_normalization_none_ --mode mesma --normalization none --max_combinations 100 --n_mc 25 --spectral_starting_col 12
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_LOCAL_LIB_PATH} level_1 ${mesma_out_directory}/local_${ext_filebase_name}_normalization_none_ --mode mesma --normalization none --max_combinations 100 --n_mc 25 --spectral_starting_col 12 --reflectance_uncertainty_file ${sensor_rfl_unc}

# these are kalahari unmix calls
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_Kalahari_LIB_PATH} level_1 ${mesma_out_directory}/kalahari_${filebase_name}_normalization_brightness_ --mode mesma --normalization brightness --max_combinations 100 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_Kalahari_LIB_PATH} level_1 ${mesma_out_directory}/kalahari_${ext_filebase_name}_normalization_brightness_ --mode mesma --normalization brightness --max_combinations 100 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}
julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_Kalahari_LIB_PATH} level_1 ${mesma_out_directory}/kalahari_${filebase_name}_normalization_none_ --mode mesma --normalization none --max_combinations 100 --n_mc 25 --spectral_starting_col 11
julia -p 1 ../SpectralUnmixing/unmix.jl ${sensor_rfl_ext_file} ${NORMALIZED_Kalahari_LIB_PATH} level_1 ${mesma_out_directory}/kalahari_${ext_filebase_name}_normalization_none_ --mode mesma --normalization none --max_combinations 100 --n_mc 25 --spectral_starting_col 11 --reflectance_uncertainty_file ${sensor_rfl_unc}

echo " "

# run tetracorder
tetracorder_out_directory=${out_base}/tetracorder/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# augment rfl data
python ./utils/augment_file.py ${rfl_img} ${tetracorder_out_directory} --augment # slpit data
em_filebase_name=$(basename "$em_rfl_file")
python ./utils/augment_file.py ${em_rfl_file} ${tetracorder_out_directory} --augment # em data

./tetracorder/tetracorder.sh "${tetracorder_out_directory}/${filebase_name}_augmented" ${tetracorder_out_directory}
./tetracorder/tetracorder.sh "${tetracorder_out_directory}/${em_filebase_name}_augmented" ${tetracorder_out_directory}

# deaugment data in tetracorder output directory
python ./utils/augment_file.py ${rfl_img} ${tetracorder_out_directory} --deaugment
python ./utils/augment_file.py ${em_rfl_file} ${tetracorder_out_directory} --deaugment

# push data to drive
out_base_name=$(basename "${out_base}")
/store/shared/rclone/bin/rclone sync ${out_base} cdrive:terraspec_output/slpit/output/spectral_transects/${out_base_name}/ -P
