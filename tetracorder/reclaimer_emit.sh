#!/bin/sh
echo "################# Running RECLAIMER  ################"
echo " "

# Script variables
out_directory=$1
sensor=$2
rfl_img=$3
unmixing_library=$4
unmixing_library_envi=$5
rock_lib=$6
rock_lib_envi=$7

unmixing_filebase_name=$(basename "$unmixing_library" .csv)
filebase_name=$(basename "$rfl_img")
local_lib_basename=$(basename "$local_library" .csv)
rock_lib_basename=$(basename "$rock_lib" .csv)

echo "RFL img: $rfl_img"
echo "Out base Directory: $out_directory"
echo "Basename_RFL: $filebase_name"
echo "Sensor: $sensor"
CURRENT_DIR=$(pwd)
echo "You are currently in: $CURRENT_DIR"

emc_out_directory=${out_directory}/emc2/

if [ -d "$emc_out_directory" ]; then
    echo "Directory '$emc_out_directory' exists. Removing..."
        rm -rf "$emc_out_directory"
fi

# unmixing call
echo "Created emc2 dir: ${emc_out_directory}"
mkdir -p ${emc_out_directory}

if [ -f ${rfl_img} ]; then
    echo "$rfl_img File exists."
    julia -p 20 ../SpectralUnmixing/unmix.jl ${rfl_img} ${unmixing_library} level_1 "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 11 --write_complete_fractions
    julia -p 20 ../SpectralUnmixing/unmix.jl ${rfl_img} ${rock_lib} level_1 "${emc_out_directory}/${rock_lib_basename}_${filebase_name}" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 11 --write_complete_fractions
else
    echo "$rfl_img does not exist."
fi

echo '======================Tetracorder on Rfl img (uncorrected)=============================='

# tetracorder uncorrected
tetracorder_out_directory=${out_directory}/tc_unc/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run tetracorder
./tetracorder/tetracorder.sh "$rfl_img" ${tetracorder_out_directory} ${sensor} --delete_tc_output 
    
echo '====================Tetracorder on Extracted veg (global)==============================='

# run Tetracorder on extracted signal of vegetation
tetracorder_out_directory=${out_directory}/tc_global/

if [ -d "$tetracorder_out_directory" ]; then
        echo "Directory '$tetracorder_out_directory' exists. Removing..."
        rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run Vegetation Extractor.py
python ./tetracorder/vegetation_extractor.py -out_dir ${tetracorder_out_directory} -sns ${sensor} -veg_fracs "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_complete_fractions" -rfl ${rfl_img} -unmix_lib_csv ${unmixing_library} -unmix_lib_envi ${unmixing_library_envi} -three_comp_frac "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_fractional_cover" --tetracorder

extracted_vegetation_rfl_img=${tetracorder_out_directory}/ext_veg_${filebase_name}_${unmixing_filebase_name}_tc

if [ -f ${extracted_vegetation_rfl_img} ]; then
    echo "$extracted_vegetation_rfl_img File exists."     
    vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")
    
    # augment rfl data and run tetracorder
    ./tetracorder/tetracorder.sh ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} ${sensor} --delete_tc_output
    ./tetracorder/tetracorder.sh ${tetracorder_out_directory}/recon_rho_${filebase_name}_${unmixing_filebase_name} ${tetracorder_out_directory} ${sensor} --delete_tc_output

else
    echo "$extracted_vegetation_rfl_img File does not exist."
fi

rm ${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_complete_fractions
rm ${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_complete_fractions.hdr

python ./tetracorder/reclaimer.py -out_dir ${tetracorder_out_directory} -tc_out ${tetracorder_out_directory}/ext_veg_${filebase_name:0:-2}_min.hdr  -rfl ${rfl_img}.hdr -um_out ${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_fractional_cover.hdr -rho_gv ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_lib_signal.hdr -rho_npv ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_lib_signal.hdr -g_num 1 
python ./tetracorder/reclaimer.py -out_dir ${tetracorder_out_directory} -tc_out ${tetracorder_out_directory}/ext_veg_${filebase_name:0:-2}_min.hdr  -rfl ${rfl_img}.hdr -um_out ${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_fractional_cover.hdr -rho_gv ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_lib_signal.hdr -rho_npv ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_lib_signal.hdr -g_num 2

# remove non critical data 
rm ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_lib_signal
rm ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_lib_signal.hdr

rm ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_lib_signal
rm ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_lib_signal.hdr

rm ${tetracorder_out_directory}/extracted_${filebase_name}_soil_global_lib_signal
rm ${tetracorder_out_directory}/extracted_${filebase_name}_soil_global_lib_signal.hdr

echo '====================Tetracorder on Extracted veg (rock)==============================='
# run Tetracorder on extracted signal of vegetation
tetracorder_out_directory=${out_directory}/tc_rock/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi
mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run Vegetation Extractor.py
python ./tetracorder/vegetation_extractor.py -out_dir ${tetracorder_out_directory} -sns ${sensor} -veg_fracs "${emc_out_directory}/${rock_lib_basename}_${filebase_name}_complete_fractions" -rfl ${rfl_img} -unmix_lib_csv ${rock_lib} -unmix_lib_envi ${rock_lib_envi} -three_comp_frac "${emc_out_directory}/${rock_lib_basename}_${filebase_name}_fractional_cover" --tetracorder

extracted_vegetation_rfl_img=${tetracorder_out_directory}/ext_veg_${filebase_name}_${rock_lib_basename}_tc

if [ -f ${extracted_vegetation_rfl_img} ]; then
    echo "$extracted_vegetation_rfl_img File exists."
       
    vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")
    
    # augment rfl data and run tetracorder
    ./tetracorder/tetracorder.sh ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} ${sensor} --delete_tc_output
    ./tetracorder/tetracorder.sh ${tetracorder_out_directory}/recon_rho_${filebase_name}_${rock_lib_basename} ${tetracorder_out_directory} ${sensor} --delete_tc_output

else
    echo "$extracted_vegetation_rfl_img File does not exist."
fi

rm ${emc_out_directory}/${rock_lib_basename}_${filebase_name}_complete_fractions
rm ${emc_out_directory}/${rock_lib_basename}_${filebase_name}_complete_fractions.hdr

python ./tetracorder/reclaimer.py -out_dir ${tetracorder_out_directory} -tc_out ${tetracorder_out_directory}/ext_veg_${filebase_name:0:-2}_min.hdr  -rfl ${rfl_img}.hdr -um_out ${emc_out_directory}/${rock_lib_basename}_${filebase_name}_fractional_cover.hdr -rho_gv ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_rock_signal.hdr -rho_npv ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_rock_signal.hdr -g 1 
python ./tetracorder/reclaimer.py -out_dir ${tetracorder_out_directory} -tc_out ${tetracorder_out_directory}/ext_veg_${filebase_name:0:-2}_min.hdr  -rfl ${rfl_img}.hdr -um_out ${emc_out_directory}/${rock_lib_basename}_${filebase_name}_fractional_cover.hdr -rho_gv ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_rock_signal.hdr -rho_npv ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_rock_signal.hdr -g 2


# remove non critical data 
rm ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_rock_signal
rm ${tetracorder_out_directory}/extracted_${filebase_name}_npv_global_rock_signal.hdr

rm ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_rock_signal
rm ${tetracorder_out_directory}/extracted_${filebase_name}_pv_global_rock_signal.hdr

rm ${tetracorder_out_directory}/extracted_${filebase_name}_soil_global_rock_signal
rm ${tetracorder_out_directory}/extracted_${filebase_name}_soil_global_rock_signal.hdr

