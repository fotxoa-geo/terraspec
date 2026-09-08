#!/bin/sh
echo "################# Running RECLAIMER  ################"
echo " "

# Script variables
out_directory=$1
sensor=$2
rfl_img=$3
unmixing_library=$4
unmixing_library_envi=$5
local_library=$6
local_library_envi=$7
rock_lib=$8
rock_lib_envi=$9

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

IFS='_' read -ra parts <<< "$filebase_name"
type=${parts[1]}

echo "Type:${parts[1]}"

rfl_img_aug=${out_directory}/${filebase_name}_augmented

if [ -d "$rfl_img_aug" ]; then
    echo "Image '$rfl_img_aug' exists... Skipping augmentation"
else
    python ./utils/augment_file.py ${rfl_img} ${out_directory} --augment --vertical_avg # augment rfl file
    echo "Created augmented file: ${rfl_img_aug}"
 
fi

emc_out_directory=${out_directory}/emc2_${type}/

if [ -d "$emc_out_directory" ]; then
    echo "Directory '$emc_out_directory' exists. Removing..."
        rm -rf "$emc_out_directory"
fi

# unmixing call
echo "Created emc2 dir: ${emc_out_directory}"
mkdir -p ${emc_out_directory}

if [ -f ${rfl_img_aug} ]; then
    echo "$rfl_img_aug File exists."
    julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img_aug} ${unmixing_library} level_1 "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 11 --write_complete_fractions
    julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img_aug} ${local_library} level_1 "${emc_out_directory}/local_${filebase_name}" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 12 --write_complete_fractions
    julia -p 1 ../SpectralUnmixing/unmix.jl ${rfl_img_aug} ${rock_lib} level_1 "${emc_out_directory}/${rock_lib_basename}_${filebase_name}" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 11 --write_complete_fractions
else
    echo "$rfl_img_aug does not exist."
fi


echo '======================Tetracorder on contact probe/local unmixing lib===================='
tetracorder_out_directory=${out_directory}/tc_contact/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Skipping..."
else
    mkdir -p ${tetracorder_out_directory}
    echo "Created tetracorder dir: ${tetracorder_out_directory}"
    
    python ./utils/augment_file.py ${local_library_envi} ${tetracorder_out_directory} --augment --vertical_avg --em_file ${local_library}

    # run tetracorder
    ./tetracorder/tetracorder.sh "$tetracorder_out_directory/${local_lib_basename}_augmented" ${tetracorder_out_directory} ${sensor} --delete_tc_output 
    
    # deaugment data in tetracorder output director
    cp ${local_library_envi} ${tetracorder_out_directory}
    cp ${local_library_envi}.hdr ${tetracorder_out_directory}

    #python ./utils/augment_file.py $local_library_envi ${tetracorder_out_directory} --deaugment
    rm ${tetracorder_out_directory}/${local_lib_basename}
    rm ${tetracorder_out_directory}/${local_lib_basename}.hdr
    rm ${tetracorder_out_directory}/${local_lib_basename}_augmented
    rm ${tetracorder_out_directory}/${local_lib_basename}_augmented.hdr

fi


echo '======================Tetracorder on Rfl img (uncorrected)=============================='

# tetracorder uncorrected
tetracorder_out_directory=${out_directory}/${type}_unc/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run tetracorder
./tetracorder/tetracorder.sh "$rfl_img_aug" ${tetracorder_out_directory} ${sensor} --delete_tc_output 
    
echo '====================Tetracorder on Extracted veg (global)==============================='

# run Tetracorder on extracted signal of vegetation
tetracorder_out_directory=${out_directory}/${type}_global/

if [ -d "$tetracorder_out_directory" ]; then
        echo "Directory '$tetracorder_out_directory' exists. Removing..."
        rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run Vegetation Extractor.py
python ./tetracorder/vegetation_extractor.py -out_dir ${tetracorder_out_directory} -sns ${sensor} -veg_fracs "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_complete_fractions" -rfl ${rfl_img_aug} -unmix_lib_csv ${unmixing_library} -unmix_lib_envi ${unmixing_library_envi} -three_comp_frac "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_fractional_cover" --tetracorder

extracted_vegetation_rfl_img=${tetracorder_out_directory}/ext_veg_${filebase_name}_augmented_${unmixing_filebase_name}_tc

if [ -f ${extracted_vegetation_rfl_img} ]; then
    echo "$extracted_vegetation_rfl_img File exists."     
    vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")
    
    # augment rfl data and run tetracorder
    ./tetracorder/tetracorder.sh ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} ${sensor} --delete_tc_output
    ./tetracorder/tetracorder.sh ${tetracorder_out_directory}/recon_rho_${filebase_name}_augmented_${unmixing_filebase_name} ${tetracorder_out_directory} ${sensor} --delete_tc_output

else
    echo "$extracted_vegetation_rfl_img File does not exist."
fi

echo '====================Tetracorder on Extracted veg (rock)==============================='


# run Tetracorder on extracted signal of vegetation
tetracorder_out_directory=${out_directory}/${type}_rock/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi
mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run Vegetation Extractor.py
python ./tetracorder/vegetation_extractor.py -out_dir ${tetracorder_out_directory} -sns ${sensor} -veg_fracs "${emc_out_directory}/${rock_lib_basename}_${filebase_name}_complete_fractions" -rfl ${rfl_img_aug} -unmix_lib_csv ${rock_lib} -unmix_lib_envi ${rock_lib_envi} -three_comp_frac "${emc_out_directory}/${rock_lib_basename}_${filebase_name}_fractional_cover" --tetracorder
extracted_vegetation_rfl_img=${tetracorder_out_directory}/ext_veg_${filebase_name}_augmented_${rock_lib_basename}_tc

if [ -f ${extracted_vegetation_rfl_img} ]; then
    echo "$extracted_vegetation_rfl_img File exists."
       
    vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")
    
    # augment rfl data and run tetracorder
    ./tetracorder/tetracorder.sh ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} ${sensor} --delete_tc_output
    ./tetracorder/tetracorder.sh ${tetracorder_out_directory}/recon_rho_${filebase_name}_augmented_${rock_lib_basename} ${tetracorder_out_directory} ${sensor} --delete_tc_output

else
    echo "$extracted_vegetation_rfl_img File does not exist."
fi

echo '====================Tetracorder on Extracted veg (local)==============================='

# run Tetracorder on extracted signal of vegetation
tetracorder_out_directory=${out_directory}/${type}_lcl/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# run Vegetation Extractor.py
python ./tetracorder/vegetation_extractor.py -out_dir ${tetracorder_out_directory} -sns ${sensor} -veg_fracs "${emc_out_directory}/local_${filebase_name}_complete_fractions" -rfl ${rfl_img_aug} -unmix_lib_csv ${local_library} -unmix_lib_envi ${local_library_envi} -three_comp_frac "${emc_out_directory}/local_${filebase_name}_fractional_cover" --tetracorder
extracted_vegetation_rfl_img=${tetracorder_out_directory}/ext_veg_${filebase_name}_augmented_${local_lib_basename}_tc

echo "Tetracorder out: $tetracorder_out_directory"
echo "Veg_fracs: ${emc_out_directory}/local_${filebase_name}_complete_fractions"
echo "Basename_RFL: $rfl_img_aug"
echo "csv_unmix: $local_library"
echo "envi_unmix: $local_library_envi"
echo "3 fracs: ${emc_out_directory}/local_${filebase_name}_fractional_cover"
echo "ext veg: $extracted_vegetation_rfl_img"

if [ -f ${extracted_vegetation_rfl_img} ]; then
    echo "$extracted_vegetation_rfl_img File exists."
    vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")
    
    # augment rfl data and run tetracorder
    ./tetracorder/tetracorder.sh ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} ${sensor} --delete_tc_output
    ./tetracorder/tetracorder.sh ${tetracorder_out_directory}/recon_rho_${filebase_name}_augmented_${local_lib_basename} ${tetracorder_out_directory} ${sensor} --delete_tc_output

else
    echo "$extracted_vegetation_rfl_img File does not exist."
fi


