#!/bin/sh
echo "################# Running RECLAIMER  ################"
echo " "

# Script variables
out_directory=$1
sensor=$2
veg_fractions=$3
rfl_img=$4
unmixing_library_global_csv=$5
unmixing_library_envi=$6
three_component_fractions=$7

unmixing_filebase_name=$(basename "$unmixing_library_global_csv" .csv)
filebase_name=$(basename "$rfl_img")

echo "RFL img: $rfl_img"
echo "Out base Directory: $out_directory"
echo "Basename_RFL: $filebase_name"
echo "Unmixing file basename: $unmixing_filebase_name"
echo "Three component fractions: $three_component_fractions"
echo "Sensor: $sensor"
CURRENT_DIR=$(pwd)
echo "You are currently in: $CURRENT_DIR"

# run Vegetation Extractor.py
python ./tetracorder/vegetation_extractor.py -out_dir ${out_directory} -sns ${sensor} -veg_fracs ${veg_fractions} -rfl ${rfl_img} -unmix_lib_csv ${unmixing_library_global_csv} -unmix_lib_envi ${unmixing_library_envi} -three_comp_frac ${three_component_fractions} --tetracorder

# run Tetracorder on extracted signal of vegetation
tetracorder_out_directory=${out_directory}/${filebase_name}_${unmixing_filebase_name}_veg_ext/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

extracted_vegetation_rfl_img=${out_directory}/ext_veg_${filebase_name}_${unmixing_filebase_name}_tc

if [ -f ${extracted_vegetation_rfl_img} ]; then
    echo "$extracted_vegetation_rfl_img File exists."

    mkdir -p ${tetracorder_out_directory}
    echo "Created tetracorder dir: ${tetracorder_out_directory}"
    vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")

    # augment rfl data and run tetracorder
    python ./utils/augment_file.py ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} --augment # augment rfl file
    ./tetracorder/tetracorder.sh "${tetracorder_out_directory}/${vegetation_extracted_basename}_aug" ${tetracorder_out_directory} ${sensor} --delete_tc_output

    #deaugment data in tetracorder output director
    cp ${extracted_vegetation_rfl_img} ${tetracorder_out_directory}
    cp ${extracted_vegetation_rfl_img}.hdr ${tetracorder_out_directory}
    python ./utils/augment_file.py ${tetracorder_out_directory}/${vegetation_extracted_basename} ${tetracorder_out_directory} --deaugment
    rm ${tetracorder_out_directory}/${vegetation_extracted_basename}    
    rm ${tetracorder_out_directory}/${vegetation_extracted_basename}.hdr


else
    echo "$extracted_vegetation_rfl_img File does not exist."
fi
