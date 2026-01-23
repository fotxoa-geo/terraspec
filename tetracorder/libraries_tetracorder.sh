#!/bin/shG
echo "################# Running Tetracorder  ################"
echo " "
# Script variables
rfl_img=$1
out_base=$2
filebase_name=$(basename "$rfl_img")

# run tetracorder
tetracorder_out_directory=${out_base}/tetracorder/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# augment rfl data and run tetracorder
python ./utils/augment_file.py ${rfl_img} ${tetracorder_out_directory} --augment # augment rfl file
./tetracorder/tetracorder.sh "${tetracorder_out_directory}/${filebase_name}_augmented" ${tetracorder_out_directory}

# deaugment data in tetracorder output directory
python ./utils/augment_file.py ${rfl_img} ${tetracorder_out_directory} --deaugment

# push data to drive
out_base_name=$(basename "${out_base}")
/store/shared/rclone/bin/rclone copy ${out_base} cdrive:terraspec_output/tetracorder/output/${out_base_name} -P