#!/bin/sh
echo "################# Running Tetracorder  ################"
echo " "
# Script variables
rfl_img=$1
out_base=$2
filebase_name=$(basename "$rfl_img")
unmixing_library_global=$3
unmixing_filebase_name=$(basename "$unmixing_library_global" .csv)

echo $rfl_img
echo $out_base
echo $filebase_name
echo $unmixing_filebase_name

UNMIX=false
TETRACORDER=false

POSITIONAL_ARGS=()

# Combine into ONE loop to catch all flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --unmix)
      UNMIX=true
      shift
      ;;
    --tetracorder)
      TETRACORDER=true
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # Save non-flag arguments
      shift
      ;;
  esac
done

# Optional: Restore positional arguments if needed for the rest of the script
set -- "${POSITIONAL_ARGS[@]}"

if [ "$UNMIX" = true ]; then
    echo "Process: Unmixing enabled."

    emc_out_directory=${out_base}/emc2/
    #if [ -d "$emc_out_directory" ]; then
    #    echo "Directory '$emc_out_directory' exists. Removing..."
    #    rm -rf "$emc_out_directory"
    #fi

    echo "Created emc2 dir: ${emc_out_directory}"
    mkdir -p ${emc_out_directory}
    
    julia -p 20 ../SpectralUnmixing/unmix.jl ${rfl_img} ${unmixing_library_global} level_1 "${emc_out_directory}/${unmixing_filebase_name}_${filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 11

else
  echo "Unmixing disabled!"
fi


if [ "$TETRACORDER" = true ]; then
    echo "Process: Tetracorder enabled."

    # run tetracorder
    tetracorder_out_directory=${out_base}/${filebase_name}_${unmixing_filebase_name}/

    if [ -d "$tetracorder_out_directory" ]; then
        echo "Directory '$tetracorder_out_directory' exists. Removing..."
        rm -rf "$tetracorder_out_directory"
    fi

    mkdir -p ${tetracorder_out_directory}
    echo "Created tetracorder dir: ${tetracorder_out_directory}"

    # augment rfl data and run tetracorder
    python ./utils/augment_file.py ${rfl_img} ${tetracorder_out_directory} --augment # augment rfl file
    ./tetracorder/tetracorder.sh "${tetracorder_out_directory}/${filebase_name}_aug" ${tetracorder_out_directory} emit --delete_tc_output 

    # deaugment data in tetracorder output director
    cp ${rfl_img} ${tetracorder_out_directory}
    cp ${rfl_img}.hdr ${tetracorder_out_directory}
    python ./utils/augment_file.py ${tetracorder_out_directory}/${filebase_name} ${tetracorder_out_directory} --deaugment
    rm ${tetracorder_out_directory}/${filebase_name}
    #${tetracorder_out_directory}/${filebase_name}.hdr

else
  echo "Tetracorder disabled!"
fi

# push data to drive
#/store/shared/rclone/bin/rclone copy ${out_base}/ "cdrive:${out_base#./}" -P 
