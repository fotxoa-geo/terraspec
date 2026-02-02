#!/bin/sh
echo "################# Running Tetracorder  ################"
echo " "
# Script variables
rfl_img=$1
out_base=$2
filebase_name=$(basename "$rfl_img")


echo $rfl_img
echo $out_base
echo $filebase_name

UNMIX=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --unmix)
      UNMIX=true
      shift # Move to the next argument
      ;;
    *)
      # This handles positional arguments or unknown flags
      POSITIONAL_ARGS+=("$1") 
      shift
      ;;
  esac
done

if [ "$UNMIX" = true ]; then
    echo "Process: Unmixing enabled."

    emc_out_directory=${out_base}/emc2/
    if [ -d "$emc_out_directory" ]; then
        echo "Directory '$emc_out_directory' exists. Removing..."
        rm -rf "$emc_out_directory"
    fi

    echo "Created emc2 dir: ${emc_out_directory}"
    mkdir -p ${emc_out_directory}
    
    unmixing_library_global=./terraspec_output/simulation/output/endmember_libraries/convex_hull__n_dims_4_sensor_emit_geofilter_True_unmix_library.csv
    julia -p 20 ../SpectralUnmixing/unmix.jl ${rfl_img} ${unmixing_library_global} level_1 "${emc_out_directory}/global_${filebase_name}_normalization_brightness_" --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 11

else
  echo "Unmixing disabled!"
fi

# run tetracorder
tetracorder_out_directory=${out_base}/tetracorder/

if [ -d "$tetracorder_out_directory" ]; then
    echo "Directory '$tetracorder_out_directory' exists. Removing..."
    rm -rf "$tetracorder_out_directory"
fi

mkdir -p ${tetracorder_out_directory}
echo "Created tetracorder dir: ${tetracorder_out_directory}"

# augment rfl data and run tetracorder
python ./utils/augment_file.py ${rfl_img} ${out_base} --augment # augment rfl file
./tetracorder/tetracorder.sh "${out_base}/${filebase_name}_augmented" ${tetracorder_out_directory}

# deaugment data in tetracorder output directory
python ./utils/augment_file.py ${rfl_img} ${out_base} --deaugment

# push data to drive
/store/shared/rclone/bin/rclone copy "${out_base}" "cdrive:${out_base#./}" -P
