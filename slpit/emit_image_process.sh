#!/bin/bash
START_TIME=$SECONDS

nc_file=$1
global_unmixing_library=$2
out_base=$3
NORMALIZED_PATH_OUTBASE=$(echo "${out_base}" | tr '\\' '/')
NORMALIZED_GLOBAL_LIB_PATH=$(echo "${global_unmixing_library}" | tr '\\' '/')

filebase_name=$(basename --suffix=".nc" "$nc_file")

# This seperates the nc file by delimter
IFS='_' read -r -a SPLIT_ARRAY <<< "$filebase_name"
SPLICE='.' read -r -a EXTENSION_ARRAY <<< "$filebase_name"

file_basename="${EXTENSION_ARRAY[0]}"
product="${SPLIT_ARRAY[1]}"
fid="${SPLIT_ARRAY[4]}"
data_type="${SPLIT_ARRAY[2]}"

echo ${filebase_name}
echo ${product}
echo ${fid}
echo ${data_type}

# setup directory for output nc file data; this is the root tree
nc_fid_directory=${NORMALIZED_PATH_OUTBASE}/${fid}
echo "FID Directory: ${nc_fid_directory}"
mkdir ${nc_fid_directory} -p

# Run geoprocessing to create envi outputs image from nc file
nc_out_directory=${nc_fid_directory}/${product}/
mkdir ${nc_out_directory} -p

if [ "${product}" = "L1B" ]; then
    
    # only geoprocess the obs file
    if ["${data_type}" = "OBS" ]; then
        python ./emit-utils/emit_utils/reformat.py ${nc_file} ${nc_out_directory} --orthorectify --overwrite
    else
        echo "Radiance detected...Skipping"
    fi
    
    rm $nc_file
else
    python ./emit-utils/emit_utils/reformat.py ${nc_file} ${nc_out_directory} --overwrite
    
    # Run unmixing on image with global library using Ochoa et al. (2025) EMC^2; this will not use uncertainty!
    # Uncertainty runs will only be used with extracted SLPIT runs. Run Tetracorder as well.
    if [ "${data_type}" = "RFL" ]; then

        rfl_img=${nc_out_directory}/${filebase_name}_reflectance
        if [ -f ${rfl_img} ]; then
            echo "$rfl_img File exists."
        else
            echo "$rfl_img File does not exist."
        fi

        #create rgbs of images
        python slpit/envi_to_rgb.py -rfl_img $rfl_img -nc_file ${nc_file} -out ${nc_out_directory}
    
        # unmixing code
        unmix_out_directory=${nc_fid_directory}/emc2/
        mkdir -p ${unmix_out_directory}
        julia -p 40 ../SpectralUnmixing/unmix.jl ${rfl_img} ${NORMALIZED_GLOBAL_LIB_PATH} level_1 ${unmix_out_directory}/${filebase_name} --mode sma --normalization brightness --num_endmember 30 --n_mc 25 --spectral_starting_col 8
    
        # ortho-rectify fractional cover
        python slpit/ortho_frac_cover.py -frac_img ${unmix_out_directory}/${filebase_name}_fractional_cover -nc_file ${nc_file} -out ${unmix_out_directory}

        # run Tetracorder
        tetracorder_out_directory=${nc_fid_directory}/tetracorder/
        if [ -d "$tetracorder_out_directory" ]; then
            echo "Directory '$tetracorder_out_directory' exists. Removing..."
            rm -rf "$tetracorder_out_directory"
        fi
    
        mkdir -p ${tetracorder_out_directory}
        echo "Created directory: $tetracorder_out_directory"
        ############# uncorrected 
        tetracorder/tetracorder.sh ${rfl_img} ${tetracorder_out_directory} --emit --delete_tc_output
        
        #ortho-rectify mineral outputs
        python slpit/ortho_tetracorder.py -tc_dir ${tetracorder_out_directory} -nc_file ${nc_file}

        ############# global lib 
        tetracorder_out_directory=${nc_fid_directory}/tetracorder_veg_ext/
        if [ -d "$tetracorder_out_directory" ]; then
            echo "Directory '$tetracorder_out_directory' exists. Removing..."
            rm -rf "$tetracorder_out_directory"
        fi
    
        mkdir -p ${tetracorder_out_directory}
        echo "Created directory: $tetracorder_out_directory"
        
        python ./tetracorder/vegetation_extractor.py -out_dir ${tetracorder_out_directory} -sns emit -veg_fracs "${emc_out_directory}/${filebase_name}_complete_fractions" -rfl ${rfl_img} -unmix_lib_csv ${NORMALIZED_GLOBAL_LIB_PATH} -unmix_lib_envi ./terraspec_output/simulation/output/endmember_libraries/convex_hull__n_dims_4_sensor_emit_geofilter_True_unmix_library  -3_comp_frac "${emc_out_directory}/${filebase_name}_fractional_cover" --tetracorder
        extracted_vegetation_rfl_img=${tetracorder_out_directory}/ext_veg_${filebase_name}_augmented_tc
        
        if [ -f ${extracted_vegetation_rfl_img} ]; then
            echo "$extracted_vegetation_rfl_img File exists."

            vegetation_extracted_basename=$(basename "$extracted_vegetation_rfl_img")

            # augment rfl data and run tetracorder
            ./tetracorder/tetracorder.sh ${extracted_vegetation_rfl_img} ${tetracorder_out_directory} --emit --delete_tc_output
            ./tetracorder/tetracorder.sh ${tetracorder_out_directory}/recon_rho_${filebase_name}_augmented ${tetracorder_out_directory} --emit --delete_tc_output

        else
            echo "$extracted_vegetation_rfl_img File does not exist."
        fi
       

        #ortho-rectify mineral outputs
        python slpit/ortho_tetracorder.py -tc_dir ${tetracorder_out_directory} -nc_file ${nc_file}
    
    else
        echo "Reflectance data not detected. Skipping spectral processes!!"
    fi   
    
    # extract windows from images
    ext_out_directory=terraspec_output/slpit/output/spectral_transects/
    mkdir -p ${ext_out_directory}
    declare -A file_endings
    file_endings=([RFL]="reflectance" [MASK]="mask" [RFLUNCERT]="reflectance_uncertainty")
    
    img_name=${nc_out_directory}/${filebase_name}_${file_endings[$data_type]}
    
    python ./slpit/window_extract.py -rfl_img ${img_name} -nc_file ${nc_file} -w_size 3 -shp ./gis/Observation.json -pad 1 -out ${ext_out_directory}
fi

echo "Geoprocess complete!"

DURATION=$(( $SECONDS - $START_TIME ))
echo "processing time: $DURATION seconds."
