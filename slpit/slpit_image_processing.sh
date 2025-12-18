#!/bin/sh
START_TIME=$SECONDS

echo "################# Running unmixing and tetracorder call ################"
echo " "
# These are the variables
rfl_file=$1
global_unmixing_library=$2
local_umixing_library=$3
out_base=$4

filebase_name=$(basename "$rfl_file")
NORMALIZED_PATH_OUTBASE=$(echo "${out_base}" | tr '\\' '/')
NORMALIZED_GLOBAL_LIB_PATH=$(echo "${unmixing_library}" | tr '\\' '/')
NORMALIZED_LOCAL_LIB_PATH=$(echo "${unmixing_library}" | tr '\\' '/')

echo "basename: ${rfl_file}"

# Seperate basename into components
IFS='_' read -r -a SPLIT_ARRAY <<< "$filebase_name"
