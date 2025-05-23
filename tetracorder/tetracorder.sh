#!/bin/sh
rfl_file=$1
out_base=$2

filebase=`basename ${rfl_file}`

tmp_rfl_path=/tmp/`basename ${rfl_file}`
tmp_tetra_path=/tmp/${filebase}_tetra_output
out_tetra_path=${out_base}${filebase}_tetra
out_min_path=${out_base}${filebase}_min
out_minunc_path=${out_base}${filebase}_minunc
out_abun_path=${out_base}${filebase}_abun

echo $tmp_tetra_path
echo $out_tetra_path

cp ${rfl_file} $tmp_rfl_path
cp ${rfl_file}.hdr ${tmp_rfl_path}.hdr

export SP_LOCAL=/beegfs/store/shared/specpr
export SP_BIN="${SP_LOCAL}/bin"
export TETRA=/beegfs/store/shared/spectroscopy-tetracorder/tetracorder5.27
export TETRA_CMDS=/beegfs/store/shared/spectroscopy-tetracorder/tetracorder.cmds/tetracorder5.27c.cmds
export PATH="${PATH}:${SP_LOCAL}/bin:${TETRA}/bin:/usr/bin"

cpwd=$PWD
$TETRA_CMDS/cmd-setup-tetrun $tmp_tetra_path emit_e cube $tmp_rfl_path 1 -T -20 80 C -P .5 1.5 bar
cd $tmp_tetra_path
time ${tmp_tetra_path}/cmd.runtet cube $tmp_rfl_path band 20 gif >& cmd.runtet.out
cd $cpwd

cp ${tmp_tetra_path} ${out_tetra_path} -r

python /beegfs/scratch/brodrick/emit/emit-sds-l2b/group_aggregator.py $out_tetra_path /beegfs/scratch/brodrick/emit/emit-sds-l2b/mineral_grouping_matrix_20230503.csv $out_min_path $out_minunc_path --reflectance_file $tmp_rfl_path --reflectance_uncertainty_file $tmp_rfl_path --reference_library /beegfs/store/shared/tetracorder_libraries/s06emitd_envi --research_library /beegfs/store/shared/tetracorder_libraries/r06emitd_envi --expert_system_file cmd.lib.setup.t5.27d1

python /beegfs/scratch/brodrick/emit/emit-sds-l2b/abundance_from_min.py $out_abun_path $out_min_path --mineral_groupings_matrix /beegfs/scratch/brodrick/emit/emit-sds-l2b/data/mineral_grouping_matrix_20230503.csv
rm $tmp_rfl_path
rm ${tmp_rfl_path}.hdr

rm -rf $tmp_tetra_path
rm -rf $out_tetra_path
