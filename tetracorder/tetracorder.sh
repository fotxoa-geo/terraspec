#!/bin/sh
rfl_file=$1
out_base=$2

filebase=`basename ${rfl_file}`

tmp_rfl_path=/local/`basename ${rfl_file}`
tmp_tetra_path=/local/${filebase}_tetra_output
out_tetra_path=${out_base}${filebase}_tetra
out_min_path=${out_base}${filebase}_min
out_minunc_path=${out_base}${filebase}_minunc
out_abun_path=${out_base}${filebase}_abun
tc_out_path=/store/fochoa/terraspec_output/terraspec/tetracorder/output/outlogs/

echo $tmp_tetra_path
echo $out_tetra_path

cp ${rfl_file} $tmp_rfl_path
cp ${rfl_file}.hdr ${tmp_rfl_path}.hdr

export SP_LOCAL=/store/shared/spectroscopy-tetracorder/specpr
export SP_BIN=${SP_LOCAL}/bin
export TETRA=/store/shared/spectroscopy-tetracorder/tetracorder5.27 # ops_sds_config.json:tetracorder_path
export TETRA_CMDS=/store/shared/tetracorder5.27c.cmds/ #tetracorder_cmds_path
export PYTHONPATH=/store/brodrick/repos/emit-utils/
export /usr/bin/rclone

export PATH="${PATH}:${SP_LOCAL}/bin:${TETRA}/bin:/usr/bin"

cpwd=$PWD
$TETRA_CMDS/cmd-setup-tetrun $tmp_tetra_path emit_e cube $tmp_rfl_path 1 -T -20 80 C -P .5 1.5 bar
cd $tmp_tetra_path
time ${tmp_tetra_path}/cmd.runtet cube $tmp_rfl_path band 20 gif >& ${tc_out_path}_cmd.runtet.out
cd $cpwd

cp ${tmp_tetra_path} ${out_tetra_path} -r

python /store/brodrick/repos/emit-sds-l2b/group_aggregator.py $out_tetra_path /store/fochoa/terraspec/utils/tetracorder/mineral_grouping_matrix_20230503.csv $out_min_path $out_minunc_path --reflectance_file $tmp_rfl_path --reflectance_uncertainty_file $tmp_rfl_path --reference_library /store/shared/tetracorder_libraries/s06emitd_envi --research_library /store/shared/tetracorder_libraries/r06emitd_envi --expert_system_file cmd.lib.setup.t5.27d1 --calculate_uncertainty 

#python /beegfs/scratch/brodrick/emit/emit-sds-l2b/abundance_from_min.py $out_abun_path $out_min_path --mineral_groupings_matrix /beegfs/scratch/brodrick/emit/emit-sds-l2b/data/mineral_grouping_matrix_20230503.csv
rm $tmp_rfl_path
rm ${tmp_rfl_path}.hdr


rm -rf $tmp_tetra_path
cpwd=$PWD

cd $out_tetra_path/cmds.abundances/lists.of.files.by.mineral/
mkdir ${out_base}${filebase}_minerals/ -p 
cp . ${out_base}${filebase}_minerals/ -r
cd $cpwd

rm -rf $out_tetra_path
