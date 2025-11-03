#!/bin/bash
#激活环境

base_dir="./distance/data"
###########################################以下是修改部分############################################################################
#target_name_txt是target的id，存放在一个txt文件中
target_name_txt="$base_dir/dataset/lst.txt"
#PDB_file是一个文件夹，里面的子文件夹名是target的id，存放内容是所有model的pdb
PDB_file="$base_dir/dataset"
#特征文件输出总路径
npz_file="$base_dir/dataset"
#中间文件路径
temp_output="$base_dir/dataset/voro/tmp"
###########################################以上是修改部分############################################################################
#step1：提取中间文件
script_path="$base_dir/feature/bin/voro_script"
area_output="$temp_output/area_feature"
normal_output="$temp_output/normal_feature"

python $script_path/area_mutigetfeature.py $target_name_txt $PDB_file $area_output
python $script_path/extract_normal.py $target_name_txt $PDB_file $normal_output
#step2：生成npz文件
normal_npz_output="$npz_file"
area_npz_output="$npz_file"

python $script_path/normal.py $target_name_txt $normal_output $normal_npz_output
python $script_path/area_complex.py $target_name_txt $area_output $area_npz_output

rm -rf "$temp_output"