#!/bin/bash

module load anaconda
source activate DeepTMP
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_dir="./distance/save_model/best.pkl"
base_dir="./distance/data/dataset"
output_dir="./distance/predict/output"
lst_file="./distance/data/dataset/lst.txt"

# 逐行读取文件内容并处理
while IFS= read -r pdb_name || [[ -n "$pdb_name" ]]; do

	pdb_name=$(echo "$pdb_name" | tr -d '\r')

  prediction_file="$output_dir/$pdb_name/${pdb_name}_prediction.adist"

	if [ -f "$prediction_file" ]; then
        echo "File $prediction_file exists, skipping..."
    else
		echo "processing pdb_name：$pdb_name"
		cd $base_dir/$pdb_name

		python $SCRIPT_DIR/predict.py \
		-m $model_dir\
		-dp $base_dir/$pdb_name\
		-t $pdb_name\
		-n all\
		-op $output_dir/$pdb_name\
		-gpu True\
    -dev 0

		cd "$base_dir"
	fi

done < "$lst_file"