
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base_dir="./distance/data"
data_path="./distance/data/dataset"

module load anaconda
source activate DeepTMP
mkdir -p $data_path

python $SCRIPT_DIR/process_pdb.py --base_dir $base_dir
python $SCRIPT_DIR/get_label_mask.py --base_dir $base_dir

# 遍历每个子文件夹
for pdb_folder in $data_path/*; do
    # 提取文件夹名，按"_"分割
    IFS='_' read -ra folder_parts <<< "$(basename "$pdb_folder")"
    antibody_name="${folder_parts[0]}"
    antigen_name="${folder_parts[1]}"

    echo "Processing files in folder: $(basename "$pdb_folder")"

    # 处理 antibody_name.pdb
    pdb_path_antibody="$pdb_folder/$antibody_name.pdb"
    if [ -f "$pdb_path_antibody" ]; then
        mkdir -p "$data_path/$antibody_name"_"$antigen_name"
        python $SCRIPT_DIR/renum_chain.py -p "$pdb_path_antibody" -o "$data_path/$antibody_name"_"$antigen_name" -op process_pdbfile -al true
        echo "Processed $antibody_name.pdb and saved in folder: $antibody_name"_"$antigen_name"
    else
        echo "$antibody_name.pdb not found in folder: $(basename "$pdb_folder")"
    fi

    # 处理 antigen_name.pdb
    pdb_path_antigen="$pdb_folder/$antigen_name.pdb"
    if [ -f "$pdb_path_antigen" ]; then
        mkdir -p "$data_path/$antibody_name"_"$antigen_name"
        python $SCRIPT_DIR/renum_chain.py -p "$pdb_path_antigen" -o "$data_path/$antibody_name"_"$antigen_name" -op process_pdbfile -al false
        echo "Processed $antigen_name.pdb and saved in folder: $antibody_name"_"$antigen_name"
    else
        echo "$antigen_name.pdb not found in folder: $(basename "$pdb_folder")"
    fi
done


