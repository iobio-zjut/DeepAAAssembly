#!/bin/bash

# 自动获取脚本所在目录（保证无论从哪里执行都能定位到 feature 路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查输入参数
if [ $# -lt 1 ]; then
    echo "Usage: bash $(basename "$0") <base_dir>"
    echo "Example: bash $(basename "$0") ./distance/data/dataset"
    exit 1
fi

# 从外部传入 base_dir
base_dir="$1"

# 检查 base_dir 是否存在
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory '$base_dir' not found!"
    exit 2
fi

bash "$SCRIPT_DIR/get_voro.sh"

# 遍历目录下的所有子目录
for dir in "$base_dir"/*/; do
    echo "==========================================="
    echo "Checking directory: $dir"
#
    # 切换到当前子目录
    cd "$dir" || continue

    # 提取目录名作为 pdb_name
    pdb_name=${dir%/}
    pdb_name=${pdb_name##*/}

    # 用 '_' 分割 pdb_name
    IFS="_" read -ra names <<< "$pdb_name"

    pdb_name1="${names[0]}"
    pdb_name2="${names[1]}"

    echo "pdb_name:  $pdb_name"
    echo "pdb_name1: $pdb_name1"
    echo "pdb_name2: $pdb_name2"

    # 调用主特征生成脚本（路径用 SCRIPT_DIR 替代）
    bash "$SCRIPT_DIR/process_feature.sh" "$pdb_name1" "$pdb_name2" "$pdb_name" -dp "$dir"

    # 返回父目录
    cd "$base_dir" || exit
done
