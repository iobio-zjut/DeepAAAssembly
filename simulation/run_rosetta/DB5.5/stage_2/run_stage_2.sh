#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
file_path="$SCRIPT_DIR/lst.txt"
target_dir="$SCRIPT_DIR"
list_name=()
parallel_limit=10 # 设置并行处理的限制，这里是一次运行的目标数

while IFS= read -r line; do
  list_name+=("$line")
done < "$file_path"

for name in "${list_name[@]}"; do
  if [ ! -d "$target_dir/${name}/tbm/ensemble_pdb" ]; then
    echo "Running domain assembly (tbm)...${name}"
    mkdir $target_dir/${name}/tbm/ensemble_pdb
    cp $target_dir/flags $target_dir/${name}/tbm
    cp $target_dir/stage2.sh $target_dir/${name}/tbm
    cd $target_dir/${name}/tbm
    chmod 755 $target_dir/${name}/tbm/stage2.sh
    ./stage2.sh &
    ((i=i+1)) # 计数器，记录已经启动的任务数
    if [ "$i" -ge "$parallel_limit" ]; then
      i=0 # 达到并行处理限制后，等待所有任务完成
      wait
    fi
    cd $target_dir
  fi
done

# 等待剩余的任务完成
wait
