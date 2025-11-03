import os
import numpy as np
import argparse
from utils import get_inter_dist
def generate_label(pdb_folder, output_dir, lst_file):
    """第一阶段：根据 PDB 生成 label 文件（距离矩阵）"""
    # 从 lst.txt 读取需要处理的列表
    with open(lst_file, "r") as f:
        pdb_list = [line.strip() for line in f if line.strip()]  # 去掉空行/换行符

    for pdb_name in pdb_list:
        print(f"[Label] Extracting distance for: {pdb_name}")

        pdb_path = os.path.join(pdb_folder, pdb_name, f'{pdb_name}.pdb')
        if not os.path.exists(pdb_path):
            print(f" Warning: {pdb_path} not found, skipped.")
            continue

        try:
            ca_dist = get_inter_dist(pdb_path)
        except Exception as e:
            print(f" Error processing {pdb_name}: {e}")
            continue

        output_subdir = os.path.join(output_dir, pdb_name)
        os.makedirs(output_subdir, exist_ok=True)
        ca_dist_file = os.path.join(output_subdir, f'{pdb_name}.txt')
        np.savetxt(ca_dist_file, ca_dist, fmt='%.3f')

    print(" Label generation completed.\n")


def generate_mask(label_dir, mask_dir):
    """第二阶段：根据 label 文件生成 mask 文件"""
    os.makedirs(mask_dir, exist_ok=True)

    for pdb_name in os.listdir(label_dir):
        label_file = os.path.join(label_dir, pdb_name, f"{pdb_name}.txt")
        if not os.path.exists(label_file):
            print(f" Missing label file for {pdb_name}, skipped.")
            continue

        try:
            data = np.loadtxt(label_file)
        except Exception as e:
            print(f" Error loading {label_file}: {e}")
            continue

        # 阈值映射
        data[(data > 0) & (data <= 12.0)] = 1.0
        data[(data > 12.0) & (data <= 16.0)] = 0.7
        data[(data > 16.0) & (data <= 20.0)] = 0.5
        data[data > 20.0] = 0.1

        if np.any(data == 0):
            print(f" Warning: {pdb_name} still has 0 values after processing.")

        output_subdir = os.path.join(mask_dir, pdb_name)
        os.makedirs(output_subdir, exist_ok=True)
        output_file = os.path.join(output_subdir, f"{pdb_name}.txt")
        np.savetxt(output_file, data, fmt='%.1f')

        print(f"[Mask] Processed: {pdb_name}")

    print(" Mask generation completed.")



def main():
    parser = argparse.ArgumentParser(description="Generate label and mask files for antibody-antigen complexes.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the base data directory (e.g., ./distance/data)"
    )
    args = parser.parse_args()

    # 根据 base_dir 自动生成子路径
    base_dir = os.path.abspath(args.base_dir)
    pdb_folder = os.path.join(base_dir, "dataset")
    label_dir = os.path.join(base_dir, "label")
    mask_dir = os.path.join(base_dir, "mask")
    lst_file = os.path.join(pdb_folder, "lst.txt")

    print("📂 Paths Configuration:")
    print(f"  Base Dir : {base_dir}")
    print(f"  PDB Dir  : {pdb_folder}")
    print(f"  Label Dir: {label_dir}")
    print(f"  Mask Dir : {mask_dir}")
    print(f"  LST File : {lst_file}\n")

    # 阶段1：生成label
    generate_label(pdb_folder, label_dir, lst_file)

    # 阶段2：生成mask
    generate_mask(label_dir, mask_dir)


if __name__ == "__main__":
    main()