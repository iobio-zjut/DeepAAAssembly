import os
import argparse

def combine_antibody_antigen(base_path, input_subdir, output_subdir):
    """
    Combine antibody (dom1.pdb) with transformed antigen pdbs under tbm/ folders.
    """

    # 遍历输入路径中的每个文件夹
    for foldername in os.listdir(base_path):
        folder_path = os.path.join(base_path, foldername)

        if not os.path.isdir(folder_path):
            continue

        # 读取 dom1.pdb
        antibody_source = os.path.join(folder_path, 'dom1.pdb')
        if not os.path.exists(antibody_source):
            print(f"[WARN] {antibody_source} does not exist.")
            continue

        # tbm/trans_pos_5model_bestscore
        output_folder_path = os.path.join(folder_path, 'tbm', input_subdir)
        if not os.path.exists(output_folder_path):
            print(f"[WARN] {output_folder_path} does not exist.")
            continue

        output_pdb_files = [f for f in os.listdir(output_folder_path) if f.endswith('.pdb')]
        if not output_pdb_files:
            print(f"[INFO] No .pdb files found in {output_folder_path}")
            continue

        # tbm/after_tran_complex_5model_bestscore
        output_complex_folder_path = os.path.join(folder_path, 'tbm', output_subdir)
        os.makedirs(output_complex_folder_path, exist_ok=True)

        for output_pdb_file in output_pdb_files:
            antigen_source = os.path.join(output_folder_path, output_pdb_file)

            # 读取 antibody (dom1.pdb)
            with open(antibody_source, 'r') as antibody_file:
                antibody_content = antibody_file.readlines()
                # 删除以 'END' 开头的行
                antibody_content = [line for line in antibody_content if not line.startswith('END')]

            # 读取 antigen pdb
            with open(antigen_source, 'r') as antigen_file:
                antigen_content = antigen_file.readlines()

            # 合并为复合物
            new_content = antibody_content + antigen_content

            # 构建输出文件名
            base_name = output_pdb_file[:-4]  # 去除 .pdb
            if base_name.startswith('trans_pose_'):
                new_pdb_filename = f'complex{base_name[10:]}.pdb'
            else:
                new_pdb_filename = f'complex{base_name}.pdb'

            new_pdb_path = os.path.join(output_complex_folder_path, new_pdb_filename)

            # 写入新文件
            with open(new_pdb_path, 'w') as new_pdb_file:
                new_pdb_file.writelines(new_content)

            print(f"[OK] Generated: {new_pdb_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine antibody (dom1.pdb) with transformed antigen pdbs.")
    parser.add_argument(
        "--base_path", type=str, required=True,
        help="Base directory containing subfolders (each with dom1.pdb and tbm folders)."
    )
    parser.add_argument(
        "--input_subdir", type=str, default="trans_pos_5model_bestscore",
        help="Subdirectory under tbm containing transformed antigen pdb files."
    )
    parser.add_argument(
        "--output_subdir", type=str, default="after_tran_complex_5model_bestscore",
        help="Subdirectory name under tbm to store combined complex pdb files."
    )

    args = parser.parse_args()
    combine_antibody_antigen(args.base_path, args.input_subdir, args.output_subdir)


if __name__ == "__main__":
    main()
