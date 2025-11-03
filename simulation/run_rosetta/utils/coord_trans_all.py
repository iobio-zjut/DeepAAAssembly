import os
import argparse
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser
import re

def process_trans_mat_file(trans_mat_file, structure):
    with open(trans_mat_file, "r") as file:
        content = file.read()
    try:
        rotMat_str = content.split("    ")[0].split(":")[1].strip()
        rotMat = eval(rotMat_str)
        traVec_str = content.split("    ")[1].split(":")[1].strip()
        traVec = eval(traVec_str)
    except IndexError:
        print(f"Error processing file: {trans_mat_file}. Check the file format.")
        return structure

    result = {"rotMat": rotMat, "traVec": traVec}

    rotation_matrix = np.array(result['rotMat'])
    translation_vector = np.array(result['traVec'])

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    transformed_coordinates = rotation_matrix @ np.array([
                        atom.coord[0] - translation_vector[0],
                        atom.coord[1] - translation_vector[1],
                        atom.coord[2] - translation_vector[2]])
                    atom.coord = transformed_coordinates

    return structure


def main(base_path, trans_mat_subdir, output_subdir):
    parser = PDBParser(QUIET=True)


    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            current_dir = os.path.join(root, dir_name)
            tbm_trans_mat_dir = os.path.join(current_dir, "tbm", trans_mat_subdir)
            dom2_pdb = os.path.join(current_dir, "dom2.pdb")

            if not os.path.exists(tbm_trans_mat_dir) or not os.path.exists(dom2_pdb):
                continue

            output_dir = os.path.join(current_dir, "tbm", output_subdir)
            os.makedirs(output_dir, exist_ok=True)

            structure = parser.get_structure("protein", dom2_pdb)

            # -------- 新逻辑：筛选并排序 trans_mat 文件 --------
            txt_files = [
                f for f in os.listdir(tbm_trans_mat_dir)
                if f.startswith("trans_mat") and f.endswith(".txt")
            ]

            def extract_score(filename):
                """从文件名中提取 score 后的数值"""
                match = re.search(r"score_([-+]?[0-9]*\.?[0-9]+)", filename)
                return float(match.group(1)) if match else -9999.0  # 若未找到则置为极小值

            # 按 score 值从大到小排序
            txt_files_sorted = sorted(txt_files, key=extract_score, reverse=False)

            # 只保留前 6 个（或不足 6 个时全部）
            top_files = txt_files_sorted[:6]

            print(f"\nProcessing directory: {current_dir}")
            print(f"Found {len(txt_files)} transformation files, processing top {len(top_files)} by score...\n")

            for file_name in top_files:
                trans_mat_file = os.path.join(tbm_trans_mat_dir, file_name)
                print(f"Processing {file_name} (score={extract_score(file_name)})")

                transformed_structure = process_trans_mat_file(trans_mat_file, structure)

                # 构建输出文件名（保留全部后缀部分）
                base_parts = file_name.replace('.txt', '').split('_')[2:]
                suffix = '_'.join(base_parts)
                output_file = os.path.join(output_dir, f"trans_pose_{suffix}.pdb")

                io = PDB.PDBIO()
                io.set_structure(transformed_structure)
                io.save(output_file)

            print(f"✅ Finished {current_dir}: {len(top_files)} PDBs saved.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply transformation matrices to PDB structures.")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory containing subdirectories with dom2.pdb and tbm folders.")
    parser.add_argument("--trans_mat_subdir", type=str, default="trans_mat_5model_bestscore",
                        help="Subdirectory under 'tbm' containing transformation matrix files.")
    parser.add_argument("--output_subdir", type=str, default="trans_pos_5model_bestscore",
                        help="Subdirectory name under 'tbm' where transformed PDB files will be saved.")

    args = parser.parse_args()

    main(args.base_path, args.trans_mat_subdir, args.output_subdir)
