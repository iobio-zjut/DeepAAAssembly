import os
import shutil
import numpy as np
import pandas as pd
import argparse
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model


class ChainSelector(Select):
    def __init__(self, target_chains):
        self.target_chains = target_chains

    def accept_chain(self, chain):
        return chain.id in self.target_chains


# -----------------------------
# 主功能函数
# -----------------------------
def process_pdbs(summary_file, pdb_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # 读取 summary 表
    summary_df = pd.read_csv(summary_file, sep='\t')

    for pdb_file in os.listdir(pdb_directory):
        pdb_name = os.path.splitext(pdb_file)[0]
        pdb_path = os.path.join(pdb_directory, pdb_file)
        if not os.path.isfile(pdb_path):
            continue

        matching_rows = summary_df[summary_df['pdb'].str.startswith(pdb_name)]

        for _, matching_row in matching_rows.iterrows():
            h_chain = str(matching_row['Hchain'])
            l_chain = str(matching_row['Lchain'])
            antigen_chain = str(matching_row['antigen_chain'])

            if h_chain == antigen_chain:
                print(f"⚠️ Hchain == antigenchain, skipped: {pdb_name}_{h_chain}")
                continue

            structure = parser.get_structure(pdb_name, pdb_path)


            # ========================
            # 第1阶段：生成 H/L + antigen 的完整复合物 pdb（脚本1逻辑）
            # ========================
            output_structure = Structure.Structure(structure.id)
            for model in structure:
                output_model = Model.Model(model.id)
                for chain_id in [h_chain, l_chain, antigen_chain]:
                    for chain in model:
                        if chain.id == chain_id:
                            output_model.add(chain)
                            break
                output_structure.add(output_model)

            # 输出路径
            subdir_name = f"{pdb_name}-{h_chain}{l_chain}_{pdb_name}-{antigen_chain}"
            subdir_path = os.path.join(output_directory, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)

            output_pdb_file = os.path.join(subdir_path, f"{subdir_name}.pdb")
            io.set_structure(output_structure)
            io.save(output_pdb_file)

            # ========================
            # 第2阶段：生成 H/L 和 antigen 的单独 pdb 文件（脚本2逻辑）
            # ========================
            for model in structure:
                # 生成抗体部分 (H/L)
                output_model_ab = Model.Model(model.id)
                if h_chain != 'NA':
                    for chain in model:
                        if chain.id == h_chain:
                            output_model_ab.add(chain)
                if l_chain != 'NA' and l_chain != h_chain:
                    for chain in model:
                        if chain.id == l_chain:
                            output_model_ab.add(chain)

                output_ab_name = f"{pdb_name}-{h_chain}{l_chain}"
                ab_pdb_path = os.path.join(subdir_path, f"{output_ab_name}.pdb")
                io.set_structure(output_model_ab)
                io.save(ab_pdb_path)

                # 生成抗原部分
                output_model_ag = Model.Model(model.id)
                for chain in model:
                    if chain.id == antigen_chain:
                        output_model_ag.add(chain)
                        break

                output_ag_name = f"{pdb_name}-{antigen_chain}"
                ag_pdb_path = os.path.join(subdir_path, f"{output_ag_name}.pdb")
                io.set_structure(output_model_ag)
                io.save(ag_pdb_path)

            print(f" Processed: {subdir_name}")


def main():
    parser = argparse.ArgumentParser(description="Extract antibody-antigen PDBs with chain selection.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing example and dataset folders.")
    args = parser.parse_args()

    # 根据传入的 base_dir 自动构建下游路径
    summary_file = os.path.join(args.base_dir, "example", "summary.tsv")
    pdb_directory = os.path.join(args.base_dir, "example")
    output_directory = os.path.join(args.base_dir, "dataset")

    print(f"Summary file: {summary_file}")
    print(f"PDB directory: {pdb_directory}")
    print(f"Output directory: {output_directory}")

    process_pdbs(
        summary_file=summary_file,
        pdb_directory=pdb_directory,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()