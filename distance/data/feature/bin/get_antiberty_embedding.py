import os
import torch
import argparse
from antiberty import AntiBERTyRunner


def process_antiberty_embeddings(input_path: str, output_path: str):
    """
    处理输入文件夹下的每个抗体目录，提取 a3m 序列并生成 AntiBERTy embedding。
    """
    antiberty = AntiBERTyRunner()

    # 遍历输入路径下的每个文件夹
    for folder_name in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 从文件夹名中提取 antibody 名称
        antibody_name = folder_name.split("_")[0]
        a3m_file_path = os.path.join(folder_path, f"{antibody_name}.a3m")

        if not os.path.exists(a3m_file_path):
            print(f"[Warning] Missing A3M file: {a3m_file_path}")
            continue

        # 读取前两行
        with open(a3m_file_path, 'r') as f:
            first_two_lines = f.readlines()[:2]

        # 创建输出文件夹
        output_folder_path = os.path.join(output_path, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        # 生成临时 fasta 文件
        fasta_file_path = os.path.join(output_folder_path, f"{antibody_name}_antiberty.fasta")
        with open(fasta_file_path, 'w') as output_file:
            output_file.writelines(first_two_lines)
        print(f"[FASTA OK] {folder_name}")

        # 读取第二行序列
        with open(fasta_file_path, 'r') as fasta_file:
            lines = fasta_file.readlines()
            if len(lines) < 2:
                print(f"[Error] Invalid FASTA format in {fasta_file_path}")
                os.remove(fasta_file_path)
                continue
            sequences = [lines[1].strip()]

        # 生成嵌入
        embeddings = antiberty.embed(sequences)

        # 写入嵌入到输出文件
        embedding_file_path = os.path.join(output_folder_path, f"{antibody_name}_antiberty.txt")
        with open(embedding_file_path, 'w') as output_file:
            for embedding in embeddings:
                for i, sublist in enumerate(embedding.tolist()):
                    if i != 0 and i != len(embedding) - 1:  # 跳过首尾 token
                        embedding_str = ' '.join(map(str, sublist))
                        output_file.write(embedding_str + '\n')

        print(f"[Embedding OK] {folder_name}")

        # 删除中间生成的 fasta
        os.remove(fasta_file_path)
        print(f"[Removed] {fasta_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate AntiBERTy embeddings for antibody sequences.")
    parser.add_argument('-d', '--input_dir', type=str, default=None, help='the dir of a3m')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='the dir of out seq_embedding txt')
    args = parser.parse_args()

    process_antiberty_embeddings(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
