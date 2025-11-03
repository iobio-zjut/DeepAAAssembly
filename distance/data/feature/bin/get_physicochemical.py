import os, sys
import numpy as np
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def seq_property(target,a3m_dir_path,txt_folder):
    with open(a3m_dir_path, 'r') as file:
        lines = file.readlines()
    sequence = lines[1].strip()
    length = len(sequence)
    result_array = np.zeros((7, length), dtype=float)
    # 字典映射
    AA_to_num = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
                 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    all_for_assign = np.loadtxt(os.path.join(SCRIPT_DIR, "all_assign.txt"))
    # 遍历序列变量
    for seq_index, aa in enumerate(sequence):
        # 检查aa是否在AA_to_num中
        if aa not in AA_to_num:
            continue
        # 获取对应的数字
        row_match = AA_to_num[aa]
        # 将数组的对应位置置为匹配值
        result_array[:, seq_index] = np.transpose(all_for_assign[row_match,:])
    embedding_file = f'{txt_folder}/{target}_seqproperty.txt'
    with open(embedding_file, "w") as file:
        pass
    np.savetxt(embedding_file, result_array)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setting parameters")
    parser.add_argument('-t', '--target', type=str, default=None, help='the target name')
    parser.add_argument('-d', '--a3m_dir', type=str, default=None, help='the dir of a3m')
    parser.add_argument('-o', '--out_txtdir', type=str, default=None, help='the dir of out seq_embedding txt')

    options = parser.parse_args()
    target = options.target
    a3m_dir_path = os.path.join(options.a3m_dir, target + '.a3m')
    txt_folder = options.out_txtdir
    seq_property(target, a3m_dir_path, txt_folder)







