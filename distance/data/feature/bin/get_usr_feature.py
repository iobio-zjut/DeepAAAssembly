import os
import sys
import numpy as np
from Bio.PDB import *

def extract_USR(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_file)

    # 获取结构中的所有残基
    residues = list(structure.get_residues())
    hang = len(residues)

    # 获取距离矩阵
    distance = get_dist_from_pdbfile(pdb_file, hang)

    avg1 = []
    avg2 = []
    avg3 = []

    for i in range(hang):
        avg1.append(np.average(distance[i]))

        # 计算与当前残基最远的残基的距离平均值
        idx2 = np.argmax(distance, axis=1)
        avg2.append(np.average(distance[idx2[i]]))

        # 计算与当前残基最远的残基的最远残基的距离平均值
        idx3 = np.argmax(distance[idx2[i]], axis=0)
        avg3.append(np.average(distance[idx3]))

    usr = np.vstack((avg1, avg2, avg3))
    return usr

def get_dist_from_pdbfile(pdb_file, length):
    seq_name = os.path.basename(pdb_file).split('.')[0]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(seq_name, pdb_file)
    model = structure[0]
    chain_id = list(model.child_dict.keys())
    chain = model[chain_id[0]]

    dist_map = np.zeros((length, length))

    L = len(chain)
    if L != length:
        print('Length error in %s, expected %s, got %s' % (pdb_file, length, L))
        return dist_map

    chain_dict = list(chain.child_dict.keys())
    for i in range(0, length):
        for j in range(i, length):
            if i == j:
                continue
            residue_i = chain[chain_dict[i]]
            residue_j = chain[chain_dict[j]]
            if "CB" in residue_i:
                ca_i = residue_i["CB"]
            elif "CA" in residue_i:
                ca_i = residue_i["CA"]
            elif "N" in residue_i:
                ca_i = residue_i["N"]
            else:
                dist_list = []
                for atom_i in residue_i:
                    for atom_j in residue_j:
                        if (
                                'C' in atom_i.name or 'N' in atom_i.name or 'O' in atom_i.name or 'S' in atom_i.name) and \
                                (
                                        'C' in atom_j.name or 'N' in atom_j.name or 'O' in atom_j.name or 'S' in atom_j.name):
                            dist_list.append(atom_i - atom_j)
                        else:
                            continue
                caca_dist = np.min(dist_list)
                dist_map[i, j] = caca_dist
                continue

            if "CB" in residue_j:
                ca_j = residue_j["CB"]
            elif "CA" in residue_j:
                ca_j = residue_j["CA"]
            elif "N" in residue_j:
                ca_j = residue_j["N"]
            else:
                dist_list = []
                for atom_i in residue_i:
                    for atom_j in residue_j:
                        if (
                                'C' in atom_i.name or 'N' in atom_i.name or 'O' in atom_i.name or 'S' in atom_i.name) and \
                                (
                                        'C' in atom_j.name or 'N' in atom_j.name or 'O' in atom_j.name or 'S' in atom_j.name):
                            dist_list.append(atom_i - atom_j)
                        else:
                            continue
                caca_dist = np.min(dist_list)
                dist_map[i, j] = caca_dist
                continue
            caca_dist = ca_i - ca_j
            dist_map[i, j] = caca_dist
    dist_map += dist_map.T
    return dist_map

def process_pdb_files(input_dir, output_dir, target):
    usr_features = extract_USR(input_dir)
    output_file = os.path.join(output_dir, f"{target}_USR.txt")
    np.savetxt(output_file, usr_features, fmt="%f")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_usr_feature.py <target> <input_path> <output_path>")
        sys.exit(1)

    target=sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]


    # If input_path is a single PDB file, process that file
    input_path=os.path.join(input_path,target+'_renum.pdb')
    process_pdb_files(input_path, output_path, target)

