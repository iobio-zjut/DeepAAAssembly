import os, sys, glob, re, platform
import torch
from torch.autograd import Variable
import numpy as np
from Bio.PDB.PDBParser import PDBParser

def chkdirs(fn):
    if not fn.endswith('/'):
        fn += '/'
    dn = os.path.dirname(fn)
    if not os.path.exists(dn):
        os.makedirs(dn)

def create_variable(tensor, use_gpu=False):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available() and use_gpu:
        if isinstance(tensor, dict):
            for key, value in tensor.items():
                if not isinstance(value, list):
                    tensor[key] = Variable(tensor[key].float().cuda())
        else :
            if not isinstance(value, list):
                Variable(tensor.float().cuda())

        return tensor

    else:
        if isinstance(tensor, dict):
            for key, value in tensor.items():
                if not isinstance(value, list):
                    tensor[key] = Variable(tensor[key].float())
        else :
            if not isinstance(value, list):
                Variable(tensor.float())

        return tensor

def read_single_fasta_file(fasta_file):
    with open(fasta_file, 'r') as file:
        fasta_lines = file.readlines()
        if len(fasta_lines) >= 2:
            fasta = fasta_lines[1].strip()
    return fasta


def get_inter_dist(pdb_file):
    '''这个脚本用于对pdb的两条或三条链没有进行重新编号,没有变换链ID,不提供length'''
    seq_name = os.path.basename(pdb_file).split('.')[0]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(seq_name, pdb_file)
    model = structure[0]
    chain_id = list(model.child_dict.keys())


    if len(chain_id) == 3:
        chain1 = model[chain_id[0]]
        chain1_dict = list(chain1.child_dict.keys())
        chain2 = model[chain_id[1]]
        chain2_dict = list(chain2.child_dict.keys())
        chain3 = model[chain_id[2]]
        chain3_dict = list(chain3.child_dict.keys())
        print("chain1 : ", chain_id[0])
        print("chain2 : ", chain_id[1])
        print("chain3 : ", chain_id[2])
        flag = True
    else:
        chain1 = model[chain_id[0]]
        chain2 = model[chain_id[1]]
        chain1_dict = list(chain1.child_dict.keys())
        chain2_dict = list(chain2.child_dict.keys())
        print("chain1 : ", chain_id[0])
        print("chain2 : ", chain_id[1])
        flag = False

    if flag:

        dist_map = np.zeros((len(chain1_dict)+len(chain2_dict), len(chain3_dict)))
        print("len(chain1_dict): ", len(chain1_dict))
        print("len(chain2_dict): ", len(chain2_dict))
        print("len(chain3_dict): ", len(chain3_dict))
        for i in range(0, len(chain1_dict)):
            residue_i = chain1[chain1_dict[i]]
            true_index_i = i
            for j in range(0, len(chain3_dict)):
                residue_j = chain3[chain3_dict[j]]
                true_index_j = j
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
                    dist_map[true_index_i, true_index_j] = caca_dist
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
                    dist_map[true_index_i, true_index_j] = caca_dist
                    continue
                caca_dist = ca_i - ca_j
                dist_map[true_index_i, true_index_j] = caca_dist

        for i in range(0, len(chain2_dict)):
            residue_i = chain2[chain2_dict[i]]
            true_index_i = len(chain1_dict)+i
            for j in range(0, len(chain3_dict)):
                residue_j = chain3[chain3_dict[j]]
                true_index_j = j
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
                    dist_map[true_index_i, true_index_j] = caca_dist
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
                    dist_map[true_index_i, true_index_j] = caca_dist
                    continue
                caca_dist = ca_i - ca_j
                dist_map[true_index_i, true_index_j] = caca_dist
    else:
        dist_map = np.zeros((len(chain1_dict), len(chain2_dict)))
        print("len(chain1_dict): ", len(chain1_dict))
        print("len(chain2_dict): ", len(chain2_dict))
        for i in range(0, len(chain1_dict)):
            residue_i = chain1[chain1_dict[i]]
            true_index_i = i
            for j in range(0, len(chain2_dict)):
                residue_j = chain2[chain2_dict[j]]
                true_index_j = j
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
                    dist_map[true_index_i, true_index_j] = caca_dist
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
                    dist_map[true_index_i, true_index_j] = caca_dist
                    continue
                caca_dist = ca_i - ca_j
                dist_map[true_index_i, true_index_j] = caca_dist
    return dist_map


