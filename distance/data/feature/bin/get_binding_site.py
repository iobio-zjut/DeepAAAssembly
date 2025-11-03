import os
import sys
import h5py
import json
import numpy as np
import torch as pt
import argparse

# 当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 添加 bin/PeSTo 到 sys.path
pe_sto_dir = os.path.join(SCRIPT_DIR, "PeSTo")
if pe_sto_dir not in sys.path:
    sys.path.append(pe_sto_dir)

# save_path 相对于 SCRIPT_DIR 动态指定
save_path = os.path.join(pe_sto_dir, "model", "save", "i_v4_1_2021-09-07_11-21")
if save_path not in sys.path:
    sys.path.append(save_path)

from tqdm import tqdm
from src.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_interface_types
from src.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ
from src.structure import data_to_structure, encode_bfactor, concatenate_chains, split_by_chain
from src.structure_io import save_pdb, read_pdb





# select saved model
model_filepath = os.path.join(save_path, 'model_ckpt.pt')
# add module to path
if save_path not in sys.path:
    sys.path.insert(0, save_path)

# load functions
from config import config_model

from model import Model

parser = argparse.ArgumentParser(description = "Setting parameters")
parser.add_argument('-t',  '--target',      type=str, default=None, help='the target name')
parser.add_argument('-d',  '--pdb_dir',      type=str, default=None, help='the dir of pdbs')
parser.add_argument('-o',  '--out_txtdir',      type=str, default=None, help='the dir of out pesto txt')
parser.add_argument('-gpu', '--use_gpu',    type=bool, default=True, help='choose the gpu or cpu to inference')

options = parser.parse_args()
target=options.target
pdb_dir_path = options.pdb_dir
txt_folder = options.out_txtdir

# %%

# create model
model = Model(config_model)

# reload model
if options.use_gpu == False:
    # print('PeSTo using the CPU to predict')
    model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))
    device = pt.device("cpu")
if options.use_gpu == True:
    if pt.cuda.is_available():
        # print('PeSTo using the GPU to predict')
        model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cuda")))
        device = pt.device("cuda:0")
    else:
        print('The state of torch.cuda', pt.cuda.is_available())
        exit(1)

# set model to inference
model = model.eval().to(device)
# find pdb files and ignore already predicted oins
pdb_filepaths=[]
pdb_filepaths.append(os.path.join(pdb_dir_path, target+"_renum.pdb"))
pdb_filepaths = [fp for fp in pdb_filepaths if "_i" not in fp]


# create dataset loader with preprocessing
dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

# %%
# run model on all subunits
with pt.no_grad():
    for subunits, filepath in tqdm(dataset):
        # concatenate all chains together
        structure = concatenate_chains(subunits)

        # encode structure and features
        X, M = encode_structure(structure)
        # q = pt.cat(encode_features(structure), dim=1)
        q = encode_features(structure)[0]

        # extract topology
        ids_topk, _, _, _, _ = extract_topology(X, 64)

        # pack data and setup sink (IMPORTANT)
        X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

        # run model
        z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

        # for all predictions
        for i in range(z.shape[1]):
            # prediction
            p = pt.sigmoid(z[:, i])

            # encode result
            structure = encode_bfactor(structure, p.cpu().numpy())

            # save results
            output_pdbpath = filepath[:-4] + '_i{}.pdb'.format(i)
            save_pdb(split_by_chain(structure), output_pdbpath)

        # 设置文件路径
        pdb_withprob_path = filepath[:-4]+'_i0.pdb'
        prob_list = []
        # 打开文件
        with open(pdb_withprob_path, 'r') as file:
            # 读取第一行
            first_line = file.readline()

            if len(first_line) >= 80:
                # 截取第24到第27个字符转为int类型，作为atom_idx
                atom_idx = int(first_line[23:26])

                # 截取第63到第67个字符转为float类型，作为contact_prob
                contact_prob = round(float(first_line[62:66]), 2)
                prob_list.append(contact_prob)

            else:
                print(pdb_withprob_path, 'first line not have enough length!')
                exit(1)

            # 遍历剩余的行
            for line in file:
                if len(line) >= 80:
                    # 截取第24到第27个字符转为int类型，作为当前行的atom_idx
                    current_atom_idx = int(line[23:26])

                    # 检查是否与目前的atom_idx相同
                    if current_atom_idx == atom_idx:
                        # 如果相同，则跳过当前行
                        continue
                    else:
                        # 如果不同，则截取第63到第67个字符转为float类型，更新contact_prob
                        contact_prob = round(float(line[62:66]), 2)
                        atom_idx = current_atom_idx
                        prob_list.append(contact_prob)
            if len(prob_list) != atom_idx:
                print(pdb_withprob_path, 'len(prob_list)!=atom_idx,need check!!!')
            else:
                os.makedirs(txt_folder,exist_ok=True)
                output_txtpath = os.path.join(txt_folder, target+'_pesto.txt')
                # 将prob_list写入文件
                with open(output_txtpath, 'w') as file:
                    for prob in prob_list:
                        file.write(f"{prob:.2f}\n")

            for i in range(5):
                os.remove(filepath[:-4] + '_i{}.pdb'.format(i))
