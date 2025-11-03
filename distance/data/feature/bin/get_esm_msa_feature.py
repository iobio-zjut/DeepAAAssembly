#
# This file (esm_msa_feature.py) is modified by the ESM-MSA example code
# https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
#
import os, sys
import esm
import torch
import argparse
import string
import itertools
import numpy as np
import pickle as pkl
from Bio import SeqIO
from typing import List, Tuple

# from torch.cuda.amp import autocast as autocast

torch.set_num_threads(8)


# read the Multiple Sequence Alignment (MSA)
def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


# load model
def load_esm(path):
    model, alphabet = esm.pretrained.load_model_and_alphabet(path)
    # model = model.eval().cuda()
    # model = model.to('cuda:0')
    model = model.to('cpu')
    batch_converter = alphabet.get_batch_converter()

    return model, batch_converter


# inference of ESM-MSA-1b
def get_esm_msa_feats(esm1b, esm1b_batch_converter, seq_list):
    # convert the sequence to tokens
    esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(seq_list)
    # esm1b_batch_tokens = esm1b_batch_tokens.to('cuda:0')
    esm1b_batch_tokens = esm1b_batch_tokens.to('cpu')
    with torch.no_grad():
        results = esm1b(esm1b_batch_tokens, repr_layers=[12], return_contacts=True)

    # esm-msa-1b sequence representation
    token_representations = results["representations"][12].mean(1)

    sequence_representations = []
    for i, seq in enumerate(seq_list):
        sequence_representations.append(np.array(token_representations[i, 1: len(seq[0][1]) + 1].cpu()))

    # return the esm-msa-1d and row-attentions
    return sequence_representations[0], np.squeeze(np.array(results['row_attentions'].cpu()))[:, :, 1:, 1:]


def generate_data_from_file(model_path, data_path, target):
    # load model and read msa
    esm1b, esm1b_batch_converter = load_esm(model_path)
    file_path = os.path.join(data_path, target + "_filter.a3m")

    with open(file_path, 'r') as file:
        # 读取文件的第二行
        file.readline()  # 跳过第一行
        second_line = file.readline()
    if len(second_line) <= 200:
        msa_data = [read_msa(os.path.join(data_path, target + "_filter.a3m"), 512)]
    elif 200 < len(second_line) <= 450:
        msa_data = [read_msa(os.path.join(data_path, target + "_filter.a3m"), 384)]
    else:
        print("***esm msa 1b only use 256 sequence***")
        msa_data = [read_msa(os.path.join(data_path, target + "_filter.a3m"), 256)]

    # inference
    esm_msa_1d, row_attentions = get_esm_msa_feats(esm1b, esm1b_batch_converter, msa_data)
    data = {'esm_msa_1d': esm_msa_1d, 'row_attentions': row_attentions}

    # save into pkl file
    with open(os.path.join(data_path, target + "_esm_msa.pkl"), 'wb') as f:
        pkl.dump(data, f, protocol=3)


if __name__ == "__main__":

    # translation for read sequence
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    # input args
    if len(sys.argv) != 4:
        print("USAGE ERROR")
        print("python esm_msa_feature.py model_path data_path target ")
        print("")
        exit()

    # generate the esm-msa features
    generate_data_from_file(sys.argv[1], sys.argv[2], sys.argv[3])



