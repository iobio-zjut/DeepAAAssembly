import os, sys, glob, re, platform
import time
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import warnings
import json
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Chain
import os
from Bio.PDB import PDBParser, PDBIO, Structure, Model
from Bio.PDB.Chain import Chain

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


def combine_chains(chain1, chain2, new_chain):
    """
    Combine three chains into one.

    Parameters:
    chain1 (Chain): The first chain to combine.
    chain2 (Chain): The second chain to combine.
    chain3 (Chain): The third chain to combine.
    new_chain (Chain): The empty chain to which residues will be added.

    Returns:
    Chain: The combined chain.
    """
    max_residue_id = max(residue.id[1] for residue in chain1)

    # Add residues from chain1 to new_chain
    for residue in chain1:
        new_chain.add(residue)

    # Add residues from chain2 to new_chain with adjusted residue ids
    for residue in chain2:
        new_residue_id = (residue.id[0], residue.id[1] + max_residue_id, residue.id[2])
        new_residue = residue.copy()
        new_residue.id = new_residue_id
        new_chain.add(new_residue)

    return new_chain