import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 2., 22., 64
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count

    D = D.transpose(1,2,0)
    RBF = np.exp(-((D - D_mu) / D_sigma)**2)
    return RBF.transpose(2,0,1)


def generate_features_homo(pdb, data, max_len=256, dist=False, train=True):

    ret = { 'rec1d' : data['rec1d'],
            'rec2d' : data['rec2d'],
            'rec_sa' : data['recsa'],
            'intra_distA' : data['intra_distA'],

    }

    seq_len = ret['intra_distA'].shape[-1]

    # mask for rec, lig and com
    mask_rec = np.ones((1, seq_len))
    diagA = ret['intra_distA'][:, np.arange(seq_len), np.arange(seq_len)]
    mask_rec[ np.where(diagA == -1.0) ] = 0.0
    mask_lig = mask_rec


    ret['mask_rec'] = mask_rec
    ret['mask_lig'] = mask_lig
    ret['mask_com'] = mask_rec[:,:,None] * mask_lig[:,None,:] 

    # mask for SA by the all_atom
    rec_sa = data['recsa']
    rec_sa = np.expand_dims(rec_sa[:,0].reshape(seq_len, 1), 0)
    rec_sa = np.where(rec_sa>0, 1.0, 0.0)
    ret['mask_sa'] = rec_sa * rec_sa.transpose(0,2,1)


    ret['com2d'] = ret['rec2d']
    ret['intra_distA_gs'] = rbf(ret['intra_distA'])
    ret['rec2d'] = np.concatenate([ret['rec2d'], ret['intra_distA_gs']], axis=0)
    ret['lig2d'] = ret['rec2d']


    ret['lig1d'] = ret['rec1d']
    ret['intra_distB'] = ret['intra_distA']

    return ret

def generate_features_hetero(pdb, data, max_len=256, dist=False, train=True):

    # 在此读取更新后的重轻链对应特征 接收 data 数据字典并提取其中的特征数据（读取npz文件中的特征）
    ret = { 'rec1d' : data['rec1d'],
            'rec2d' : data['rec2d'],
            'lig1d' : data['lig1d'],
            'lig2d' : data['lig2d'],
            'com2d' : data['com2d'],
            'intra_distA' : data['intra_distA'],
            'intra_distB' : data['intra_distB'],
    }


    seq_lenA = ret['intra_distA'].shape[-1]
    seq_lenB = ret['intra_distB'].shape[-1]

    # mask for rec, lig, and com
    mask_rec = np.ones((1, seq_lenA))
    diagA = ret['intra_distA'][:, np.arange(seq_lenA), np.arange(seq_lenA)]
    mask_rec[ np.where(diagA == -1.0) ] = 0.0

    mask_lig = np.ones((1, seq_lenB))
    diagB = ret['intra_distB'][:, np.arange(seq_lenB), np.arange(seq_lenB)]
    mask_lig[ np.where(diagB == -1.0) ] = 0.0

    mask_com = mask_rec[:,:,None] * mask_lig[:,None,:]

    ret['mask_rec'] = mask_rec
    ret['mask_lig'] = mask_lig
    ret['mask_com'] = mask_com

    # mask for SA by the all_atom
    rec_sa = data['recsa']
    rec_sa = np.expand_dims(rec_sa[:,0].reshape(seq_lenA, 1), 0)
    rec_sa = np.where(rec_sa>0, 1.0, 0.0)

    lig_sa = data['ligsa']
    lig_sa = np.expand_dims(lig_sa[:,0].reshape(seq_lenB, 1), 0).transpose(0,2,1)
    lig_sa = np.where(lig_sa>0, 1.0, 0.0)

    ret['mask_sa'] = rec_sa * lig_sa

    ret['intra_distA_gs'] = rbf(ret['intra_distA'])
    ret['intra_distB_gs'] = rbf(ret['intra_distB'])
    ret['rec2d'] = np.concatenate([ret['rec2d'], ret['intra_distA_gs']], axis=0)
    ret['lig2d'] = np.concatenate([ret['lig2d'], ret['intra_distB_gs']], axis=0)

    return ret


class DeepAAA_DataSet(Dataset):

    def __init__(self, homo_pdb=None, homo_files=None, hetero_pdb=None, hetero_files=None, dist=False, is_train=False):

        if homo_pdb == None:
            homo_pdb = []
            homo_files = []
        if hetero_pdb == None:
            hetero_pdb = []
            hetero_files = []

        self.data_pdb = homo_pdb + hetero_pdb
        self.data_files = homo_files + hetero_files
        self.data_sym = [ 1 for i in range(len(homo_pdb)) ] + [ 0 for i in range(len(hetero_pdb))]
        self.dist = dist
        self.is_train = is_train
        self.len =  len(self.data_pdb)

    def __getitem__(self, index):
        pdb = self.data_pdb[index]
        file = self.data_files[index]
        sym = self.data_sym[index]
        data = np.load(file)     

        if sym == 1:
            return [pdb, generate_features_homo(pdb, data, 256, self.dist, self.is_train) ]
        else:
            return [pdb, generate_features_hetero(pdb, data, 256, self.dist, self.is_train) ]

    def __len__(self):

        return self.len


