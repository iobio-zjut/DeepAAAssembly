import os, sys
import numpy as np
import argparse
import time
import torch
import random
import pickle as pkl
from data_generator import *
from utils import *
from model import *
from config import *

def bins2distmap(mul_class):
    mul_class = mul_class.transpose(1, 2, 0)
    L1 = mul_class.shape[0]
    L2 = mul_class.shape[1]
    _class = mul_class.shape[-1]
    if _class == 37:
        mul_thred = [20.0,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75,16.25,16.75,17.25,17.75,
        18.25,18.75,19.25,19.75]
    else:
        mul_thred = [0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.0, 15.1]

    mul_class_weighted = np.zeros((L1, L2, _class))
    for i in range(_class):
        mul_class_single = np.copy(mul_class[:,:,i])
        mul_class_single *= mul_thred[i]
        mul_class_weighted[:,:,i] = mul_class_single
    dist_from_mulclass = mul_class_weighted.sum(axis=-1)

    return dist_from_mulclass

def predict(pdb_list, testLoader, options):
    """
    Arguments:
        pdb_list   : id of target proteins -> list()
        testLoader : test_loader of test dataset -> iterator
        options    : parameters of input -> dict()
    Return:
        output the average precision into the specified file
    """


    if options.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{options.device}")
        print('Using the GPU to calculate the inter-protein contacts with DeepAAA')
    else:
        device = torch.device('cpu')
        if options.use_gpu:
            print('torch.cuda is not available. Using the CPU instead.')
        else:
            print('Using the CPU to calculate the inter-protein contacts with DeepAAA')

    model_config = config_parse_args(options)
    model = DeepAAA_Triangle(model_config)


    model.eval()
    with torch.no_grad():

        checkpoint = torch.load(os.path.join("models", options.model), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        for idx, (pdb, feats) in enumerate(testLoader):
            feats = create_variable(feats, options.use_gpu)

            pdb_name = options.target
            pdb_name1, pdb_name2 = pdb_name.split('_')

            data_path = options.data_path


            rec_antiberty = torch.from_numpy(np.loadtxt(os.path.join(data_path, pdb_name1 + '_antiberty.txt'))).float().to(device)  # (L1*512)

            # antibody
            voro_area1 = torch.tensor(np.load(os.path.join(data_path, pdb_name1 + '_voro_area.npz'))['voro_area'].astype(np.float64)).to(device)
            ab_voro_area = voro_area1.sum(dim=1)
            ab_voro_area = ab_voro_area.unsqueeze(1)

            solvent_area1 = torch.tensor(np.load(os.path.join(data_path, pdb_name1 + '_voro_area.npz'))['voro_solvent_area'].astype(np.float64)).to(device)
            ab_voro_solvent_area = solvent_area1.unsqueeze(1)

            voro_orien1 = torch.tensor(np.load(os.path.join(data_path, pdb_name1 + '_voro_normal.npz'))['normal'].astype(np.float64)).to(device)
            ab_voro_orien = voro_orien1.sum(dim=1)

            # antigen
            voro_area2 = torch.tensor(np.load(os.path.join(data_path, pdb_name2 + '_voro_area.npz'))['voro_area'].astype(np.float64)).to(device)
            ag_voro_area = voro_area2.sum(dim=1)
            ag_voro_area = ag_voro_area.unsqueeze(1)

            solvent_area2 = torch.tensor(np.load(os.path.join(data_path, pdb_name2 + '_voro_area.npz'))['voro_solvent_area'].astype(np.float64)).to(device)
            ag_voro_solvent_area = solvent_area2.unsqueeze(1)

            voro_orien2 = torch.tensor(np.load(os.path.join(data_path, pdb_name2 + '_voro_normal.npz'))['normal'].astype(np.float64)).to(device)
            ag_voro_orien = voro_orien2.sum(dim=1)


            y_pred = model(feats['rec1d'], feats['rec2d'], feats['lig1d'], feats['lig2d'], feats['com2d'],
                           feats['intra_distA'], feats['intra_distB'], rec_antiberty, ab_voro_area, ab_voro_solvent_area, ab_voro_orien, ag_voro_area, ag_voro_solvent_area, ag_voro_orien, None, feats['mask_sa'])
            y_pred1 = y_pred.view((y_pred.size()[0], y_pred.size()[1], -1))  # B*C*H*W->B*C*(H*W)
            act = nn.Softmax(1)
            y_pred_3D = act(y_pred1)

            y_pred2 = y_pred_3D.view_as(y_pred)
            prediction = np.array(y_pred2.cpu()).squeeze()

            if not os.path.exists(options.output_path + '/test_result'):
                os.makedirs(options.output_path + '/test_result')

            prediction_dist = bins2distmap(prediction)
            dist_inter_file = os.path.join(options.output_path + '/test_result',
                                           pdb_list[idx] + "_prediction.adist")
            np.savetxt(dist_inter_file, prediction_dist, fmt='%.4f')


if __name__ == "__main__" :

    # Input parameters
    parser = argparse.ArgumentParser(description = "Setting parameters")
    parser.add_argument('-mtype', '--model_type', type=str, default='hetero',help='Setting the model type')
    parser.add_argument('-m',  '--model',       type=str, default='hetero_param.pkl',help='Setting the file of model (homo_param.pkl or hetero_param.pkl)')
    parser.add_argument('-t',  '--target',      type=str, default=None, help='the id of target protein')
    parser.add_argument('-dp', '--data_path',   type=str, default='  ', help='the path of data files')
    parser.add_argument('-op', '--output_path', type=str, default='Outputs', help='the path of the output')
    parser.add_argument('-gpu', '--use_gpu',    type=bool, default=True, help='choose the gpu or cpu to inference')
    parser.add_argument('-dev', '--device',     type=int,  default=0, help='choose the gpu device')
    parser.add_argument('-n', '--ntop', type=str, default="all", help='output the top n predicted contacts')
    options = parser.parse_args()


    if options.target != None:

        pdb_list = [ options.target ]
        if options.model_type == "homo":
            test_dataset = DeepAAA_DataSet(homo_pdb = pdb_list, homo_files=[ os.path.join(options.data_path, options.target+'.npz') ] )
        else:
            test_dataset = DeepAAA_DataSet(hetero_pdb = pdb_list, hetero_files=[ os.path.join(options.data_path, options.target+'.npz') ] )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        predict(pdb_list, test_loader, options)

