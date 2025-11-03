import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Module import *

# from torchvision.transforms import Resize\
class MCNN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(MCNN_Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MCNN(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=True):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(MCNN_Conv2d(64, 32, 9, same_padding=True, bn=bn),
                                     #  nn.MaxPool2d(2),
                                     MCNN_Conv2d(32, 64, 7, same_padding=True, bn=bn),
                                     #  nn.MaxPool2d(2),
                                     MCNN_Conv2d(64, 32, 7, same_padding=True, bn=bn),
                                     MCNN_Conv2d(32, 16, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(MCNN_Conv2d(64, 40, 7, same_padding=True, bn=bn),
                                     #  nn.MaxPool2d(2),
                                     MCNN_Conv2d(40, 80, 5, same_padding=True, bn=bn),
                                     #  nn.MaxPool2d(2),
                                     MCNN_Conv2d(80, 40, 5, same_padding=True, bn=bn),
                                     MCNN_Conv2d(40, 20, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(MCNN_Conv2d(64, 56, 5, same_padding=True, bn=bn),
                                     #  nn.MaxPool2d(2),
                                     MCNN_Conv2d(56, 112, 3, same_padding=True, bn=bn),
                                     #  nn.MaxPool2d(2),
                                     MCNN_Conv2d(112, 56, 3, same_padding=True, bn=bn),
                                     MCNN_Conv2d(56, 28, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(MCNN_Conv2d(30, 1, 1, same_padding=True, bn=bn))
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, im_data):
        identity = im_data
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        # print('x shape:',x.shape)
        # x = self.fuse(x)

        return self.act(x + identity)
        # return x


class DeepAAA_middle(nn.Module):

    def __init__(self, model_args):
        super(DeepAAA_middle, self).__init__()

        args1d = model_args['BasicBlock1D']
        args2d = model_args['BasicBlock2D']

        """******************DMNet*******************"""
        # self.DMNet = DMNet(64,32)
        """******************DMNet*******************"""

        """******************MCNN*******************"""
        self.mcnns = nn.Sequential(
            MCNN(bn=True),
            MCNN(bn=True),
            MCNN(bn=True),
            MCNN(bn=True)
        )
        """******************MCNN*******************"""

        self.identity1 = conv_identity_2d(args1d['InChannels'] * 2, args1d['Channels'][0], 1, 1, bias=False)
        self.identity2 = conv_identity_2d(args2d['InChannels'], args2d['Channels'][0], 1, 1, bias=False)
        self.identity3 = conv_identity_2d(args2d['Channels'][0] * 2, args2d['Channels'][0], 1, 1, bias=False)
        # self.layer2 = self._make_layer(conv2d, args2d)

        # output
        if model_args['dist'] == True:
            self.conv = conv2d(args2d['Channels'][-1], model_args['dist_bins'], 1, 1)
            self.acti = nn.Softmax(1)
        else:
            self.conv = conv2d(args2d['Channels'][-1], 1, 1, 1)
            self.acti = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    # downsample
    def _downsample(self, conv, in_channels, out_channels, stride):

        if in_channels == out_channels and stride == 1:
            return None
        else:
            return nn.Sequential(conv(in_channels, out_channels, kernel_size=1, stride=stride))

    # make layers
    def _make_layer(self, fn, config):

        conv = fn
        Block = config['name']
        Num_Blocks = len(config['Channels'])
        Block_Cycle = config['num_Cycle']
        in_channels = config['Channels'][0]
        out_channels = config['Channels']
        kernel_size = config['Kernel_size']
        dilations = config['Dilation']
        group = config['Group']
        bias = config['Bias']
        track_running_stats = config['track_running_stats']
        stride = 1
        padding = "same"

        layers = []
        for i in range(Num_Blocks):

            n_dilation = len(dilations)
            dilation = dilations[i % n_dilation]

            if i == 0:
                downsample = self._downsample(conv, in_channels, out_channels[0], stride)
                layers.append(
                    Block(in_channels, out_channels[0], kernel_size, stride, downsample, padding, dilation, group, bias,
                          track_running_stats))

                for j in range(1, Block_Cycle):
                    layers.append(
                        Block(out_channels[0], out_channels[0], kernel_size, stride, None, padding, dilation, group,
                              bias, track_running_stats))
            else:
                downsample = self._downsample(conv, out_channels[i - 1], out_channels[i], stride)
                layers.append(
                    Block(out_channels[i - 1], out_channels[i], kernel_size, stride, downsample, padding, dilation,
                          group, bias, track_running_stats))

                for j in range(1, Block_Cycle):
                    layers.append(
                        Block(out_channels[i], out_channels[i], kernel_size, stride, None, padding, dilation, group,
                              bias, track_running_stats))

        return nn.Sequential(*layers)

    def forward(self, rec1d, lig1d, com2d):

        pair1 = seq2pairwise_v3(rec1d, lig1d) #(1638, L1, L1)
        pair1 = self.identity1(pair1)

        pair2 = self.identity2(com2d)

        pair = torch.cat([pair1, pair2], dim=1)
        pair = self.identity3(pair)
        out = self.mcnns(pair)

        out_act = self.conv(out)
        out_act = self.acti(out_act)

        return out, out_act


class DeepAAA_Triangle(nn.Module):

    def __init__(self, model_config):
        super(DeepAAA_Triangle, self).__init__()

        self.linear_layer_1 = nn.Linear(512, 64)
        self.linear_layer_2 = nn.Linear(5, 64)

        self.model_args = model_config['model_args']
        self.rec_args = model_config['rec_args']
        args1d = self.model_args['BasicBlock1D']
        args2d = self.model_args['BasicBlock2D']
        self.triangle_args = model_config['triangle_args']

        # 添加了新的rec_args
        self.resnet_rec = DeepAAA_middle(self.rec_args)
        self.resnet_lig = DeepAAA_middle(self.model_args)
        self.resnet_com = DeepAAA_middle(model_config['triangle_conv_args'])

        self.identity_com = conv_identity_2d(args1d['InChannels'] + args2d['InChannels'],
                                             self.triangle_args['Channel_z'], 1, 1, bias=False)

        self.TriangleMulti = nn.ModuleList(
            [TriangleMultiplication(self.triangle_args) for _ in range(self.triangle_args['num_TriangleMulti'])])
        self.TriangleSelfR = nn.ModuleList(
            [TriangleSelfAttention(self.triangle_args) for _ in range(self.triangle_args['num_TriangleSelfR'])])
        if self.triangle_args['num_TriangleSelfC'] > 0:
            self.TriangleSelfC = nn.ModuleList(
                [TriangleSelfAttention(self.triangle_args) for _ in range(self.triangle_args['num_TriangleSelfC'])])
        self.Transition = nn.ModuleList(
            [Transition(self.triangle_args) for _ in range(self.triangle_args['num_Transition'])])

        self.norm_final = nn.LayerNorm(self.triangle_args['final'])
        if self.triangle_args['dist'] == True:
            self.Linear_final = nn.Linear(self.triangle_args['final'], self.triangle_args['dist_bins'])
        else:
            self.Linear_final = nn.Linear(self.triangle_args['final'], 1)

        if self.triangle_args['dist'] == False:
            self.act = nn.Sigmoid()

        self.drop = nn.Dropout(0.10)

    def forward(self, rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig, anti_data_ori1, ab_voro_area, ab_voro_solvent_area, ab_voro_orien, ag_voro_area, ag_voro_solvent_area, ag_voro_orien, mask=None, mask_sa=None):

        out_anti_data_ori1 = self.linear_layer_1(anti_data_ori1)
        anti_data_ori1 = torch.transpose(out_anti_data_ori1, 0, 1)
        anti_data_ori1 = anti_data_ori1.unsqueeze(0)  # 1,64,L
        append_data1 = torch.cat([ab_voro_area, ab_voro_solvent_area, ab_voro_orien], dim=1)
        out_append_data1 = self.linear_layer_2(append_data1.float())
        out_append_data1 = out_append_data1.unsqueeze(0)
        out_append_data1 = out_append_data1.permute(0, 2, 1)  # 现在维度为 (1, 64, L)
        final_append_data1 = torch.cat([anti_data_ori1, out_append_data1], dim=1)
        rec1d = torch.cat([rec1d[:, :819, :], final_append_data1], dim=1).float()

        append_data2 = torch.cat([ag_voro_area, ag_voro_solvent_area, ag_voro_orien], dim=1)
        out_append_data2 = self.linear_layer_2(append_data2.float())
        out_append_data2 = out_append_data2.unsqueeze(0)
        out_append_data2 = out_append_data2.permute(0, 2, 1)  # 现在维度为 (1, C, L)
        lig1d = torch.cat([lig1d[:, :819, :], out_append_data2], dim=1).float()


        # intra_contact/intra_distanc
        rec2d, rec2d_pred = self.resnet_rec(rec1d, rec1d, rec2d)
        lig2d, lig2d_pred = self.resnet_lig(lig1d, lig1d, lig2d)
        rec2d = rec2d.permute(0, 2, 3, 1)
        lig2d = lig2d.permute(0, 2, 3, 1)

        # inter_contact/inter_distance
        z_com, _ = self.resnet_com(rec1d, lig1d, com2d)
        z_com = z_com.permute(0, 2, 3, 1)

        mask = None
        for idx in range(self.triangle_args['num_TriangleMulti']):
            z_com = z_com + self.drop(self.TriangleMulti[idx](z_com, rec2d, lig2d, mask, mask_sa.permute(0, 2, 3, 1)))
            z_com = z_com + self.drop(self.TriangleSelfR[idx](z_com, intra_lig, mask, mask_sa))

            if self.triangle_args['num_TriangleSelfC'] > 0:
                _mask = None
                _mask_sa = None if mask_sa == None else mask_sa.permute(0, 1, 3, 2)
                z_com_T = z_com.permute(0, 2, 1, 3)
                z_com_T = self.TriangleSelfC[idx](z_com_T, intra_rec, _mask, _mask_sa)
                z_com = z_com + self.drop(z_com_T.permute(0, 2, 1, 3))
            z_com = z_com + self.Transition[idx](z_com)

        z_final_norm = self.norm_final(z_com)
        if self.triangle_args['dist'] == True:
            z_final = self.Linear_final(z_final_norm)
            # print('z_final:no softmax,no sigmoid,train for bins.')
        else:
            z_final = self.act(self.Linear_final(z_final_norm))
            # print('z_final:use sigmoid,train for binary.')
        return z_final.permute(0, 3, 1, 2)

