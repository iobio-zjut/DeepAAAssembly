import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

""" conv1d """
def conv1d( in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: str = "same", 
            dilation: int = 1, 
            group: int = 1, 
            bias: bool = False) -> nn.Conv1d:

    if padding == "same":
        padding = int((kernel_size - 1)/2)

    return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

""" conv2d """
def conv2d(in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: str = "same", 
            dilation: int = 1, 
            group: int = 1, 
            bias: bool = False) -> nn.Conv2d:

    if padding == "same":
        padding = int((kernel_size - 1)/2)

    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

""" conv1d 1x1 """
def conv_identity_1d( in_channels  : int,
                      out_channels : int,
                      kernel_size  : int = 1,
                      stride       : int = 1,
                      padding      : str = "same",
                      dilation     : int = 1,
                      group        : int = 1,
                      bias         : bool = False,
                      norm         : str = "IN",
                      activation   : str = "Relu",
                      track_running_stats_ : bool = True):
    layers = []

    # convolution
    layers.append( conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias))

    # normalization
    if norm == "BN":
       layers.append( nn.BatchNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_))
    elif norm == "IN":
        layers.append( nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_))
       
    # activation
    if activation == "ELU":
        layers.append( nn.ELU())
    elif activation == "Relu":
        layers.append(nn.LeakyReLU(negative_slope=0.01,inplace=True))

    return nn.Sequential(*layers)

""" conv2d 1x1"""
def conv_identity_2d( in_channels  : int,
                      out_channels : int,
                      kernel_size  : int = 1,
                      stride       : int = 1,
                      padding      : str = "same",
                      dilation     : int = 1,
                      group        : int = 1,
                      bias         : bool = False,
                      norm         : str = "IN",
                      activation   : str = "Relu",
                      track_running_stats_ : bool = True):
    layers = []

    # convolution
    layers.append(conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias))

    # normalization
    if norm == "BN":
       layers.append( nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_))
    elif norm == "IN":
        layers.append( nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_))

    # activation
    if activation == "ELU":
        layers.append( nn.ELU())
    elif activation == "Relu":
        layers.append( nn.LeakyReLU(negative_slope=0.01,inplace=True))

    return nn.Sequential(*layers)

""" ResNetv2 BasicBlock1D """
class BasicBlock_ResNetV2_1D(nn.Module):

    def __init__(self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : int,
        stride       : int = 1,
        downsample = None,
        padding      : str = "same",
        dilation     : int = 1,
        group        : int = 1,
        bias         : bool = False,
        track_running_stats_ : bool = True,
        norm         : str = "BN",
        activation   : str = "ELU"):

        super(BasicBlock_ResNetV2_1D, self).__init__()

        if norm == "BN":
            self.bn1 = nn.BatchNorm1d(in_channels, affine=True, track_running_stats=track_running_stats_)
            self.bn2 = nn.BatchNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm1d(in_channels, affine=True, track_running_stats=track_running_stats_)
            self.bn2 = nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_)

        if activation == "ELU":
            self.relu1 = nn.ELU()
            self.relu2 = nn.ELU()
        elif activation == "Relu":
            self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
            self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.conv1 = conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        self.conv2 = conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

        self.downsample = downsample

    def forward(self, x):

        identity = x

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample != None :
            identity = self.downsample(identity)

        x += identity

        return x


class Res2NetBlock(nn.Module):
    def __init__(self, inplanes, outplanes, scales=4):
        super(Res2NetBlock, self).__init__()

        if outplanes % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')

        self.scales = scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True, track_running_stats=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True, track_running_stats=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True, track_running_stats=True)
        )
        self.act = nn.ELU()

    def forward(self, x):
        input = x

        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.act(self.conv1(xs[1])))
        ys.append(self.act(self.conv2(xs[2]) + ys[1]))
        ys.append(self.act(self.conv2(xs[3]) + ys[2]))
        y = torch.cat(ys, 1)

        output = self.act((y + input))

        return output

class BasicBlock_Inception2D_V1(nn.Module):

    def __init__(self,
        in_channels  : int,#resnet_rec64
        out_channels : int,#
        kernel_size  : int,
        stride       : int = 1,
        downsample = None,
        padding      : str = "same",
        dilation     : int = 1,
        group        : int =1,
        bias         : bool = False,
        track_running_stats_ : bool = True,
        norm         : str = "IN",
        activation   : str = "Relu"):

        super(BasicBlock_Inception2D_V1, self).__init__()

        if norm == "BN":
            self.bns1 = nn.ModuleList( [ nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3) ])
            self.bns2 = nn.ModuleList( [ nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3) ])
        elif norm == "IN":
            self.bns1 = nn.ModuleList( [ nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3)])
            self.bns2 = nn.ModuleList( [ nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3)])

        if activation == "ELU":
            self.acts1 = nn.ModuleList( [ nn.ELU() for _ in range(3) ] )
            self.act = nn.ELU()
        elif activation == "Relu":
            self.acts1 = nn.ModuleList( [ nn.LeakyReLU(negative_slope=0.01,inplace=True)  for _ in range(3) ] )
            self.act = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (1,9), stride, (0,4), dilation, group, bias), \
                                    nn.Conv2d(in_channels, out_channels, (9,1), stride, (4,0), dilation, group, bias), \
                                    nn.Conv2d(in_channels, out_channels, (3,3), stride, (1,1), dilation, group, bias) ] )

        self.convs2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (1,9), stride, (0,4), dilation, group, bias), \
                                    nn.Conv2d(in_channels, out_channels, (9,1), stride, (4,0), dilation, group, bias), \
                                    nn.Conv2d(in_channels, out_channels, (3,3), stride, (1,1), dilation, group, bias) ] )

        self.downsample = downsample

    def forward(self, x):

        identity = x

        xs = None
        for i in range(3):

            xsi = self.convs1[i](x)
            xsi = self.bns1[i](xsi)
            xsi = self.acts1[i](xsi)

            xsi = self.convs2[i](xsi)
            xsi = self.bns2[i](xsi)

            if xs == None:
                xs = xsi
            else:
                xs = xs + xsi

        if self.downsample != None:
            identity = self.downsample(identity)

        return self.act(xs + identity)


""" concatenate 1D -> 2D """
def seq2pairwise_v3(rec1d, lig1d):

    device = rec1d.device
    b, c, L1 = rec1d.size()  # [1, 1331, 320]
    _, _, L2 = lig1d.size()
    # print("lig1d.size()", lig1d.size())

    out1 = rec1d.unsqueeze(3).to(device)
    repeat_idx = [1] * out1.dim()
    repeat_idx[3] = L2
    out1 = out1.repeat(*(repeat_idx))

    out2 = lig1d.unsqueeze(2).to(device)
    repeat_idx = [1] * out2.dim()
    repeat_idx[2] = L1
    out2 = out2.repeat(*(repeat_idx))

    return torch.cat([out1, out2], dim=1)


#####################################################################################################################

class TriangleMultiplication(nn.Module):
    def __init__(self, model_args):
        super(TriangleMultiplication, self).__init__()

        self.dz = model_args['Channel_z']
        self.dc = model_args['Channel_z']

        # init norm
        self.norm_com = nn.LayerNorm(self.dz)
        self.norm_rec = nn.LayerNorm(self.dz)
        self.norm_lig = nn.LayerNorm(self.dz)

        # linear * gate for com_rec, com_lig
        self.Linear_com_rec = nn.Linear(self.dz, self.dc)
        self.Linear_com_lig = nn.Linear(self.dz, self.dc)
        self.gate_com_rec = nn.Linear(self.dz, self.dc)
        self.gate_com_lig = nn.Linear(self.dz, self.dc)

        # linear * gate for rec, lig
        self.Linear_rec = nn.Linear(self.dz, self.dc)
        self.Linear_lig = nn.Linear(self.dz, self.dc)
        self.gate_rec = nn.Linear(self.dz, self.dc)
        self.gate_lig = nn.Linear(self.dz, self.dc)

        # final output
        self.norm_all = nn.LayerNorm(self.dc)
        self.Linear_all = nn.Linear(self.dc, self.dz)
        self.gate_all = nn.Linear(self.dz, self.dz)

    def forward(self, z_com, z_rec, z_lig, mask=None, mask_sa=None):
        """
        Argument:
            z_com : (B, nrec, nlig, dz)
            z_rec : (B, nrec, nrec, dz)
            z_lig : (B, nlig, nlig, dz)
            mask  : (B, nrec, nlig)
        return:
            z_com : (B, nrec, nlig, dz)
        """
        z_com = self.norm_com(z_com)
        z_rec = self.norm_rec(z_rec)
        z_lig = self.norm_lig(z_lig)
        z_com_init = z_com

        if mask != None:
            z_com_rec = self.Linear_com_rec(z_com) * \
                        ( self.gate_com_rec(z_com).sigmoid() * mask)
            z_com_lig = self.Linear_com_lig(z_com) * \
                        ( self.gate_com_lig(z_com).sigmoid() * mask)
        else:
            z_com_rec = self.Linear_com_rec(z_com) * \
                        ( self.gate_com_rec(z_com).sigmoid())
            z_com_lig = self.Linear_com_lig(z_com) * \
                        ( self.gate_com_lig(z_com).sigmoid())

        if mask_sa != None:
            z_com_rec = z_com_rec * mask_sa
            z_com_lig = z_com_lig * mask_sa


        z_rec = self.Linear_rec(z_rec) * self.gate_rec(z_rec).sigmoid()
        z_lig = self.Linear_lig(z_lig) * self.gate_lig(z_lig).sigmoid()

        z_com_rec = torch.einsum(f"bikc,bkjc->bijc", z_rec, z_com_rec)
        z_com_lig = torch.einsum(f"bikc,bjkc->bjic", z_lig, z_com_lig)
        z_all = z_com_rec + z_com_lig

        z_com = self.gate_all(z_com_init).sigmoid() * self.Linear_all( self.norm_all(z_all))

        return z_com


class TriangleSelfAttention(nn.Module):
    def __init__(self, model_args):
        super(TriangleSelfAttention, self).__init__()

        self.dz = model_args['Channel_z']
        self.dc = model_args['Channel_c']
        self.num_head = model_args['num_head']
        self.dhc = self.num_head * self.dc

        self.norm_com = nn.LayerNorm(self.dz)
        self.Linear_Q = nn.Linear(self.dz, self.dhc)
        self.Linear_K = nn.Linear(self.dz, self.dhc)
        self.Linear_V = nn.Linear(self.dz, self.dhc)
        #self.Linear_bias = nn.Linear(self.dz, self.num_head)

        self.softmax = nn.Softmax(-1)
        self.gate_v = nn.Linear(self.dz, self.dhc)
        self.Linear_final = nn.Linear(self.dhc, self.dz)


    def reshape_dim(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.dc)
        return x.view(*new_shape)

    def forward(self, z_com, dist, mask=None, mask_sa=None, eps=5e4):

        B, row, col, _ = z_com.shape
        z_com = self.norm_com(z_com)

        scalar = torch.sqrt( torch.tensor(1.0/self.dc) )#根号c^(-1)
        q = self.reshape_dim(self.Linear_Q(z_com))
        k = self.reshape_dim(self.Linear_K(z_com))
        v = self.reshape_dim(self.Linear_V(z_com))
        #bias = self.Linear_bias(z_com).permute(0,3,1,2)

        coef = torch.exp(-(dist/8.0)**2.0/2.0).unsqueeze(2).type_as(q)

        attn = torch.einsum(f"bnihc, bnjhc->bhnij", q * scalar, k)
        if mask != None:
            attn = attn - ((1-mask[:,:,None,None,:])*eps).type_as(attn)
        if mask_sa != None:
            attn = attn - ((1-mask_sa[:,:,:,None,:])*eps).type_as(attn)
        attn = attn * coef

        if attn.dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
            #attn_weights = self.softmax(attn)
                attn_weights = torch.nn.functional.softmax(attn, -1)
        else:
            attn_weights = torch.nn.functional.softmax(attn, -1)

        v_avg = torch.einsum(f"bhnij, bnjhc->bnihc",attn_weights, v)
        gate_v = (self.reshape_dim(self.gate_v(z_com))).sigmoid()
        z_com = (v_avg * gate_v).contiguous().view( v_avg.size()[:-2] + (-1,) )

        z_final = self.Linear_final(z_com)

        return  z_final

class Transition(nn.Module):

    def __init__(self, model_args):
        super(Transition, self).__init__()

        self.dz = model_args['Channel_z']
        self.n = model_args['Transition_n']

        self.norm = nn.LayerNorm(self.dz)
        self.transition = nn.Sequential(   nn.Linear(self.dz, self.dz*self.n),
                                           nn.ReLU(),
                                           nn.Linear(self.dz*self.n, self.dz)
                                        )
    def forward(self, z_com):

        z_com = self.norm(z_com)
        z_com = self.transition(z_com)

        return z_com
