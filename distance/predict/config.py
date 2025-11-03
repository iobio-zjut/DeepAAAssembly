import torch
from Module import *
import Module
from utils import *

def config_parse_args(options):

    # dist_bool = True

    model_config = {
            'model_args': {
                'BasicBlock1D': {
                    'name': Module.BasicBlock_ResNetV2_1D,
                    'InChannels': 883,  # 788
                    'Channels': [64, 64, 64],
                    'OutChannels': 2,
                    'num_Cycle': 1,
                    'Kernel_size': 17,
                    'Dilation': [1],
                    'Group': 1,
                    'Bias': False,
                    'track_running_stats': False
                },
                'BasicBlock2D': {
                    'name': Module.BasicBlock_Inception2D_V1,
                    'InChannels': 208,  # 210
                    'Channels': [64, 64, 64, 64],  # 4个64
                    'OutChannels': 64,
                    'num_Cycle': 1,
                    'Kernel_size': 3,
                    'Dilation': [1],
                    'Group': 1,
                    'Bias': False,
                    'track_running_stats': False
                },
                'dist': True,
                'dist_bins': 37
            },
            'rec_args': {
                'BasicBlock1D': {
                    'name': Module.BasicBlock_ResNetV2_1D,
                    'InChannels': 947,  # 768+20+1+3+20+7+512=(1331+819)/2=1075 ##768+20+1+3+20+7+256=(1075+819)/2=947
                    'Channels': [64, 64, 64],
                    'OutChannels': 2,
                    'num_Cycle': 1,
                    'Kernel_size': 17,
                    'Dilation': [1],
                    'Group': 1,
                    'Bias': False,
                    'track_running_stats': False
                },
                'BasicBlock2D': {
                    'name': Module.BasicBlock_Inception2D_V1,
                    'InChannels': 208,  # 210
                    'Channels': [64, 64, 64, 64],  # 4个64
                    'OutChannels': 64,
                    'num_Cycle': 1,
                    'Kernel_size': 3,
                    'Dilation': [1],
                    'Group': 1,
                    'Bias': False,
                    'track_running_stats': False
                },
                'dist': True,
                'dist_bins': 37
            },
            'triangle_conv_args': {
                'BasicBlock1D': {
                    'name': Module.BasicBlock_ResNetV2_1D,
                    'InChannels': 915,  # 788
                    'Channels': [64, 64, 64],
                    'OutChannels': 2,
                    'num_Cycle': 1,
                    'Kernel_size': 17,
                    'Dilation': [1],
                    'Group': 1,
                    'Bias': False,
                    'track_running_stats': False
                },
                'BasicBlock2D': {
                    'name': Module.BasicBlock_Inception2D_V1,
                    'InChannels': 144,  # 146
                    'Channels': [64, 64, 64, 64],  # 4个64
                    'OutChannels': 64,
                    'num_Cycle': 1,
                    'Kernel_size': 3,
                    'Dilation': [1],
                    'Group': 1,
                    'Bias': False,
                    'track_running_stats': False
                },
                'dist': True,
                'dist_bins': 37
            },
            'triangle_args': {
                'Channel_z': 64,
                'num_head': 4,
                'Channel_c': 8,
                'Transition_n': 4,
                'num_TriangleMulti': 20,  # 20
                'num_TriangleSelfR': 20, #20
                'num_TriangleSelfC': 20, # 20
                'num_Transition': 20,  # 20
                'final': 64,
                'dist': True,
                'dist_bins': 37
            }
    }

    return model_config