import torch.nn as nn
from . import pointnet2_utils
from .activation import build_activation_layer


def build_grouper(grouper_cfg):
    radius = grouper_cfg.get('radius', 0.1)
    nsample = grouper_cfg.get('nsample', 20)
    normalize_xyz = grouper_cfg.get('normalize_xyz', False)
    grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=False,
                                            ret_grouped_xyz=True, normalize_xyz=normalize_xyz)
    return grouper


def build_conv(conv_cfg, last_act=True):
    channels = conv_cfg.get('channels', None)
    assert channels is not None
    method = conv_cfg.get('method', 'mlp').lower()
    use_bn = conv_cfg.get('use_bn', True)
    act_name = conv_cfg.get('activation', 'relu')
    activation = build_activation_layer(act_name)
    groups = conv_cfg.get('groups', 1)
    shared_mlps = []
    if method == 'conv2d':
        for k in range(len(channels) - 1):
            shared_mlps.append(nn.Conv2d(channels[k], channels[k + 1], kernel_size=(1, 1), groups=groups, bias=False))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(channels[k + 1]))
            if k != len(channels) - 2 or last_act:
                shared_mlps.append(activation)

    elif method == 'conv1d':
        for k in range(len(channels) - 1):
            shared_mlps.append(nn.Conv1d(channels[k], channels[k + 1], kernel_size=1, groups=groups, bias=False))
            if use_bn:
                shared_mlps.append(nn.BatchNorm1d(channels[k + 1]))
            if k != len(channels) - 2 or last_act:
                shared_mlps.append(activation)
    else:
        raise NotImplementedError(f'{method} in local aggregation transform is not supported currently')
    return nn.Sequential(*shared_mlps)

