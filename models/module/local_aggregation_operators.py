import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import build_grouper, build_conv, build_activation_layer
from . import pointnet2_utils
import numpy as np


CHANNEL_MAP = {
    'fj': lambda x: x,
    'assa': lambda x: x,
    'dp_fj': lambda x: 3 + x,
}


class PreConv(nn.Module):
    def __init__(self,
                 conv_cfg,
                 grouper_cfg,
                 config):
        """A PreConv operator for local aggregation

        Args:
            config: config file
        """
        super(PreConv, self).__init__()
        self.feature_type = config.feature_type
        self.reduction = config.reduction
        self.post_res = config.get('post_res', False)
        self.pre_res = config.get('pre_res', False)  # use residual convs/mlps in preconv and postconv.

        post_layers = config.get('post_layers', 0)
        if post_layers == -1:
            post_layers = config.layers//2
        # build grouper
        group_method = grouper_cfg.get('method', 'ball_query').lower()
        self.use_mask = 'mask' in group_method
        self.grouper = build_grouper(grouper_cfg)
        self.nsample = grouper_cfg.get('nsample', 20)
        self.radius = grouper_cfg.get('radius', None)
        assert self.radius is not None

        # only supports conv1d in PreConv Module. 
        assert '1d' in conv_cfg['method']

        # Build PreConvs before Pooling
        channels = conv_cfg.channels
        pre_conv_cfg = conv_cfg.copy()
        if post_layers != 0:
            pre_conv_cfg['channels'] = pre_conv_cfg['channels'][:-(post_layers)]
        if self.feature_type == 'assa':
            pre_conv_cfg['channels'][-1] = int(np.ceil(pre_conv_cfg['channels'][-1]/3.0))
        self.pre_conv = build_conv(pre_conv_cfg, last_act=not self.pre_res)

        # Build skip connection layer for the pre convs
        if self.pre_res:
            if pre_conv_cfg['channels'][-1] != pre_conv_cfg['channels'][0]:
                short_cut_cfg = conv_cfg.copy()
                short_cut_cfg['channels'] = [pre_conv_cfg['channels'][0], pre_conv_cfg['channels'][-1]]
                self.pre_short_cut = build_conv(short_cut_cfg, last_act=not self.pre_res)
            else:
                self.pre_short_cut = nn.Sequential()
        act_name = conv_cfg.get('activation', 'relu')
        self.pre_activation = build_activation_layer(act_name) if self.pre_res else nn.Sequential()

        # Build PostConvs 
        if post_layers != 0:
            post_conv_cfg = conv_cfg.copy()
            post_channel_in = CHANNEL_MAP[config.feature_type](pre_conv_cfg['channels'][-1])
            if self.feature_type == 'assa':
                post_channel_in *= 3
            post_conv_cfg['channels'] = [post_channel_in] + [channels[-1]] * post_layers
            self.post_conv = build_conv(post_conv_cfg, last_act=not self.post_res)
        else:
            self.post_conv = nn.Sequential()
        self.post_activation = build_activation_layer(act_name) if self.post_res else nn.Sequential()

        # Build skip connection layer for the post convs
        if self.post_res:
            if pre_conv_cfg['channels'][-1] != post_conv_cfg['channels'][-1]:
                short_cut_cfg = conv_cfg.copy()
                short_cut_cfg['channels'] = [pre_conv_cfg['channels'][-1], channels[-1]]
                self.post_short_cut = build_conv(short_cut_cfg, last_act=False)
            else:
                self.post_short_cut = nn.Sequential()

        # reduction layer
        if self.reduction == 'max':
            self.reduction_layer = lambda x: F.max_pool2d(
                x, kernel_size=[1, self.nsample]
            ).squeeze(-1)

        elif self.reduction == 'avg' or self.reduction == 'mean':
            self.reduction_layer = lambda x: torch.mean(x, dim=-1, keepdim=False)

        elif self.reduction == 'sum':
            self.reduction_layer = lambda x: torch.sum(x, dim=-1, keepdim=False)
        else:
            raise NotImplementedError(f'reduction {self.reduction} not implemented')

    def forward(self, query_xyz, support_xyz, features, query_idx=None):
        """
        Args:

        Returns:
           output features of query points: [B, C_out, 3]
        """
        # PreConv Layer with possible residual connection
        if self.pre_res:
            features = self.pre_conv(features) + self.pre_short_cut(features)
        else:
            features = self.pre_conv(features)
        features = self.pre_activation(features)

        # subsampling + grouping layer.
        # subsampling has already been executed outside this module.
        # here, we directly use the precomputed the query_xyz and the idx
        neighborhood_features, relative_position = self.grouper(query_xyz, support_xyz, features)
        B, C, npoint, nsample = neighborhood_features.shape

        # subsample layer if query_idx is not None.
        if query_idx is not None:
            # torch gather is slower than this c++ implementation
            # center_feature = torch.gather(features, 2, query_idx.unsqueeze(1).repeat(1, features.shape[1], 1))
            features = pointnet2_utils.gather_operation(features, query_idx)

        # Anisotropic Reduction layer
        neighborhood_features = neighborhood_features.unsqueeze(1).expand(-1, 3, -1, -1, -1) \
                                * relative_position.unsqueeze(2)
        neighborhood_features = neighborhood_features.view(B, -1, npoint, nsample)
        neighborhood_features = self.reduction_layer(neighborhood_features)

        # Post Conv layer with possible residual connection
        if self.post_res:
            features = self.post_conv(neighborhood_features) + self.post_short_cut(features)
        else:
            features = self.post_conv(neighborhood_features)
        features = self.post_activation(features)
        return features


class ConvPool(nn.Module):
    def __init__(self,
                 conv_cfg,
                 grouper_cfg,
                 config):
        """A PosPool operator for local aggregation

        Args:
            config: config file
        """
        super(ConvPool, self).__init__()
        self.feature_type = config.feature_type
        self.reduction = config.reduction

        # use conv2d is wrongly used.
        channel_in = CHANNEL_MAP[config.feature_type](conv_cfg['channels'][0])
        conv_cfg.channels[0] = channel_in
        if conv_cfg['method'] == 'conv1d':
            conv_cfg['method'] = 'conv2d'

        self.convs = build_conv(conv_cfg)

        # build grouper
        self.grouper = build_grouper(grouper_cfg)

    def forward(self, query_xyz, support_xyz, support_features, query_idx=None):
        """
        Args:

        Returns:
           output features of query points: [B, C_out, 3]
        """
        neighborhood_features, relative_position = self.grouper(query_xyz, support_xyz, support_features)

        B, C, npoint, nsample = neighborhood_features.shape

        if 'df' not in self.feature_type:
            if self.feature_type == 'assa':
                if C >= 3:
                    repeats = C // 3
                    repeat_tensor = torch.tensor([repeats, repeats, C - repeats * 2], dtype=torch.long,
                                                 device=relative_position.device, requires_grad=False)
                    position_embedding = torch.repeat_interleave(relative_position, repeat_tensor, dim=1)
                    aggregation_features = position_embedding * neighborhood_features  # (B, C//3, 3, npoint, nsample)
                else:
                    attn = torch.sum(relative_position, dim=1, keepdims=True)
                    aggregation_features = neighborhood_features * attn

            elif self.feature_type == 'fj':
                aggregation_features = neighborhood_features

            elif self.feature_type == 'dp_fj':
                aggregation_features = torch.cat([relative_position, neighborhood_features], 1)

        aggregation_features = self.convs(aggregation_features)

        if len(aggregation_features.size()) == 4:
            if self.reduction == 'max':
                out_features = F.max_pool2d(
                    aggregation_features, kernel_size=[1, nsample]
                ).squeeze(-1)

            elif self.reduction == 'avg' or self.reduction == 'mean':
                out_features = torch.mean(aggregation_features, dim=-1, keepdim=False)

            elif self.reduction == 'sum':
                out_features = torch.sum(aggregation_features, dim=-1, keepdim=False)
            else:
                raise NotImplementedError(f'reduction {self.reduction} not implemented')
        else:
            out_features = aggregation_features

        return out_features


class LocalAggregation(nn.Module):
    def __init__(self,
                 conv_cfg,
                 grouper_cfg,
                 config):
        """LocalAggregation operators

        Args:
            config: config file
        """
        super(LocalAggregation, self).__init__()
        self.conv_cfg = conv_cfg
        self.grouper_cfg = grouper_cfg
        if config.type.lower() == 'preconv':
            self.SA_CONFIG_operator = PreConv(conv_cfg, grouper_cfg, config)
        elif config.type.lower() == 'convpool':
            self.SA_CONFIG_operator = ConvPool(conv_cfg, grouper_cfg, config)
        else:
            raise NotImplementedError(f'LocalAggregation {config.type.lower()} not implemented')

    def forward(self, query_xyz, support_xyz, support_features, query_idx=None):
        """
        Args:


        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.SA_CONFIG_operator(query_xyz, support_xyz, support_features, query_idx)
