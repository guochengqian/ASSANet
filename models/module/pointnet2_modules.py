from typing import List

import torch
import torch.nn as nn

from . import pointnet2_utils
from models.module.local_aggregation_operators import LocalAggregation, build_conv
from easydict import EasyDict as edict


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.depth = 1
        self.sample_method = 'fps'
        self.sampler = None
        self.local_aggregations = None

    def forward(self, support_xyz: torch.Tensor,
                support_features: torch.Tensor = None, query_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param support_xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param query_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        if query_xyz is None:
            if self.npoint is not None:
                if self.sample_method.lower() == 'fps':
                    xyz_flipped = support_xyz.transpose(1, 2).contiguous()
                    idx = pointnet2_utils.furthest_point_sample(support_xyz, self.npoint)
                    query_xyz = pointnet2_utils.gather_operation(
                        xyz_flipped,
                        idx).transpose(1, 2).contiguous()
                elif self.sample_method.lower() == 'random':
                    query_xyz, idx = self.sampler(support_xyz, support_features)
                    idx = idx.to(torch.int32)
            else:
                query_xyz = support_xyz
                idx = None
        for i in range(self.depth):
            # grouper outputs the concatenation of (grouped_xyz, grouped_features)
            # every time, use the old features, but query new points (different radius)
            new_features = self.local_aggregations[i](query_xyz, support_xyz, support_features,
                                                      query_idx=idx)

            new_features_list.append(new_features)
        # return query_xyz, None  # concatenate
        # return query_xyz, new_features  # concatenate
        return query_xyz, torch.cat(new_features_list, dim=1)  # concatenate


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping

        PointNet++ Set Abstraction Module:
        1. For each module, downsample the point cloud once
        2. For each downsampled point cloud, query neighbors from the HR point cloud multiple times
        3. In each neighbor querying, build the aggregation_features, perform local aggregations
    """

    def __init__(self,
                 npoint: int,
                 radii: List[float],
                 nsamples: List[int],
                 channel_list: List[List[int]],
                 local_aggregation_cfg: dict,
                 sample_method='fps'
                 ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(channel_list)  # time for querying and performing local aggregations

        self.npoint = npoint  # the number of sampled points
        self.depth = len(radii)
        self.sample_method = sample_method

        if self.sample_method.lower() == 'random':
            self.sampler = pointnet2_utils.DenseRandomSampler(num_to_sample=self.npoint)

        self.local_aggregation_cfg = local_aggregation_cfg
        self.reduction = local_aggregation_cfg.get('reduction', 'avg')

        # holder for the grouper and convs (MLPs, \etc)
        self.local_aggregations = nn.ModuleList()

        grouper_cfg = local_aggregation_cfg.get('grouper', edict())
        conv_cfg = local_aggregation_cfg.get('conv', edict())

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            channels = channel_list[i]

            grouper_cfg.radius = radius
            grouper_cfg.nsample = nsample
            grouper_cfg.npoint = npoint

            # build the convs
            conv_cfg.channels = channels

            self.local_aggregations.append(LocalAggregation(conv_cfg,
                                                            grouper_cfg,
                                                            self.local_aggregation_cfg))


class PointnetFPModule(nn.Module):
    r"""Propagates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True, local_aggregation_cfg):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        conv_cfg = local_aggregation_cfg.get('conv', edict())
        if conv_cfg['method'] == 'conv2d':  # use 1d
            conv_cfg['method'] = 'conv1d'
        conv_cfg.channels = mlp
        self.convs = build_conv(conv_cfg, last_act=True)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features. To upsample!!!
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            # can we use other poitshuffle for upsampling?
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = self.convs(new_features)

        return new_features
        # return unknow_feats


if __name__ == "__main__":
    pass
