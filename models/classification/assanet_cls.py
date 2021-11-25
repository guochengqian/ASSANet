import numpy as np
import torch.nn as nn
from easydict import EasyDict as edict
from ..module.ASSA import  PointnetSAModuleMSG
from ..module.local_aggregation_operators import LocalAggregation
from ..module.activation import build_activation_layer
from .clss_head import ClassifierPointNet


class ASSANetCls(nn.Module):
    def __init__(self, cfg):
        """ASSA-Net implementation with point cloud classification for paper:
        Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning

        Args:
            cfg (dict): configuration
        """
        super().__init__()
        self.model_cfg = cfg.model
        self.local_aggregation_cfg = cfg.model.sa_config.local_aggregation
        self.SA_modules = nn.ModuleList()

        mlps = self.model_cfg.sa_config.get('mlps', None)
        if mlps is None:
            width = self.model_cfg.get('width', None)
            depth = self.model_cfg.get('depth', 2)
            layers = self.local_aggregation_cfg.get('layers', None)
            assert width is not None
            assert layers is not None
            mlps = [[[width] * layers]*depth,
                    [[width * depth] * layers]*depth,
                    [[width * depth ** 2] * layers]*depth,
                    [[width * depth ** 3] * layers]*depth]
            self.local_aggregation_cfg.post_layers = self.local_aggregation_cfg.get('post_layers', layers//2)
            self.model_cfg.sa_config.mlps = mlps
            print(f'channels for the current model is modified to {self.model_cfg.sa_config.mlps}')

            # revise the radius, and nsample
            for i in range(len(self.model_cfg.sa_config.radius)):
                self.model_cfg.sa_config.radius[i] = self.model_cfg.sa_config.radius[i] + \
                                                     [self.model_cfg.sa_config.radius[i][-1]]*(depth-2)
            for i in range(len(self.model_cfg.sa_config.nsample)):
                self.model_cfg.sa_config.nsample[i] = self.model_cfg.sa_config.nsample[i] + [
                    self.model_cfg.sa_config.nsample[i][-1]] * (depth - 2)

        # build the first conv and local aggregations on the input points. (this is to be similar to close3d).
        width = self.model_cfg.sa_config.mlps[0][0][0]
        activation = self.model_cfg.get('activation', 'relu')
        self.conv1 = nn.Sequential(*[nn.Conv1d(self.model_cfg.in_channel, width, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(width),
                                     build_activation_layer(activation, inplace=True)])

        grouper_cfg = self.local_aggregation_cfg.get('grouper', edict())
        grouper_cfg.radius = self.model_cfg.sa_config.radius[0][0]
        grouper_cfg.nsample = self.model_cfg.sa_config.nsample[0][0]
        grouper_cfg.npoint = self.model_cfg.sa_config.npoints[0]

        conv_cfg = self.local_aggregation_cfg.get('conv', edict())
        conv_cfg.channels = [width] * 4 
        la1_cfg = edict(self.local_aggregation_cfg.copy())
        la1_cfg.post_layers = 1
        self.la1 = LocalAggregation(conv_cfg, grouper_cfg, la1_cfg)

        skip_channel_list = [width]
        for k in range(self.model_cfg.sa_config.npoints.__len__()):  # sample times

            # obtain the in_channels and output channels from the configuration
            channel_list = self.model_cfg.sa_config.mlps[k].copy()
            channel_out = 0
            for idx in range(channel_list.__len__()):
                channel_list[idx] = [width] + channel_list[idx]
                channel_out += channel_list[idx][-1]  # concatenate
                width = channel_list[idx][-1]
            width = channel_out

            # for each sample, may query points multiple times, the query radii and nsamples may be different
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=self.model_cfg.sa_config.npoints[k],
                    radii=self.model_cfg.sa_config.radius[k],
                    nsamples=self.model_cfg.sa_config.nsample[k],
                    channel_list=channel_list,
                    local_aggregation_cfg=self.local_aggregation_cfg,
                    sample_method=self.model_cfg.sa_config.get('sample_method', 'fps')
                )
            )
            skip_channel_list.append(channel_out)

        self.classifier = ClassifierPointNet(cfg.data.num_classes,
                                             in_channels=np.array(cfg.model.sa_config.mlps)[-1, :, -1].sum())

    def forward(self, xyz, features):
        """
        Args:

        Returns:

        """
        features = self.la1(xyz, xyz, self.conv1(features))

        for i in range(len(self.SA_modules)):
            xyz, features = self.SA_modules[i](xyz, features)  # query_xyz is None. query for neighbors

        return self.classifier(features)
