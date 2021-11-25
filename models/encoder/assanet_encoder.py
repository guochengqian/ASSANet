import torch.nn as nn
from easydict import EasyDict as edict
from ..module.ASSA import PointnetSAModuleMSG
from ..module.local_aggregation_operators import LocalAggregation
from ..module.activation import build_activation_layer


class ASSANetEncoder(nn.Module):
    def __init__(self, cfg):
        """ASSA-Net implementation for paper:
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
            post_layers = self.local_aggregation_cfg.get('post_layers', 0)
            if post_layers == -1:
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
        self.encoder_layers = len(self.SA_modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz, features):
        """
        Args:

        Returns:

        """
        l_xyz, l_features = [[] for _ in range(self.encoder_layers + 1)], [[] for _ in range(self.encoder_layers + 1)]

        # first SA layer for processing the whole xyz without subsampling
        l_xyz[0] = xyz
        l_features[0] = self.la1(xyz, xyz, self.conv1(features))

        # repeated SA encoder modules
        for i in range(self.encoder_layers):
            l_xyz[i + 1], l_features[i + 1] = self.SA_modules[i](l_xyz[i], l_features[i])

        return l_xyz, l_features

