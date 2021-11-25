import torch.nn as nn
from ..module.ASSA import PointnetFPModule


class ASSANetDecoder(nn.Module):
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
            skip_channel_list.append(channel_out)

        self.decoders = nn.ModuleList()
        for k in range(self.model_cfg.fp_mlps.__len__()):
            pre_channel = self.model_cfg.fp_mlps[k + 1][-1] if k + 1 < len(self.model_cfg.fp_mlps) else channel_out
            self.decoders.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.fp_mlps[k],
                    local_aggregation_cfg=self.local_aggregation_cfg
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, l_xyz, l_features):
        """
        Args:

        Returns:

        """
        # repeated decoder modules
        for i in range(-1, -(len(self.decoders) + 1), -1):
            l_features[i - 1] = self.decoders[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        return l_features[0]
