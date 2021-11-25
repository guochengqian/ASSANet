import torch.nn as nn
from ..encoder.assanet_encoder import ASSANetEncoder
from ..decoder.assanet_decoder import ASSANetDecoder
from .segmentation_head import SceneSegHeadPointNet


class ASSANetSeg(nn.Module):
    def __init__(self, cfg):
        """ASSA-Net implementation for paper:
        Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning

        Args:
            cfg (dict): configuration
        """
        super().__init__()
        self.encoder = ASSANetEncoder(cfg)
        self.decoder = ASSANetDecoder(cfg)
        self.head = SceneSegHeadPointNet(cfg.data.num_classes, in_channles=cfg.model.fp_mlps[0][0])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz, features):
        l_xyz, l_features = self.encoder(xyz, features)
        return self.head(self.decoder(l_xyz, l_features))

