import torch.nn as nn
import torch.nn.functional as F


class SceneSegHeadPointNet(nn.Module):
    def __init__(self, num_classes, in_channles):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super(SceneSegHeadPointNet, self).__init__()
        self.head = nn.Sequential(nn.Conv1d(in_channles, 32, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(32, num_classes, kernel_size=1, bias=True))

    def forward(self, end_points):
        logits = self.head(end_points)
        return logits


class MultiPartSegHeadPointNet(nn.Module):
    def __init__(self, num_classes, in_channles, num_parts):
        """A scene segmentation head for ResNet backbone.
        Args:
            num_classes: class num.
            width: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super(MultiPartSegHeadPointNet, self).__init__()
        self.num_classes = num_classes
        self.multi_shape_heads = nn.ModuleList()
        for i in range(num_classes):
            self.multi_shape_heads.append(
                nn.Sequential(nn.Conv1d(in_channles, 32, kernel_size=1, bias=False),
                              nn.BatchNorm1d(32),
                              nn.ReLU(inplace=True),
                              nn.Conv1d(32, num_parts[i], kernel_size=1, bias=True)))

    def forward(self, end_points):
        logits_all_shapes = []
        for i in range(self.num_classes):
            logits_all_shapes.append(self.multi_shape_heads[i](end_points))
        return logits_all_shapes
