import torch.nn as nn

from .registry import build_from_cfg, ACTIVATION_LAYERS

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh
]:
    ACTIVATION_LAYERS.register_module(module=module)


ACTIVATION_MAPPING = {
    'relu': 'ReLU',
    'leakyrelu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu6': 'ReLU6',
    'elu6': 'ELU',
    'sigmoid': 'Sigmoid',
    'tanh': 'Tanh',
}


def activation_str2dict_mapping(act='relu'):
    """ Map the input args of act (string) into the dictionary for building activation layer.
    Args:
        act (string): lowercase string for activation layer

    Returns:

    """
    assert isinstance(act, str) or act is None
    if act is not None:
        act = {'type': ACTIVATION_MAPPING[act.lower()]}
    return act


def build_activation_layer(cfg, inplace=True):
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
        inplace (bool): Set to use inplace operation.

    Returns:
        nn.Module: Created activation layer.

    Example:
            >>> # ---------------- Activation
            >>> n_feats = 512
            >>> n_points = 128
            >>> n_batch = 4
            >>> device = torch.device('cuda')
            >>> feats = torch.rand((n_batch, n_feats, n_points), dtype=torch.float).to(device) - 0.5
            >>> print(f"before operation, "
            >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")

            >>> # build activation
            >>> act_cfg = {'type': 'LeakyReLU'}
            >>> activation = build_activation_layer(act_cfg).to(device)

            >>> # test activation
            >>> feats = activation(feats)
            >>> print(f"after operation, "
            >>>       f"the minimum value of feats is {feats.min()}, and the maximum value of feats is {feats.max()}")

    """
    if not isinstance(cfg, dict):
        assert isinstance(cfg, str)
        cfg = activation_str2dict_mapping(cfg)

    if cfg['type'] not in [
        'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
    ]:
        cfg.setdefault('inplace', inplace)

    return build_from_cfg(cfg, ACTIVATION_LAYERS)

