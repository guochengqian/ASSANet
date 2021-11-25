# Copyright (c) OpenMMLab. All rights reserved.
from torch import distributed as dist


def get_dist_info():

    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

