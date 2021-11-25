import os
import torch
import random
import numpy as np
import logging
from collections import OrderedDict
import shutil


# ================ model related ==================
def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(config, model, optimizer=None, scheduler=None, load_path=None, printer=logging.info):
    if load_path is None:
        load_path = config.load_path
        assert load_path is not None
    printer("=> loading checkpoint '{}'".format(load_path))

    checkpoint = torch.load(load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            printer('optimizer does not match')
    if scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            printer('scheduler does not match')

    ckpt_state = checkpoint['model']
    model_dict = model.state_dict()
    # rename ckpt (avoid name is not same because of multi-gpus)
    is_model_multi_gpus = True if list(model_dict)[0].split('.')[0] == 'module' else False
    is_ckpt_multi_gpus = True if list(ckpt_state)[0].split('.')[0] == 'module' else False

    if not (is_model_multi_gpus == is_ckpt_multi_gpus):
        temp_dict = OrderedDict()
        for k, v in ckpt_state.items():
            if is_ckpt_multi_gpus:
                name = k[7:]  # remove 'module.'
            else:
                name = 'module.' + k  # add 'module'
            temp_dict[name] = v
        ckpt_state = temp_dict

    model.load_state_dict(ckpt_state)

    printer("=> loaded successfully '{}' (epoch {})".format(load_path, checkpoint['epoch']))
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, scheduler, save_name=None,
                    is_best=False, printer=logging.info):
    if save_name is None:
        save_name = config.logname
    printer('==> Saving...')
    state = {
        'config': config,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    current_ckpt_name = f'{save_name}_ckpt_latest.pth'
    current_ckpt_path = os.path.join(config.ckpt_dir, current_ckpt_name)
    torch.save(state, current_ckpt_path)

    if config.save_freq > 0 and epoch % config.save_freq == 0:
            milestone_ckpt_name = f'{save_name}_E{epoch}.pth'
            milestone_ckpt_path = os.path.join(config.ckpt_dir, milestone_ckpt_name)
            shutil.copyfile(current_ckpt_path, milestone_ckpt_path)
            printer("Saved in {}".format(milestone_ckpt_path))

    if is_best:
        best_ckpt_name = f'{save_name}_ckpt_best.pth' if save_name else 'ckpt_best.pth'
        best_ckpt_path = os.path.join(config.ckpt_dir, best_ckpt_name)
        shutil.copyfile(current_ckpt_path, best_ckpt_path)
        printer("Found the best model and saved in {}".format(best_ckpt_path))
