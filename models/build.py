import torch
from .classification import ASSANetCls
from .segmentation import ASSANetSeg
from .losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR


def build_classification(config):
    if config.model.name == 'assanet':
        model = ASSANetCls(config)
    else:
        raise NotImplementedError(f'{config.model.name} is not support yet')
    criterion = LabelSmoothingCrossEntropyLoss()
    return model, criterion


def build_multi_part_segmentation(config):
    if config.model.name == 'assanet':
        model = ASSANetPartSeg(config)
    else:
        raise NotImplementedError(f'{config.model.name} is not support yet')
    criterion = MultiShapeCrossEntropy(config.data.num_classes)
    return model, criterion



def build_scene_segmentation(config):
    if config.model.name == 'assanet':
        model = ASSANetSeg(config)
    else:
        raise NotImplementedError(f'{config.model.name} is not support yet')
    criterion = MaskedCrossEntropy()
    return model, criterion


def build_optimizer(model, config):
    optim_config = config.optimizer
    lr = optim_config.lr  # linear rule does not apply here

    if optim_config.name == 'sgd':
        # lr = optim_config.batch_size * dist.get_world_size() / 8 * optim_config.lr
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=optim_config.momentum,
                                    weight_decay=optim_config.weight_decay)
    elif optim_config.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=optim_config.weight_decay)
    elif optim_config.name == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr,
                                      weight_decay=optim_config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optim_config.name} not supported")

    return optimizer


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def build_scheduler(optimizer, config, n_iter_per_epoch=1):
    """ build the lr scheduler
    Args:
        optimizer:
        config:
        n_iter_per_epoch: set to 1 if perform lr scheduler per epoch

    Returns:

    """
    assert config.lr_scheduler.on_epoch == (n_iter_per_epoch == 1)
    if "cosine" in config.lr_scheduler.name:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=(config.epochs - config.warmup_epoch)*n_iter_per_epoch)

    elif "multistep" in config.lr_scheduler.name:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=config.lr_scheduler.decay_rate,
            milestones=[int(x) for x in config.lr_scheduler.decay_steps.split(',')])

    elif "step" in config.lr_scheduler.name:
        lr_decay_epochs = [config.lr_scheduler.decay_steps * i
                           for i in range(1, config.epochs // config.lr_scheduler.decay_steps)]
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=config.lr_scheduler.decay_rate,
            milestones=[(m - config.warmup_epoch)*n_iter_per_epoch for m in lr_decay_epochs])
    else:
        raise NotImplementedError(f"scheduler {config.lr_scheduler.name} not supported")

    if config.warmup_epoch > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=config.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=config.warmup_epoch*n_iter_per_epoch)
    return scheduler

