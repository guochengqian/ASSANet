"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import os
import sys
import time
import json
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_scene_segmentation, build_scheduler, build_optimizer
from datasets import build_s3dis_loader, data_utils as d_utils
from utils import AverageMeter, s3dis_metrics, sub_s3dis_metrics, s3dis_part_metrics, \
    cal_model_parm_nums, set_seed, save_checkpoint, load_checkpoint
from utils.logger import setup_logger, generate_exp_directory, resume_exp_directory
from utils.wandb import Wandb
from utils.config import config


def parse_option():
    parser = argparse.ArgumentParser('S3DIS scene-segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    config.local_rank = int(os.environ['LOCAL_RANK'])

    if args.profile:
        if config.model.sa_config.local_aggregation.get('conv', None) is not None:
            if config.model.sa_config.local_aggregation.conv.get('use_bn', False):
                config.model.sa_config.local_aggregation.conv.use_bn = False  # NO BN when test inference
    return args, config


def main(config, profile=False):
    train_loader, val_loader = build_s3dis_loader(config)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    n_data = len(val_loader.dataset)
    logger.info(f"length of validation dataset: {n_data}")

    model, criterion = build_scene_segmentation(config)
    model.cuda()
    criterion.cuda()
    logger.info(model)
    model_size = cal_model_parm_nums(model)
    logger.info('Number of params: %.4f M' % (model_size / (1e6)))

    if profile:
        model.eval()
        total_time = 0.
        points, mask, features, points_labels, cloud_label, input_inds = iter(val_loader).next()
        points = points.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        print(points.shape, features.shape)
        # B, N, C = config.batch_size, 15000, 4
        # points = torch.randn(B, N, 3).cuda()
        # features = torch.randn(B, C, N).cuda()
        # mask = None

        n_runs = 200
        with torch.no_grad():
            for idx in range(n_runs):
                start_time = time.time()
                model(points, features)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                total_time += time_taken
        print(f'inference time: {total_time / float(n_runs)}')
        return False

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, 1 if config.lr_scheduler.on_epoch else len(train_loader.dataset))
    model = DistributedDataParallel(model,  broadcast_buffers=False, device_ids=[config.local_rank], output_device=config.local_rank)

    runing_vote_logits = [np.zeros((config.data.num_classes, l.shape[0]), dtype=np.float32) for l in
                          val_loader.dataset.sub_clouds_points_labels]
    # optionally resume from a checkpoint
    if config.load_path:
        load_checkpoint(config, model, optimizer, scheduler, printer=logger.info)
        if 'train' in config.mode:
            val_miou = validate('resume', val_loader, model, criterion, runing_vote_logits, config, num_votes=2)
            logger.info(f'\nresume val mIoU is {val_miou}\n ')
        else:
            val_miou_20 = validate('Test', val_loader, model, criterion, runing_vote_logits, config, num_votes=20)
            logger.info(f'\nval mIoU is {val_miou_20}\n ')
            return val_miou_20

    # ===> start training
    val_miou = 0.
    best_val = 0.
    is_best = False
    for epoch in range(config.start_epoch, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch - 1
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, config)

        if epoch % config.val_freq == 0:
            val_miou = validate(0, val_loader, model, criterion, runing_vote_logits, config, num_votes=1)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou

        logger.info('epoch {}, total time {:.2f}, lr {:.5f}, '
                    'best val mIoU {:3f}'.format(epoch,
                                                 (time.time() - tic),
                                                 optimizer.param_groups[0]['lr'],
                                                 best_val))
        if dist.get_rank() == 0:
            # save model
            save_checkpoint(config, epoch, model, optimizer, scheduler, is_best=is_best)
            is_best = False

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('best_val_miou', best_val, epoch)
            summary_writer.add_scalar('val_miou', val_miou, epoch)
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if config.lr_scheduler.on_epoch:
            scheduler.step()

    if summary_writer is not None:
        Wandb.add_file(os.path.join(config.ckpt_dir, f'{config.logname}_ckpt_best.pth'))
        Wandb.add_file(os.path.join(config.ckpt_dir, f'{config.logname}_ckpt_latest.pth'))

    load_checkpoint(config, model,
                    load_path=os.path.join(config.ckpt_dir, f'{config.logname}_ckpt_best.pth'),
                    printer=logger.info)
    set_seed(config.rng_seed)
    best_miou_20 = validate('Best', val_loader, model, criterion, runing_vote_logits, config, num_votes=20)
    if summary_writer is not None:
        summary_writer.add_scalar('val_miou20', best_miou_20, config.epochs + 50)


def train(epoch, train_loader, model, criterion, optimizer, scheduler, config):
    """
    One epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = points.size(0)
        # forward
        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)

        pred = model(points, features)
        loss = criterion(pred, points_labels, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if not config.lr_scheduler.on_epoch:
            scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            logger.info(f'Train: [{epoch}/{config.epochs + 1}][{idx}/{len(train_loader)}]\t'
                        f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
            # logger.info(f'[{cloud_label}]: {input_inds}')
    return loss_meter.avg


def validate(epoch, test_loader, model, criterion, runing_vote_logits, config, num_votes=1):
    """
    One epoch validating
    """
    vote_logits_sum = [np.zeros((config.data.num_classes, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.sub_clouds_points_labels]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.sub_clouds_points_labels]
    vote_logits = [np.zeros((config.data.num_classes, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.sub_clouds_points_labels]
    validation_proj = test_loader.dataset.projections
    validation_labels = test_loader.dataset.clouds_points_labels
    test_smooth = 0.95

    val_proportions = np.zeros(config.data.num_classes, dtype=np.float32)
    for label_value in range(config.data.num_classes):
        val_proportions[label_value] = np.sum(
            [np.sum(labels == label_value) for labels in test_loader.dataset.clouds_points_labels])

    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        RT = d_utils.BatchPointcloudRandomRotate(x_range=config.data.x_angle_range, y_range=config.data.y_angle_range,
                                                 z_range=config.data.z_angle_range)
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.data.scale_low, scale_high=config.data.scale_high,
                                                   std=config.data.noise_std, clip=config.data.noise_clip,
                                                   augment_symmetries=config.data.augment_symmetries)
        for v in range(num_votes):
            test_loader.dataset.epoch = (0 + v) if isinstance(epoch, str) else (epoch + v) % 20
            predictions = []
            targets = []

            for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(test_loader):
                # augment for voting
                if v > 0:
                    points = RT(points)
                    points = TS(points)
                    if config.data.input_features_dim <= 5:
                        pass
                    elif config.data.input_features_dim == 6:
                        color = features[:, :3, :]
                        features = torch.cat([color, points.transpose(1, 2).contiguous()], 1)
                    elif config.data.input_features_dim == 7:
                        color_h = features[:, :4, :]
                        features = torch.cat([color_h, points.transpose(1, 2).contiguous()], 1)
                    else:
                        raise NotImplementedError(
                            f"input_features_dim {config.data.input_features_dim} in voting not supported")
                # forward
                points = points.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                cloud_label = cloud_label.cuda(non_blocking=True)
                input_inds = input_inds.cuda(non_blocking=True)

                pred = model(points, features)

                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item(), points.size(0))

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(bool)
                    logits = pred[ib].cpu().numpy()[:, mask_i]
                    inds = input_inds[ib].cpu().numpy()[mask_i]
                    c_i = cloud_label[ib].item()
                    vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits
                    vote_counts[c_i][:, inds] += 1
                    vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i]
                    runing_vote_logits[c_i][:, inds] = test_smooth * runing_vote_logits[c_i][:, inds] + \
                                                       (1 - test_smooth) * logits
                    predictions += [logits]
                    targets += [test_loader.dataset.sub_clouds_points_labels[c_i][inds]]

            predictions = collect_results_gpu(predictions, test_loader.dataset.__len__())
            targets = collect_results_gpu(targets, test_loader.dataset.__len__())

            mIoU = torch.tensor(0., device=torch.device('cuda'), dtype=torch.float64)
            if dist.get_rank() == 0:
                pIoUs, pmIoU = s3dis_part_metrics(config.data.num_classes, predictions, targets, val_proportions)
                runsubIoUs, runsubmIoU = sub_s3dis_metrics(config.data.num_classes, runing_vote_logits,
                                                           test_loader.dataset.sub_clouds_points_labels,
                                                           val_proportions)
                subIoUs, submIoU = sub_s3dis_metrics(config.data.num_classes, vote_logits,
                                                     test_loader.dataset.sub_clouds_points_labels, val_proportions)
                IoUs, mIoU = s3dis_metrics(config.data.num_classes, vote_logits, validation_proj, validation_labels)

                mIoU = torch.as_tensor(mIoU, device=torch.device('cuda'))
                logger.info(f'E{epoch} V{v} * part_mIoU {pmIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * part_msIoU {pIoUs}')

                logger.info(f'E{epoch} V{v} * running sub_mIoU {runsubmIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * running sub_msIoU {runsubIoUs}')

                logger.info(f'E{epoch} V{v} * sub_mIoU {submIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * sub_msIoU {subIoUs}')

                logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * msIoU {IoUs}')
            dist.broadcast(mIoU, 0)
    return mIoU


def collect_results_gpu(result_part, size):
    import pickle
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # dump result part to tensor with pickle
    if world_size == 1:
        return result_part
    else:
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_list.append(
                    pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            original_size = len(ordered_results)
            ordered_results = ordered_results[:size]
            print(f'total length of preditions of {original_size} is reduced to {size}')
            return ordered_results


if __name__ == "__main__":
    opt, config = parse_option()

    # random seed
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["JOB_LOG_DIR"] = config.log_dir
    set_seed(config.rng_seed)

    # LR rule
    config.batch_size *= dist.get_world_size()
    config.optimizer.lr = config.optimizer.lr * config.batch_size

    # logger
    if dist.get_rank() == 0:
        if config.load_path is None:
            local_aggregation_cfg = config.model.sa_config.local_aggregation
            tags = [config.data.datasets,
                    config.mode,
                    opt.cfg.split('.')[-2].split('/')[-1],
                    local_aggregation_cfg.type,
                    local_aggregation_cfg.feature_type,
                    local_aggregation_cfg.reduction,
                    f'C{config.model.width}', f'L{local_aggregation_cfg.layers}', f'D{config.model.depth}',
                    f'B{config.batch_size}', f'LR{config.optimizer.lr}',
                    f'Epoch{config.epochs}', f'Seed{config.rng_seed}',
                    f'GPUS{dist.get_world_size()}'
                    ]
            generate_exp_directory(config, tags)
            config.wandb.tags = tags

        else:  # resume from the existing ckpt and reuse the folder.
            resume_exp_directory(config, config.load_path)
            config.wandb.tags = ['resume']
    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="s3dis")  # stdout master only!

    # wandb and tensorboard
    if dist.get_rank() == 0:
        cfg_path = os.path.join(config.log_dir, "config.json")
        with open(cfg_path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (opt.cfg, config.log_dir))
        config.cfg_path = cfg_path

        # wandb config
        config.wandb.name = config.logname
        Wandb.launch(config, config.wandb.use_wandb)

        # tensorboard
        summary_writer = SummaryWriter(log_dir=config.log_dir)
    else:
        summary_writer = None

    logger.info(config)
    main(config, profile=opt.profile)
