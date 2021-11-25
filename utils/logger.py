# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import os.path as osp
import sys
from termcolor import colored

import time
import shortuuid
import pathlib
import shutil


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="moco", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


# ================ experiment folder ==================
def generate_exp_directory(config, expname=None, expid=None, logname=None):
    """Function to create checkpoint folder.

    Args:
        config:
        tags: tags for saving and generating the expname
        expid: id for the current run
        logname: the name for the current run. None if auto

    Returns:
        the expname, jobname, and folders into config
    """

    if logname is None:
        if expid is None:
            expid = time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid())
        if isinstance(expname, list):
            expname = '-'.join(expname)
        logname = '-'.join([expname, expid])
    config.logname = logname
    config.log_dir = os.path.join(config.log_dir, config.logname)
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoint')
    config.code_dir = os.path.join(config.log_dir, 'code')
    pathlib.Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.code_dir).mkdir(parents=True, exist_ok=True)

    shutil.copytree('models', osp.join(config.code_dir, 'models'))
    shutil.copytree('function', osp.join(config.code_dir, 'function'))


def resume_exp_directory(config, load_path):
    """Function to resume the exp folder from the checkpoint folder.

    Args:
        config:
        load_path: should have such structure. Logfolder/Expfolder/checkpoint/checkpoint_file

    Returns:
        the expname, jobname, and folders into config
    """
    
    if os.path.basename(os.path.dirname(config.load_path)) == 'checkpoint':
        config.log_dir = os.path.dirname(os.path.dirname(config.load_path))
        config.logname = os.path.basename(config.log_dir)
        config.ckpt_dir = os.path.join(config.log_dir, 'checkpoint')
    else:
        expid = time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid())
        config.logname = '_'.join([os.path.basename(config.load_path), expid])
        config.log_dir = os.path.join(config.log_dir, config.logname)
        config.ckpt_dir = os.path.join(config.log_dir, 'checkpoint')
    os.makedirs(config.ckpt_dir, exist_ok=True)
    config.wandb.tags = ['resume']

