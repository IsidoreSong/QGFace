import os
import pickle
import torch
from torch import nn
import torch.distributed as dist
from typing import Any, Callable, Dict, List
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig
import hydra
import logging

log = logging.getLogger()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def l2_norm(input, axis=1):
    """l2 normalize"""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class OutFeatureHook(object):
    def __init__(self, module: nn.Module, layer_id_of_interest=None) -> None:
        self.features = torch.empty(0)  # placeholder
        # add hook here
        if layer_id_of_interest is not None:
            module = dict(module.named_modules())[layer_id_of_interest]
        module.register_forward_hook(hook=self.forward_wrapper())

    def forward_wrapper(self):
        def hook(_module, fea_in, fea_out):
            self.features = fea_out.clone().detach()
        return hook
    
def fuse_features_with_norm(stacked_embeddings, stacked_norms):

    assert stacked_embeddings.ndim == 3  # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3  # (n_features_to_fuse, batch_size, 1)

    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    local_rank = get_local_rank()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(local_rank)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=torch.device("cuda", local_rank))
    size_list = [torch.tensor([0], device=torch.device("cuda", local_rank)) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=torch.device("cuda", local_rank)))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=torch.device("cuda", local_rank))
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def get_num_class(hparams):
    # getting number of subjects in the dataset
    if hparams.custom_num_class != -1:
        return hparams.custom_num_class

    if "faces_emore" in hparams.train_data_path.lower():
        # MS1MV2
        class_num = 70722 if hparams.train_data_subset else 85742
    elif "ms1m-retinaface-t1" in hparams.train_data_path.lower():
        # MS1MV3
        assert not hparams.train_data_subset
        class_num = 93431
    elif "faces_vgg_112x112" in hparams.train_data_path.lower():
        # VGGFace2
        assert not hparams.train_data_subset
        class_num = 9131
    elif "faces_webface_112x112" in hparams.train_data_path.lower():
        # CASIA-WebFace
        assert not hparams.train_data_subset
        class_num = 10572
    elif "webface4m" in hparams.train_data_path.lower():
        assert not hparams.train_data_subset
        class_num = 205990
    elif "webface12m" in hparams.train_data_path.lower():
        assert not hparams.train_data_subset
        class_num = 617970
    elif "webface42m" in hparams.train_data_path.lower():
        assert not hparams.train_data_subset
        class_num = 2059906
    else:
        raise ValueError("Check your train_data_path", hparams.train_data_path)

    return class_num

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks = {}

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks[cb_name] = hydra.utils.instantiate(cb_conf)

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig):
    """Instantiates loggers from config."""
    logger = {}

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_name, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger[lg_name] = hydra.utils.instantiate(config=lg_conf)
    return logger

def wandb_hook(wandb_logger, trainer_mod):
    # wandb_logger.watch(trainer_mod, log="all", log_freq=50)
    wandb_logger.define_metric("val/acc", summary="max")
    wandb_logger.define_metric("val/cfpfp_acc", summary="max")
    wandb_logger.define_metric("val/agedb30_acc", summary="max")
    wandb_logger.define_metric("val/calfw_acc", summary="max")
    wandb_logger.define_metric("val/cplfw_acc", summary="max")
    wandb_logger.define_metric("val/lfw_acc", summary="max")
    wandb_logger.define_metric("tinyface/rank_1", summary="max")
    wandb_logger.define_metric("tinyface/rank_2", summary="max")
    wandb_logger.define_metric("tinyface/rank_3", summary="max")

def wandb_sync(wandb_path="/public/home/fwang/workspace/experiments/wandb/latest-run"):
    if not os.path.exists(wandb_path):
        log.warning("The path is not available!")
        exit(1)
    code = os.system(f'wandb sync {wandb_path}')
    log.info(f"Sync Status: {code}")

@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
        
if __name__ == "__main__":
    wandb_sync()