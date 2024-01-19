from pytorch_lightning import seed_everything, LightningModule, Trainer
import sys
import torch
import os
sys.path.append("/workspace")
sys.path.append("~/workspace/QGFace/QGFace")
sys.path.append("/workspace/QGFace")
import utils
import framework as framework
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)
import hydra
from pylogger import init_logger
from validation import validate


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def hydra_main(cfg: DictConfig):

    if "LOCAL_RANK" not in os.environ:
        os.makedirs(cfg.job_storage_dir, exist_ok=True)
        init_logger(cfg.job_storage_dir)

    trainer_mod: LightningModule = framework.FaceModel(**cfg)
    data_mod = hydra.utils.instantiate(cfg.data, validation=cfg.validation)

    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    callbacks_dict = utils.instantiate_callbacks(cfg.callbacks)
    callbacks = list(callbacks_dict.values())
    model_checkpoint, progress_bar = callbacks_dict['model_checkpoint'], callbacks_dict['progress_bar']
    trainer_mod.progress_bar = progress_bar
    if ("NODE_RANK" not in os.environ or os.environ["NODE_RANK"] == '0') and "LOCAL_RANK" not in os.environ:
        logger_dict = utils.instantiate_loggers(cfg.logger)
        wandb_lg, local_lg = logger_dict['wandb'], logger_dict['local']
        utils.wandb_hook(wandb_lg.experiment, trainer_mod)
        trainer_mod.local_lg = local_lg
        trainer_mod.wandb_lg = wandb_lg
        logger = list(logger_dict.values())
        local_lg.info(f"wandb sync {os.path.dirname(wandb_lg.experiment.dir)}")

    else:
        logger = None
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)  # , devices=find_usable_cuda_devices(1))

    if cfg.validation.pure:
        ckpt_path = cfg.checkpoint.path_list[cfg.validation.model]
        trainer.test(trainer_mod, datamodule=data_mod, ckpt_path=ckpt_path)
        exit(0)
    # the weight and training state are loaded;
    # the hparams will not be replaced by loaded model;
    if cfg.checkpoint.model:
        ckpt_path = cfg.checkpoint.path_list[cfg.checkpoint.model]
    else:
        ckpt_path = None

    trainer.fit(trainer_mod, datamodule=data_mod, ckpt_path=ckpt_path)
    if logger is not None:
        local_lg.log_hyperparams(cfg)
        wandb_lg.experiment.finish(quiet=True)
        print(f"wandb sync \\\n{os.path.dirname(wandb_lg.experiment.dir)}")


if __name__ == "__main__":
    hydra_main()
