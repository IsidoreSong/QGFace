import os
import torch
import wandb
from functools import partialmethod
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
import validation.evaluate_utils as evaluate_utils
import net as net
import numpy as np
import utils
import hydra
import copy
from easydict import EasyDict as edict
from warnings import filterwarnings
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from validation.validation_lq import (
    validate_tinyface,
    tinyface_helper,
    scface_helper,
)
from validation.validation_mixed import validate_IJB_BC


filterwarnings("ignore", category=PossibleUserWarning)
filterwarnings(
    "ignore",
    category=UserWarning,
    message="Be aware that when using `ckpt_path`, callbacks used to create the checkpoint need to be provided during `Trainer` instantiation.",
)
filterwarnings(
    "ignore",
    category=UserWarning,
    message="Experiment logs directory .* exists and is not empty.",
)
filterwarnings(
    "ignore",
    category=UserWarning,
    message="Checkpoint directory .* exists and is not empty.",
)
filterwarnings(
    "ignore",
    category=UserWarning,
    message="UserWarning: incompatible copy of pydevd already imported",
)
filterwarnings(
    "ignore",
    category=UserWarning,
    message=".* therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.",
)


class FaceModel(LightningModule):
    def __init__(self, **kwargs):
        super(FaceModel, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        self.class_num = self.hparams.data.class_num
        self.head = hydra.utils.instantiate(self.hparams.head)
        self.c_head = hydra.utils.instantiate(self.hparams.contrast.qgface)
        self.backbone = net.build_model(model_name=self.hparams.arch)
        self.tinyface_test = tinyface_helper.TinyFaceTest(
            tinyface_root=self.hparams.validation.tinyface.data_root,
            alignment_dir_name=self.hparams.validation.tinyface.aligned_dir,
        )
        self.scface_test = scface_helper.SCFaceTest(
            data_root=self.hparams.validation.scface.data_root,
            aligned_dir=self.hparams.validation.scface.aligned_dir,
        )
        if self.hparams.contrast.queue.name == "moco":
            self.moco_backbone = copy.deepcopy(self.backbone)
            self.moco_head = copy.deepcopy(self.head)
            for param in self.moco_backbone.parameters():
                param.requires_grad = False
            for param in self.moco_head.parameters():
                param.requires_grad = False
        self.register_buffer("queue_proxies", torch.empty(0))
        self.register_buffer("queue_embeds", torch.empty(0))
        self.register_buffer("queue_labels", torch.empty(0, dtype=torch.long))
        self.validation_step_dict = {
            "hq": self.validation_high_step,
            "crlfw": self.validation_high_step,
            "tinyface": self.validation_tinyface_step,
            "scface": self.validation_tinyface_step,
            "IJBB": self.validation_IJB_step,
        }
        self.validation_list = []
        for dataset_name in self.hparams.validation.validation_list:
            if getattr(self.hparams.validation, dataset_name).apply:
                self.validation_list.append(dataset_name)

        # TODO: move
        self.norm_len, self.sim_len, self.P_len = 50, 100, 100
        self.X, self.Y = torch.meshgrid(
            torch.arange(self.norm_len), torch.arange(self.sim_len)
        )

        self.hooks = edict(
            {
                "head": {
                    "log_softmax": utils.OutFeatureHook(self.head.log_softmax)
                },
                "c_head": {
                    "log_softmax": utils.OutFeatureHook(self.c_head.log_softmax)
                },
            }
        )

        # TODO: config with gradient_clip_val in trainer
        # self.automatic_optimization = False

    def on_load_checkpoint(self, checkpoint):
        hparams = edict(checkpoint["hyper_parameters"])
        self.queue_proxies = torch.zeros(
            hparams.contrast.queue.queue_size,
            hparams.head.embedding_size,
        )
        self.queue_embeds = torch.zeros(
            hparams.contrast.queue.queue_size,
            hparams.head.embedding_size,
        )
        self.queue_labels = torch.zeros(
            hparams.contrast.queue.queue_size, dtype=torch.long
        )

    def get_current_lr(self):
        scheduler = None
        if scheduler is None:
            try:
                scheduler = self.trainer.lr_scheduler_configs[0].scheduler
            except:
                pass

        if scheduler is None:
            raise ValueError("lr calculation not successful")

        if isinstance(scheduler, lr_scheduler._LRScheduler):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = scheduler.get_epoch_values(self.current_epoch)[0]
        return lr

    def forward(self, images):
        embeddings, norms = self.backbone(images)
        return embeddings, norms

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward_16(self, images):
        embeddings, norms = self.backbone(images)
        return embeddings, norms

    def on_fit_start(self):
        if not self.hparams.data.contrast_view:
            if (
                self.hparams.contrast.queue.name is not None
                or self.hparams.contrast.apply
            ):
                print("Data Augmentation is needed for contrastive learning!")
                exit(1)

        if self.hparams.contrast.queue.sync_GPU:
            self.expel_local_group = [
                _
                for _ in range(self.trainer.world_size)
                if _ != self.global_rank
            ]
        self.contrast_queue_size = self.hparams.contrast.queue.queue_size
        self.contrast_apply_epoch = 0
        self.queue_embeds = self.queue_embeds.to(self.device)
        self.queue_labels = self.queue_labels.to(self.device)
        self.queue_proxies = self.queue_proxies.to(self.device)
        self.log_on_fit_start()

    @rank_zero_only
    def log_on_fit_start(self):
        framework_pth = os.path.join(self.hparams.project_dir, "framework.py")
        head_pth = os.path.join(self.hparams.project_dir, "head.py")
        data_pth = os.path.join(self.hparams.project_dir, "dataset/record_dataset.py")
        self.wandb_lg.experiment.save(framework_pth, policy="now")
        self.wandb_lg.experiment.save(head_pth, policy="now")
        self.wandb_lg.experiment.save(data_pth, policy="now")

    def on_train_epoch_start(self):
        # from rich.progress import Progress
        # if self.global_rank == 0:
        #     progress = Progress(console=self.trainer.progress_bar_callback.progress.console)
        #     with progress:
        #         for i, n in progress.track(enumerate(self.trainer.train_dataloader), total=len(self.trainer.train_dataloader)):
        #             if i % 100 == 0:
        #                 progress.print(i)
        # else:
        # console = live.console
        # progress2 = Progress(console=console)
        # task2 = progress2.add_task("Task 2", total=200)
        # for i in range(200):
        #     progress2.update(task2, advance=1)
        #     live.update(progress2)
        
        # from tqdm import tqdm
        # # if self.global_rank == 0:
        # for i, n in tqdm(enumerate(self.trainer.train_dataloader), total=len(self.trainer.train_dataloader), 
        #                 disable=self.global_rank!=0, miniters=50, maxinterval=10):
        #     pass

        self.norm_sim_cla_heat = torch.zeros(
            (self.norm_len, self.sim_len), dtype=torch.long
        )
        self.norm_sim_con_heat = torch.zeros(
            (self.norm_len, self.sim_len), dtype=torch.long
        )
        self.norm_P_heat = torch.zeros(
            (self.norm_len, self.P_len), dtype=torch.long
        )
        # add contrast loss after the first tailing of lr -> epoch E_th
        # lr is tailed at the end of E_th -> contrast loss is added at the beginning of E_th+1
        if (
            self.hparams.contrast.later_joint
            and self.trainer.current_epoch < self.hparams.contrast.joint_point
        ):
            self.contrast_weight = 0
            self.contrast_apply_epoch = self.current_epoch
        else:
            self.contrast_weight = self.hparams.contrast.weight
        if (
            self.hparams.contrast.later_aug
            and self.trainer.current_epoch < self.hparams.contrast.joint_point
        ):
            self.contrast_later_aug = True
        else:
            self.contrast_later_aug = False
        if (
            self.hparams.contrast.later_detach
            and self.trainer.current_epoch >= self.hparams.contrast.joint_point
        ):
            self.contrast_weight = 0
        if self.hparams.contrast.warmup_scaler:
            self.head.quality_scale = (
                min(
                    1,
                    self.trainer.current_epoch
                    / self.hparams.contrast.joint_point,
                )
                * self.hparams.contrast.quality_scale
            )
            self.c_head.quality_scale = self.head.quality_scale
        if (
            self.hparams.contrast.progressive
            and self.hparams.contrast.queue.extra_queue
        ):
            self.contrast_queue_size = int(
                (
                    self.hparams.contrast.queue.queue_size
                    - self.hparams.data.batch_size
                )
                / (self.trainer.max_epochs - self.contrast_apply_epoch)
                * (self.current_epoch - self.contrast_apply_epoch)
                + self.hparams.data.batch_size
            )
        lr = self.get_current_lr()
        self.log(
            "trainer/lr",
            lr,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # * sync all samples for negative pool
        # all_outputs = self.all_gather(outputs, sync_grads=True)
        if self.hparams.contrast.queue.name != "moco":
            batch_size = labels.shape[0]
            if self.hparams.data.contrast_view:
                if self.hparams.contrast.later_aug:
                    images = images[:, 1]
                    labels = labels
                else:
                    images = torch.cat([images[:, 1], images[:, 0]])
                    labels = torch.cat([labels, labels], dim=0)
            embeddings, norms = self.forward_16(images)
            loss_cla, margin_scaler = self.head(embeddings, norms, labels)
            if self.hparams.contrast.queue.name is not None:
                k_norms, q_norms = norms.split(batch_size)
                k_embeddings, q_embeddings = embeddings.split(batch_size)
                k_margin_scaler, q_margin_scaler = margin_scaler.split(batch_size)
                k_labels, q_labels = labels.split(batch_size)
                # if self.hparams.contrast.queue.name == "proxy":
                    # images, norms = images[: q_labels.shape[0]], q_norms
                    # embeddings, labels = k_embeddings, k_labels
                    # self.head.similarity = self.head.similarity[: q_labels.shape[0]]
                # TODO: add BroadFace (note the self.head above doesn't backwards)

        elif self.hparams.contrast.queue.name == "moco":
            q_embeddings, q_norms = self.forward_16(images[:, 1])
            loss_cla, q_margin_scaler = self.head(q_embeddings, q_norms, labels)
            with torch.no_grad():
                # this mean the images used for training classifier is reduced by half
                k_embeddings, k_norms = self.moco_backbone(images[:, 0])
                moco_loss_cla, k_margin_scaler = self.moco_head(
                    k_embeddings, k_norms, labels
                )
                self._momentum_update_key_encoder()
            images, norms = images[:, 0], q_norms
            q_labels, embeddings = labels, k_embeddings
            
        if self.hparams.contrast.queue.name is not None:
            queue_embeddings = self._dequeue_and_enqueue(embeddings, labels)
            loss_con = self.c_head(
                [q_embeddings, k_embeddings, queue_embeddings],
                [q_norms, k_norms],
                [q_labels, self.queue_labels],
                [q_margin_scaler, k_margin_scaler],
                embeddings.shape[0],
            )
        else:
            q_embeddings, k_embeddings, queue_embeddings = None, None, None
            loss_con = 0
        # loss, loss_con = self.manual_step(loss_cla, loss_con)
        loss = loss_cla + self.contrast_weight * loss_con

        self.log_training_step(
            batch_idx,
            images,
            labels,
            norms,
            loss_cla,
            loss_con,
            loss,
            q_embeddings,
            k_embeddings,
            queue_embeddings
        )
        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        # gather keys before updating queue
        proxies = self.head.kernel.data.T
        if self.hparams.contrast.queue.sync_GPU:
            all_keys = self.all_gather(keys, sync_grads=False)
            all_labels = self.all_gather(labels, sync_grads=False)
            all_keys = torch.cat([keys, *all_keys[self.expel_local_group]])
            all_labels = [labels, *all_labels[self.expel_local_group]]
            all_proxies = [proxies[l] for l in all_labels]
            all_labels = torch.cat(all_labels)
            all_proxies = torch.cat(all_proxies)

        else:
            all_keys, all_labels, all_proxies = (
                keys,
                labels,
                proxies[labels],
            )
        if self.hparams.contrast.queue.name is None:
            self.queue_labels = all_labels
            self.queue_proxies = all_proxies
            return all_keys
        cut_slice = slice(self.contrast_queue_size)
        self.queue_embeds = torch.cat(
            [all_keys.clone().detach(), self.queue_embeds]
        )[cut_slice]
        self.queue_labels = torch.cat([all_labels, self.queue_labels])[
            cut_slice
        ]
        self.queue_proxies = torch.cat([all_proxies, self.queue_proxies])[
            cut_slice
        ]
        queue_embeddings = self.queue_embeds
        if self.hparams.contrast.queue.name == "proxy":
            with torch.no_grad():
                delta_weight = proxies[self.queue_labels] - self.queue_proxies
                queue_embeddings = (
                    self.queue_embeds
                    + (
                        self.queue_embeds.norm(p=2, dim=1, keepdim=True)
                        / self.queue_proxies.norm(p=2, dim=1, keepdim=True)
                    )
                    * delta_weight
                )
        return queue_embeddings

    def log_training_step(
        self,
        batch_idx,
        images,
        labels,
        norms,
        loss_cla,
        loss_con,
        loss,
        q_embeddings,
        k_embeddings,
        queue_embeddings
    ):
        P_pos = self.hooks.head.log_softmax.features.gather(
            1, labels.view(-1, 1)
        ).exp()
        rounded_norm_cla = self.round(norms, 1, 0 + 1, self.norm_len - 1)
        rounded_sim_cla = self.round(
            self.head.similarity, 100, 0 + 1, self.sim_len - 1
        )

        rounded_P_pos = self.round(P_pos, 100, 0 + 1, self.P_len - 1)

        self.update_heat(
            rounded_norm_cla,
            rounded_sim_cla,
            "norm_sim_cla_heat",
            step=self.global_step,
        )

        self.update_heat(
            rounded_norm_cla,
            rounded_P_pos,
            "norm_P_heat",
            step=self.global_step,
        )
        hist_dict = {
            "train-stat/norm_cla": rounded_norm_cla,
            "train-stat/sim_cla": rounded_sim_cla,
        }
        if self.hparams.contrast.apply:
            rounded_norm_con = self.round(
                self.c_head.norms, 1, 0 + 1, self.norm_len - 1
            )
            rounded_sim_con = self.round(
                self.c_head.similarity, 100, 0 + 1, self.sim_len - 1
            )
            self.update_heat(
                rounded_norm_con,
                rounded_sim_con,
                "norm_sim_con_heat",
                step=self.global_step,
            )
            hist_dict["train-stat/sim_con"] = rounded_sim_con
            hist_dict["train-stat/norm_con"] = rounded_norm_con
            
        process_dict = self.get_progress_dict()
        self.log_dict(
            process_dict, on_epoch=False, logger=True, sync_dist=False
        )
        self.log_images(
            images, labels, rounded_norm_cla, rounded_P_pos, batch_idx
        )
        sync_dict = {
            "trainer/loss": loss,
            "trainer/loss_cla": loss_cla,
            "trainer/loss_con": loss_con,
            "trainer/m": self.c_head.batch_mean.item(),
        }
        if q_embeddings is not None and k_embeddings is not None:
            with torch.no_grad():
                # TODO: check
                same_embed_diff = (
                    10000 * (q_embeddings - k_embeddings).mean().clone().detach()
                )
                diff_embed_diff = 0
                # diff_embed_diff = (
                #     10000
                #     * (q_embeddings.unsqueeze(0) - queue_embeddings.unsqueeze(1))
                #     .mean()
                #     .clone()
                #     .detach()
                # )
            sync_dict["trainer/same_embed_diff"] = same_embed_diff
            sync_dict["trainer/de_diff"] = diff_embed_diff

        no_logger_dict = {"l": loss_cla, "l_C": loss_con}
        self.log_dict(sync_dict, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict(no_logger_dict, logger=False, prog_bar=True)
        self.log_histogram(hist_dict=hist_dict, step=self.global_step)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        if self.hparams.contrast.queue.name != "moco":
            return
        m = self.hparams.contrast.queue.moco_m
        for param_q, param_k in zip(
            self.backbone.parameters(),
            self.moco_backbone.parameters(),
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(
            self.head.parameters(), self.moco_head.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @rank_zero_only
    def log_images(self, images, labels, norms, P, batch_idx, N=3):
        data_step = (
            len(self.trainer.datamodule.train_dataset)
            // self.trainer.datamodule.batch_size
            // self.trainer.world_size
            // self.trainer.log_every_n_steps
            * self.trainer.log_every_n_steps
        )
        if isinstance(self.trainer.limit_train_batches, int):
            target_step = min(data_step, self.trainer.limit_train_batches)
        elif isinstance(self.trainer.limit_train_batches, float):
            target_step = int(data_step * self.trainer.limit_train_batches)

        if batch_idx + 1 != target_step:
            return
        epoch = self.trainer.current_epoch + 1
        img_path_list = self.local_lg.log_images(
            images, labels, norms, P, epoch, N
        )
        for img_path in img_path_list:
            self.wandb_lg.experiment.log(
                {"train-PoQu/Pic": wandb.Image(img_path)},
                step=self.global_step,
            )

    def round(self, tensor, scale, _min, _max):
        rounded = (tensor.clone().detach() * scale).round().to(torch.long)
        rounded = rounded.clip(_min, _max)
        rounded = rounded.cpu()
        return rounded

    def log_histogram(self, hist_dict, step=None):
        if step is None or (step + 1) % self.trainer.log_every_n_steps == 0:
            for k, v in hist_dict.items():
                self.log(
                    f"{k}",
                    v.to(torch.float).mean().item(),
                    logger=True,
                    sync_dist=True,
                )
        if (
            self.trainer.training
            and self.trainer.is_last_batch
            and self.global_rank == 0
        ):
            for k, v in hist_dict.items():
                self.wandb_lg.experiment.log(
                    {k + "_hist": wandb.Histogram(v)}, step=step
                )

    @rank_zero_only
    def update_heat(self, *args, step=None, **kwargs):
        if (step + 1) % self.trainer.log_every_n_steps == 0:
            self.local_lg.update_heat(*args, **kwargs)

    def get_progress_dict(self):
        if self.progress_bar.main_progress_bar_id is None:
            return {}
        progress = self.progress_bar.main_progress_bar
        elapsed = (
            progress.finished_time if progress.finished else progress.elapsed
        )
        remaining = progress.time_remaining
        progress_dict = {
            "p/completed": float(progress.completed),
            "p/total": float(progress.total),
            "p/elapsed": float(elapsed),
            "p/current_e": float(self.trainer.current_epoch),
            "p/max_e": float(self.trainer.max_epochs),
            "p/remaining": float(remaining) if remaining is not None else 0.0,
        }

        return progress_dict

    @rank_zero_only
    def log_train_end(self):
        self.log("contrast_queue_size", self.contrast_queue_size)
        fig_dict = self.local_lg.log_heatmap(epoch=self.current_epoch, contrast_apply = self.hparams.contrast.apply)
        fig_name3, fig_pth3 = self.local_lg.log_norm__p(self.current_epoch)
        fig_dict[fig_name3] = fig_pth3
        self.wandb_lg.experiment.log(
            {
                k: wandb.Image(v) for k, v in fig_dict.items()
            },
            step=self.current_epoch,
        )

    def training_epoch_end(self, outputs):
        # if self.current_epoch == 0:
        #     return
        if (
            self.current_epoch + 1
            == self.hparams.lr_milestones[0]
        ):
            self.trainer.save_checkpoint(
                os.path.join(self.hparams.job_storage_dir, "mid.ckpt")
            )
        self.log_train_end()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if (
            self.current_epoch > 0
            and self.current_epoch
            < self.hparams.validation.check_val_after_epoch
        ):
            return None
        images, labels, dataname, image_index = batch
        embeddings, norms = self.validation_step_dict[
            self.validation_list[dataloader_idx]
        ](images)

        step_output = {
            "output": embeddings,
            "norm": norms,
            "target": labels,
            "dataname": dataname,
            "image_index": image_index,
        }
        if (
            self.hparams.trainer.strategy is not None
            and "ddp" in self.hparams.trainer.strategy
        ):
            # to save gpu memory
            for k, v in step_output.items():
                step_output[k] = v.to("cpu")

        return step_output


    def validation_epoch_end(self, outputs):
        if (
            self.current_epoch > 0
            and self.current_epoch
            < self.hparams.validation.check_val_after_epoch
        ):
            return None
        evaluate_idx = 0
        if isinstance(outputs[0], dict):
            outputs = [outputs]
        if self.hparams.validation.hq.apply:
            val_logs = self.inference_high(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.tinyface.apply:
            val_logs = self.inference_tinyface(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.scface.apply:
            val_logs = self.inference_scface(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.crlfw.apply:
            val_logs = self.inference_high(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.IJBB.apply:
            val_logs = self.inference_IJBB(outputs[evaluate_idx])
            evaluate_idx += 1

        return None

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if (
            self.current_epoch > 0
            and self.current_epoch
            < self.hparams.validation.check_val_after_epoch
        ):
            return None
        images, labels, dataname, image_index = batch
        embeddings, norms = self.validation_step_dict[
            self.validation_list[dataloader_idx]
        ](images)
        # print(self.global_rank, batch_idx, embeddings.sum())
        # if batch_idx == 4 or self.global_rank > 0:
        #     exit(0)
        step_output = {
            "output": embeddings,
            "norm": norms,
            "target": labels,
            "dataname": dataname,
            "image_index": image_index,
        }
        if (
            self.hparams.trainer.strategy is not None
            and "ddp" in self.hparams.trainer.strategy
        ):
            # to save gpu memory
            for k, v in step_output.items():
                step_output[k] = v.to("cpu")

        return step_output

    def test_epoch_end(self, outputs):
        if (
            self.current_epoch > 0
            and self.current_epoch
            < self.hparams.validation.check_val_after_epoch
        ):
            return None
        evaluate_idx = 0
        if isinstance(outputs[0], dict):
            outputs = [outputs]
        if self.hparams.validation.hq.apply:
            val_logs = self.inference_high(outputs[evaluate_idx])
            evaluate_idx += 1
            # torch.cuda.empty_cache()
        if self.hparams.validation.tinyface.apply:
            # torch.cuda.empty_cache()
            val_logs = self.inference_tinyface(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.scface.apply:
            # torch.cuda.empty_cache()
            val_logs = self.inference_scface(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.crlfw.apply:
            # torch.cuda.empty_cache()
            val_logs = self.inference_high(outputs[evaluate_idx])
            evaluate_idx += 1
        if self.hparams.validation.IJBB.apply:
            # torch.cuda.empty_cache()
            val_logs = self.inference_IJBB(outputs[evaluate_idx])
            evaluate_idx += 1
        return None

    def validation_high_step(self, images):
        embeddings, norms = self.forward(images)

        fliped_images = torch.flip(images, dims=[3])
        flipped_embeddings, flipped_norms = self.forward(fliped_images)
        stacked_embeddings = torch.stack(
            [embeddings, flipped_embeddings], dim=0
        )
        stacked_norms = torch.stack([norms, flipped_norms], dim=0)
        embeddings, norms = utils.fuse_features_with_norm(
            stacked_embeddings, stacked_norms
        )
        return embeddings, norms

    # TODO: merge tinyface & scface
    def validation_tinyface_step(self, images):
        embeddings, norms = validate_tinyface.infer_batch(
            self.backbone,
            self.hparams.validation.tinyface.use_flip_test,
            self.hparams.validation.tinyface.fusion,
            images,
        )
        return embeddings, norms

    def validation_IJB_step(self, images):
        embeddings, norms = validate_IJB_BC.infer_batch(
            self.backbone,
            self.hparams.validation.tinyface.use_flip_test,
            self.hparams.validation.tinyface.fusion,
            images,
        )
        return embeddings, norms

    def inference_IJB(self, outputs, subset_name):
        (
            all_output_tensor,
            all_norm_tensor,
            all_target_tensor,
            all_dataname_tensor,
        ) = self.gather_outputs(outputs)
        features = all_output_tensor.cpu().numpy()
        norms = all_norm_tensor.cpu().numpy()
        save_path = f"./IJB_{subset_name}_result"
        result_dict = validate_IJB_BC.verification(
            self.hparams.validation.IJBB.data_root,
            subset_name,
            features,
            save_path,
        )
        self.log_dict(result_dict, logger=True, sync_dist=False)

    # inference_IJBB = partialmethod(inference_IJB, subset_name="IJBB")
    # inference_IJBC = partialmethod(inference_IJB, subset_name="IJBC")
    def inference_IJBB(self, outputs):
        return self.inference_IJB(outputs, subset_name="IJBB")

    def inference_IJBC(self, outputs):
        return self.inference_IJB(outputs, subset_name="IJBC")

    def inference_scface(self, outputs):
        (
            all_output_tensor,
            all_norm_tensor,
            all_target_tensor,
            all_dataname_tensor,
        ) = self.gather_outputs(outputs)
        features = all_output_tensor.cpu().numpy()
        norms = all_norm_tensor.cpu().numpy()
        results = self.scface_test.test_identification(
            features, ranks=[1], gpu_id=self.local_rank
        )
        log_dict = dict(
            zip(
                ["scface/d1", "scface/d2", "scface/d3"],
                results,
            )
        )
        self.log_dict(log_dict, logger=True, sync_dist=False)

    def inference_tinyface(self, outputs):
        (
            all_output_tensor,
            all_norm_tensor,
            all_target_tensor,
            all_dataname_tensor,
        ) = self.gather_outputs(outputs)
        features = all_output_tensor.cpu().numpy()
        norms = all_norm_tensor.cpu().numpy()
        results = self.tinyface_test.test_identification(
            features, ranks=[1, 5, 20], gpu_id=self.local_rank
        )
        log_dict = dict(
            zip(
                [
                    "tinyface/rank_1",
                    "tinyface/rank_2",
                    "tinyface/rank_3",
                ],
                results,
            )
        )
        self.log_dict(log_dict, logger=True, sync_dist=False)

    def inference_high_one(
        self, dataname_idx, idx_to_dataname, gathered_outputs
    ):
        dataname = idx_to_dataname[dataname_idx.item()]
        (
            all_output_tensor,
            all_norm_tensor,
            all_target_tensor,
            all_dataname_tensor,
        ) = gathered_outputs
        embeddings = (
            all_output_tensor[all_dataname_tensor == dataname_idx]
            .to("cpu")
            .numpy()
        )
        labels = (
            all_target_tensor[all_dataname_tensor == dataname_idx]
            .to("cpu")
            .numpy()
        )
        norms = (
            all_norm_tensor[all_dataname_tensor == dataname_idx]
            .to("cpu")
            .numpy()
        )
        issame = labels[0::2]
        (
            tpr,
            fpr,
            accuracy,
            best_thresholds,
        ) = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
        pos_sim, neg_sim = self.get_val_similarity(embeddings, issame)
        acc, best_threshold = (
            accuracy.mean(),
            best_thresholds.mean(),
        )

        rounded_norm = self.round(
            torch.tensor(norms), 1, 0 + 1, self.norm_len - 1
        )
        rounded_pos_sim = self.round(
            torch.tensor(pos_sim), 100, 0 + 1, self.sim_len - 1
        )
        rounded_neg_sim = self.round(
            torch.tensor(neg_sim), 100, 0 + 1, self.sim_len - 1
        )
        if dataname_idx <= 4:
            hist_dict = {
                f"val-stat/{dataname}_norm": rounded_norm,
                f"val-stat/{dataname}_pos_sim": rounded_pos_sim,
                f"val-stat/{dataname}_neg_sim": rounded_neg_sim,
            }
            self.log_histogram(hist_dict)
        val_logs = {
            f"{dataname}_acc": torch.tensor(acc, dtype=torch.float),
            f"{dataname}_best_threshold": torch.tensor(
                best_threshold, dtype=torch.float
            ),
            f"{dataname}_num_samples": len(embeddings),
        }
        return val_logs

    def inference_high(self, outputs):
        gathered_outputs = self.gather_outputs(outputs)
        all_dataname_tensor = gathered_outputs[-1]
        dataname_to_idx = {
            "agedb30": 0,
            "cfpfp": 1,
            "lfw": 2,
            "cplfw": 3,
            "calfw": 4,
            "8x8": 5,
            "12x12": 6,
            "16x16": 7,
            "20x20": 8,
        }
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        val_logs = {}
        for dataname_idx in all_dataname_tensor.unique():
            val_log_one = self.inference_high_one(
                dataname_idx, idx_to_dataname, gathered_outputs
            )
            val_logs.update(val_log_one)

        high_version = (
            all_dataname_tensor.unique().cpu().numpy().mean().astype(int)
        )
        # if high_version == 2:
        val_logs[f"acc_{high_version}"] = np.mean(
            [
                val_logs[f"{dataname}_acc"]
                for dataname in dataname_to_idx.keys()
                if f"{dataname}_acc" in val_logs
            ]
        )
        ordered_log_dict = self.inference_high_log(val_logs)

    def get_val_similarity(self, embeddings, actual_issame):
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        similarity = (embeddings1 * embeddings2).sum(axis=1)
        bool_idx = actual_issame.astype(bool)
        pos_sim, neg_sim = similarity[bool_idx], similarity[~bool_idx]
        return pos_sim, neg_sim

    def inference_high_log(self, log_dict):
        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            if isinstance(value, int):
                return float(value)
            return value

        val_info, val_metric = {}, {}
        if "acc_2" in log_dict:
            self.log(
                "val/acc",
                round(_handle_value(log_dict.pop("acc_2")), 4),
                logger=True,
                prog_bar=True,
            )
        if "acc_7" in log_dict:
            self.log(
                "acc-cr",
                round(_handle_value(log_dict.pop("acc_6")), 4),
                logger=True,
                prog_bar=True,
            )

        for k in list(log_dict.keys()):
            if "acc" in k:
                val_metric["val/" + k] = round(
                    _handle_value(log_dict.pop(k)), 4
                )
        for k in list(log_dict.keys()):
            val_info["val-info/" + k] = round(_handle_value(log_dict.pop(k)), 4)

        self.log_dict(
            self.get_progress_dict(),
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        self.log_dict(val_metric, logger=True)
        self.log_dict(val_info, logger=True)

        return log_dict

    def gather_outputs(self, outputs):
        if (
            self.hparams.trainer.strategy is not None
            and "ddp" in self.hparams.trainer.strategy
        ):
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        # if self.trainer.is_global_zero:
        all_output_tensor = torch.cat(
            [out["output"] for out in outputs_list], axis=0
        ).to("cpu")
        all_norm_tensor = torch.cat(
            [out["norm"] for out in outputs_list], axis=0
        ).to("cpu")
        all_target_tensor = torch.cat(
            [out["target"] for out in outputs_list], axis=0
        ).to("cpu")
        all_dataname_tensor = torch.cat(
            [out["dataname"] for out in outputs_list], axis=0
        ).to("cpu")
        all_image_index = torch.cat(
            [out["image_index"] for out in outputs_list], axis=0
        ).to("cpu")

        # get rid of duplicate index outputs
        unique_dict = {}
        for _out, _nor, _tar, _dat, _idx in zip(
            all_output_tensor,
            all_norm_tensor,
            all_target_tensor,
            all_dataname_tensor,
            all_image_index,
        ):
            unique_dict[_idx.item()] = {
                "output": _out,
                "norm": _nor,
                "target": _tar,
                "dataname": _dat,
            }
        unique_keys = sorted(unique_dict.keys())
        all_output_tensor = torch.stack(
            [unique_dict[key]["output"] for key in unique_keys],
            axis=0,
        )
        all_norm_tensor = torch.stack(
            [unique_dict[key]["norm"] for key in unique_keys], axis=0
        )
        all_target_tensor = torch.stack(
            [unique_dict[key]["target"] for key in unique_keys],
            axis=0,
        )
        all_dataname_tensor = torch.stack(
            [unique_dict[key]["dataname"] for key in unique_keys],
            axis=0,
        )

        return (
            all_output_tensor,
            all_norm_tensor,
            all_target_tensor,
            all_dataname_tensor,
        )

    def configure_optimizers(self):
        # paras_only_bn, paras_wo_bn = self.separate_bn_paras(self.model)
        params = self.get_params(self.backbone, self.head)
        sgd = optim.SGD(
            params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
        )

        sgd_scheduler = lr_scheduler.MultiStepLR(
            sgd,
            milestones=self.hparams.lr_milestones,
            gamma=self.hparams.lr_gamma,
        )

        return [sgd], [sgd_scheduler]

    def get_params(self, backbone, head):
        paras_wo_bn, paras_only_bn = self.split_parameters(backbone)
        params = [
            {"params": paras_wo_bn + [head.kernel], "weight_decay": 5e-4},
            {"params": paras_only_bn},
        ]

        return params

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(
            params_no_decay
        )
        return params_decay, params_no_decay
