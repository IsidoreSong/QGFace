import logging
import os
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from rich.logging import RichHandler
from pytorch_lightning.callbacks import RichProgressBar
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torchvision import transforms
import cv2

logger = None

def init_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    global logger
    logger = logging.getLogger()
    logger.handlers = []
    fmt_shell = logging.Formatter("%(message)s", datefmt="%m%d %H:%M:%S")
    fmt_file = logging.Formatter("[%(asctime)s]-%(message)s-[%(filename)s.%(funcName)s:%(lineno)d]", datefmt="%m-%d %H:%M:%S")
    shell_handler = RichHandler(show_level=False, show_path=False, rich_tracebacks=True)
    shell_handler.setFormatter(fmt_shell)
    logger.addHandler(shell_handler)

    if save_dir is not None:
        file_handler = logging.FileHandler(os.path.join(save_dir, "main.log"))
        file_handler.setFormatter(fmt_file)
        logger.addHandler(file_handler)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))


def recur_dict(key_info):
    info_dict = {}
    for k, v in key_info.items():
        if isinstance(v, DictConfig):
            info_dict[k] = convert_params(v)
        else:
            info_dict[k] = v
            if isinstance(v, ListConfig):
                info_dict[k] = str(v)
            # elif isinstance(v, float):
            #     info_dict[k] = round(v, 4)
    return info_dict


class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        if "v_num" in items:
            items.pop("v_num")
        if "loss" in items:
            items.pop('loss')
        return items


def convert_params(hparams, upper_level="", sep="_"):
    """
    Flatten structure hparams to dict.

    Returns:
        dict: sep_char joined key and original values
    """
    flatten_dict = {}
    for k, v in hparams.items():
        nameholder = sep.join([upper_level, k]).lstrip(sep)
        if isinstance(v, DictConfig):
            flatten_dict.update(convert_params(v, nameholder))
        else:
            flatten_dict[nameholder] = v
            if isinstance(v, ListConfig):
                flatten_dict[nameholder] = str(v)
            elif isinstance(v, float):
                flatten_dict[nameholder] = round(v, 4)

    return flatten_dict


def cmp__update_dict(dict_a, dict_b):
    # TODO: save bigger or save smaller
    for k, v in dict_b.items():
        if (k not in dict_a) or ((dict_b[k] is not None) and ((dict_a[k] is None) or (dict_b[k] > dict_a[k]))):
            dict_a[k] = v
    for k, v in dict_a.items():
        if isinstance(v, float):
            dict_a[k] = round(v, 4)


class LocalLogger(Logger):
    """
    Save experiments information locally.
    The hyperparameters are stored in a csv file.
    The metrics and running info are saved in log file.
    """

    def __init__(self, save_dir, model_info_dir, prefix="", flush_logs_every_n_steps=1, key_info=None):
        super().__init__()
        self._save_dir = os.fspath(save_dir)
        self.prefix = prefix
        self.key_info = key_info
        os.makedirs(self._save_dir, exist_ok=True)
        self._experiment = LocalExperimentWriter(self._save_dir)
        self.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.LOGGER_JOIN_CHAR = "_"
        self.model_info_dir = model_info_dir
        os.makedirs(self.model_info_dir, exist_ok=True)
        self.norm_len, self.sim_len, self.P_len = 50, 100, 100
        self.X, self.Y = torch.meshgrid(
            torch.arange(self.norm_len), torch.arange(self.sim_len)
        )
        self.norm_sim_cla_heat = torch.zeros(
            (self.norm_len, self.sim_len), dtype=torch.long
        )
        self.norm_sim_con_heat = torch.zeros(
            (self.norm_len, self.sim_len), dtype=torch.long
        )
        self.norm_P_heat = torch.zeros(
            (self.norm_len, self.P_len), dtype=torch.long
        )

    def info(self, message):
        self.experiment.logger.info(message)

    @rank_zero_only
    def update_heat(self, norm_idx, sim_idx, target_map, step=None):
        index = torch.cat([norm_idx, sim_idx], dim=1)
        norm_sequent = torch.arange(norm_idx.shape[0])
        target_map = getattr(self, target_map)
        index_map = torch.zeros_like(target_map)
        index_map[index[:, 0], index[:, 1]] = norm_sequent
        target_map[index[:, 0], index[:, 1]] += 1
        un_touched_idx = set(norm_sequent.numpy()) - set(
            index_map[index[:, 0], index[:, 1]].numpy()
        )
        # un_touched_idx = torch.tensor(list(un_touched_idx), dtype=torch.long)
        for i in un_touched_idx:
            target_map[index[i].split(1)] += 1

    @rank_zero_only
    def log_norm__p(self, epoch, cmap="YlOrRd"):
        def _get_c_lis(cmap="jet"):
            cmo = plt.cm.get_cmap(cmap)
            cs, k = list(), 256 / cmo.N
            for i in range(cmo.N):
                c = cmo(i)
                cs += [c] * (int((i + 1) * k) - int(i * k))
            cs = np.array(cs)
            return cs

        def _plot_fake_hist(X, P_means, colors):
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot()
            ax.plot(X, P_means)
            ax.bar(X, P_means, color=colors, width=1)
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

            ax.set_xlim(0, self.norm_len)
            ax.set_ylim(0, self.P_len)
            ax.set_title("Norm & P")
            ax.set_xlabel("norm")
            ax.set_ylabel("P")
            save_pth = os.path.join(
                self.model_info_dir, f"P_{epoch:02d}.jpg"
            )
            fig.savefig(save_pth)
            return save_pth

        num_P = self.norm_P_heat.sum(dim=1)
        normed_num_P = np.uint8(
            255 * (num_P - num_P.min()) / (num_P.max() - num_P.min())
        )
        P_weighted = (
            torch.arange(self.P_len, dtype=torch.float).view(1, -1)
            * self.norm_P_heat
        ).sum(dim=1)
        P_means = P_weighted / num_P
        P_means = np.where(np.isnan(P_means), 0, P_means)
        c_lis = _get_c_lis(cmap=cmap)
        colors = c_lis[normed_num_P]
        X = np.arange(self.norm_len)
        save_pth = _plot_fake_hist(X, P_means, colors)
        return "train-stat/P", save_pth
    
    @rank_zero_only
    def log_heatmap(self, epoch, contrast_apply):
        def _plot_heatmap(heat_matrix_name, epoch):
            # TODO: improve quality to paper figure
            heat_matrix = getattr(self, heat_matrix_name)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot()
            pcm = ax.pcolormesh(
                self.X, self.Y, heat_matrix, cmap="jet", shading="gouraud"
            )
            # TODO: set colorbar boundary
            fig.colorbar(pcm, extend="both")
            ax.set_title(heat_matrix_name)
            ax.set_xlabel("norm")
            ax.set_ylabel("similarity")
            save_pth = os.path.join(
                self.model_info_dir,
                f"{heat_matrix_name}_{epoch:02d}.jpg",
            )
            fig.savefig(save_pth)
            
            return save_pth

        fig_pth1 = _plot_heatmap("norm_sim_cla_heat", epoch)
        fig_dict = {"train-stat/norm_sim_cla_heat": fig_pth1}
        if contrast_apply:
            fig_pth2 = _plot_heatmap("norm_sim_con_heat", epoch)
            fig_dict["train-stat/norm_sim_con_heat"] = fig_pth2
        return fig_dict
    
    def log_images(self, images, labels, norms, P, epoch, N=3):
        ori_idx = np.arange(norms.shape[0], dtype=np.int64)
        sample_weight = (100.0 - P + norms).softmax(dim=0).numpy().reshape(-1)
        target_idx = np.random.choice(
            ori_idx, N, replace=False, p=sample_weight
        )
        m_s = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        ToImage = transforms.ToPILImage()
        sample_save_path = os.path.join(self._save_dir, "PoQu_check")
        if not os.path.isfile(sample_save_path):
            os.makedirs(sample_save_path, exist_ok=True)
        images = images.cpu()
        img_pth_list = []
        for idx in target_idx:
            img = images[idx]
            aug_img = img * m_s + m_s
            img = ToImage(aug_img)
            save_pth = os.path.join(
                sample_save_path,
                f"E{epoch:02d}_{norms[idx].item()}_{P[idx].item():02d}_{labels[idx].item()}_{idx}.jpg",
            )
            cv2.imwrite(save_pth, np.array(img))
            img_pth_list.append(save_pth)
        return img_pth_list
    
    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @property
    @rank_zero_experiment
    def experiment(self):
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the
        following.

        Example::

            self.logger.experiment.some_experiment_writer_function()
        """
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        """
        Flatten hierarchical ConfigDict to one-layer dict.
        """
        self.experiment.log_hyperparams(params.logger.local.key_info)
        self.save()

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        # if step is not None and (step + 1) % self.flush_logs_every_n_steps == 0:
        if step is not None:
            metrics = _add_prefix(metrics, self.prefix, self.LOGGER_JOIN_CHAR)
            self.experiment.log_metrics(metrics, step + 1)
            # if (step + 1) % self.flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self) -> None:
        """Save recorded hparams and metrics into files.
        The csv file saving checks the hparams for dynamic adaptation and updates the key metrics.
        The log file saving is automatic.
        """
        super().save()
        self.experiment.save()
        return

class LocalExperimentWriter:
    r"""
    Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_METRICS_FILE = "metrics.csv"

    def __init__(self, log_dir: str) -> None:
        self.project_id = os.path.basename(log_dir)
        self.metrics = {"project_id": self.project_id}
        # self.logger = logging.getLogger("LocalLogger")
        self.logger = logging.getLogger("LocalLogger")
        self.trans_dict = {"trainer/lr_step": "lr", 
                           "trainer/loss_step": "loss", 
                           "trainer/loss_cla_step": "loss-cla",
                           "trainer/loss_con_step": "loss-con",
                           "trainer/m_step": "m", 
                           "train-stat/sim_cla": "sim-cla", 
                           "train-stat/sim_con": "sim-con", 
                           "train-stat/norm_cla": "norm-cla", 
                           "train-stat/norm_con": "norm-con", 
                           }
        self.exclude_list = ["lr_step$"]
        self.log_dir = log_dir

        if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(os.path.dirname(self.log_dir), self.NAME_METRICS_FILE)

    def log_hyperparams(self, params):
        hparams = convert_params(params)
        cmp__update_dict(self.metrics, hparams)
        self.logger.info(OmegaConf.to_yaml(recur_dict(params)))
    
    def log_metrics(self, metrics, step=None) -> None:
        """Record metrics."""

        progress_keys = ["p/current_e", "p/max_e", "p/completed", "p/remaining", "p/total", "p/elapsed"]
        log_str = ""
        if progress_keys[0] in metrics:
            elapsed_delta = "-:--:--" if metrics["p/elapsed"] is None else str(timedelta(seconds=int(metrics["p/elapsed"])))
            remaining_delta = "-:--:--" if metrics["p/remaining"] is None else str(timedelta(seconds=int(metrics["p/remaining"])))
            whole_delta = "-:--:--"
            if metrics["p/elapsed"] is not None and metrics["p/remaining"] is not None:
                epoch_delta = int(metrics["p/elapsed"] + metrics["p/remaining"])
                whole_delta = epoch_delta * (metrics["p/max_e"] - metrics["p/current_e"] - 1) + metrics["p/remaining"]
                whole_delta = str(timedelta(seconds=int(whole_delta)))
            # step_num_len = int(metrics['p/total'])
            log_str += f"[E{int(metrics['p/current_e']+1):02d}/{int(metrics['p/max_e']):2d}]"
            log_str += f"[{int(metrics['p/completed']+1)}/{int(metrics['p/total'])}]"
            log_str += f"[{elapsed_delta}/{remaining_delta}/{whole_delta}] "
            for key in progress_keys:
                metrics.pop(key)
        else:
            log_str += f"[{step:6d}][E{metrics['epoch']:2d}]"
        metrics.pop("epoch")

        key_metrics = {"acc": 0, "tinyface": 0, "IJB": 0, "scface": 0, "best_threshold": 0, "num_samples": 0}
        target_keys = []
        for key_short in key_metrics:
            for key in metrics:
                if key_short in key:
                    if key_metrics[key_short] == 0:
                        self.logger.info(log_str[:-1])
                        log_str = ""
                        key_metrics[key_short] += 1
                    if key_short == "num_samples":
                        log_str += f"<{key.split('/')[-1]}:{int(metrics[key])}>-"
                    else:
                        log_str += f"<{key.split('/')[-1]}:{metrics[key]:.4f}>-"
                    target_keys.append(key)
                    
        for key in target_keys:
            metrics.pop(key)
        
        for key in metrics:
            if key in self.trans_dict:
                log_str += f"<{self.trans_dict[key]}:{metrics[key]:.2f}>-"
            elif isinstance(metrics[key], float):
                log_str += f"<{key}:{metrics[key]:.4f}>-"
            else:
                log_str += f"<{key}:{metrics[key]}>-"

        self.logger.info(log_str[:-1])
        cmp__update_dict(self.metrics, metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        
        return 
    
        if not self.metrics:
            return
        
        # TODO: if fail to open
        # @retry(delay=0.2, logger=logger)
        def _save_metric_local():
            one_df = pd.DataFrame([self.metrics])
            if os.path.isfile(self.metrics_file_path):
                metric_df = pd.read_csv(self.metrics_file_path)
                project_list = metric_df["project_id"].to_list()
                if self.project_id in project_list:
                    # metric_df.loc[metric_df["project_id"] == self.project_id] = self.metrics
                    metric_df = pd.concat([metric_df, one_df])
                    metric_df = metric_df.drop_duplicates("project_id", keep="last")
                else:
                    metric_df = pd.concat([metric_df, one_df])
            else:
                metric_df = one_df
            metric_df.to_csv(self.metrics_file_path, index=None)
        
        _save_metric_local()
