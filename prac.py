import sys

sys.path.append("/workspace")
sys.path.append("/public/home/fwang/workspace/QGFace/QGFace")
sys.path.append("/workspace/QGFace")
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)
import hydra
from pylogger import init_logger
from retry import retry
from time import sleep
import os


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print("START")
    from validation.validation_lq import tinyface_helper
    from dataset import convert

    tinyface_image_paths = tinyface_helper.get_all_files(
        os.path.join(
            cfg.validation.tinyface.data_root,
            cfg.validation.tinyface.aligned_dir,
        )
    )
    scface_image_paths = tinyface_helper.get_all_files(
        os.path.join(
            cfg.validation.scface.data_root,
            cfg.validation.scface.aligned_dir,
        )
    )
    lab_func = None
    # convert.make_rec(tinyface_image_paths, cfg.validation.tinyface.data_root, lab_func, dataset_name='tinyface', is_origin=True)


if __name__ == "__main__":
    main()
