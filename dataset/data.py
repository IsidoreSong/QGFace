import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torch
from torchvision import transforms
import numpy as np
import time
import pandas as pd
import validation.evaluate_utils as evaluate_utils
from dataset.image_folder_dataset import CustomImageFolderDataset, ContrastFolderDataset
from dataset.five_validation_dataset import (
    FiveValidationDataset,
    FiveBinValDataset,
)
import shutil
from dataset.record_dataset import (AugmentRecordDataset,    ContrastDataset,)
import logging
from validation.validation_lq import (
    data_utils,
    tinyface_helper,
    scface_helper,
    crlfw_helper,
)
from validation.validation_mixed import insightface_ijb_helper


logger = logging.getLogger()


class DataModule(pl.LightningDataModule):
    def __init__(self, validation, **kwargs):
        super().__init__()
        self.output_dir = kwargs["job_storage_dir"]
        self.data_root = kwargs["data_root"]
        self.train_data_path = kwargs["train_data_path"]
        self.val_data_path = kwargs["val_data_path"]
        self.train_data_subset = kwargs["train_data_subset"]
        self.swap_color_channel = kwargs["swap_color_channel"]
        self.use_mxrecord = kwargs["use_mxrecord"]

        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.rotation_augmentation_prob = kwargs["rotation_augmentation_prob"]
        self.crop_augmentation_prob = kwargs["crop_augmentation_prob"]
        self.photometric_augmentation_prob = kwargs[
            "photometric_augmentation_prob"
        ]
        self.contrast_view = kwargs["contrast_view"]
        self.LH_align = kwargs["LH_align"]
        self.validation = validation

        # self.keep_batch_size = kwargs["keep_batch_size"]
        # if self.contrast_view and not self.keep_batch_size:
        #     self.batch_size //= 2

        concat_mem_file_name = os.path.join(
            self.data_root, self.val_data_path, "concat_validation_memfile"
        )
        self.concat_mem_file_name = concat_mem_file_name
        self.prepare_data_per_node = False

    def prepare_data(self):
        logger.info("Preparing Data...")

    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            logger.info("Creating Training Dataset")
            self.train_dataset = train_dataset(
                self.data_root,
                self.train_data_path,
                self.rotation_augmentation_prob,
                self.crop_augmentation_prob,
                self.photometric_augmentation_prob,
                self.swap_color_channel,
                self.use_mxrecord,
                self.output_dir,
                self.contrast_view,
                self.LH_align,
            )

            if "faces_emore" in self.train_data_path and self.train_data_subset:
                # subset ms1mv2 dataset for reproducing the same setup in AdaFace ablation experiments.
                with open(os.path.join("/workspace/QGFace", "assets/ms1mv2_train_subset_index.txt"), "r") as f:
                    subset_index = [int(i) for i in f.read().split(",")]
                    self.subset_ms1mv2_dataset(subset_index)

            # logger.info("Creating Validataion Dataset")
        #     self.val_dataset = val_dataset(
        #         self.data_root, self.val_data_path, self.concat_mem_file_name
        #     )
        # # Assign Test split(s) for use in Dataloaders
        # if stage == "test" or stage is None:
        #     self.val_dataset = val_dataset(
        #         self.data_root, self.val_data_path, self.concat_mem_file_name
        #     )

    def train_dataloader(self):
        return DataLoader(
            pin_memory=True,
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataloader_list = []
        if self.validation.hq.apply:
            self.val_dataset = FiveBinValDataset(
                lfw_root=self.validation.hq.lfw_root,
                agedb_root=self.validation.hq.agedb_root,
                cfpfp_root=self.validation.hq.cfpfp_root,
                calfw_root=self.validation.hq.calfw_root,
                cplfw_root=self.validation.hq.cplfw_root,)
            
            five_dataloader = DataLoader(
                dataset=self.val_dataset,
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            dataloader_list.append(five_dataloader)
        if self.validation.tinyface.apply:
            image_paths = tinyface_helper.get_all_files(
                os.path.join(
                    self.validation.tinyface.data_root,
                    self.validation.tinyface.aligned_dir,
                )
            )
            image_paths = np.array(image_paths).astype(np.object).flatten()
            # TODO: fix cluster validation
            tinyface_dataloader = data_utils.prepare_dataloader(
                image_paths, self.batch_size, num_workers=self.num_workers,
                path_imgidx=self.validation.tinyface.path_imgidx,
                path_imgrec=self.validation.tinyface.path_imgrec,
            )
            dataloader_list.append(tinyface_dataloader)
        if self.validation.scface.apply:
            image_paths = scface_helper.get_all_files(self.validation.scface.data_root, 
                self.validation.scface.aligned_dir,
                )
            image_paths = np.array(image_paths).flatten()
            scface_dataloader = data_utils.prepare_dataloader(image_paths, self.batch_size, 
                num_workers=self.num_workers,
                path_imgidx=self.validation.scface.path_imgidx,
                path_imgrec=self.validation.scface.path_imgrec,
            )
            dataloader_list.append(scface_dataloader)
        if self.validation.crlfw.apply:
            crlfw_dataset = crlfw_helper.CRLFWTest(self.validation.crlfw.data_root)
            crlfw_dataloader = DataLoader(dataset=crlfw_dataset, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
            dataloader_list.append(crlfw_dataloader)
        if self.validation.IJBB.apply:
            img_paths, landmarks, faceness_scores = insightface_ijb_helper.dataloader.get_IJB_info(
                self.validation.IJBB.data_root, 'IJBB',
            )
            IJBB_dataloader = insightface_ijb_helper.dataloader.prepare_dataloader(
                img_paths,
                landmarks,
                self.batch_size,
                num_workers=self.num_workers,
                image_size=(112, 112),
                path_imgidx=self.validation.IJBB.path_imgidx,
                path_imgrec=self.validation.IJBB.path_imgrec,
                # image_is_saved_with_swapped_B_and_R = True
            )
            dataloader_list.append(IJBB_dataloader)
        return dataloader_list

    def test_dataloader(self):
        # return self.val_dataloader()
        # save_path = os.path.join("./tinyface_result", self.validation.model, self.validation.tinyface.fusion)

        # image_paths = tinyface_helper.get_all_files(
        #     os.path.join(
        #         self.validation.tinyface.data_root,
        #         self.validation.tinyface.aligned_dir,
        #     )
        # )
        # image_paths = np.array(image_paths).astype(np.object).flatten()
        # dataloader = data_utils.prepare_dataloader(
        #     image_paths, self.batch_size, num_workers=self.num_workers
        # )

        # image_paths = scface_helper.get_all_files(self.validation.scface.data_root, self.validation.scface.aligned_dir)
        # image_paths = np.array(image_paths).flatten()
        # dataloader = data_utils.prepare_dataloader(image_paths, self.batch_size, num_workers=self.num_workers)

        # crlfw_dataset = crlfw_helper.CRLFWTest(self.validation.crlfw.data_root)
        # dataloader = DataLoader(dataset=crlfw_dataset, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        # img_paths, landmarks, faceness_scores = insightface_ijb_helper.dataloader.get_IJB_info(
        #     self.validation.IJBB.data_root, 'IJBB'
        # )
        # dataloader = insightface_ijb_helper.dataloader.prepare_dataloader(
        #     img_paths,
        #     landmarks,
        #     self.batch_size,
        #     num_workers=self.num_workers,
        #     image_size=(112, 112),
        # )
        # return dataloader
        # return DataLoader(self.test_dataset, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return self.val_dataloader()

    def subset_ms1mv2_dataset(self, subset_index):
        # TODO: check
        # remove too few example identites
        self.train_dataset.samples = [
            self.train_dataset.samples[idx] for idx in subset_index
        ]
        self.train_dataset.targets = [
            self.train_dataset.targets[idx] for idx in subset_index
        ]
        value_counts = pd.Series(self.train_dataset.targets).value_counts()
        to_erase_label = value_counts[value_counts < 5].index
        e_idx = [i in to_erase_label for i in self.train_dataset.targets]
        self.train_dataset.samples = [
            i
            for i, erase in zip(self.train_dataset.samples, e_idx)
            if not erase
        ]
        self.train_dataset.targets = [
            i
            for i, erase in zip(self.train_dataset.targets, e_idx)
            if not erase
        ]

        # label adjust
        max_label = np.max(self.train_dataset.targets)
        adjuster = {}
        new = 0
        for orig in range(max_label + 1):
            if orig in to_erase_label:
                continue
            adjuster[orig] = new
            new += 1

        # readjust class_to_idx
        self.train_dataset.targets = [
            adjuster[orig] for orig in self.train_dataset.targets
        ]
        self.train_dataset.samples = [
            (sample[0], adjuster[sample[1]])
            for sample in self.train_dataset.samples
        ]
        new_class_to_idx = {}
        for label_str, label_int in self.train_dataset.class_to_idx.items():
            if label_int in to_erase_label:
                continue
            else:
                new_class_to_idx[label_str] = adjuster[label_int]
        self.train_dataset.class_to_idx = new_class_to_idx


def train_dataset(
    data_root,
    train_data_path,
    rotation_augmentation_prob,
    crop_augmentation_prob,
    photometric_augmentation_prob,
    swap_color_channel,
    use_mxrecord,
    output_dir,
    contrast_view,
    LH_align,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if use_mxrecord:
        train_dir = os.path.join(data_root, train_data_path)
        if os.environ['USER'] == 'fwang':
            train_dir = os.path.join('/dev/shm', train_data_path)
        train_dataset = ContrastDataset(
            root_dir=train_dir,
            transform=None,
            swap_color_channel=swap_color_channel,
            rotation_augmentation_prob=rotation_augmentation_prob,
            crop_augmentation_prob=crop_augmentation_prob,
            photometric_augmentation_prob=photometric_augmentation_prob,
            output_dir=output_dir,
            contrast_view=contrast_view,
            LH_align=LH_align,
        )

    else:
        train_dir = os.path.join(data_root, train_data_path, "imgs")
        train_dataset = ContrastFolderDataset(
            root_dir=train_dir,
            transform=train_transform,
            swap_color_channel=swap_color_channel,
            output_dir=output_dir,
            contrast_view=contrast_view,
            LH_align=LH_align,
        )

    return train_dataset


def val_dataset(data_root, val_data_path, concat_mem_file_name):

    val_data = evaluate_utils.get_val_data(
        os.path.join(data_root, val_data_path)
    )
    # theses datasets are already normalized with mean 0.5, std 0.5
    (
        age_30,
        cfp_fp,
        lfw,
        age_30_issame,
        cfp_fp_issame,
        lfw_issame,
        cplfw,
        cplfw_issame,
        calfw,
        calfw_issame,
    ) = val_data
    val_data_dict = {
        "agedb_30": (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
    # val_dataset = SingleValidationDataset(val_data_dict, concat_mem_file_name)

    return val_dataset


def test_dataset(data_root, val_data_path, concat_mem_file_name):
    val_data = evaluate_utils.get_val_data(
        os.path.join(data_root, val_data_path)
    )
    # theses datasets are already normalized with mean 0.5, std 0.5
    (
        age_30,
        cfp_fp,
        lfw,
        age_30_issame,
        cfp_fp_issame,
        lfw_issame,
        cplfw,
        cplfw_issame,
        calfw,
        calfw_issame,
    ) = val_data
    val_data_dict = {
        "agedb_30": (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
    return val_dataset

def tensor2img(img):
    m_s = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    # img = torch.stack([img[2], img[1], img[0]], dim=0)
    ToImage = transforms.ToPILImage()
    aug_img = img * m_s + m_s
    aug_img = ToImage(aug_img)
    return aug_img