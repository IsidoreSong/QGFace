import os

import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from .record_dataset import RandomRescale
from .augmenter import Augmenter
import torch


class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        swap_color_channel=False,
        output_dir="./",
    ):
        super(CustomImageFolderDataset, self).__init__(
            root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file
        )
        self.root = root
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir  # for checking the sanity of input images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        sample = self.augmenter.augment(sample)

        sample_save_path = os.path.join(self.output_dir, "training_samples", "sample.jpg")
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ContrastFolderDataset(datasets.ImageFolder):
    def __init__(
        self,
        root_dir,
        transform=None,
        swap_color_channel=False,
        output_dir="./",
        contrast_view=False,
        LH_align=False,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
    ):
        # root_dir = os.path.join(root_dir, 'imgs')
        super(datasets.ImageFolder, self).__init__(root_dir, loader=loader, is_valid_file=is_valid_file, extensions='jpg')
        self.root = root_dir
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir  # for checking the sanity of input images

        self.ada_transform = transforms.Compose(
            [
                transforms.RandomApply([transforms.RandomResizedCrop(size=(112, 112), scale=(0.2, 1.0), ratio=(0.75, 1.33))], p=0.2),
                transforms.RandomApply([RandomRescale(original_size=112, scale=(0.2, 1.0))], p=0.2),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        p = 0.2
        self.L_transform = [
            transforms.RandomApply([RandomRescale(original_size=112, scale=(0.1, 0.2))], p=1),
            # transforms.RandomApply([RandomRescale(original_size=112, scale=(0.1, 0.5))], p=1),
            # transforms.RandomApply([RandomRescale(original_size=112, scale=(0.2, 1.0))], p=p),
            transforms.RandomApply([transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.2), ratio=(0.75, 1.33))], p=p),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)], p=p),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=p),
            # transforms.RandomRotation(degrees=30),
            # transforms.RandomResizedCrop(size=112, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.GaussianBlur(11),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.H_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.transform = transform
        self.output_dir = output_dir
        self.contrast_view = contrast_view
        self.LH_align = LH_align

        # sample, target = self.read_sample(np.random.randint(len(self.imgidx)))
        # L_sample = transforms.Compose(self.L_transform[:-3])(sample)
        # H_sample = transforms.Compose(self.H_transform[:-2])(sample)
        self.L_transform = transforms.Compose(self.L_transform)
        self.H_transform = transforms.Compose(self.H_transform)
        # sample_save_path = os.path.join(self.output_dir, "training_samples")
        # if not os.path.isfile(sample_save_path):
        #     if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
        #         os.makedirs(sample_save_path, exist_ok=True)
        #         cv2.imwrite(
        #             os.path.join(sample_save_path, "L_sample.jpg"), np.array(L_sample)
        #         )  # the result has to look okay (Not color swapped)
        #         cv2.imwrite(
        #             os.path.join(sample_save_path, "H_sample.jpg"), np.array(H_sample)
        #         )  # the result has to look okay (Not color swapped)

    def __getitem__(self, index):
        # img, target = self.read_sample(index)
        path, target = self.samples[index]
        sample = self.loader(path)
        img = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        if self.contrast_view and not self.LH_align:
            L_sample = self.L_transform(img)
            L_sample2 = self.L_transform(img)
            sample = torch.stack([L_sample, L_sample2])
        elif self.contrast_view and self.LH_align:
            L_sample = self.L_transform(img)
            H_sample = self.H_transform(img)
            sample = torch.stack([L_sample, H_sample])
        # elif self.transform is not None:
        #     sample = self.transform(img)
        else:
            sample = self.ada_transform(img)
        return sample, target
