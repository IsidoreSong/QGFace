import numbers
import mxnet as mx
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2
from .augmenter import Augmenter
from tqdm import tqdm
import io
from collections.abc import Iterable


class BaseMXDataset(Dataset):
    def __init__(self, root_dir, swap_color_channel=False):
        super(BaseMXDataset, self).__init__()
        self.root_dir = root_dir
        self.path_imgrec = os.path.join(root_dir, "train.rec")
        self.path_imgidx = os.path.join(root_dir, "train.idx")
        path_imglst = os.path.join(root_dir, "train.lst")
        path_csv = os.path.join(root_dir, "train.csv")

        self.swap_color_channel = swap_color_channel
        if self.swap_color_channel:
            print("[INFO] Train data in swap_color_channel")
        self.init_record()

    def init_record(self):
        self.record = mx.recordio.MXIndexedRecordIO(self.path_imgidx, self.path_imgrec, "r")
        # grad image index from the record and know how many images there are.
        # image index could be occasionally random order. like [4,3,1,2,0]
        s = self.record.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.record.keys))

    def read_sample(self, index):
        idx = self.imgidx[index]
        s = self.record.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])
        return sample, label

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.imgidx)


class AugmentRecordDataset(BaseMXDataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        low_res_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
        swap_color_channel=False,
        output_dir="./",
        contrast_view=False,
        *args,
        **kwargs,
    ):
        super(AugmentRecordDataset, self).__init__(
            root_dir,
            swap_color_channel=swap_color_channel,
        )
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.transform = transform
        self.output_dir = output_dir
        self.contrast_view = contrast_view

        sample, target = self.read_sample(np.random.randint(len(self.imgidx)))
        sample = self.augmenter.augment(sample)
        sample_save_path = os.path.join(self.output_dir, "training_samples", "sample.jpg")
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

    def __getitem__(self, index):
        img, target = self.read_sample(index)

        sample = self.augmenter.augment(img)
        sample = self.transform(sample)
        if self.contrast_view:
            sample2 = self.transform(img)
            return torch.stack([sample, sample2]), target

        return sample, target


class RandomRescale(torch.nn.Module):
    def __init__(self, original_size, scale, is_jpeg=False, quality=75, max_size=None, antialias="warn"):
        super().__init__()
        self.original_size = original_size
        self.scale = scale
        self.is_jpeg = is_jpeg
        self.quality = quality
        self.interpolation_lis = np.array(
            [
                InterpolationMode.BICUBIC,
                InterpolationMode.BILINEAR,
                InterpolationMode.NEAREST,
                InterpolationMode.BOX,
                InterpolationMode.LANCZOS,
            ]
        )
        self.max_size = max_size
        self.antialias = antialias

    def get_params(self, scale):
        if isinstance(scale, Iterable):
            target_scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
        else:
            target_scale = scale
        target_size = int(target_scale * self.original_size)
        return target_size

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        target_size = self.get_params(self.scale)

        interpolation = np.random.choice(self.interpolation_lis)
        img = F.resize(img, (target_size, target_size), interpolation, self.max_size, self.antialias)
        if self.is_jpeg:
            img_b = io.BytesIO()
            img.save(img_b, format="jpeg", quality=self.quality)
            img = Image.open(img_b)
        interpolation = np.random.choice(self.interpolation_lis)
        img = F.resize(img, (self.original_size, self.original_size), interpolation)
        return img


class ContrastDataset(BaseMXDataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        swap_color_channel=False,
        output_dir="./",
        contrast_view=False,
        LH_align=False,
        rotation_augmentation_prob=0.0,
        crop_augmentation_prob=0.0,
        photometric_augmentation_prob=0.0,
    ):
        super(ContrastDataset, self).__init__(
            root_dir,
            swap_color_channel=swap_color_channel,
        )
        p = 0.2
        self.ada_transform = [
            transforms.RandomApply([transforms.RandomResizedCrop(size=(112, 112), scale=(0.2, 1.0), ratio=(0.75, 1.33))], p=p),
            transforms.RandomApply([RandomRescale(original_size=112, scale=(0.2, 1.0), is_jpeg=False)], p=p),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)], p=p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.L_transform = [
            RandomRescale(original_size=112, scale=(0.1, 0.5), is_jpeg=True, quality=75),
            transforms.RandomApply(
                [transforms.RandomResizedCrop(size=(112, 112), scale=(0.8, 1.0), ratio=(0.75, 1.33))], p=crop_augmentation_prob
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)], p=photometric_augmentation_prob
            ),
            transforms.RandomApply([transforms.RandomRotation(degrees=45)], p=rotation_augmentation_prob),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        # self.L_transform = self.ada_transform
        self.H_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.transform = transform
        self.output_dir = output_dir
        self.contrast_view = contrast_view
        self.LH_align = LH_align

        sample, target = self.read_sample(np.random.randint(len(self.imgidx)))
        L_sample = transforms.Compose(self.L_transform[:-3])(sample)
        H_sample = transforms.Compose(self.H_transform[:-2])(sample)
        self.ada_transform = transforms.Compose(self.ada_transform)
        self.L_transform = transforms.Compose(self.L_transform)
        self.H_transform = transforms.Compose(self.H_transform)
        sample_save_path = os.path.join(self.output_dir, "training_samples")
        if not os.path.isfile(sample_save_path):
            if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
                os.makedirs(sample_save_path, exist_ok=True)
                cv2.imwrite(
                    os.path.join(sample_save_path, "L_sample.jpg"), np.array(L_sample)
                )  # the result has to look okay (Not color swapped)
                cv2.imwrite(
                    os.path.join(sample_save_path, "H_sample.jpg"), np.array(H_sample)
                )  # the result has to look okay (Not color swapped)

    def __getitem__(self, index):
        img, target = self.read_sample(index)
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
