import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from dataset import face_align
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import mxnet as mx
import io

class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False):
        super(ListDatasetWithIndex, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.image_is_saved_with_swapped_B_and_R = (
            image_is_saved_with_swapped_B_and_R
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.image_is_saved_with_swapped_B_and_R:
            with open(self.img_list[idx], "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
            img = self.transform(img)

        else:
            # ArcFace Pytorch
            img = cv2.imread(self.img_list[idx])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[:, :, :3]

            img = Image.fromarray(img)
            # img = np.moveaxis(img, -1, 0)
            img = self.transform(img)
        return img, idx


class ListDataset(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False, path_imgidx=None, path_imgrec=None):
        super(ListDataset, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.path_imgrec = path_imgrec
        self.path_imgidx = path_imgidx
        if path_imgidx is not None:
            self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
            self.imgidx = np.array(list(self.record.keys))
        self.image_is_saved_with_swapped_B_and_R = (
            image_is_saved_with_swapped_B_and_R
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        if self.path_imgidx is None:
            img = cv2.imread(image_path)
            img = img[:, :, :3]
            label = idx
        else:
            idx = self.imgidx[idx]
            s = self.record.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label
            img = Image.open(io.BytesIO(img))
            # img = np.array(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if self.image_is_saved_with_swapped_B_and_R:
            print("check if it really should be on")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label, 0, idx


def prepare_imagelist_dataloader(
    img_list,
    batch_size,
    num_workers=0,
    image_is_saved_with_swapped_B_and_R=False,
):
    # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
    # face_emore/img has images saved with B and G (of RGB) swapped.
    # Since training data loader uses PIL (results in RGB) to read image
    # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
    # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
    # then you should set image_is_saved_with_swapped_B_and_R=True

    image_dataset = ListDatasetWithIndex(
        img_list, image_is_saved_with_swapped_B_and_R
    )
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader


def prepare_dataloader(
    img_list,
    batch_size,
    num_workers=0,
    image_is_saved_with_swapped_B_and_R=False,
    path_imgidx=None,
    path_imgrec=None
):
    # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
    # face_emore/img has images saved with B and G (of RGB) swapped.
    # Since training data loader uses PIL (results in RGB) to read image
    # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
    # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
    # then you should set image_is_saved_with_swapped_B_and_R=True

    image_dataset = ListDataset(
        img_list,
        image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R,
        path_imgidx=path_imgidx,
        path_imgrec=path_imgrec
    )
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def count_imgs(img_pth):
    image_id_list = os.listdir(img_pth)
    image_list = []
    for image_id in tqdm(image_id_list):
        for image in os.listdir(os.path.join(img_pth, image_id)):
            image_path = os.path.join(img_pth, image_id, image)
            image_list.append(image_path)
    print(len(image_list))


def align_img_one(row, img_pth, save_pth, suffix, points_num):
    id_img_name, test_img_lmd = row[0] + suffix, np.array(
        row[1:], dtype=np.float64
    ).reshape(-1, 2)
    img = cv2.imread(os.path.join(img_pth, id_img_name))
    aligned_face = face_align.norm_crop(
        img, test_img_lmd, points_num=points_num
    )
    id_dir = os.path.join(save_pth, os.path.dirname(id_img_name))
    os.makedirs(id_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_pth, id_img_name.lower()), aligned_face)


def submit_one(row, executor, task_lis):
    task = executor.submit(align_img_one, row)
    task_lis.append(task)
