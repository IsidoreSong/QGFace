from torch.utils.data import Dataset
import numpy as np
import validation.evaluate_utils as evaluate_utils
import torch
import pickle
from torchvision import transforms
import cv2
from PIL import Image
import mxnet as mx

class FiveValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name,):
        """
        concatenates all validation datasets from emore
        val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
        }
        agedb_30: 0
        cfp_fp: 1
        lfw: 2
        cplfw: 3
        calfw: 4
        """
        self.dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}

        self.val_data_dict = val_data_dict
        # concat all dataset
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
            for same in issame:
                dup_issame.append(same)
                dup_issame.append(same)
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            key_orders.append(key)
        assert key_orders == ["agedb_30", "cfp_fp", "lfw", "cplfw", "calfw"]

        if isinstance(all_imgs[0], np.memmap):
            self.all_imgs = evaluate_utils.read_memmap(concat_mem_file_name)
        else:
            self.all_imgs = np.concatenate(all_imgs)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

        assert len(self.all_imgs) == len(self.all_issame)
        assert len(self.all_issame) == len(self.all_dataname)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = torch.tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)

class FiveBinValDataset(Dataset):
    def __init__(self, lfw_root, agedb_root, cfpfp_root, cplfw_root, calfw_root):
        self.dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        self.dataset_root_list = [agedb_root, cfpfp_root, lfw_root, cplfw_root, calfw_root]
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        for key, dataset_root in zip(self.dataname_to_idx.keys(), self.dataset_root_list):
            imgs, issame_list = pickle.load(open(dataset_root, "rb"), encoding="bytes")
            all_imgs.append(imgs)
            dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
            for same in issame_list:
                dup_issame.append(same)
                dup_issame.append(same)
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            key_orders.append(key)
            
        self.all_imgs = np.concatenate(all_imgs)
        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        img = mx.image.imdecode(self.all_imgs[index]).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        x = self.transform(img)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)
    
    
class SingleValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name):
        """
        concatenates all validation datasets from emore
        val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        }
        agedb_30: 0
        """
        single_mem_file_name = concat_mem_file_name.rsplit("/", 1)[0] + "/agedb_30/memfile/mem_file.dat"
        self.dataname_to_idx = {"agedb_30": 0}
        val_data_dict = val_data_dict["agedb_30"]
        self.val_data_dict = val_data_dict
        # concat all dataset
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        key, (imgs, issame) = "agedb_30", val_data_dict
        # for key, (imgs, issame) in val_data_dict.items():
        all_imgs.append(imgs)
        dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
        for same in issame:
            dup_issame.append(same)
            dup_issame.append(same)
        all_issame.append(dup_issame)
        all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
        key_orders.append(key)
        assert key_orders == ["agedb_30"]

        if isinstance(all_imgs[0], np.memmap):
            self.all_imgs = evaluate_utils.read_memmap(single_mem_file_name)
        else:
            self.all_imgs = np.concatenate(all_imgs)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

        assert len(self.all_imgs) == len(self.all_issame)
        assert len(self.all_issame) == len(self.all_dataname)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = torch.tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)
