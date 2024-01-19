import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import mxnet as mx
from PIL import Image

class CRLFWTest(Dataset):
    def __init__(self, data_root, size_list=[8, 12, 16, 20]):
        super(CRLFWTest, self).__init__()
        self.bins, self.issame_list = pickle.load(
            open(data_root, "rb"), encoding="bytes"
        )
        self.dataname_to_idx = {"8x8": 5, "12x12": 6, "16x16": 7, "20x20": 8}

        self.size_transforms = [
            transforms.Resize((size, size)) for size in size_list
            # transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST) for size in size_list
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.bins) * len(self.size_transforms)

    def __getitem__(self, idx):
        dataset_idx = idx // len(self.bins)
        bin_idx = idx % len(self.bins)
        img = mx.image.imdecode(self.bins[bin_idx]).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        if bin_idx % 2 == 0:
            img = self.size_transforms[dataset_idx](img)
        img = self.transform(img)
        label = self.issame_list[bin_idx // 2]

        return img, label, dataset_idx + 5, idx