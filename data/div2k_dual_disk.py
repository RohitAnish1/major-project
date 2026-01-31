import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class DIV2KDualDisk(Dataset):
    def __init__(self, root_dir):
        self.hr_dir = os.path.join(root_dir, "HR")
        self.lr1_dir = os.path.join(root_dir, "LR1")
        self.lr2_dir = os.path.join(root_dir, "LR2")
        self.images = sorted(os.listdir(self.hr_dir))

    def __len__(self):
        return len(self.images)

    def _read_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __getitem__(self, idx):
        name = self.images[idx]
        hr  = self._read_img(os.path.join(self.hr_dir, name))
        lr1 = self._read_img(os.path.join(self.lr1_dir, name))
        lr2 = self._read_img(os.path.join(self.lr2_dir, name))
        return lr1, lr2, hr
