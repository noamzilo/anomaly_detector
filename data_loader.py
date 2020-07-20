from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import random
import os
import posixpath
from PIL import Image
from image_paths import read_links_file_to_list

train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

class ChipDataset(Dataset):
    def __init__(self,
                 data_dir,
                 transform,
                 train_percent=100.,
                 use_train=True,
                 seed=17):
        assert 0. <= train_percent <= 100.

        self._data_dir = data_dir
        self._transform = transform

        # allowing a sub sample of the dataset for training and validation
        filename_list = read_links_file_to_list()
        self._length = int(len(filename_list) * train_percent // 100)
        random.seed(seed)
        train_inds = set(random.sample(range(len(filename_list)), self._length))
        validation_inds = set(range(len(filename_list))) - train_inds

        if use_train:
            filename_list = [filename_list[i] for i in train_inds]
        else:
            filename_list = [filename_list[i] for i in validation_inds]
            self._length = len(filename_list)

        self._x_train = filename_list
        self._y_train = filename_list

    def __getitem__(self, index):
        #NOAM
        path = os.path.join(self._data_dir, self._x_train[index] + self._img_ext)
        path = posixpath.join(*path.split('\\'))
        img = Image.open(path)
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self._data_dir, self._y_train[index] + self.annot_ext)
        mat_path = posixpath.join(*mat_path.split('\\'))

        if self._transform is not None:
            img = self._transform(img)

        return img, self._x_train[index]

    def __len__(self):
        # 122,450
        return self._length