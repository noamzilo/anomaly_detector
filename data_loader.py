from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import random
import os
import posixpath
from PIL import Image

train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self,
                 data_dir,
                 filename_path,
                 transform,
                 img_ext='.jpg',
                 annot_ext='.mat',
                 image_mode='RGB',
                 train_percent=100.,
                 use_train=True,
                 seed=17):
        assert 0. <= train_percent <= 100.

        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        # allowing a sub sample of the dataset for training and validation
        filename_list = get_list_from_filenames(filename_path)
        self.length = int(len(filename_list) * train_percent // 100)
        random.seed(seed)
        train_inds = set(random.sample(range(len(filename_list)), self.length))
        validation_inds = set(range(len(filename_list))) - train_inds

        if use_train:
            filename_list = [filename_list[i] for i in train_inds]
        else:
            filename_list = [filename_list[i] for i in validation_inds]
            self.length = len(filename_list)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        # self.length = len(filename_list)

    def __getitem__(self, index):
        #NOAM
        path = os.path.join(self.data_dir, self.X_train[index] + self.img_ext)
        path = posixpath.join(*path.split('\\'))
        img = Image.open(path)
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        mat_path = posixpath.join(*mat_path.split('\\'))

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length