import os
import torch
import numpy as np
import scipy.misc as m
import cv2

from torch.utils import data

from data.city_utils import recursive_glob
from data.augmentations import *

class target_4modal(data.Dataset):
    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],}

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        img_norm=False,
        augmentations=None,
        version="cityscapes",
        return_id=False,
        img_mean = np.array([73.15835921, 82.90891754, 72.39239876])
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 4
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = img_mean
        self.files = {}
        self.files_T1 = {}
        self.files_T1c = {}
        self.files_T2 = {}

        self.images_base = os.path.join(self.root,"Flair")
        self.T1 = os.path.join(self.root,"T1")
        self.T1c = os.path.join(self.root,"T1c")
        self.T2 = os.path.join(self.root,"T2")


        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.files_T1[split] = recursive_glob(rootdir=self.T1, suffix=".png")
        self.files_T1c[split] = recursive_glob(rootdir=self.T1c, suffix=".png")
        self.files_T2[split] = recursive_glob(rootdir=self.T2, suffix=".png")

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        T1_path = self.files_T1[self.split][index].rstrip()
        T1c_path = self.files_T1c[self.split][index].rstrip()
        T2_path = self.files_T2[self.split][index].rstrip()

        img = Image.open(img_path).convert('RGB')
        img_T1 = Image.open(T1_path).convert('RGB')
        img_T1c = Image.open(T1c_path).convert('RGB')
        img_T2 = Image.open(T2_path).convert('RGB')

        if self.is_transform:
            img = self.transform(img)
            img_T1 = self.transform(img_T1)
            img_T1c = self.transform(img_T1c)
            img_T2 = self.transform(img_T2)
        
        img_name = img_path.split('\\')[-1]
        return img,img_T1,img_T1c,img_T2, img_path, img_name

        #return img, img_path, img_name
        # return img_T1c, img_path, img_name

    def transform(self, img):

        img = img.resize(self.img_size, Image.BICUBIC)
        img = np.array(img)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.transpose((2, 0, 1))
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255
        return img

