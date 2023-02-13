import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
class Four_modal_DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, augmentations = None, img_size=(321, 321), mean=IMG_MEAN, scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.img_size = img_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentations = augmentations
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        T1_path = '%s'%(datafiles["img"].replace('images','T1'))
        T1c_path = '%s'%(datafiles["img"].replace('images','T1c'))
        T2_path = '%s'%(datafiles["img"].replace('images','T2'))
        
        image = Image.open(datafiles["img"]).convert('RGB')
        T1 = Image.open(T1_path).convert('RGB')
        T1c = Image.open(T1c_path).convert('RGB')
        T2 = Image.open(T2_path).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        T1_image = T1.resize(self.img_size, Image.BICUBIC)
        T1c_image = T1c.resize(self.img_size, Image.BICUBIC)
        T2_image = T2.resize(self.img_size, Image.BICUBIC)
        label = label.resize(self.img_size, Image.NEAREST)

        image = np.array(image)
        T1_image = np.array(T1_image)
        T1c_image = np.array(T1c_image)
        T2_image = np.array(T2_image)
        label = np.array(label)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        image = image.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        image = (image-np.min(image))/(np.max(image)-np.min(image))*255
        
        T1_image = T1_image[:, :, ::-1]  # change to BGR
        T1_image = T1_image.transpose((2, 0, 1))
        T1_image = (T1_image-np.min(T1_image))/(np.max(T1_image)-np.min(T1_image))*255
        
        T1c_image = T1c_image[:, :, ::-1]  # change to BGR
        T1c_image = T1c_image.transpose((2, 0, 1))
        T1c_image = (T1c_image-np.min(T1c_image))/(np.max(T1c_image)-np.min(T1c_image))*255
        
        T2_image = T2_image[:, :, ::-1]  # change to BGR
        T2_image = T2_image.transpose((2, 0, 1))
        T2_image = (T2_image-np.min(T2_image))/(np.max(T2_image)-np.min(T2_image))*255
        
        
        label = label//63
        label[label==4]=3
        return image.copy(),T1_image.copy(),T1c_image.copy(),T2_image.copy(),label.copy(), np.array(size), name


