import os
import glob
import json
import random
import pickle
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from data.data import get_img_loader
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch
import cv2
import math
import copy
from skimage import morphology

import imgaug.augmenters as iaa
import glob
from torchvision import transforms as T
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# from . import DATA
from data import DATA

import os
import random
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
import numpy as np


URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

def downsampling(x, size, to_tensor=False, bin=True):

    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down


from torch.utils.data import Sampler

datasets_classes = {
    "unidatasets":[

        # 'cookie_mvtec3DYellowSN',
        # 'tire_mvtec3DYellowSN',
        # 'dowel_mvtec3DYellowSN',
        # 'potato_mvtec3DYellowSN',
        # 'cable_gland_mvtec3DYellowSN',
        # 'carrot_mvtec3DYellowSN',
        # 'bagel_mvtec3DYellowSN',
        # 'rope_mvtec3DYellowSN',
        # 'foam_mvtec3DYellowSN',
        # 'peach_mvtec3DYellowSN',

        'GummyBear_EyeRGBNorm',
        'Lollipop_EyeRGBNorm',
        'Marshmallow_EyeRGBNorm', 'LicoriceSandwich_EyeRGBNorm',
        'ChocolatePraline_EyeRGBNorm', 'ChocolateCookie_EyeRGBNorm',
        'PeppermintCandy_EyeRGBNorm', 'HazelnutTruffle_EyeRGBNorm',
        'Confetto_EyeRGBNorm',
        'CandyCane_EyeRGBNorm',


    ]}

class WeightedClassSampler(Sampler):
    def __init__(self, dataset, num_samples=None):
        """
        自定义采样器，每个 epoch 重新按类别平衡采样数据。

        Args:
            dataset (UniDataset): 需要进行加权采样的数据集
            num_samples (int): 每个 epoch 采样的数据量（默认和 dataset 设定的相同）
        """
        self.dataset = dataset
        self.num_samples = num_samples if num_samples else dataset.sample_size
        self.indices = list(range(len(self.dataset)))  # 初始化索引

    def set_epoch_samples(self):
        """在每个 epoch 重新进行加权随机抽样"""
        self.indices = self.dataset.get_indices()  # 让数据集按照类别均衡采样
        # = list(range(len(self.dataset)))  # 重新计算索引，防止索引超出范围

    def __iter__(self):
        """返回新的索引"""
        return iter(self.indices)

    def __len__(self):
        """返回当前 epoch 的样本数"""
        return len(self.indices)

def get_data_transforms(size, isize, mean_train=None, std_train=None):
    
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    normal_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

import tifffile


@DATA.register
class RGBSNDataset(Dataset):
    def __init__(self, cfg, train=True, transform=None, target_transform=None, get_mask=True, get_features=False):
        super(RGBSNDataset, self).__init__()
        self.train = train
        self.imgsize = cfg["img_size"][0]
        self.crop_size = cfg["crop_size"][0]
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.data_all = []
        self.root = cfg["data_root"]
        meta_info = json.load(open(cfg["data_meta"]))
        self.ref_meta_info = meta_info['train']
        self.meta_info = meta_info = meta_info['train' if self.train else 'test']
        self.cls_names = cfg["all_datasets_classes"]
        if not isinstance(self.cls_names, list):
            self.cls_names = [self.cls_names]
        self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
        for cls_name in self.cls_names:
            if train and cfg["shot"]!=-1:
                # self.data_all.extend(self.meta_info[cls_name])
                # self.data_all.extend(random.sample(self.meta_info[cls_name],cfg["shot"]))
                sort_datas =sorted( self.meta_info[cls_name],key=lambda x:x['img_path'][0])#sorted( self.meta_info[cls_name])

                self.data_all.extend(sort_datas[-cfg["shot"]:])
            else:
                self.data_all.extend(self.meta_info[cls_name])
        self.sample_size = self.total_size = len(self.data_all)
        random.shuffle(self.data_all) if self.train else None
        self.dataset_weight = cfg["dataset_weight"]
        self.mean_train = [0.485, 0.456, 0.406]
        self.std_train = [0.229, 0.224, 0.225] 
        self.image_transforms,self.target_transform = get_data_transforms(self.imgsize,self.crop_size,self.mean_train,self.std_train)

        if train:


            # self.probabilities = torch.ones((len(self.data_all)))
            probabilities = []

            # self.data_all = filter(lambda data:  data['cls_name'], self.data_all)
            for data in self.data_all:
                cls = data['cls_name']
                # pro = len(self.cls_names) / self.sample_size



                pro = 1 / len(self.meta_info[cls])
                probabilities.append(pro)
            self.probabilities = torch.tensor(probabilities)
        else:
            self.probabilities = torch.ones((len(self.data_all)))
        # self.image_transforms = transforms.Compose([transforms.Resize(cfg["img_size"]), transforms.ToTensor(),
        #                                             transforms.Normalize(norm_mean, norm_std)], )

    def get_indices(self):
        samples_indices = torch.multinomial(self.probabilities, num_samples=self.sample_size, replacement=True)
        return samples_indices

    def __len__(self):
        return len(self.data_all)

    def get_3D(self, path, cls_name):
        if cls_name == "brats_BratsAD":
            img = Image.open(path).convert('RGB')
            depth = self.image_transforms(img).mean(0, keepdim=True)
        elif path.endswith("png"):
            img = Image.open(path).convert('RGB')
            depth = self.image_transforms(img).mean(0, keepdim=True)
        else:
            sample = np.load(path)
            depth = sample[:, :, 0]
            fg = sample[:, :, -1]
            # depth = (depth-self.mean)/self.std
            mean_fg = np.sum(fg * depth) / np.sum(fg)
            depth = fg * depth + (1 - fg) * mean_fg
            depth = (depth - mean_fg) * 100
            depth = self.transform(depth, self.imgsize, binary=False)
        return depth

    def transform(self, x, img_len, binary=False):
        x = x.copy()
        x = torch.FloatTensor(x)
        if len(x.shape) == 2:
            x = x[None, None]
            channels = 1
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1)[None]
            channels = x.shape[1]
        else:
            raise Exception(f'invalid dimensions of x:{x.shape}')

        x = downsampling(x, (img_len, img_len), bin=binary)
        x = x.reshape(channels, img_len, img_len)
        return x

    def read_mask(self, mask_path):
        # print(mask_path)
        if not self.train and mask_path:
            with open(f"{self.root}/{mask_path}", 'rb') as f:
                mask = Image.open(f)
                mask = self.transform(np.array(mask), self.imgsize, binary=True)[:1]
                mask[mask > 0] = 1
        else:
            mask = torch.zeros((1, self.imgsize, self.imgsize))
        return mask

    def get_img(self, path, cls_name):
        if cls_name == "brats_BratsAD":
            flair = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            t1 = cv2.cvtColor(cv2.imread(path.replace("flair", "t1")), cv2.COLOR_BGR2GRAY)
            t2 = cv2.cvtColor(cv2.imread(path.replace("flair", "t2")), cv2.COLOR_BGR2GRAY)
            return Image.fromarray(np.stack([flair, t1, t2], -1))
        else:
            return Image.open(path).convert('RGB')

    def read_img(self, path, typ,depth_mean_std):
        if typ.lower() == "rgb":
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.image_transforms(img)
            return img, img
        elif typ.lower() == "normal":
            img = np.load(os.path.join(self.root, path))
            img = torch.tensor(img).permute(2, 0, 1)
            img = F.interpolate(img[None], size=(self.crop_size, self.crop_size))
            img = transforms.Normalize(self.mean_train,self.std_train)(img[0])
            return img, img
        elif typ.lower() == "gray":
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.image_transforms(img)
            return img.mean(0, keepdim=True), img
        elif typ.lower() == "depth":
            sample = np.load(os.path.join(self.root, path))
            depth = sample[:, :, 0]
            fg = sample[:, :, -1]
            # depth = (depth-self.mean)/self.std
            mean_fg = np.sum(fg * depth) / np.sum(fg)
            depth = fg * depth + (1 - fg) * mean_fg
            # mean,std = depth_mean_std
            # depth =(depth-mean)/std
            depth = (depth - mean_fg) * 100
            depth = (depth-depth.min())/(depth.max()-depth.min())

            depth = self.transform(depth, self.imgsize, binary=False)
            return depth, depth.repeat(3, 1, 1)
        elif typ.lower() == "wofg_depth":
            sample = np.load(os.path.join(self.root, path))
            depth = sample[:, :, 0]
            mean_fg = depth.mean()
            # mean,std = depth_mean_std
            # depth =(depth-mean)/std
            depth = (depth - mean_fg) * 100
            depth = self.transform(depth, self.imgsize, binary=False)
            return depth, depth.repeat(3, 1, 1)
        else:
            raise Exception(f'invalid type:{typ}')


    def get_imgs(self, img_paths, img_types,depth_mean_std):
        share_img, specific_img = [], []
        for path, typ in zip(img_paths, img_types):
            share, spec = self.read_img(path, typ,depth_mean_std)
            share_img.append(share)
            specific_img.append(spec)
        return  specific_img

    def __getitem__(self, index):
        data = self.data_all[index]
        
        img_paths,  mask_path, img_types,cls_name, anomaly = (data['img_path'],  data['mask_path'], data['img_type'],
                                                                                          data['cls_name'],
                                                                                          data['anomaly'])

        img_path, gt, label = img_paths[0], os.path.join(self.root,mask_path), anomaly

        img,normal = self.get_imgs(img_paths, img_types,None)

        fg_mask =(normal[0]==normal[0,0,0])&(normal[1]==normal[1,0,0])&(normal[2]==normal[2,0,0])

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt).convert("L")
            gt = self.target_transform(gt)
            gt[gt>=0.5]=1
            gt[gt<0.5]=0
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        res = {
                "RGB": img,
                "Depth": normal    ,
                'anomaly_mask': gt,
                'class_name': cls_name, 'has_anomaly': label,
                'path': img_path,
                'fg_mask': fg_mask,
            }


        return res


