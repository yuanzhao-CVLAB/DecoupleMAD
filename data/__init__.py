import glob
import importlib
import torch
import torchvision.transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np

import torchvision.transforms.functional as F
from myutils.register import Registry
import glob
import importlib
from timm.data.distributed_sampler import RepeatAugSampler
TRANSFORMS = Registry('Transforms')
DATA = Registry('Data')

files = glob.glob('data/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))


def get_transforms(cfg_transforms):
	transform_list = []
	for t in cfg_transforms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_tran = TRANSFORMS.get_module(t_type)(**t)
		transform_list.extend(t_tran) if isinstance(t_tran, list) else transform_list.append(t_tran)
	transform_out = TRANSFORMS.get_module('Compose')(transform_list)

	return transform_out
def get_dataset(cfg):
	# torchvision.transforms.RandomVerticalFlip
	train_transforms = [
		dict(type='Resize', size=cfg["img_size"], interpolation=F.InterpolationMode.BILINEAR),
		dict(type='CenterCrop', size=cfg["img_size"]),
		# dict(type='ColorJitter', brightness=0.2,contrast=0.2,saturation=0.2),
		# dict(type='RandomHorizontalFlip'),
		# dict(type='RandomVerticalFlip'),
		# dict(type='RandomRotation', degrees=(-30,30)),
		dict(type='ToTensor'),
		dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
	]
	target_transforms = [
		dict(type='Resize', size=cfg["img_size"], interpolation=F.InterpolationMode.BILINEAR),
		dict(type='CenterCrop', size=cfg["img_size"]),
		dict(type='ToTensor'),
	]

	train_transforms,test_transforms = get_transforms(cfg_transforms=train_transforms),get_transforms( cfg_transforms=train_transforms)
	target_transforms = get_transforms( cfg_transforms=target_transforms)
	train_set = DATA.get_module(cfg["data_type"])(cfg, train=True, transform=train_transforms, target_transform=target_transforms)
	test_set = DATA.get_module(cfg["data_type"])(cfg, train=False, transform=test_transforms, target_transform=target_transforms)
	return train_set, test_set
