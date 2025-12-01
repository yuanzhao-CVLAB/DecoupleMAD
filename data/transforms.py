import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from timm.data import create_transform

import cv2
import numpy as np
from PIL import Image
from . import TRANSFORMS

tv_tran = transforms.transforms.__all__

for tv_tran_name in tv_tran:
	tv_transform = getattr(transforms, tv_tran_name, None)
	TRANSFORMS.register(tv_transform, name=tv_tran_name) if tv_transform else None

# for timm
TRANSFORMS.register(create_transform, name='timm_create_transform')


class vt_TransBase(object):

	def __init__(self):
		pass

	def pre_process(self):
		pass

	def __call__(self, img):
		pass


@TRANSFORMS.register
class vt_Identity(vt_TransBase):

	def __call__(self, img):
		return img


@TRANSFORMS.register
class vt_Resize(vt_TransBase):
	"""
	Args:
		size    : h | (h, w)
		img     : PIL Image
	Returns:
		PIL Image
	"""
	def __init__(self, size, interpolation=F.InterpolationMode.BICUBIC):
		super().__init__()
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		return F.resize(img, self.size, self.interpolation)


@TRANSFORMS.register
class vt_Compose(vt_TransBase):
	def __init__(self, transforms):
		super().__init__()
		self.transforms = transforms

	def pre_process(self):
		for t in self.transforms:
			t.pre_process()

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img


