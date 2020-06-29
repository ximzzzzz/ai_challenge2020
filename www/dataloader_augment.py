import sys
sys.path.append('./www')
sys.path.append('./')
from PIL import Image
import evaluation
import os
import cv2
import numpy as np
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils import data
import math

import random
from torch.utils import data
from PIL import Image
import glob
import json
import collections
import math
from fmix import  sample_mask

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
#     shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
#     targets = (target, shuffled_target, lam)

    return new_data, target

def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
#     shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.4,0.7)
    data = lam*data + (1-lam)*shuffled_data
#     targets = (target, shuffled_target, lam)

    return data, target


def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft)
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
#     shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask)*data
    x2 = torch.from_numpy(1-mask)*shuffled_data
#     targets=(targets, shuffled_targets, lam)

    return (x1+x2), targets



class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29
    
    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)
    
    Targets:
        image, mask
    
    Image types:
        uint8, float32
    
    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """
    
    def __init__(self, num_grid=3, div_val=[2,2], fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []
        self.div_val= div_val # div_val[0] : height div value, [1] : width div value
    
    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
#                 print(this_mask.shape)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / self.div_val[0]),
                             int(j * grid_w) : int(j * grid_w + grid_w / self.div_val[1])
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask
    
                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)
    
    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image
    
    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)
    
        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0
    
        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}
    
    @property
    def targets_as_params(self):
        return ['image']
    
    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


​    

class CustomDataset(data.Dataset):
    
    def __init__(self, root, phase='train',resize_shape = (32, 250, 1), transform=None, target_transform=None):
        
        self.root = os.path.join(root, phase)
        print(self.root)
        self.imgs = sorted(glob.glob(self.root + '/*.png')+glob.glob(self.root + '/*.jpg'))
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
        self.resize_imgH = resize_shape[0]
        self.resize_imgW = resize_shape[1]
        self.channels = resize_shape[2]
        self.toTensor = transforms.ToTensor()
        
        # Mix-up
        self.mix_up_alpha = 0.4
        
        # Cutmix
        self.cutmix_alpha = 0.4
        
        annotations = None
    
        with open(os.path.join(self.root, phase + '.json'), 'r') as label_json :
            label_json = json.load(label_json)
            annotations = label_json['annotations']
        annotations = sorted(annotations, key=lambda x: x['file_name'])
        for anno in annotations :
            if phase == 'test' :
                self.labels.append('dummy')
            else :
                self.labels.append(anno['text'])


    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('L')
        img_tensor = self.toTensor(img)
        imgH_orig = img_tensor.shape[1]
        imgW_orig = img_tensor.shape[2]
        ratio = imgW_orig / float(imgH_orig)
        
        if math.ceil(self.resize_imgH * ratio) > self.resize_imgW:
            resized_w = self.resize_imgW
        else :
            resized_w = math.ceil(self.resize_imgH * ratio)
            
        label = self.labels[index]
        img = np.array(img.resize((self.resize_imgW, self.resize_imgH), Image.BICUBIC))
        
        if self.transform is not None:
            img = self.transform(**{'image' : img, 'label' : label })['image']
        
        return (img, label)
    
    def get_root(self) :
        return self.root
    
    def get_img_path(self, index) :
        return self.imgs[index]


​    
#     def mix_up(self, X1, y1, X2, y2):
#         assert X1.shape[0] == X2.shape[0]
#         batch_size = X1.shape[0]
#         l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
#         X_l = l.reshape(batch_size, 1, 1, 1)
#         y_l = l.reshape(batch_size, 1)
#         X = X1 * X_l + X2 * (1-X_l)
#         return X, y1

#     def cutmix(self, X1, y1, X2, y2):
#         assert X1.shape[0] == X2.shape[0]
#         lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
#         width = X1.shape[1]
#         height = X1.shape[0]
#         r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
#         bx1 = np.clip(r_x - r_w // 2, 0, width)
#         by1 = np.clip(r_y - r_h // 2, 0, height)
#         bx2 = np.clip(r_x + r_w // 2, 0, width)
#         by2 = np.clip(r_y + r_h // 2, 0, height)
#         X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
#         X = X1
#         return X, y1    


​    
def data_loader(root, batch_size, imgH, imgW, phase='train', transform=None, target_transform=None, ) :
​    dataset = CustomDataset(root=root, phase=phase, transform=transform, target_transform=target_transform)
​    shuffle = False
​    if phase == 'train' :
​        shuffle = True
​    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=AlignCollate(imgH=imgH, imgW=imgW, input_channel=1),   
​                                 pin_memory=True, num_workers=5)
​    
    return dataloader




class NormalizePAD(object):
    
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2]/2)
        self.PAD_type = PAD_type
        
    def __call__(self, img):
        img = self.toTensor(img)
#         img.div_(255)
        img.sub_(0.5).div_(0.5) #빼고 나누는 연산을 inplace
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:,:, :w] = img # right pad
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            
        return Pad_img


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
    
    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class AlignCollate(object):
    
    def __init__(self,  imgH = 193, imgW = 370, input_channel=1, keep_ratio_with_pad = True, ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.totensor = transforms.ToTensor()
        self.input_channel = input_channel


​        
    def __call__(self, batch):
       
        batch = filter(lambda x : x is not None, batch)
        images, labels = zip(*batch)
    
        if self.keep_ratio_with_pad :
            resized_max_w = self.imgW
            transform = NormalizePAD((self.input_channel, self.imgH, resized_max_w))
            
            resized_images = []
            for image in images:
                h, w = image.shape
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

#                 resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_image = cv2.resize(image, (resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

#         else:
#             transform = ResizeNormalize((self.imgW, self.imgH))
#             image_tensors = [transform(image) for image in images]
#             image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
#             resized_images.append(self.totensor(image.resize((self.imgW, self.imgH), Image.BICUBIC)))
        aug_idx = np.random.choice(range(3), size=1)[0]
        if aug_idx==0:
            image_tensors, labels = cutmix(image_tensors, labels, 1.0)
        elif aug_idx==1:
            image_tensors, labels = mixup(image_tensors, labels, 1.0)
        else:
            img_tensors, labels = fmix(image_tensors, labels,  alpha = 1., decay_power=2, shape=(image_tensors.size(2), image_tensors.size(3)))


​            
        return image_tensors, labels

#         return images, labels