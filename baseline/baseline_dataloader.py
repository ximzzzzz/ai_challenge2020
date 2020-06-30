#!/usr/bin/python

# encoding: utf-8

import random
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from glob import glob
import json
import collections
import os
import math
import cv2

class CustomDataset(data.Dataset):

    def __init__(self, root, phase='train', transform=None, target_transform=None):
        
        self.root = os.path.join(root, phase)
        self.imgs = sorted(glob(self.root + '/*.png')+glob(self.root + '/*.jpg'))
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
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
#         img = np.array(Image.open(img_path).convert('L'))
        img = Image.open(img_path).convert('L')
        label = self.labels[index]
    
        if self.transform is not None:
            img = self.transform(**{'image' : img, 'label' : label })['image']
    
        if self.target_transform is not None:
            label = self.target_transform(label)
    
        return (img, label)


​    
    def get_root(self) :
        return self.root
    
    def get_img_path(self, index) :
        return self.imgs[index]


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
    
    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


​    
​    
​    
​    
class alignCollate(object):

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
    
        transform = resizeNormalize((imgW, imgH))
        
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
    
        return images, labels


​    
#     def __init__(self,  imgH = 193, imgW = 370, input_channel=1, keep_ratio_with_pad = True, ):
#         self.imgH = imgH
#         self.imgW = imgW
#         self.keep_ratio_with_pad = keep_ratio_with_pad
#         self.totensor = transforms.ToTensor()
#         self.input_channel = input_channel


#     def __call__(self, batch):
#         batch = filter(lambda x : x is not None, batch)
#         images, labels = zip(*batch)
#         if self.keep_ratio_with_pad :
#             resized_max_w = self.imgW
#             transform = NormalizePAD((self.input_channel, self.imgH, resized_max_w))

#             resized_images = []
#             for image in images:
#                 h, w = image.size
#                 ratio = w / float(h)
#                 if math.ceil(self.imgH * ratio) > self.imgW:
#                     resized_w = self.imgW
#                 else:
#                     resized_w = math.ceil(self.imgH * ratio)

#                 resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
# #                 resized_image = cv2.resize(image, (resized_w, self.imgH), Image.BICUBIC)
#                 resized_images.append(transform(resized_image))

#             image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

#         return image_tensors, labels

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    
    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
    
    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += lf
            return texts



def loadData(v, data):
    d_size = data.size()
    v.resize_(d_size).copy_(data)

def data_loader(root, batch_size, imgH, imgW, phase='train', transform=None, target_transform=None) :
    dataset = CustomDataset(root=root, phase=phase, transform=transform, target_transform=target_transform)
    shuffle = False
    if phase == 'train' :
        shuffle = True
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=alignCollate(imgH=imgH, imgW=imgW),   
                                 pin_memory=True, num_workers=2)
#     dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=AlignCollate(imgH=imgH, imgW=imgW, input_channel = input_channel, keep_ratio_with_pad=True), pin_memory=True, num_workers=2)

    return dataloader



#####################custom#########################

class NormalizePAD(object):
    
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2]/2)
        self.PAD_type = PAD_type
        
    def __call__(self, img):
        img = self.toTensor(img)
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
    
    def __init__(self,  imgH = 193, imgW = 370, input_channel=3, keep_ratio_with_pad = True, ):
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
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
    
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        
        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
            resized_images.append(self.totensor(image.resize((self.imgW, self.imgH), Image.BICUBIC)))
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)


​    
        del(images)
        del(resized_images)
        
        return image_tensors, labels