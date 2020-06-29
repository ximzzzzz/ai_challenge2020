import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F

import re
import six
import math
import sys
# import torchvision.transforms as transforms
sys.path.append('../scatter')
sys.path.append('./scatter')
sys.path.append('../')

import utils
import Trans
import Extract
from Seq import BidirectionalLSTM
import Seq
import Pred


class STR(nn.Module):
    def __init__(self, opt, device):
        super(STR, self).__init__()
        self.opt = opt
        
#         Trans
#         self.Trans = Trans.TPS_SpatialTransformerNetwork(F = opt.num_fiducial,
#                                                   i_size = (opt.imgH, opt.imgW), 
#                                                   i_r_size= (opt.imgH, opt.imgW), 
#                                                   i_channel_num=opt.input_channel,
#                                                         device = device)
        #Extract
        if self.opt.extract =='RCNN':
            self.Extract = self.Extract = Extract.RCNN_extractor(opt.input_channel, opt.output_channel)
        elif 'efficientnet' in self.opt.extract :
            self.Extract = Extract.EfficientNet(opt)
        elif 'resnet' in self.opt.extract :
            self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise print('invalid extract model name!')

#         self.Extract = Extract.RCNN_extractor(opt.input_channel, opt.output_channel)
#         self.Extract = Extract.ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel # (imgH/16 -1 )* 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None,1)) # imgH/16-1   ->  1
            
        # Sequence
        self.Seq = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size,  opt.hidden_size),
#             BidirectionalLSTM(1536, opt.hidden_size,  opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.Seq_output = opt.hidden_size
        
        #Pred
        self.Pred = Pred.Attention(self.Seq_output, opt.hidden_size, opt.num_classes, device=device)


​        
    def forward(self, input, text, is_train=True):
        #Trans stage
#         input = self.Trans(input)

        #Extract stage
        visual_feature = self.Extract(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3) #(batch_size, num_seq, vectors) ex) 192, 23, 512
    
        #Seq stage
        contextual_feature = self.Seq(visual_feature) # same shape as previous stage ex) 192, 23, 512
        #Pred stage
        prediction = self.Pred(contextual_feature.contiguous(), text, is_train, batch_max_length = self.opt.batch_max_length)
    
        return prediction