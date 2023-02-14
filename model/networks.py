import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)  # 对颜色通道的归一化操作




def vgg_preprocess(img,opt):
    tensortype = type(img.data)
    (r,g,b) = torch.chunk(img,3,dim=1)
    img = torch.cat((b,g,r),dim=1)  # 转换格式,rgb-bgr
    img = (img+1)*255*0.5 # [-1,1] => [0,255]
    if opt.vgg_mean:
        mean = tensortype(img.data.size())
        mean[:,0,:,:] = 103.939
        mean[:,1,:,:] = 116.779
        mean[:,2,:,:] = 123.680
        img = img.sub(Variable(mean)) # subtract mean
    return img


class PerceptualLoss(nn.Module):
    def __init__(self,opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        # 归一化
        self.instancenorm = nn.InstanceNorm2d(512,affine=False)

    def compute(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target,self.opt)
        img_fea = vgg(img_vgg, self.opt)

        target_fea = vgg(target_vgg, self.opt)

        # normalize
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea))**2)
