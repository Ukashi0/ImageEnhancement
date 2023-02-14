import numpy as np

import torch

from collections import OrderedDict
from torch.autograd import Variable
import  random
import  sys
from baseModel import BaseModel
from . import networks


class Model(BaseModel):
    def name(self):
        return 'model'


    def initialize(self, opt):
        BaseModel.initialize(self,opt)

        self.opt = opt
        nb = opt.batchSize
        size = opt.fineSize

        self.opt = opt
        # 初始化输入空间
        self.input_A = self.Tensor(nb,opt.input_nc, size,size)
        self.input_B = self.Tensor(nb, opt.input_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, opt.input_nc, size, size)

        # 设置vgg loss

        self.vgg_loss = networks.PerceptualLoss(opt)
        if self.opt.IN_vgg:
            self.vgg_patch_loss = networks.PerceptualLoss(opt)
            self.vgg_patch_loss.cuda()
        self.vgg_loss.cuda()
        self.vgg = networks.load_vgg16("./vggModel",self.gpu_id)
        self.vgg.eval()
        for i in self.vgg.parameters():
            i.requires_grad = False


        # 模型构建
        if self.isTrain:
            # self.netD_A = networks



        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

