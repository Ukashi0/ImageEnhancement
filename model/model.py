import numpy as np

import torch

from collections import OrderedDict
from torch.autograd import Variable
import  random
import  sys
from baseModel import BaseModel
from . import networks
from util.imagePool import ImagePool
import util.utils as util

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
        self.inputA = self.Tensor(nb,opt.input_nc, size,size)
        self.inputB = self.Tensor(nb, opt.input_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)

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

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)
        # 鉴别器
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
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fakeB_pool = ImagePool(opt.pool_size)

            # loss
            self.criterionGAN = networks.GANLoss()
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # optimizer
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            if self.isTrain:
                networks.print_network(self.netD_A)
                if self.opt.patchD:
                    networks.print_network(self.netD_P)
            if opt.isTrain:
                self.netG_A.train()
            else:
                self.netG_A.eval()
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputA = input['A' if AtoB else 'B']
        inputB = input['B' if AtoB else 'A']
        input_img = input['input_img']
        self.inputA.resize_(inputA.size()).copy_(inputA) # inputA数据赋值给self.inputA
        self.inputB.resize_(inputB.size()).copy_(inputB)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def test(self):
        self.realA = Variable(self.inputA, volatitle=True)
        if self.opt.noise > 0:
            self.noise = Variable(
                torch.cuda.FloatTensor(self.realA.size()).normal_(mean=0, std=self.opt.noise / 255.))

        if self.opt.input_linear:
            self.realA = (self.realA - torch.min(self.realA)) / (torch.max(self.realA)- torch.min(self.realA))
        if self.opt.skip == 1:
            self.fakeB, self.latentRealA = self.netG_A.forward(self.realA)

        else:
            self.fakeB = self.netG_A.forward(self.realA)

        self.realB = Variable(self.inputB, volatile=True)

    def predict(self):
        with torch.no_grad():
            self.realA = Variable(self.inputA)
        if self.opt.noise > 0:
            self.noise = Variable(
                torch.cuda.FloatTensor(self.realA.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.realA = self.realA + self.noise
        if self.opt.input_linear:
            self.realA = (self.realA - torch.min(self.realA)) / (
                    torch.max(self.realA) - torch.min(self.realA))
        if self.opt.skip == 1:
            self.fakeB, self.latentRealA = self.netG_A.forward(self.realA)
        else:
            self.fakeB = self.netG_A.forward(self.realA)
        realA = util.tensor2im(self.realA.data)
        fakeB = util.tensor2im(self.fakeB.data)
        return OrderedDict([('realA',realA),('fakeB',fakeB)])

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self,netD,real,fake):
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())

        lossD_real = self.criterionGAN(pred_real,True)  # .
        lossD_fake = self.criterionGAN(pred_fake,False)
        lossD = (lossD_fake + lossD_real) / 2

        return lossD

    def backward_D_A(self,netD,):
        fakeB = self.fakeB_pool.query(self.fakeB)
        fakeB = self.fakeB
        self.lossD_A = self.backward_D_basic(self.netD_A,self.realB,fakeB)
        self.lossD_A.backward()

    def backward_D_P(self):
        lossD_P = self.backward_D_basic(self.netD_P,self.real_patch,self.fake_patch)
        for i in range(self.opt.patchD_3):
            lossD_P += self.backward_D_basic(self.netD_P,self.real_patch[i], self.fake_patch[i])
        self.lossD_P = lossD_P / float(self.opt.patchD_3 + 1)
        self.lossD_P.backward()

    def backward_G(self,epoch):
        pred_fake = self.netD_A.forward(self.fakeB)
        pred_real = self.netD_A.forward(self.realB)
        self.lossG_A = (self.criterionGAN(pred_fake - torch.mean(pred_fake), False) +
                        (self.criterionGAN(pred_fake) - torch.mean(pred_real),True))/2
        loss = 0






