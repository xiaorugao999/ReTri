"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import glob
from networks import AdaINGen, MsImageDis, VAEGen
from segnetworks.losses import get_seg_loss_criterion
from segnetworks.unet_2d import UNet2D, ResidualUNet2D, ReTri
# from segnetworks.unet_2d_TSNE import UNet2D, ResidualUNet2D, ReTri
from segnetworks.deeplab_xception_skipconnection_2d import DeepLabv3_plus_skipconnection_2d
from segnetworks.deeplabv2_2d import Deeplabv2_2d
from architectures import network_architectures
import torch.utils.data
import torchvision.transforms as tvt
from segnetworks import seg_data, seg_transforms, seg_transforms_cv
import evaluation
import optim_weight_ema
import mask_gen
from utils import weights_init, get_model_list, get_test_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import numpy as np
import torch.fft
import cv2
import random
from torch import nn
import torch.nn.functional as F
import math
import kornia as K

sup_augmentations = K.augmentation.container.AugmentationSequential(
    K.augmentation.RandomHorizontalFlip(p=0.75),
    K.augmentation.RandomAffine(
        [-15., 15.], [0., 0.05], [0.9, 1.1], [0., 0.15], p=0.75),

    # K.augmentation.ColorJitter(0.6, 0.6, 0.6, 0.1, p=0.1),
    K.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.3),
    K.augmentation.RandomPlasmaShadow(
        roughness=(0.4, 0.7), shade_intensity=(-0.15, 0.0), shade_quantity=(0, 0.5), p=0.5),

    data_keys=["input", "mask"],
    same_on_batch=False,
    random_apply=2,
)

unsup_augmentations_weak = K.augmentation.container.AugmentationSequential(
    K.augmentation.RandomHorizontalFlip(p=0.75),
    K.augmentation.RandomAffine(
        [-10., 10.], [0., 0.05], [0.9, 1.1], [0., 0.15], p=0.75),
    data_keys=["input", "mask"],
    same_on_batch=False,
    random_apply=3,
)

unsup_augmentations_strongcolor = K.augmentation.container.AugmentationSequential(

    K.augmentation.ColorJitter(0.6, 0.6, 0.6, 0.1, p=0.0),
    K.augmentation.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.3),
    K.augmentation.RandomPlasmaShadow(
        roughness=(0.4, 0.7), shade_intensity=(-0.15, 0.0), shade_quantity=(0, 0.5), p=0.3),
    data_keys=["input", "mask"],
    same_on_batch=False,
    random_apply=3,
)

test_augmentations_color = K.augmentation.container.AugmentationSequential(
    K.augmentation.ColorJitter(0.6, 0.6, 0.6, 0.1, p=0.1),
    data_keys=["input", "mask"],
    same_on_batch=False,
)


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        self.hyperparameters = hyperparameters
        self.loss_conf_rate_acc = 0.0
        self.loss_conf_rate_acc_self0 = 0.0
        self.loss_conf_rate_acc_self1 = 0.0
        lr = hyperparameters['lr']
        lr_seg = hyperparameters['lr_seg']
        self.n_classes = hyperparameters['seg']['n_classes']
        self.root_n_classes = math.sqrt(self.n_classes)

        self.in_channels = self.hyperparameters['seg']['in_channels']
        if self.hyperparameters['augmentation']['crop_size'] == '':
            self.crop_size = None
        else:
            self.crop_size = [self.hyperparameters['augmentation']
                              ['crop_size'], self.hyperparameters['augmentation']['crop_size']]

        self.prop_range_cutmix = (self.hyperparameters['cutmix']['mask_prop_range_min'],
                                  self.hyperparameters['cutmix']['mask_prop_range_max'])
        self.mask_generator = mask_gen.BoxMaskGenerator(prop_range=self.prop_range_cutmix, n_boxes=self.hyperparameters['cutmix']['boxmask_n_boxes'],
                                                        random_aspect_ratio=not self.hyperparameters[
            'cutmix']['boxmask_fixed_aspect_ratio'],
            prop_by_area=not self.hyperparameters['cutmix']['boxmask_by_size'], within_bounds=not self.hyperparameters['cutmix']['boxmask_outside_bounds'],
            invert=not self.hyperparameters['cutmix']['boxmask_no_invert'])

        self.clss_generate_mask_params = mask_gen.GenerateMaskParamsToBatch_our(
            self.mask_generator)

        self.prop_range_cutout = (self.hyperparameters['cutmix']['cutout_prop_range_min'],
                                  self.hyperparameters['cutmix']['cutout_prop_range_max'])

        self.cutout_mask_generator = mask_gen.BoxMaskGenerator(prop_range=self.prop_range_cutout, n_boxes=self.hyperparameters['cutmix']['boxmask_n_boxes'],
                                                               random_aspect_ratio=False,
                                                               prop_by_area=not self.hyperparameters['cutmix']['boxmask_by_size'], within_bounds=not self.hyperparameters['cutmix']['boxmask_outside_bounds'],
                                                               invert=not self.hyperparameters['cutmix']['boxmask_no_invert'])

        self.clss_generate_cutout_mask_params = mask_gen.GenerateMaskParamsToBatch_our(
            self.cutout_mask_generator)

        # Initiate the networks
        # auto-encoder for domain a
        self.gen_a = AdaINGen(
            hyperparameters['input_dim_a'], hyperparameters['gen'])
        # auto-encoder for domain b
        self.gen_b = AdaINGen(
            hyperparameters['input_dim_b'], hyperparameters['gen'])
        # discriminator for domain a
        self.dis_a = MsImageDis(
            hyperparameters['input_dim_a'], hyperparameters['dis'])
        # discriminator for domain b
        self.dis_b = MsImageDis(
            hyperparameters['input_dim_b'], hyperparameters['dis'])

        # Create the segmentation model
        if self.hyperparameters['seg']['segmentor'] == 'UNet2D':
            self.seg_student = UNet2D(in_channels=self.in_channels, out_channels=self.n_classes, final_sigmoid=False, f_maps=32, layer_order='cbr',
                                      num_groups=8)
            self.seg_teacher = UNet2D(in_channels=self.in_channels, out_channels=self.n_classes, final_sigmoid=False, f_maps=32, layer_order='cbr',
                                      num_groups=8)
        elif self.hyperparameters['seg']['segmentor'] == 'ResidualUNet2D':
            print('segmentation network is ResidualUNet2D!!!!')
            self.seg_student = ResidualUNet2D(in_channels=self.in_channels, out_channels=self.n_classes, final_sigmoid=False, f_maps=32,
                                              conv_layer_order='cbr', num_groups=8)
            self.seg_teacher = ResidualUNet2D(in_channels=self.in_channels, out_channels=self.n_classes, final_sigmoid=False, f_maps=32,
                                              conv_layer_order='cbr', num_groups=8)

        elif self.hyperparameters['seg']['segmentor'] == 'ReTri':
            print('segmentation network is ReTri!!!!')
            self.seg_student = ReTri(in_channels=self.in_channels, out_channels=self.n_classes, final_sigmoid=False, f_maps=32,
                                                             conv_layer_order='cbr', num_groups=8)
            self.seg_teacher = ReTri(in_channels=self.in_channels, out_channels=self.n_classes, final_sigmoid=False, f_maps=32,
                                                             conv_layer_order='cbr', num_groups=8)

        elif self.hyperparameters['seg']['segmentor'] == 'DeepLabv3_plus_skipconnection_2d':
            self.seg_student = DeepLabv3_plus_skipconnection_2d(nInputChannels=self.in_channels, n_classes=self.n_classes, os=16, pretrained=False,
                                                                _print=True, final_sigmoid=False)
            self.seg_teacher = DeepLabv3_plus_skipconnection_2d(nInputChannels=self.in_channels, n_classes=self.n_classes, os=16, pretrained=False,
                                                                _print=True, final_sigmoid=False)

        elif self.hyperparameters['seg']['segmentor'] == 'Deeplabv2_2d':
            self.seg_student = Deeplabv2_2d(
                in_channels=self.in_channels, num_classes=self.n_classes, init_weights=None, restore_from=None)
            self.seg_teacher = Deeplabv2_2d(
                in_channels=self.in_channels, num_classes=self.n_classes, init_weights=None, restore_from=None)

        elif self.hyperparameters['seg']['segmentor'] == 'student_teacher':
            NetClass = network_architectures.seg.get(
                self.hyperparameters['mean_teacher']['mean_teacher_arch'])
            self.seg_student = NetClass(self.n_classes)
            self.seg_teacher = NetClass(self.n_classes).cuda()
        else:
            print('Unknown model type {}'.format(
                self.hyperparameters['seg']['segmentor']))
            return

        for p in self.seg_teacher.parameters():
            p.requires_grad = False

        self.freeze_bn = self.hyperparameters['mean_teacher']['freeze_bn']
        if self.freeze_bn:
            if not hasattr(self.seg_student, 'freeze_batchnorm'):
                raise ValueError(
                    'Network {} does not support batchnorm freezing'.format(self.hyperparameters['seg']['segmentor']))
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + \
            list(self.gen_b.parameters())
        seg_params_student = list(self.seg_student.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.student_opt = torch.optim.Adam([p for p in seg_params_student if p.requires_grad],
                                            lr=lr_seg, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters)
        self.seg_scheduler = get_scheduler(
            self.student_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.seg_student.apply(weights_init('gaussian'))
        self.teacher_opt = optim_weight_ema.EMAWeightOptimizer(
            self.seg_teacher, self.seg_student, self.hyperparameters['mean_teacher']['teacher_alpha'])
        self.eval_net = self.seg_teacher

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(
                hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def seg_loss_criterion(self, predict, mask):
        segloss1 = get_seg_loss_criterion(self.hyperparameters, 'loss1')
        segloss2 = get_seg_loss_criterion(self.hyperparameters, 'loss2')
        return segloss1(predict, mask), segloss2(predict, mask)
        # return segloss1(predict, mask)

    def feacut_loss_criterion(self, predict, target, eps=1e-6):
        # normalize teacher features
        # target_mean = target.mean(dim=[2, 3], keepdim=False)
        # target_std = (target.var(dim=[2, 3], keepdim=False) + eps).sqrt()
        # target = (target - target_mean.reshape(target.shape[0], target.shape[1], 1, 1)) / target_std.reshape(target.shape[0], target.shape[1], 1, 1)
        segloss1 = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        cosine_similarity = segloss1(predict, target)
        return -1.0*cosine_similarity.mean()

    def get_phase_loss(self, img1, img2):
        # note that img2 has to be real image
        # idea come from PCEDA Phase Consistent Ecological Domain Adaptation
        # Transform T should be phase preserving

        fft_fake_img1 = torch.fft.fft2(img1, dim=(-2, -1))
        fft_img1 = torch.stack((fft_fake_img1.real, fft_fake_img1.imag), -1)
        fft_real_img2 = torch.fft.fft2(img2, dim=(-2, -1))
        fft_img2 = torch.stack((fft_real_img2.real, fft_real_img2.imag), -1)

        # amp1, pha1 = self.extract_ampl_phase(fft1)
        amp2, _ = self.extract_ampl_phase(fft_img2)
        amp2max, _ = torch.max(amp2, dim=2, keepdim=True)
        amp2max, _ = torch.max(amp2max, dim=3, keepdim=True)
        w2 = amp2 / (amp2max + 1e-20)

        inner_product = (fft_img1 * fft_img2).sum(dim=-1)
        norm1 = (fft_img1.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        norm2 = (fft_img2.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        cos = inner_product / (norm1*norm2 + 1e-20)

        cos = cos * w2
        return -1.0*cos.mean()

    def get_log_amp_loss(self, im1, im2):
        # fft_im1 = torch.rfft(im1, signal_ndim=2, onesided=False, normalized=True ) # fft tranform
        # fft_im2 = torch.rfft(im2, signal_ndim=2, onesided=False, normalized=True ) # fft tranform
        fft_fake_img1 = torch.fft.fft2(im1, dim=(-2, -1))
        fft_img1 = torch.stack((fft_fake_img1.real, fft_fake_img1.imag), -1)
        fft_real_img2 = torch.fft.fft2(im2, dim=(-2, -1))
        fft_img2 = torch.stack((fft_real_img2.real, fft_real_img2.imag), -1)
        fft_amp1 = fft_img1[:, :, :, :, 0]**2 + fft_img1[:, :, :, :, 1]**2
        fft_amp1 = torch.sqrt(fft_amp1+1e-20)
        log_amp1 = torch.log(1 + fft_amp1)

        fft_amp2 = fft_img2[:, :, :, :, 0]**2 + fft_img2[:, :, :, :, 1]**2
        fft_amp2 = torch.sqrt(fft_amp2+1e-20)
        log_amp2 = torch.log(1 + fft_amp2)
        return self.recon_criterion(log_amp1, log_amp2)

    def extract_ampl_phase(self, fft_im):
        # fft_im: size should be bx3xhxwx2
        fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
        fft_amp = torch.sqrt(fft_amp+1e-20)
        fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
        return fft_amp, fft_pha

    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i !=
                      axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(
            n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
    # math for fft

    def fftwraper(self, netG, img):
        fft = torch.rfft(img, signal_ndim=2, onesided=False,
                         normalized=True)  # fft tranform
        fft = self.batch_fftshift2d(fft)
        amp, pha = self.extract_ampl_phase(fft)  # extract amplitude and phase
        # transform the amplitude
        _, _, imgH, imgW = list(img.size())
        imgHc = int(imgH/2)
        imgWc = int(imgW/2)

        amp = torch.log(amp+1.0e-8)    # log of the amplitude

        amp /= 10.0

        delta_amp = netG(amp)
        # amp_ = delta_amp + amp
        # amp_ = delta_amp * amp
        amp_ = amp
        # mybeta = int(100) # 50
        # amp_[:,:,imgHc-mybeta:imgHc+mybeta,imgWc-mybeta:imgWc+mybeta] += delta_amp[:,:,imgHc-mybeta:imgHc+mybeta,imgWc-mybeta:imgWc+mybeta]
        amp_ += delta_amp

        amp_ *= 10.0

        amp_ = torch.exp(amp_)-1.0e-8

        # generate the transformed image using amp_
        fft_ = torch.zeros(fft.size(), dtype=torch.float)
        fft_[:, :, :, :, 0] = torch.cos(pha) * amp_
        fft_[:, :, :, :, 1] = torch.sin(pha) * amp_
        fft_ = self.batch_ifftshift2d(fft_)
        _, _, imgH, imgW = img.size()
        img_ = torch.irfft(fft_, signal_ndim=2, onesided=False,
                           signal_sizes=[imgH, imgW], normalized=True)

        return torch.clamp(img_, min=0.0, max=1.0)

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def validate_seg(self, x, objectnet):
        self.eval()
        NetSeg = getattr(self, 'seg_%s' % objectnet)
        seg_activate = NetSeg(x)[0]
        out_seg = self.validate_mask(seg_activate)
        self.train()
        return out_seg

    def validate_seg_sup(self, x, objectnet):
        self.eval()
        # aug
        s0 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s1 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        loss_mask_ones = torch.ones_like(x).cuda().detach()
        GenNet = getattr(self, 'gen_%s' % objectnet)
        c_b, s_b_prime = GenNet.encode(x)
        # decode (within domain) ####self reconstruction#####
        x_b_recon0 = GenNet.decode(c_b, s0)
        x_ba = self.gen_a.decode(c_b, s1)
        cb_rec, s_rec = self.gen_a.encode(x_ba)
        x_bab = self.gen_b.decode(cb_rec, s0)
        x_coloraug = test_augmentations_color(
            self.un_normalize(x), loss_mask_ones)
        x_coloraug_img = x_coloraug[0]
        # seg
        # logits_coloraug = self.eval_net(x_coloraug_img)
        # logits_rec = self.eval_net(self.un_normalize(x_b_recon0))
        # logits_stu = self.seg_student(self.un_normalize(x))
        # logits_cyc = self.eval_net(self.un_normalize(x_bab))
        # logits_ori = self.eval_net(self.un_normalize(x))
        # prob_coloraug = F.softmax(logits_coloraug, dim=1)
        # out_seg_coloraug = self.validate_mask(prob_coloraug)
        # prob_rec = F.softmax(logits_rec, dim=1)
        # out_seg_rec = self.validate_mask(prob_rec)
        # prob_stu = F.softmax(logits_stu, dim=1)
        # out_seg_stu = self.validate_mask(prob_stu)
        # prob_cyc = F.softmax(logits_cyc, dim=1)
        # out_seg_cyc = self.validate_mask(prob_cyc)
        # prob_ori = F.softmax(logits_ori, dim=1)
        # out_seg_ori = self.validate_mask(prob_ori)

        prob_coloraug = self.eval_net(x_coloraug_img)[0]
        prob_rec = self.eval_net(self.un_normalize(x_b_recon0))[0]
        prob_cyc = self.eval_net(self.un_normalize(x_bab))[0]
        prob_ori = self.eval_net(self.un_normalize(x))[0]

        out_seg_coloraug = self.validate_mask(prob_coloraug)
        out_seg_rec = self.validate_mask(prob_rec)
        out_seg_cyc = self.validate_mask(prob_cyc)
        out_seg_ori = self.validate_mask(prob_ori)

        prob_out_seg_mean_tea = prob_rec + prob_cyc + prob_ori
        out_seg_mean_tea = self.validate_mask(prob_out_seg_mean_tea)
        prob_out_seg_mean_tea_aug = prob_rec + prob_cyc + prob_ori + prob_coloraug
        out_seg_mean_tea_aug = self.validate_mask(prob_out_seg_mean_tea_aug)
        self.train()
        return out_seg_rec, out_seg_cyc, out_seg_ori, out_seg_coloraug, out_seg_mean_tea, out_seg_mean_tea_aug

    def validate_seg_SA_haveself(self, x, objectnet):
        self.eval()
        # aug
        s0 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s1 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s2 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        GenNet = getattr(self, 'gen_%s' % objectnet)
        c_b, s_b_prime = GenNet.encode(x)
        # decode (within domain) ####self reconstruction#####
        x_b_recon0 = GenNet.decode(c_b, s0)
        x_b_recon1 = GenNet.decode(c_b, s1)
        x_b_recon2 = GenNet.decode(c_b, s2)
        # seg
        NetSeg = getattr(self, 'seg_%s' % objectnet)
        seg_activate0 = NetSeg(x)[0]
        seg_activate1 = NetSeg(x_b_recon0)[0]
        seg_activate2 = NetSeg(x_b_recon1)[0]
        seg_activate3 = NetSeg(x_b_recon2)[0]
        seg_activate = seg_activate0 + seg_activate1 + seg_activate2 + seg_activate3
        out_seg = self.validate_mask(seg_activate)
        self.train()
        return out_seg

    def validate_seg_SA_simply(self, x, objectnet):
        self.eval()
        # aug
        s0 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s1 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s2 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        GenNet = getattr(self, 'gen_%s' % objectnet)
        c_b, s_b_prime = GenNet.encode(x)
        # decode (within domain) ####self reconstruction#####
        x_b_recon0 = GenNet.decode(c_b, s0)
        x_b_recon1 = GenNet.decode(c_b, s1)
        x_b_recon2 = GenNet.decode(c_b, s2)
        # seg
        NetSeg = getattr(self, 'seg_%s' % objectnet)

        seg_activate1 = NetSeg(x_b_recon0)[0]
        seg_activate2 = NetSeg(x_b_recon1)[0]
        seg_activate3 = NetSeg(x_b_recon2)[0]
        seg_activate = seg_activate1 + seg_activate2 + seg_activate3
        out_seg = self.validate_mask(seg_activate)
        self.train()
        return out_seg

    def validate_seg_cutmix(self, x, objectnet):
        self.eval()
        # aug
        sb0 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        sa0 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        sb1 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())

        GenNet = getattr(self, 'gen_%s' % objectnet)
        c_b, s_b_prime = GenNet.encode(x)
        # decode (within domain) ####self reconstruction#####
        x_b0_stu_b = self.gen_b.decode(c_b, sb0)
        x_ba0_2b = self.gen_a.decode(c_b, sa0)
        c_ba0b_rec, _ = self.gen_a.encode(x_ba0_2b)
        x_ba0b_tea_b = self.gen_b.decode(c_ba0b_rec, sb1)

        # seg
        logits_b0 = self.eval_net(x_b0_stu_b)
        prob_eval_b0 = F.softmax(logits_b0, dim=1)
        out_seg_b0 = self.validate_mask(prob_eval_b0)

        logits_bab = self.eval_net(x_ba0b_tea_b)
        prob_eval_bab = F.softmax(logits_bab, dim=1)
        out_seg_bab = self.validate_mask(prob_eval_bab)
        self.train()
        return out_seg_b0, out_seg_bab

    def sample_SA_seg_simply(self, x_a, x_b, test_display_masks_a, test_display_masks_b):
        self.eval()
        # cross_aug
        s_a1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())

        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        x_a1, x_a2, x_a3, x_ab1, x_ab2 = [], [], [], [], []
        x_b1, x_b2, x_b3, x_ba1, x_ba2 = [], [], [], [], []
        x_a1_seg, x_a2_seg, x_a3_seg, x_ab1_seg, x_ab2_seg = [], [], [], [], []
        x_b1_seg, x_b2_seg, x_b3_seg, x_ba1_seg, x_ba2_seg = [], [], [], [], []
        x_a_seg, x_b_seg, x_a_mask, x_b_mask = [], [], [], []
        aug_a, aug_b, self_seg_pred_a, self_seg_pred_b, a_final, b_final = [], [], [], [], [], []
        x_b1_seg_volume = []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))

            # self_aug
            x_a1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_a, s_a1[i].unsqueeze(0)))[0])))
            x_a2_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_a, s_a2[i].unsqueeze(0)))[0])))
            x_a3_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_a, s_a2[i].unsqueeze(0)))[0])))

            x_b1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_b, s_b1[i].unsqueeze(0)))[0])))
            x_b1_seg_volume.append(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_b, s_b1[i].unsqueeze(0)))[0]))
            x_b2_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_b, s_b2[i].unsqueeze(0)))[0])))
            x_b3_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))[0])))

            x_a_mask.append(self.mask2color_single(
                (test_display_masks_a[i].unsqueeze(0))[:, 0, :, :]))
            x_b_mask.append(self.mask2color_single(
                (test_display_masks_b[i].unsqueeze(0))[:, 0, :, :]))

            x_a1.append(self.gen_a.decode(c_a, s_a1[i].unsqueeze(0)))
            x_a2.append(self.gen_a.decode(c_a, s_a2[i].unsqueeze(0)))
            x_a3.append(self.gen_a.decode(c_a, s_a3[i].unsqueeze(0)))

            x_b1.append(self.gen_b.decode(c_b, s_b1[i].unsqueeze(0)))
            x_b2.append(self.gen_b.decode(c_b, s_b2[i].unsqueeze(0)))
            x_b3.append(self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))

            seg_activate1 = self.seg_a(
                self.gen_a.decode(c_a, s_a1[i].unsqueeze(0)))[0]
            seg_activate2 = self.seg_a(
                self.gen_a.decode(c_a, s_a2[i].unsqueeze(0)))[0]
            seg_activate3 = self.seg_a(
                self.gen_a.decode(c_a, s_a3[i].unsqueeze(0)))[0]
            seg_activate_a = seg_activate1 + seg_activate2 + seg_activate3

            aug_a.append(self.mask2color_single(
                self.validate_mask(seg_activate_a)))
            a_final.append(self.validate_mask(seg_activate_a))

            seg_activate1 = self.seg_b(
                self.gen_b.decode(c_b, s_b1[i].unsqueeze(0)))[0]
            seg_activate2 = self.seg_b(
                self.gen_b.decode(c_b, s_b2[i].unsqueeze(0)))[0]
            seg_activate3 = self.seg_b(
                self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))[0]

            seg_activate_b = seg_activate1 + seg_activate2 + seg_activate3

            aug_b.append(self.mask2color_single(
                self.validate_mask(seg_activate_b)))
            b_final.append(self.validate_mask(seg_activate_b))

        x_a_mask, x_b_mask = torch.cat(x_a_mask), torch.cat(x_b_mask)
        x_a1_seg, x_b1_seg = torch.cat(x_a1_seg), torch.cat(x_b1_seg)
        x_a2_seg, x_b2_seg = torch.cat(x_a2_seg), torch.cat(x_b2_seg)
        x_a3_seg, x_b3_seg = torch.cat(x_a3_seg), torch.cat(x_b3_seg)
        x_a1, x_a2, x_a3 = torch.cat(x_a1), torch.cat(x_a2), torch.cat(x_a3)
        x_b1, x_b2, x_b3 = torch.cat(x_b1), torch.cat(x_b2), torch.cat(x_b3)

        a_final, b_final = torch.cat(a_final), torch.cat(b_final)
        aug_a, aug_b = torch.cat(aug_a), torch.cat(aug_b)
        x_b1_seg_volume = torch.cat(x_b1_seg_volume)
        self.train()
        return x_a, x_a_mask, x_a1, x_a1_seg, x_a2, x_a2_seg, x_a3, x_a3_seg, aug_a,\
            x_b, x_b_mask, x_b1, x_b1_seg, x_b2, x_b2_seg, x_b3, x_b3_seg, aug_b,\
            x_b1_seg_volume, a_final, b_final

    def sample_SA_seg_cutmix(self, x_a, x_b, test_display_masks_a, test_display_masks_b):
        self.eval()
        # cross_aug
        s_a1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())

        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        x_a1, x_a2, x_a3, x_ab1, x_ab2 = [], [], [], [], []
        x_b1, x_b2, x_b3, x_ba1, x_ba2 = [], [], [], [], []
        x_a1_seg, x_a2_seg, x_a3_seg, x_ab1_seg, x_ab2_seg = [], [], [], [], []
        x_b1_seg, x_b2_seg, x_b3_seg, x_ba1_seg, x_ba2_seg = [], [], [], [], []
        x_a_seg, x_b_seg, x_a_mask, x_b_mask = [], [], [], []
        aug_a, aug_b, self_seg_pred_a, self_seg_pred_b, a_final, b_final = [], [], [], [], [], []
        x_b1_seg_volume = []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))

            # self_aug
            x_a1_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_a, s_a1[i].unsqueeze(0)))))))
            x_a2_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_a, s_a2[i].unsqueeze(0)))))))
            x_a3_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_a, s_a2[i].unsqueeze(0)))))))

            x_b1_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_b, s_b1[i].unsqueeze(0)))))))
            x_b1_seg_volume.append(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_b, s_b1[i].unsqueeze(0))))))
            # print(x_b1_seg_volume[0].shape)
            x_b2_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_b, s_b2[i].unsqueeze(0)))))))
            x_b3_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))))))

            x_a_mask.append(self.mask2color_single(
                (test_display_masks_a[i].unsqueeze(0))[:, 0, :, :]))
            x_b_mask.append(self.mask2color_single(
                (test_display_masks_b[i].unsqueeze(0))[:, 0, :, :]))

            x_a1.append(self.un_normalize(
                self.gen_b.decode(c_a, s_a1[i].unsqueeze(0))))
            x_a2.append(self.un_normalize(
                self.gen_b.decode(c_a, s_a2[i].unsqueeze(0))))
            x_a3.append(self.un_normalize(
                self.gen_b.decode(c_a, s_a3[i].unsqueeze(0))))

            x_b1.append(self.un_normalize(
                self.gen_b.decode(c_b, s_b1[i].unsqueeze(0))))
            x_b2.append(self.un_normalize(
                self.gen_b.decode(c_b, s_b2[i].unsqueeze(0))))
            x_b3.append(self.un_normalize(
                self.gen_b.decode(c_b, s_b3[i].unsqueeze(0))))

            seg_activate1 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_a, s_a1[i].unsqueeze(0))))
            seg_activate2 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_a, s_a2[i].unsqueeze(0))))
            seg_activate3 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_a, s_a3[i].unsqueeze(0))))
            seg_activate_a = seg_activate1 + seg_activate2 + seg_activate3

            aug_a.append(self.mask2color_single(
                self.validate_mask(seg_activate_a)))
            a_final.append(self.validate_mask(seg_activate_a))

            seg_activate1 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_b, s_b1[i].unsqueeze(0))))
            seg_activate2 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_b, s_b2[i].unsqueeze(0))))
            seg_activate3 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_b, s_b3[i].unsqueeze(0))))

            seg_activate_b = seg_activate1 + seg_activate2 + seg_activate3

            aug_b.append(self.mask2color_single(
                self.validate_mask(seg_activate_b)))
            b_final.append(self.validate_mask(seg_activate_b))

        x_a_mask, x_b_mask = torch.cat(x_a_mask), torch.cat(x_b_mask)
        x_a1_seg, x_b1_seg = torch.cat(x_a1_seg), torch.cat(x_b1_seg)
        x_a2_seg, x_b2_seg = torch.cat(x_a2_seg), torch.cat(x_b2_seg)
        x_a3_seg, x_b3_seg = torch.cat(x_a3_seg), torch.cat(x_b3_seg)
        x_a1, x_a2, x_a3 = torch.cat(x_a1), torch.cat(x_a2), torch.cat(x_a3)
        x_b1, x_b2, x_b3 = torch.cat(x_b1), torch.cat(x_b2), torch.cat(x_b3)

        a_final, b_final = torch.cat(a_final), torch.cat(b_final)
        aug_a, aug_b = torch.cat(aug_a), torch.cat(aug_b)
        x_b1_seg_volume = torch.cat(x_b1_seg_volume)
        self.train()
        return x_a, x_a_mask, x_a1, x_a1_seg, x_a2, x_a2_seg, x_a3, x_a3_seg, aug_a,\
            x_b, x_b_mask, x_b1, x_b1_seg, x_b2, x_b2_seg, x_b3, x_b3_seg, aug_b,\
            x_b1_seg_volume, a_final, b_final

    def sample_test_b(self, input_x_b, test_display_masks_b):
        self.eval()
        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        x_babs, x_bab_segs = [], []
        x_b0s, x_b0_segs = [], []
        x_b_segs = []
        x_b_segs2 = []
        x_bab_segs2 = []
        x_ab0s, x_ab0_segs = [], []
        x_a_masks, x_b_masks = [], []
        x_b0_segs2, x_ab0_segs2 = [], []
        x_as, x_bs = [], []
        x_b_coloraugs = []
        x_b_coloraug_segs = []
        combine_pred_b_teas = []
        x_b_seg_volume = []
        x_b_seg_volume_aug = []
        loss_mask_ones = torch.ones_like(
            input_x_b[0].unsqueeze(0)).cuda().detach()
        for i in range(input_x_b.size(0)):

            c_b, s_b_fake = self.gen_b.encode(input_x_b[i].unsqueeze(0))
            x_b0 = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_ba = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            # encode again    ####  cycle reconstruct result
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            # decode again (if needed)
            x_bab = self.gen_b.decode(c_b_recon, s_b0[i].unsqueeze(0))
            x_bab = self.un_normalize(x_bab)
            x_b0 = self.un_normalize(x_b0)

            x_babs.append(x_bab)
            x_b0s.append(x_b0)

            x_b_ori = self.un_normalize(input_x_b[i]).unsqueeze(0).float()
            x_b_coloraug = test_augmentations_color(x_b_ori, loss_mask_ones)
            x_b_coloraug_img = x_b_coloraug[0]
            x_b_coloraugs.append(x_b_coloraug_img)
            x_b_coloraug_seg = self.eval_net(x_b_coloraug_img)
            x_b_seg = self.eval_net(x_b_ori)
            x_b0_seg = self.eval_net(x_b0.float())
            x_bab_seg = self.eval_net(x_bab.float())
            prob_x_b_seg = F.softmax(x_b_seg, dim=1)
            prob_x_b0_seg = F.softmax(x_b0_seg, dim=1)
            prob_x_bab_seg = F.softmax(x_bab_seg, dim=1)
            prob_x_b_coloraug_seg = F.softmax(x_b_coloraug_seg, dim=1)
            mean_logits_b_tea = (
                prob_x_b_seg + prob_x_b0_seg + prob_x_bab_seg + prob_x_b_coloraug_seg) / 4
            combine_pred_b_teas.append(self.mask2color_single(
                self.validate_mask(mean_logits_b_tea)))
            x_b_seg_volume_aug.append(self.validate_mask(mean_logits_b_tea))
            x_b_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_b_seg)))
            x_b_seg_volume.append(self.validate_mask(prob_x_b_seg))
            x_b0_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_b0_seg)))
            x_bab_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_bab_seg)))
            x_b_coloraug_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_b_coloraug_seg)))
            x_b_mask = self.mask2color_single(
                (test_display_masks_b[i].unsqueeze(0))[:, 0, :, :])
            x_b_masks.append(x_b_mask)
            x_bs.append(x_b_ori)
        x_b0s = torch.cat(x_b0s)
        x_babs = torch.cat(x_babs)
        x_b0_segs = torch.cat(x_b0_segs)
        x_bab_segs = torch.cat(x_bab_segs)
        x_b_segs = torch.cat(x_b_segs)
        x_b_masks = torch.cat(x_b_masks)
        x_bs = torch.cat(x_bs)
        x_b_coloraugs = torch.cat(x_b_coloraugs)
        x_b_coloraug_segs = torch.cat(x_b_coloraug_segs)
        x_b_seg_volume_aug = torch.cat(x_b_seg_volume_aug)
        x_b_seg_volume = torch.cat(x_b_seg_volume)
        self.train()
        return x_bs, x_b_segs, x_b0s, x_b0_segs, x_babs, x_bab_segs, x_b_coloraugs, x_b_coloraug_segs, x_b_masks,\
            x_b_seg_volume_aug, x_b_seg_volume

    def sample_SA_quadtree(self, input_x_a, input_x_b, test_display_masks_a, test_display_masks_b):
        self.eval()
        # cross_aug
        s_a0 = Variable(torch.randn(input_x_a.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a1 = Variable(torch.randn(input_x_a.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a2 = Variable(torch.randn(input_x_a.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a3 = Variable(torch.randn(input_x_a.size(0),
                        self.style_dim, 1, 1).cuda())

        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        x_as = []
        x_bs = []
        x_a1, x_a2, x_a3, x_ab1, x_ab2 = [], [], [], [], []
        x_b1, x_b2, x_b3, x_ba1, x_ba2 = [], [], [], [], []
        x_a1_seg, x_a2_seg, x_a3_seg, x_ab1_seg, x_ab2_seg = [], [], [], [], []
        x_b1_seg, x_b2_seg, x_b3_seg, x_ba1_seg, x_ba2_seg = [], [], [], [], []
        x_a_seg, x_b_seg, x_a_mask, x_b_mask = [], [], [], []
        aug_a, aug_b, self_seg_pred_a, self_seg_pred_b, a_final, b_final = [], [], [], [], [], []
        x_b1_seg_volume = []
        pred_ab0_teas = []
        x_ab0s, x_ab0_segs = [], []
        x_a_masks, x_b_masks = [], []
        x_bs = []
        pred_b_teas = []
        x_b0s = []
        pred_b0_teas = []
        x_cycs = []
        pred_cycs_teas = []
        combine_pred_b_teas = []
        x_as = []
        x_b_seg_volume = []
        for i in range(input_x_a.size(0)):
            x_a = input_x_a[i].unsqueeze(0)
            x_as.append(self.un_normalize(x_a))
            x_b = input_x_b[i].unsqueeze(0)
            x_bs.append(self.un_normalize(x_b))
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)

            # self_aug
            x_a1_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_a, s_a1[i].unsqueeze(0)))))))
            x_a2_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_a, s_a2[i].unsqueeze(0)))))))
            x_a3_seg.append(self.mask2color_single(self.validate_mask(
                self.eval_net(self.un_normalize(self.gen_b.decode(c_a, s_a2[i].unsqueeze(0)))))))

            x_b_tea = x_b.clone()
            x_b_tea = self.un_normalize(x_b_tea)
            x_b0_tea = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_b0_tea = self.un_normalize(x_b0_tea)
            x_b0s.append(x_b0_tea)
            x_ba0_2b = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            c_ba0b_rec, _ = self.gen_a.encode(x_ba0_2b)
            x_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b1[i].unsqueeze(0))
            x_ba0b_tea = self.un_normalize(x_ba0b_tea)
            x_cycs.append(x_ba0b_tea)

            logits_b_tea = self.eval_net(
                x_b_tea).detach()
            pred_b_teas.append(self.mask2color_single(
                self.validate_mask(logits_b_tea)))
            x_b_seg_volume.append(self.validate_mask(logits_b_tea))
            # print(x_b_seg_volume[0].shape)
            logits_b0_tea = self.eval_net(
                x_b0_tea).detach()
            pred_b0_teas.append(self.mask2color_single(
                self.validate_mask(logits_b0_tea)))
            logits_ba0b_tea = self.eval_net(
                x_ba0b_tea).detach()
            pred_cycs_teas.append(self.mask2color_single(
                self.validate_mask(logits_ba0b_tea)))
            mean_logits_b_tea = (
                logits_b_tea + logits_b0_tea + logits_ba0b_tea) / 3
            combine_pred_b_teas.append(self.mask2color_single(
                self.validate_mask(mean_logits_b_tea)))
            x_a_mask.append(self.mask2color_single(
                (test_display_masks_a[i].unsqueeze(0))[:, 0, :, :]))
            x_b_mask.append(self.mask2color_single(
                (test_display_masks_b[i].unsqueeze(0))[:, 0, :, :]))

            x_a1.append(self.un_normalize(
                self.gen_b.decode(c_a, s_a1[i].unsqueeze(0))))
            x_a2.append(self.un_normalize(
                self.gen_b.decode(c_a, s_a2[i].unsqueeze(0))))
            x_a3.append(self.un_normalize(
                self.gen_b.decode(c_a, s_a3[i].unsqueeze(0))))

            seg_activate1 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_a, s_a1[i].unsqueeze(0))))
            seg_activate2 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_a, s_a2[i].unsqueeze(0))))
            seg_activate3 = self.eval_net(
                self.un_normalize(self.gen_b.decode(c_a, s_a3[i].unsqueeze(0))))
            seg_activate_a = seg_activate1 + seg_activate2 + seg_activate3

            aug_a.append(self.mask2color_single(
                self.validate_mask(seg_activate_a)))
            a_final.append(self.validate_mask(seg_activate_a))

            aug_b.append(self.mask2color_single(
                self.validate_mask(mean_logits_b_tea)))
            b_final.append(self.validate_mask(mean_logits_b_tea))

        x_a_mask, x_b_mask = torch.cat(x_a_mask), torch.cat(x_b_mask)
        x_a1_seg = torch.cat(x_a1_seg)
        x_a2_seg = torch.cat(x_a2_seg)
        x_a3_seg = torch.cat(x_a3_seg)
        x_a1, x_a2, x_a3 = torch.cat(x_a1), torch.cat(x_a2), torch.cat(x_a3)
        x_as = torch.cat(x_as)
        x_bs = torch.cat(x_bs)
        pred_b_teas = torch.cat(pred_b_teas)
        x_b0s = torch.cat(x_b0s)
        pred_b0_teas = torch.cat(pred_b0_teas)
        x_cycs = torch.cat(x_cycs)
        pred_cycs_teas = torch.cat(pred_cycs_teas)
        combine_pred_b_teas = torch.cat(combine_pred_b_teas)

        a_final, b_final = torch.cat(a_final), torch.cat(b_final)
        aug_a, aug_b = torch.cat(aug_a), torch.cat(aug_b)
        x_b_seg_volume = torch.cat(x_b_seg_volume)
        self.train()
        return x_as, x_a_mask, x_a1, x_a1_seg, x_a2, x_a2_seg, x_a3, x_a3_seg, aug_a,\
            x_bs, x_b_mask, x_bs, pred_b_teas, x_b0s, pred_b0_teas, x_cycs, pred_cycs_teas, aug_b,\
            x_b_seg_volume, a_final, b_final

    def validate_seg_TTA_CTA(self, x, objectnet):
        self.eval()
        # aug
        name = ['a', 'b']
        name.remove(objectnet)
        s0 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s1 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s_a = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        s_a1 = Variable(torch.randn(x.size(0), self.style_dim, 1, 1).cuda())
        GenNet = getattr(self, 'gen_%s' % objectnet)
        GenNet_source = getattr(self, 'gen_%s' % name[0])
        c_b, s_b_prime = GenNet.encode(x)
        # decode (within domain) ####self reconstruction#####
        x_b_recon0 = GenNet.decode(c_b, s0)
        x_b_recon1 = GenNet.decode(c_b, s1)
        x_ba = GenNet_source.decode(c_b, s_a)
        x_ba1 = GenNet_source.decode(c_b, s_a1)
        # seg
        NetSeg = getattr(self, 'seg_%s' % objectnet)
        NetSeg_source = getattr(self, 'seg_%s' % name[0])
        seg_activate0 = NetSeg(x)[0]
        seg_activate1 = NetSeg(x_b_recon0)[0]
        seg_activate2 = NetSeg(x_b_recon1)[0]
        seg_activate3 = NetSeg_source(x_ba)[0]
        seg_activate4 = NetSeg_source(x_ba1)[0]
        seg_activate = seg_activate0 + seg_activate1 + \
            seg_activate2 + seg_activate3 + seg_activate4
        out_seg = self.validate_mask(seg_activate)
        self.train()
        return out_seg

    def gen_update(self, x_a, x_b, hyperparameters, gauss_kernel):
        # print('self.training gen_update:', self.training)
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # diverse
        s_a1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain) ####self reconstruction#####
        self.x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        self.x_b_recon = self.gen_b.decode(c_b, s_b_prime)

        # decode (cross domain)  translation
        self.x_ba = self.gen_a.decode(c_b, s_a)
        self.x_ba1 = self.gen_a.decode(c_b, s_a1)
        self.x_ba2 = self.gen_a.decode(c_b, s_a2)
        # different appearance with same gaussian structure L1
        gaussian_ba = self.find_fake_freq(self.x_ba, gauss_kernel, index=None)
        gaussian_ba1 = self.find_fake_freq(
            self.x_ba1, gauss_kernel, index=None)
        gaussian_ba2 = self.find_fake_freq(
            self.x_ba2, gauss_kernel, index=None)

        gaus_b_ba = self.find_fake_freq(x_b, gauss_kernel, index=None)

        self.x_ab = self.gen_b.decode(c_a, s_b)
        self.x_ab1 = self.gen_a.decode(c_a, s_b1)
        self.x_ab2 = self.gen_a.decode(c_a, s_b2)
        # different appearance with same gaussian structure L1
        gaussian_ab = self.find_fake_freq(self.x_ab, gauss_kernel, index=None)
        gaussian_ab1 = self.find_fake_freq(
            self.x_ab1, gauss_kernel, index=None)
        gaussian_ab2 = self.find_fake_freq(
            self.x_ab2, gauss_kernel, index=None)

        gaus_a_ab = self.find_fake_freq(x_a, gauss_kernel, index=None)

        # encode again    ####  cycle reconstruct result
        c_b_recon, s_a_recon = self.gen_a.encode(self.x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(self.x_ab)

        # decode again (if needed)
        self.x_aba = self.gen_a.decode(
            c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        self.x_bab = self.gen_b.decode(
            c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss with different appearance code
        self.loss_gaussian_ba = self.recon_criterion(gaussian_ba, gaussian_ba1) + \
            self.recon_criterion(
                gaussian_ba1, gaussian_ba2) if hyperparameters['gaussian_diffappear_w'] > 0 else 0
        self.loss_gaussian_ab = self.recon_criterion(gaussian_ab, gaussian_ab1) + self.recon_criterion(
            gaussian_ab1, gaussian_ab2) if hyperparameters['gaussian_diffappear_w'] > 0 else 0

        # different modality should preserve the structure
        self.loss_gaussian_b_ba = self.recon_criterion(
            gaussian_ba, gaus_b_ba) if hyperparameters['gaussian_crossmodal_w'] > 0 else 0
        self.loss_gaussian_a_ab = self.recon_criterion(
            gaussian_ab, gaus_a_ab) if hyperparameters['gaussian_crossmodal_w'] > 0 else 0

        self.loss_phase_ba = self.get_phase_loss(self.x_ba1, self.x_ba) + \
            self.get_phase_loss(
                self.x_ba2, self.x_ba1) if hyperparameters['phase_ba_w'] > 0 else 0
        self.loss_phase_ab = self.get_phase_loss(self.x_ab1, self.x_ab) + self.get_phase_loss(
            self.x_ab2, self.x_ab1) if hyperparameters['phase_ba_w'] > 0 else 0
        # within domain
        self.loss_gen_recon_x_a = self.recon_criterion(self.x_a_recon,  x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(self.x_b_recon,  x_b)
        # style and content reconstruction
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(
            self.x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(
            self.x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(self.x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(self.x_ab)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(
            self.vgg,  self.x_ba,  x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(
            self.vgg,  self.x_ab,  x_a) if hyperparameters['vgg_w'] > 0 else 0
        # phase consitency loss  soft phase
        # soft phase contraint
        if hyperparameters['softphase_w'] > 0:
            self.loss_phase_a = self.get_phase_loss(self.x_a_recon, x_a) + \
                self.get_phase_loss(self.x_aba, x_a) + \
                self.get_phase_loss(self.x_ab, x_a)

            self.loss_phase_b = self.get_phase_loss(
                self.x_b_recon, x_b) + self.get_phase_loss(self.x_bab, x_b) + self.get_phase_loss(self.x_ba, x_b)
        else:
            self.loss_phase_a = 0
            self.loss_phase_b = 0

        if hyperparameters['logamp_w'] > 0:
            self.loss_logamp_a = self.get_log_amp_loss(self.x_a_recon, x_a) + \
                self.get_log_amp_loss(self.x_aba, x_a)
            self.loss_logamp_b = self.get_log_amp_loss(
                self.x_b_recon, x_b) + self.get_log_amp_loss(self.x_bab, x_b)
        else:
            self.loss_logamp_a = 0
            self.loss_logamp_b = 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
            hyperparameters['gan_w'] * self.loss_gen_adv_b + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
            hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
            hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
            hyperparameters['softphase_w'] * self.loss_phase_a + \
            hyperparameters['softphase_w'] * self.loss_phase_b + \
            hyperparameters['gaussian_diffappear_w'] * self.loss_gaussian_ba + \
            hyperparameters['gaussian_diffappear_w'] * self.loss_gaussian_ab + \
            hyperparameters['gaussian_crossmodal_w'] * self.loss_gaussian_b_ba + \
            hyperparameters['gaussian_crossmodal_w'] * self.loss_gaussian_a_ab + \
            hyperparameters['phase_ba_w'] * self.loss_phase_ba + \
            hyperparameters['phase_ba_w'] * self.loss_phase_ab + \
            hyperparameters['logamp_w'] * self.loss_logamp_a + \
            hyperparameters['logamp_w'] * self.loss_logamp_b
        # self.loss_gen_total.backward(retain_graph=True)
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def seg_update_BSFE(self, x_a, y_a, x_b, hyperparameters):
        self.seg_student.train()
        if self.seg_teacher is not self.seg_student:
            self.seg_teacher.train()
        self.student_opt.zero_grad()
        if hyperparameters['update_withgen']:
            self.gen_opt.zero_grad()
        s_b0 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())

        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        # decode (cross domain)  translation [-1,1]
        x_ab0 = self.gen_b.decode(c_a, s_b0)
        
        ##[0,1]
        x_ab0 = self.un_normalize(x_ab0)
        x_b0 = self.un_normalize(x_b)
       
        x_ab0_aug = unsup_augmentations_weak(x_ab0.float(), y_a.float())

        if self.hyperparameters['strong_aug_para']['sup_quadtree']:
            qt_ab0 = QTree(0.0005, 2, x_ab0.squeeze(0))
            qt_ab0.subdivide()
            flag_quadtree_aug = random.randint(0, 1)
            if flag_quadtree_aug:
                quad_mask_ab0 = qt_ab0.gen_quadaug_mask(
                    ratio=hyperparameters['strong_aug_para']['quad_ratio'])
                quad_mask_ab0 = quad_mask_ab0.unsqueeze(0)
                # print(quad_mask_ab0.shape)

                quadtree_maskab0_aug = sup_augmentations(
                    quad_mask_ab0, quad_mask_ab0, params=sup_augmentations._params)
                aug_img_ab0 = quadtree_maskab0_aug[1] * x_ab0_aug[0]
            else:
                aug_img_ab0 = x_ab0_aug[0]
        else:
            aug_img_ab0 = x_ab0_aug[0]

        y_ab0 = x_ab0_aug[1][:, 0, :, :]
        y_a = y_a[:, 0, :, :]
        # print('y_ab0.shape:*******', y_ab0.shape)
        # print('y_a.shape:*******', y_a.shape)
        
        # print('torch.unique(y_ab0)***:', torch.unique(y_ab0))
        #
        # Supervised branch
        #
        logits_sup = self.seg_student(aug_img_ab0.float().contiguous(), x_b0)
        self.celoss, self.diceloss = self.seg_loss_criterion(
            logits_sup, y_ab0.long())
        self.loss_supervise1 = self.celoss + self.diceloss

        ## self_attention
        logits_sup_self = self.seg_student(x_ab0.float().contiguous(), x_b0, if_self_attention = True)
        # celoss2, diceloss2 = self.seg_loss_criterion(
        #     logits_sup_self, y_a.long())
        # self.loss_supervise2 = celoss2 + diceloss2
        ## cross_attention
        logits_sup_cross = self.seg_student(x_ab0.float().contiguous(), x_b0, if_cross_attention = True)
        # celoss3, diceloss3 = self.seg_loss_criterion(
        #     logits_sup_cross, y_a.long())
        delta_prob = logits_sup_self - logits_sup_cross
        loss_var =  delta_prob * delta_prob
        self.loss_supervise3 =  loss_var.mean()

        self.loss_supervise = self.loss_supervise1 + hyperparameters['attn_consis_weight'] * self.loss_supervise3
        self.loss_supervise.backward()
        self.student_opt.step()
        if self.teacher_opt is not None:
            self.teacher_opt.step(self.seg_teacher, self.seg_student)
        if hyperparameters['update_withgen']:
            self.gen_opt.step()

    def seg_update_Multiview_Duallevel(self, x_a, x_b, x_b1, x_b2, hyperparameters):
        # y_a.shape: BHW
        self.seg_student.train()
        if self.seg_teacher is not self.seg_student:
            self.seg_teacher.train()
        self.student_opt.zero_grad()

        if hyperparameters['update_withgen']:
            self.gen_opt.zero_grad()
        # Input CT-->MR and origin MR, MR1, MR2
        # First, Get CT-->MR weak aug img and corresponding strong aug img, and get teachers' predction
        # for origin input img and afterwards weak aug the prediction
        # Second, Get MR weak aug img and corresponding strong aug img, and get teachers' predction
        # for origin input img, cyc img and self recon img and sum mean the prediction, afterwards weak aug the prediction
        # Third, cutmix and input the student network
        # Fourth, random int 01 and apply the quadtree augmentation on cutmix img
        # For first pair CT-->MR AND MR
        s_b0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b4 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        s_a0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_a1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain) ####self reconstruction#####
        # [-1,1] [1,3,224,256]
        x_b_tea = x_b.clone()
        x_b_tea = self.un_normalize(x_b_tea)
        x_b0_tea = self.gen_b.decode(c_b, s_b0)
        x_b0_tea = self.un_normalize(x_b0_tea)
        x_ba0_2b = self.gen_a.decode(c_b, s_a0)
        c_ba0b_rec, _ = self.gen_a.encode(x_ba0_2b)
        x_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b1)
        x_ba0b_tea = self.un_normalize(x_ba0b_tea)
        x_ab0_tea = self.gen_b.decode(c_a, s_b2)
        x_ab0_tea = self.un_normalize(x_ab0_tea)

        loss_mask_ones = torch.ones_like(x_b_tea).cuda().detach()
        if self.hyperparameters['mean_teacher']['cons_weight']:
            #
            # Unsupervised branch consistency branch
            #
            if self.hyperparameters['mean_teacher']['cross_cutmix']:

                x_ab0_weak = unsup_augmentations_weak(
                    x_ab0_tea, loss_mask_ones)
                params_weak_ab0_tea = unsup_augmentations_weak._params
                img_ab0_weak = x_ab0_weak[0]
                loss_mask_ab0_weak = x_ab0_weak[1]
                x_ab0_weak_strong_stu = unsup_augmentations_strongcolor(
                    img_ab0_weak, loss_mask_ones)
                img_ab0_weak_strong_stu = x_ab0_weak_strong_stu[0]
                with torch.no_grad():
                    # logits_ab0_tea = self.seg_teacher(img_ab0_weak).detach()
                    # prob_ab0_tea_weak = F.softmax(logits_ab0_tea, dim=1)
                    # print('no pai*********!!!!')
                    if self.hyperparameters['mean_teacher']['feature_cutmix']:
                        logits_ab0_tea_FC, fea_ab0_tea_FC = self.seg_teacher(
                            img_ab0_weak, [], True)
                        logits_ab0_tea_FC, fea_ab0_tea_FC0 = logits_ab0_tea_FC.detach(
                        ), fea_ab0_tea_FC[0].detach()
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                            fea_ab0_tea_FC2 = fea_ab0_tea_FC[2].detach()
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                            fea_ab0_tea_FC2 = fea_ab0_tea_FC[1].detach()
                        print('****** no pai for feature cutmix ******!!!!')
                    logits_ab0_tea = self.seg_teacher(x_ab0_tea, [])
                    logits_ab0_tea = logits_ab0_tea.detach()
                    prob_ab0_tea = F.softmax(logits_ab0_tea, dim=1)
                    # print('logits_ab0_tea.shape:', logits_ab0_tea.shape)
                    prob_ab0_tea_weak = unsup_augmentations_weak(
                        prob_ab0_tea, loss_mask_ones, params=params_weak_ab0_tea)
                    prob_ab0_tea_weak = prob_ab0_tea_weak[0]
                    print('****** have pai ******!!!!')
                x_b_weak = unsup_augmentations_weak(x_b_tea, loss_mask_ones)
                params_weak_b_tea = unsup_augmentations_weak._params
                img_b_weak = x_b_weak[0]
                loss_mask_b_weak = x_b_weak[1]
                x_b_weak_strong_stu = unsup_augmentations_strongcolor(
                    img_b_weak, loss_mask_ones)
                img_b_weak_strong_stu = x_b_weak_strong_stu[0]
                with torch.no_grad():
                    if self.hyperparameters['mean_teacher']['feature_cutmix']:
                        logits_b_tea_FC, fea_b_tea_FC = self.seg_teacher(
                            img_b_weak,[],  True)
                        logits_b_tea_FC, fea_b_tea_FC0 = logits_b_tea_FC.detach(
                        ), fea_b_tea_FC[0].detach()
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                            fea_b_tea_FC2 = fea_b_tea_FC[2].detach()
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                            fea_b_tea_FC2 = fea_b_tea_FC[1].detach()
                        # print('****** no pai for feature cutmix ******!!!!')
                        # print(fea_b_tea_FC0.shape)
                    if hyperparameters['mean_teacher']['tea_perturb_mean']:
                        x_b_coloraug = test_augmentations_color(
                            x_b_tea, loss_mask_ones)
                        x_b_coloraug_img = x_b_coloraug[0]
                        logits_b_color_tea = self.seg_teacher(
                            x_b_coloraug_img, []).detach()
                        logits_b_tea = self.seg_teacher(
                            x_b_tea, [])
                        logits_b_tea = logits_b_tea.detach()
                        # logits_b0_tea = self.seg_teacher(
                        #     x_b0_tea)
                        # logits_b0_tea = logits_b0_tea.detach()
                        logits_ba0b_tea = self.seg_teacher(
                            x_ba0b_tea, [])
                        logits_ba0b_tea = logits_ba0b_tea.detach()
                        prob_b_tea = F.softmax(logits_b_tea, dim=1)
                        # prob_b0_tea = F.softmax(logits_b0_tea, dim=1)
                        prob_ba0b_tea = F.softmax(logits_ba0b_tea, dim=1)
                        prob_b_color_tea = F.softmax(logits_b_color_tea, dim=1)
                        # mean_prob_b_tea = (
                        #     prob_b_tea + prob_b0_tea + prob_ba0b_tea + prob_b_color_tea) / 4
                        mean_prob_b_tea = (
                            prob_b_tea + prob_b_color_tea + prob_ba0b_tea) / 3
                        # mean_prob_b_tea = (prob_b_tea + prob_ba0b_tea ) / 2
                        prob_b_tea_weak = unsup_augmentations_weak(
                            mean_prob_b_tea, loss_mask_ones, params=params_weak_b_tea)
                        prob_b_tea_weak = prob_b_tea_weak[0]
                        print('have tea_perturb_mean!!*********!!!!')
                    else:
                        # logits_b_tea_weak = self.seg_teacher(
                        #     img_b_weak).detach()
                        # prob_b_tea_weak = F.softmax(logits_b_tea_weak, dim=1)
                        # print('no pai*********!!!!')
                        logits_b_tea = self.seg_teacher(
                            x_b_tea, [])
                        logits_b_tea = logits_b_tea.detach()
                        prob_b_tea = F.softmax(logits_b_tea, dim=1)
                        prob_b_tea_weak = unsup_augmentations_weak(
                            prob_b_tea, loss_mask_ones, params=params_weak_b_tea)
                        prob_b_tea_weak = prob_b_tea_weak[0]
                        print('have pai*********!!!!')
                batch_mix_masks = torch.from_numpy(
                    self.clss_generate_mask_params(x_ab0_tea)).cuda()
                # print('batch_mix_masks.shape', batch_mix_masks.shape)
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    featuresize = (fea_b_tea_FC0).shape[2:]
                    fea_batch_mix_masks = K.geometry.transform.resize(
                        batch_mix_masks, featuresize, interpolation='nearest')
                    print('fea_batch_mix_masks.shape',
                          fea_batch_mix_masks.shape)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        featuresize_de1 = (fea_b_tea_FC2).shape[2:]
                        fea_batch_mix_masks_de1 = K.geometry.transform.resize(
                            batch_mix_masks, featuresize_de1, interpolation='nearest')
                        print('fea_batch_mix_masks_de1.shape',
                              fea_batch_mix_masks_de1.shape)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        featuresize_de1 = (fea_b_tea_FC2).shape[2:]
                        fea_batch_mix_masks_de1 = K.geometry.transform.resize(
                            batch_mix_masks, featuresize_de1, interpolation='nearest')
                        print('fea_batch_mix_masks_en1.shape',
                              fea_batch_mix_masks_de1.shape)
                # Convert mask parameters to masks of shape (N,1,H,W)
                # torch.Size([1, 1, 224, 256]) tensor([0., 1.], device='cuda:0'

                # Mix images with masks, cross mixed
                ab0_b_stu_mixed = img_ab0_weak_strong_stu * \
                    (1 - batch_mix_masks) + \
                    img_b_weak_strong_stu * batch_mix_masks
                ab0_b_stu_mixed = ab0_b_stu_mixed.float().contiguous()

                loss_mask_mixed = loss_mask_ab0_weak * \
                    (1 - batch_mix_masks) + \
                    loss_mask_b_weak * batch_mix_masks
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    fea_loss_mask_mixed = K.geometry.transform.resize(
                        loss_mask_mixed, featuresize, interpolation='nearest')
                    print('fea_loss_mask_mixed.shape',fea_loss_mask_mixed.shape)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        fea_loss_mask_mixed_de1 = K.geometry.transform.resize(
                            loss_mask_mixed, featuresize_de1, interpolation='nearest')
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        fea_loss_mask_mixed_de1 = K.geometry.transform.resize(
                            loss_mask_mixed, featuresize_de1, interpolation='nearest')

                if self.hyperparameters['strong_aug_para']['cutmix_quadtree']:
                    qt_mix = QTree(0.0005, 2, ab0_b_stu_mixed.squeeze(0))
                    qt_mix.subdivide()
                    quad_mask_stu = qt_mix.gen_quadaug_mask(
                        ratio1=hyperparameters['strong_aug_para']['quad_ratio1'], ratio2=hyperparameters['strong_aug_para']['quad_ratio2'])
                    quad_mask_stu1 = quad_mask_stu[0].unsqueeze(0)
                    quad_mask_stu2 = quad_mask_stu[1].unsqueeze(0)
                    ab0_b_stu_mixed1 = quad_mask_stu1 * ab0_b_stu_mixed
                    ab0_b_stu_mixed2 = quad_mask_stu2 * ab0_b_stu_mixed
                    ab0_b_stu_mixed = ab0_b_stu_mixed
                    print('have cutmix quadtree!!!**********')

                else:
                    ab0_b_stu_mixed = ab0_b_stu_mixed

                # Get student prediction for mixed images
                logits_cons_stu, fea_cons_stu = self.seg_student(
                    ab0_b_stu_mixed.float().contiguous(), [], True)
                if hyperparameters['strong_aug_para']['cutmix_quadtree']:
                    logits_cons_stu1, fea_cons_stu1 = self.seg_student(
                        ab0_b_stu_mixed1.float().contiguous(), [], True)
                    logits_cons_stu2, fea_cons_stu2 = self.seg_student(
                        ab0_b_stu_mixed2.float().contiguous(), [], True)
                # print('logits_cons_stu.shape:', logits_cons_stu.shape)
                # logits_cons_stu.shape: torch.Size([1, 2, 224, 224])
                # Mix teacher predictions using same mask
                # It makes no difference whether we do this with logits or probabilities as
                # the mask pixels are either 1 or 0

                # print('logits_cons_tea.shape:', logits_cons_tea.shape)
                # Logits -> probs

                prob_cons_stu = F.softmax(logits_cons_stu, dim=1)
                if hyperparameters['strong_aug_para']['cutmix_quadtree']:
                    prob_cons_stu1 = F.softmax(logits_cons_stu1, dim=1)
                    prob_cons_stu2 = F.softmax(logits_cons_stu2, dim=1)
                prob_cons_tea = prob_ab0_tea_weak * \
                    (1 - batch_mix_masks) + \
                    prob_b_tea_weak * batch_mix_masks
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    fea_cons_tea = fea_ab0_tea_FC0 * \
                        (1 - fea_batch_mix_masks) + \
                        fea_b_tea_FC0 * fea_batch_mix_masks
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        fea_cons_tea_de1 = fea_ab0_tea_FC2 * \
                            (1 - fea_batch_mix_masks_de1) + \
                            fea_b_tea_FC2 * fea_batch_mix_masks_de1
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        fea_cons_tea_de1 = fea_ab0_tea_FC2 * \
                            (1 - fea_batch_mix_masks_de1) + \
                            fea_b_tea_FC2 * fea_batch_mix_masks_de1
                        print('fea_cons_tea_de1.shape:',
                              fea_cons_tea_de1.shape)
                # prob_cons_tea.shape: torch.Size([1, 6, 224, 224])

            if self.hyperparameters['mean_teacher']['self_cutmix']:
                # encode
                c_b1, s_b1_prime = self.gen_b.encode(x_b1)
                c_b2, s_b2_prime = self.gen_b.encode(x_b2)

                b1_b_tea = x_b1.clone()
                b1_b_tea = self.un_normalize(b1_b_tea)
                b1_b0_tea = self.gen_b.decode(c_b1, s_b3)
                b1_b0_tea = self.un_normalize(b1_b0_tea)
                b1_ba0_2b = self.gen_a.decode(c_b1, s_a1)
                c_ba0b_rec, _ = self.gen_a.encode(b1_ba0_2b)
                b1_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b4)
                b1_ba0b_tea = self.un_normalize(b1_ba0b_tea)

                b2_b_tea = x_b2.clone()
                b2_b_tea = self.un_normalize(b2_b_tea)
                b2_b0_tea = self.gen_b.decode(c_b2, s_b3)
                b2_b0_tea = self.un_normalize(b2_b0_tea)
                b2_ba0_2b = self.gen_a.decode(c_b2, s_a1)
                c_ba0b_rec, _ = self.gen_a.encode(b2_ba0_2b)
                b2_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b4)
                b2_ba0b_tea = self.un_normalize(b2_ba0b_tea)

                x_b1_weak = unsup_augmentations_weak(
                    b1_b_tea, loss_mask_ones)
                params_weak_b1_tea = unsup_augmentations_weak._params
                img_b1_weak = x_b1_weak[0]
                loss_mask_b1_weak = x_b1_weak[1]
                x_b1_weak_strong_stu = unsup_augmentations_strongcolor(
                    img_b1_weak, loss_mask_ones)
                img_b1_weak_strong_stu = x_b1_weak_strong_stu[0]
                with torch.no_grad():
                    if self.hyperparameters['mean_teacher']['feature_cutmix']:
                        logits_b1_tea_FC, fea_b1_tea_FC = self.seg_teacher(
                            img_b1_weak, [], True)
                        logits_b1_tea_FC, fea_b1_tea_FC0 = logits_b1_tea_FC.detach(
                        ), fea_b1_tea_FC[0].detach()
                        print('****** no pai for feature cutmix ******!!!!')
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                            fea_b1_tea_FC2 = fea_b1_tea_FC[2].detach()
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                            fea_b1_tea_FC2 = fea_b1_tea_FC[1].detach()
                    if hyperparameters['mean_teacher']['tea_perturb_mean']:
                        x_b1_coloraug = test_augmentations_color(
                            b1_b_tea, loss_mask_ones)
                        x_b1_coloraug_img = x_b1_coloraug[0]
                        logitsb1_color_tea = self.seg_teacher(
                            x_b1_coloraug_img, []).detach()
                        logitsb1_b_tea = self.seg_teacher(
                            b1_b_tea, [])
                        logitsb1_b_tea = logitsb1_b_tea.detach()
                        # logitsb1_b0_tea = self.seg_teacher(
                        #     b1_b0_tea)
                        # logitsb1_b0_tea = logitsb1_b0_tea.detach()
                        logitsb1_ba0b_tea = self.seg_teacher(
                            b1_ba0b_tea, [])
                        logitsb1_ba0b_tea = logitsb1_ba0b_tea.detach()
                        probb1_b_tea = F.softmax(logitsb1_b_tea, dim=1)
                        # probb1_b0_tea = F.softmax(logitsb1_b0_tea, dim=1)
                        probb1_ba0b_tea = F.softmax(logitsb1_ba0b_tea, dim=1)
                        probb1_b_color_tea = F.softmax(
                            logitsb1_color_tea, dim=1)
                        # mean_prob_b1_tea = (
                        #     probb1_b_tea + probb1_b0_tea + probb1_ba0b_tea + probb1_b_color_tea) / 4
                        mean_prob_b1_tea = (
                            probb1_b_tea + probb1_b_color_tea + probb1_ba0b_tea) / 3
                        # mean_prob_b1_tea = (probb1_b_tea + probb1_ba0b_tea) / 2
                        prob_b1_tea_weak = unsup_augmentations_weak(
                            mean_prob_b1_tea, loss_mask_ones, params=params_weak_b1_tea)
                        prob_b1_tea_weak = prob_b1_tea_weak[0]
                        print('have tea_perturb_mean!!*********!!!!')

                    else:
                        # logitsb1_b_tea = self.seg_teacher(
                        #     img_b1_weak).detach()
                        # prob_b1_tea_weak = F.softmax(logitsb1_b_tea, dim=1)
                        # print('no pai*********!!!!')
                        logitsb1_b_tea = self.seg_teacher(
                            b1_b_tea, [])
                        logitsb1_b_tea = logitsb1_b_tea.detach()
                        prob_b1_tea = F.softmax(logitsb1_b_tea, dim=1)
                        prob_b1_tea_weak = unsup_augmentations_weak(
                            prob_b1_tea, loss_mask_ones, params=params_weak_b1_tea)
                        prob_b1_tea_weak = prob_b1_tea_weak[0]
                        print('have pai*********!!!!')
                x_b2_weak = unsup_augmentations_weak(
                    b2_b_tea, loss_mask_ones)
                params_weak_b2_tea = unsup_augmentations_weak._params
                img_b2_weak = x_b2_weak[0]
                loss_mask_b2_weak = x_b2_weak[1]
                x_b2_weak_strong_stu = unsup_augmentations_strongcolor(
                    img_b2_weak, loss_mask_ones)
                img_b2_weak_strong_stu = x_b2_weak_strong_stu[0]
                with torch.no_grad():
                    if self.hyperparameters['mean_teacher']['feature_cutmix']:
                        logits_b2_tea_FC, fea_b2_tea_FC = self.seg_teacher(
                            img_b2_weak,[], True)
                        logits_b2_tea_FC, fea_b2_tea_FC0 = logits_b2_tea_FC.detach(
                        ), fea_b2_tea_FC[0].detach()
                        print('****** no pai for feature cutmix ******!!!!')
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                            fea_b2_tea_FC2 = fea_b2_tea_FC[2].detach()
                        if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                            fea_b2_tea_FC2 = fea_b2_tea_FC[1].detach()
                    if hyperparameters['mean_teacher']['tea_perturb_mean']:
                        x_b2_coloraug = test_augmentations_color(
                            b2_b_tea, loss_mask_ones)
                        x_b2_coloraug_img = x_b2_coloraug[0]
                        logitsb2_color_tea = self.seg_teacher(
                            x_b2_coloraug_img, []).detach()
                        logitsb2_b_tea = self.seg_teacher(b2_b_tea, [])
                        logitsb2_b_tea = logitsb2_b_tea.detach()
                        # logitsb2_b0_tea = self.seg_teacher(b2_b0_tea)
                        # logitsb2_b0_tea = logitsb2_b0_tea.detach()
                        logitsb2_ba0b_tea = self.seg_teacher(b2_ba0b_tea, [])
                        logitsb2_ba0b_tea = logitsb2_ba0b_tea.detach()
                        probb2_b_tea = F.softmax(logitsb2_b_tea, dim=1)
                        # probb2_b0_tea = F.softmax(logitsb2_b0_tea, dim=1)
                        probb2_ba0b_tea = F.softmax(logitsb2_ba0b_tea, dim=1)
                        probb2_b_color_tea = F.softmax(
                            logitsb2_color_tea, dim=1)
                        # mean_prob_b2_tea = (
                        #     probb2_b_tea + probb2_b0_tea + probb2_ba0b_tea + probb2_b_color_tea) / 4
                        mean_prob_b2_tea = (
                            probb2_b_tea + probb2_b_color_tea + probb2_ba0b_tea) / 3
                        # mean_prob_b2_tea = (probb2_b_tea + probb2_ba0b_tea) / 2
                        prob_b2_tea_weak = unsup_augmentations_weak(
                            mean_prob_b2_tea, loss_mask_ones, params=params_weak_b2_tea)
                        prob_b2_tea_weak = prob_b2_tea_weak[0]
                        print('have tea_perturb_mean!!*********!!!!')

                    else:
                        # logitsb2_b_tea = self.seg_teacher(
                        #     img_b2_weak).detach()
                        # prob_b2_tea_weak = F.softmax(logitsb2_b_tea, dim=1)
                        # print('no pai*********!!!!')
                        logitsb2_b_tea = self.seg_teacher(b2_b_tea, [])
                        logitsb2_b_tea = logitsb2_b_tea.detach()

                        prob_b2_tea = F.softmax(logitsb2_b_tea, dim=1)
                        prob_b2_tea_weak = unsup_augmentations_weak(
                            prob_b2_tea, loss_mask_ones, params=params_weak_b2_tea)
                        prob_b2_tea_weak = prob_b2_tea_weak[0]
                        print('have pai*********!!!!')

                batch_mix_masks = torch.from_numpy(
                    self.clss_generate_mask_params(b1_b_tea)).cuda()
                # print('batch_mix_masks.shape', batch_mix_masks.shape)

                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    featuresize = (fea_b2_tea_FC0).shape[2:]
                    fea_batch_mix_masks = K.geometry.transform.resize(
                        batch_mix_masks, featuresize, interpolation='nearest')
                    print('fea_batch_mix_masks.shape',
                          fea_batch_mix_masks.shape)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        featuresize_de1 = (fea_b2_tea_FC2).shape[2:]
                        fea_batch_mix_masks_de1 = K.geometry.transform.resize(
                            batch_mix_masks, featuresize_de1, interpolation='nearest')
                        print('fea_batch_mix_masks_de1.shape',
                              fea_batch_mix_masks_de1.shape)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        featuresize_de1 = (fea_b2_tea_FC2).shape[2:]
                        fea_batch_mix_masks_de1 = K.geometry.transform.resize(
                            batch_mix_masks, featuresize_de1, interpolation='nearest')
                        print('fea_batch_mix_masks_en1.shape',
                              fea_batch_mix_masks_de1.shape)
                # Mix images with masks, cross mixed
                b1_b2_stu_mixed = img_b1_weak_strong_stu * \
                    (1 - batch_mix_masks) + \
                    img_b2_weak_strong_stu * batch_mix_masks
                b1_b2_stu_mixed = b1_b2_stu_mixed.float().contiguous()

                loss_mask_mixed_b1b2 = loss_mask_b1_weak * \
                    (1 - batch_mix_masks) + \
                    loss_mask_b2_weak * batch_mix_masks
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    fea_loss_mask_mixed_b1b2 = K.geometry.transform.resize(
                        loss_mask_mixed_b1b2, featuresize, interpolation='nearest')
                    print('fea_loss_mask_mixed_b1b2.shape',
                          fea_loss_mask_mixed_b1b2.shape)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        fea_loss_mask_mixed_b1b2_de1 = K.geometry.transform.resize(
                            loss_mask_mixed_b1b2, featuresize_de1, interpolation='nearest')
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        fea_loss_mask_mixed_b1b2_de1 = K.geometry.transform.resize(
                            loss_mask_mixed_b1b2, featuresize_de1, interpolation='nearest')
                if self.hyperparameters['strong_aug_para']['cutmix_quadtree']:
                    qt_mix_b1b2 = QTree(
                        0.0005, 2, b1_b2_stu_mixed.squeeze(0))
                    qt_mix_b1b2.subdivide()
                    quad_mask_stu = qt_mix_b1b2.gen_quadaug_mask(
                        ratio1=hyperparameters['strong_aug_para']['quad_ratio1'], ratio2=hyperparameters['strong_aug_para']['quad_ratio2'])
                    quad_mask_stu1 = quad_mask_stu[0].unsqueeze(0)
                    quad_mask_stu2 = quad_mask_stu[1].unsqueeze(0)
                    # print(quad_mask_stu.shape)

                    b1_b2_stu_mixed1 = quad_mask_stu1 * b1_b2_stu_mixed
                    b1_b2_stu_mixed2 = quad_mask_stu2 * b1_b2_stu_mixed
                    b1_b2_stu_mixed = b1_b2_stu_mixed
                    print('have cutmix quadtree!!!**********')

                else:
                    b1_b2_stu_mixed = b1_b2_stu_mixed

                # Get student prediction for mixed image
                logits_cons_stu_b1b2, fea_cons_stu_b1b2 = self.seg_student(
                    b1_b2_stu_mixed.float().contiguous(), [], True)
                if hyperparameters['strong_aug_para']['cutmix_quadtree']:
                    logits_cons_stu_b1b2_1, fea_cons_stu_b1b2_1 = self.seg_student(
                        b1_b2_stu_mixed1.float().contiguous(), [], True)
                    logits_cons_stu_b1b2_2, fea_cons_stu_b1b2_2 = self.seg_student(
                        b1_b2_stu_mixed2.float().contiguous(), [], True)
                # print('logits_cons_stu.shape:', logits_cons_stu.shape)
                # logits_cons_stu.shape: torch.Size([1, 2, 224, 224])
                # Mix teacher predictions using same mask
                # It makes no difference whether we do this with logits or probabilities as
                # the mask pixels are either 1 or 0

                # Logits -> probs
                prob_cons_stu_b1b2 = F.softmax(logits_cons_stu_b1b2, dim=1)
                if hyperparameters['strong_aug_para']['cutmix_quadtree']:
                    prob_cons_stu_b1b2_1 = F.softmax(
                        logits_cons_stu_b1b2_1, dim=1)
                    prob_cons_stu_b1b2_2 = F.softmax(
                        logits_cons_stu_b1b2_2, dim=1)
                prob_cons_tea_b1b2 = prob_b1_tea_weak * \
                    (1 - batch_mix_masks) + \
                    prob_b2_tea_weak * batch_mix_masks
                # print('prob_cons_tea_b1b2.shape:',prob_cons_tea_b1b2.shape)
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    fea_cons_tea_b1b2 = fea_b1_tea_FC0 * \
                        (1 - fea_batch_mix_masks) + \
                        fea_b2_tea_FC0 * fea_batch_mix_masks
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        fea_cons_tea_b1b2_de1 = fea_b1_tea_FC2 * \
                            (1 - fea_batch_mix_masks_de1) + \
                            fea_b2_tea_FC2 * fea_batch_mix_masks_de1
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        fea_cons_tea_b1b2_de1 = fea_b1_tea_FC2 * \
                            (1 - fea_batch_mix_masks_de1) + \
                            fea_b2_tea_FC2 * fea_batch_mix_masks_de1
            # Confidence thresholding
            if self.hyperparameters['mean_teacher']['conf_thresh'] > 0.0:
                # 0.97
                # Compute confidence of teacher predictions
                if self.hyperparameters['mean_teacher']['cross_cutmix']:
                    conf_tea = prob_cons_tea.max(dim=1)[0]
                    # print('conf_tea.shape:', conf_tea.shape)
                    # conf_tea.shape: torch.Size([1, 224, 224])
                    # Compute confidence mask
                    conf_mask = (conf_tea >= self.hyperparameters['mean_teacher']['conf_thresh']).float()[
                        :, None, :, :]
                    # print('conf_mask.shape:', conf_mask.shape)
                    # conf_mask.shape: torch.Size([1, 1, 224, 224])
                    # Average confidence mask if requested
                    if not self.hyperparameters['mean_teacher']['conf_per_pixel']:
                        conf_mask = conf_mask.mean()
                        print('conf_mask:', conf_mask)
                    loss_mask_mixed = loss_mask_mixed * conf_mask
                if self.hyperparameters['mean_teacher']['self_cutmix']:
                    conf_tea_b1b2 = prob_cons_tea_b1b2.max(dim=1)[0]
                    # print('conf_tea_b1b2.shape:', conf_tea_b1b2.shape)
                    # conf_tea.shape: torch.Size([1, 224, 224])
                    # Compute confidence mask
                    conf_mask_b1b2 = (conf_tea_b1b2 >= self.hyperparameters['mean_teacher']['conf_thresh']).float()[
                        :, None, :, :]
                    # print('conf_mask.shape:', conf_mask.shape)
                    # conf_mask.shape: torch.Size([1, 1, 224, 224])

                    # Average confidence mask if requested
                    if not self.hyperparameters['mean_teacher']['conf_per_pixel']:
                        conf_mask_b1b2 = conf_mask_b1b2.mean()
                        print('conf_mask_b1b2:', conf_mask_b1b2)
                    loss_mask_mixed_b1b2 = loss_mask_mixed_b1b2 * conf_mask_b1b2

            # Compute per-pixel consistency loss
            # Note that the way we aggregate the loss across the class/channel dimension (1)
            # depends on the loss function used. Generally, summing over the class dimension
            # keeps the magnitude of the gradient of the loss w.r.t. the logits
            # nearly constant w.r.t. the number of classes. When using logit-variance,
            # dividing by `sqrt(num_classes)` helps.
            if self.hyperparameters['mean_teacher']['cons_loss_fn'] == 'var':
                if self.hyperparameters['mean_teacher']['cross_cutmix']:
                    delta_prob = prob_cons_stu - prob_cons_tea
                    img_consistency_loss = delta_prob * delta_prob
                    img_consistency_loss = img_consistency_loss.sum(
                        dim=1, keepdim=True)
                    if self.hyperparameters['strong_aug_para']['cutmix_quadtree']:
                        delta_prob1 = prob_cons_stu1 - prob_cons_tea
                        img_consistency_loss1 = delta_prob1 * delta_prob1
                        img_consistency_loss1 = img_consistency_loss1.sum(
                            dim=1, keepdim=True)
                        delta_prob2 = prob_cons_stu2 - prob_cons_tea
                        img_consistency_loss2 = delta_prob2 * delta_prob2
                        img_consistency_loss2 = img_consistency_loss2.sum(
                            dim=1, keepdim=True)
                        # if self.hyperparameters['strong_aug_para']['quad_feature_align']:
                        #     if self.hyperparameters['mean_teacher']['fea_cons_loss_fn'] == 'var':
                        #         fea_delta_prob_quad = fea_cons_stu1[0] - fea_cons_stu2[0]
                        #         fea_consistency_loss_quad = fea_delta_prob_quad * fea_delta_prob_quad
                        #         fea_consistency_loss_quad = fea_consistency_loss_quad.sum(
                        #             dim=1, keepdim=True)
                        #         fea_delta_prob_quad1 = fea_cons_stu[0] - fea_cons_stu1[0]
                        #         fea_consistency_loss_quad1 = fea_delta_prob_quad1 * fea_delta_prob_quad1
                        #         fea_consistency_loss_quad1 = fea_consistency_loss_quad1.sum(
                        #             dim=1, keepdim=True)
                        #     else:
                        #         fea_consistency_loss_quad = self.feacut_loss_criterion(fea_cons_stu1[0], fea_cons_stu2[0])
                        #         fea_consistency_loss_quad1 = self.feacut_loss_criterion(fea_cons_stu[0], fea_cons_stu1[0])
                    if self.hyperparameters['mean_teacher']['feature_cutmix']:
                        if self.hyperparameters['mean_teacher']['tea_fea_norm']:
                            target_mean = fea_cons_tea.mean(
                                dim=[2, 3], keepdim=False)
                            target_std = (fea_cons_tea.var(
                                dim=[2, 3], keepdim=False) + 1e-6).sqrt()
                            fea_cons_tea = (fea_cons_tea - target_mean.reshape(fea_cons_tea.shape[0], fea_cons_tea.shape[1], 1, 1)) / target_std.reshape(
                                fea_cons_tea.shape[0], fea_cons_tea.shape[1], 1, 1)
                        if self.hyperparameters['mean_teacher']['fea_cons_loss_fn'] == 'var':
                            fea_delta_prob = fea_cons_stu[0] - fea_cons_tea
                            fea_consistency_loss = fea_delta_prob * fea_delta_prob
                            fea_consistency_loss = fea_consistency_loss.sum(
                                dim=1, keepdim=True)
                        else:
                            fea_consistency_loss = self.feacut_loss_criterion(
                                fea_cons_stu[0], fea_cons_tea)
                        # all feature consistencys' target is fea_cons_tea
                        if self.hyperparameters['strong_aug_para']['quad_feature_align']:
                            if self.hyperparameters['mean_teacher']['fea_cons_loss_fn'] == 'var':
                                fea_delta_prob_quad = fea_cons_stu1[0] - \
                                    fea_cons_tea
                                fea_consistency_loss_quad = fea_delta_prob_quad * fea_delta_prob_quad
                                fea_consistency_loss_quad = fea_consistency_loss_quad.sum(
                                    dim=1, keepdim=True)
                                fea_delta_prob_quad1 = fea_cons_stu2[0] - \
                                    fea_cons_tea
                                fea_consistency_loss_quad1 = fea_delta_prob_quad1 * fea_delta_prob_quad1
                                fea_consistency_loss_quad1 = fea_consistency_loss_quad1.sum(
                                    dim=1, keepdim=True)
                            else:
                                fea_consistency_loss_quad = self.feacut_loss_criterion(
                                    fea_cons_stu1[0], fea_cons_tea)
                                fea_consistency_loss_quad1 = self.feacut_loss_criterion(
                                    fea_cons_stu2[0], fea_cons_tea)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        fea_delta_prob_de1 = fea_cons_stu[2] - fea_cons_tea_de1
                        fea_consistency_loss_de1 = fea_delta_prob_de1 * fea_delta_prob_de1
                        fea_consistency_loss_de1 = fea_consistency_loss_de1.sum(
                            dim=1, keepdim=True)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        fea_delta_prob_de1 = fea_cons_stu[1] - fea_cons_tea_de1
                        fea_consistency_loss_de1 = fea_delta_prob_de1 * fea_delta_prob_de1
                        fea_consistency_loss_de1 = fea_consistency_loss_de1.sum(
                            dim=1, keepdim=True)
                if self.hyperparameters['mean_teacher']['self_cutmix']:
                    delta_prob_b1b2 = prob_cons_stu_b1b2 - prob_cons_tea_b1b2
                    img_consistency_loss_b1b2 = delta_prob_b1b2 * delta_prob_b1b2
                    img_consistency_loss_b1b2 = img_consistency_loss_b1b2.sum(
                        dim=1, keepdim=True)
                    if self.hyperparameters['strong_aug_para']['cutmix_quadtree']:
                        delta_prob_b1b2_1 = prob_cons_stu_b1b2_1 - prob_cons_tea_b1b2
                        img_consistency_loss_b1b2_1 = delta_prob_b1b2_1 * delta_prob_b1b2_1
                        img_consistency_loss_b1b2_1 = img_consistency_loss_b1b2_1.sum(
                            dim=1, keepdim=True)
                        delta_prob_b1b2_2 = prob_cons_stu_b1b2_2 - prob_cons_tea_b1b2
                        img_consistency_loss_b1b2_2 = delta_prob_b1b2_2 * delta_prob_b1b2_2
                        img_consistency_loss_b1b2_2 = img_consistency_loss_b1b2_2.sum(
                            dim=1, keepdim=True)
                        # if self.hyperparameters['strong_aug_para']['quad_feature_align']:
                        #     if self.hyperparameters['mean_teacher']['fea_cons_loss_fn'] == 'var':
                        #         fea_delta_prob_b1b2_quad = fea_cons_stu_b1b2_1[0] - \
                        #             fea_cons_stu_b1b2_2[0]
                        #         fea_consistency_loss_b1b2_quad = fea_delta_prob_b1b2_quad * fea_delta_prob_b1b2_quad
                        #         fea_consistency_loss_b1b2_quad = fea_consistency_loss_b1b2_quad.sum(
                        #             dim=1, keepdim=True)
                        #         fea_delta_prob_b1b2_quad1 = fea_cons_stu_b1b2[0] - \
                        #             fea_cons_stu_b1b2_1[0]
                        #         fea_consistency_loss_b1b2_quad1 = fea_delta_prob_b1b2_quad1 * fea_delta_prob_b1b2_quad1
                        #         fea_consistency_loss_b1b2_quad1 = fea_consistency_loss_b1b2_quad1.sum(
                        #             dim=1, keepdim=True)
                        #     else:
                        #         fea_consistency_loss_b1b2_quad = self.feacut_loss_criterion(
                        #             fea_cons_stu_b1b2_1[0], fea_cons_stu_b1b2_2[0])
                        #         fea_consistency_loss_b1b2_quad1 = self.feacut_loss_criterion(
                        #             fea_cons_stu_b1b2[0], fea_cons_stu_b1b2_1[0])

                    if self.hyperparameters['mean_teacher']['feature_cutmix']:
                        if self.hyperparameters['mean_teacher']['tea_fea_norm']:
                            target_mean = fea_cons_tea_b1b2.mean(
                                dim=[2, 3], keepdim=False)
                            target_std = (fea_cons_tea_b1b2.var(
                                dim=[2, 3], keepdim=False) + 1e-6).sqrt()
                            fea_cons_tea_b1b2 = (fea_cons_tea_b1b2 - target_mean.reshape(fea_cons_tea_b1b2.shape[0], fea_cons_tea_b1b2.shape[1], 1, 1)) / target_std.reshape(
                                fea_cons_tea_b1b2.shape[0], fea_cons_tea_b1b2.shape[1], 1, 1)
                        if self.hyperparameters['mean_teacher']['fea_cons_loss_fn'] == 'var':
                            fea_delta_prob_b1b2 = fea_cons_stu_b1b2[0] - \
                                fea_cons_tea_b1b2
                            fea_consistency_loss_b1b2 = fea_delta_prob_b1b2 * fea_delta_prob_b1b2
                            fea_consistency_loss_b1b2 = fea_consistency_loss_b1b2.sum(
                                dim=1, keepdim=True)
                        else:
                            print('*************feacosine')
                            fea_consistency_loss_b1b2 = self.feacut_loss_criterion(
                                fea_cons_stu_b1b2[0], fea_cons_tea_b1b2)
                        if self.hyperparameters['strong_aug_para']['quad_feature_align']:
                            if self.hyperparameters['mean_teacher']['fea_cons_loss_fn'] == 'var':
                                fea_delta_prob_b1b2_quad = fea_cons_stu_b1b2_1[0] - \
                                    fea_cons_tea_b1b2
                                fea_consistency_loss_b1b2_quad = fea_delta_prob_b1b2_quad * fea_delta_prob_b1b2_quad
                                fea_consistency_loss_b1b2_quad = fea_consistency_loss_b1b2_quad.sum(
                                    dim=1, keepdim=True)
                                fea_delta_prob_b1b2_quad1 = fea_cons_stu_b1b2_2[0] - \
                                    fea_cons_tea_b1b2
                                fea_consistency_loss_b1b2_quad1 = fea_delta_prob_b1b2_quad1 * fea_delta_prob_b1b2_quad1
                                fea_consistency_loss_b1b2_quad1 = fea_consistency_loss_b1b2_quad1.sum(
                                    dim=1, keepdim=True)
                            else:
                                fea_consistency_loss_b1b2_quad = self.feacut_loss_criterion(
                                    fea_cons_stu_b1b2_1[0], fea_cons_tea_b1b2)
                                fea_consistency_loss_b1b2_quad1 = self.feacut_loss_criterion(
                                    fea_cons_stu_b1b2_2[0], fea_cons_tea_b1b2)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        fea_delta_prob_b1b2_de1 = fea_cons_stu_b1b2[2] - \
                            fea_cons_tea_b1b2_de1
                        fea_consistency_loss_b1b2_de1 = fea_delta_prob_b1b2_de1 * fea_delta_prob_b1b2_de1
                        fea_consistency_loss_b1b2_de1 = fea_consistency_loss_b1b2_de1.sum(
                            dim=1, keepdim=True)
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        fea_delta_prob_b1b2_de1 = fea_cons_stu_b1b2[1] - \
                            fea_cons_tea_b1b2_de1
                        fea_consistency_loss_b1b2_de1 = fea_delta_prob_b1b2_de1 * fea_delta_prob_b1b2_de1
                        fea_consistency_loss_b1b2_de1 = fea_consistency_loss_b1b2_de1.sum(
                            dim=1, keepdim=True)
            else:
                raise ValueError(
                    'Unknown consistency loss function {}'.format(self.hyperparameters['mean_teacher']['cons_loss_fn']))

            # Apply consistency loss mask and take the mean over pixels and images
            if self.hyperparameters['mean_teacher']['cross_cutmix']:
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        consistency_loss = (img_consistency_loss * loss_mask_mixed).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss * fea_loss_mask_mixed).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut']
                               * fea_consistency_loss_de1 * fea_loss_mask_mixed_de1).mean()
                    else:
                        self.inter_feacut_loss = (fea_consistency_loss * fea_loss_mask_mixed).mean()
                        self.inter_transcut_loss = (img_consistency_loss * loss_mask_mixed).mean()
                        # self.consistency_loss = (img_consistency_loss * loss_mask_mixed).mean() \
                        # + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss * fea_loss_mask_mixed).mean()

                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        consistency_loss = (img_consistency_loss * loss_mask_mixed).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss * fea_loss_mask_mixed).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut']
                               * fea_consistency_loss_de1 * fea_loss_mask_mixed_de1).mean()
                    else:
                        self.inter_feacut_loss = (fea_consistency_loss * fea_loss_mask_mixed).mean()
                        self.inter_transcut_loss = (img_consistency_loss * loss_mask_mixed).mean()
                        # self.consistency_loss = (img_consistency_loss * loss_mask_mixed).mean() \
                        # + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss * fea_loss_mask_mixed).mean()

                        # consistency_loss = (img_consistency_loss * loss_mask_mixed).mean() \
                        #     + (self.hyperparameters['mean_teacher']['cons_weight_feacut']
                        #        * fea_consistency_loss * fea_loss_mask_mixed).mean()
                    if self.hyperparameters['strong_aug_para']['cutmix_quadtree']:
                        consistency_loss = consistency_loss + (img_consistency_loss1 * loss_mask_mixed).mean() \
                            + (img_consistency_loss2 * loss_mask_mixed).mean()
                        if self.hyperparameters['strong_aug_para']['quad_feature_align']:
                            consistency_loss = consistency_loss + (self.hyperparameters['mean_teacher']['cons_weight_feacut_quad'] * fea_consistency_loss_quad * fea_loss_mask_mixed).mean() \
                                + (self.hyperparameters['mean_teacher']['cons_weight_feacut_quad']
                                   * fea_consistency_loss_quad1 * fea_loss_mask_mixed).mean()
                else:
                    consistency_loss = (
                        img_consistency_loss * loss_mask_mixed).mean()
            else:
                consistency_loss = 0
            if self.hyperparameters['mean_teacher']['self_cutmix']:
                if self.hyperparameters['mean_teacher']['feature_cutmix']:
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_de1':
                        consistency_loss_b1b2 = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] *
                               fea_consistency_loss_b1b2_de1 * fea_loss_mask_mixed_b1b2_de1).mean()
                        print('cut two feature layers')
                    else:
                        self.intra_transcut_loss = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() 
                        self.intra_feacut_loss = (fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean()
                        # self.consistency_loss_b1b2 = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() \
                        #                    + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean()

                        # consistency_loss_b1b2 = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() \
                        #     + (self.hyperparameters['mean_teacher']['cons_weight_feacut']
                        #        * fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean()
                    if self.hyperparameters['mean_teacher']['feature_cutmix_layers'] == 'bottle_en1':
                        consistency_loss_b1b2 = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean() \
                            + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] *
                               fea_consistency_loss_b1b2_de1 * fea_loss_mask_mixed_b1b2_de1).mean()
                        print('cut encoder and bottle feature layers')
                    else:
                        self.intra_transcut_loss = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() 
                        self.intra_feacut_loss = (fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean()
                        # self.consistency_loss_b1b2 = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() \
                        #                    + (self.hyperparameters['mean_teacher']['cons_weight_feacut'] * fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean()

                        # consistency_loss_b1b2 = (img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean() \
                        #     + (self.hyperparameters['mean_teacher']['cons_weight_feacut']
                        #        * fea_consistency_loss_b1b2 * fea_loss_mask_mixed_b1b2).mean()
                        print('only cut bottle feature layers')
                    if self.hyperparameters['strong_aug_para']['cutmix_quadtree']:
                        consistency_loss_b1b2 = consistency_loss_b1b2 + (img_consistency_loss_b1b2_1 * loss_mask_mixed_b1b2).mean() \
                            + (img_consistency_loss_b1b2_2 *
                               loss_mask_mixed_b1b2).mean()
                        if self.hyperparameters['strong_aug_para']['quad_feature_align']:
                            consistency_loss_b1b2 = consistency_loss_b1b2 + (self.hyperparameters['mean_teacher']['cons_weight_feacut_quad'] * fea_consistency_loss_b1b2_quad * fea_loss_mask_mixed_b1b2).mean() \
                                + (self.hyperparameters['mean_teacher']['cons_weight_feacut_quad']
                                   * fea_consistency_loss_b1b2_quad1 * fea_loss_mask_mixed_b1b2).mean()

                else:
                    consistency_loss_b1b2 = (
                        img_consistency_loss_b1b2 * loss_mask_mixed_b1b2).mean()
            else:
                consistency_loss_b1b2 = 0

            # Modulate with rampup if desired
            if self.hyperparameters['mean_teacher']['rampup'] > 0:
                if self.hyperparameters['mean_teacher']['cross_cutmix']:
                    consistency_loss = consistency_loss * \
                        self.hyperparameters['mean_teacher']['rampup']
                if self.hyperparameters['mean_teacher']['self_cutmix']:
                    consistency_loss_b1b2 = consistency_loss_b1b2 * \
                        self.hyperparameters['mean_teacher']['rampup']
                if self.hyperparameters['mean_teacher']['self_cutout']:
                    consistency_loss_cutout0 = consistency_loss_cutout0 * \
                        self.hyperparameters['mean_teacher']['rampup']
                if self.hyperparameters['mean_teacher']['self_quadtree']:
                    consistency_loss_quad = consistency_loss_quad * \
                        self.hyperparameters['mean_teacher']['rampup']
            # # Weight the consistency loss and back-prop
            # self.loss_unsup_consis = self.consistency_loss * self.hyperparameters['mean_teacher']['cons_weight_ab'] + \
            #     self.consistency_loss_b1b2 * \
            #     self.hyperparameters['mean_teacher']['cons_weight_b1b2']
            if self.hyperparameters['mean_teacher']['cross_cutmix']:
                if self.hyperparameters['mean_teacher']['self_cutmix']:
                    self.multi_transcut = self.inter_transcut_loss + 1 * self.intra_transcut_loss
                    self.multi_feacut = self.inter_feacut_loss + 1 * self.intra_feacut_loss
                    print('*** cutmix cross and self')
                else:
                    self.intra_transcut_loss = 0
                    self.intra_feacut_loss = 0
                    self.multi_transcut = self.inter_transcut_loss
                    self.multi_feacut = self.inter_feacut_loss
                    print('*** only cutmix cross')
            else:
                self.inter_transcut_loss = 0
                self.inter_feacut_loss = 0
                self.multi_transcut = self.intra_transcut_loss
                self.multi_feacut = self.intra_feacut_loss
                print('*** only cutmix self')

            self.loss_unsup_consis = self.multi_transcut * self.hyperparameters['mean_teacher']['cons_weight_multi_transcut'] + \
                self.multi_feacut * self.hyperparameters['mean_teacher']['cons_weight_multi_feacut']
            self.loss_unsup_consis.backward()

        self.student_opt.step()
        if self.teacher_opt is not None:
            self.teacher_opt.step(self.seg_teacher, self.seg_student)

        if hyperparameters['update_withgen']:
            self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def phase_content(self, trg_img, hyperparameters):
        # the phase content reconstruct
        # get fft
        fft_trg_img = torch.fft.fft2(trg_img, dim=(-2, -1))
        fft_img = torch.stack((fft_trg_img.real, fft_trg_img.imag), -1)

        amp_trg, pha_trg = self.extract_ampl_phase(fft_img)

        # reconstruct phase
        fft_trg_p = torch.zeros(fft_img.size(), dtype=torch.float).cuda()
        fft_trg_p[:, :, :, :, 0] = torch.cos(
            pha_trg) * hyperparameters['recon_phase_ampw']
        fft_trg_p[:, :, :, :, 1] = torch.sin(
            pha_trg) * hyperparameters['recon_phase_ampw']
        # get the recomposed image
        _, _, imgH, imgW = trg_img.size()
        # trg_p = torch.fft.irfft(fft_trg_p, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])

        trg_p = torch.fft.ifft2(torch.complex(fft_trg_p[:, :, :, :, 0], fft_trg_p[:, :, :, :, 1]), dim=(-2, -1))    # 如果运行了torch.stack()
        # trg_p = trg_p.real  * 100
        # trg_p = trg_p.detach()
        # trg_p = torch.clamp(trg_p,-1,1)
        # trg_p = (trg_p - trg_p.min())/(trg_p.max() - trg_p.min())
        # trg_p = (trg_p - trg_p.mean())/(trg_p.std() + 1e-8)
        trg_p = trg_p.real
        trg_p = trg_p.detach()
        trg_p = torch.where(torch.isnan(
            trg_p), torch.full_like(trg_p, 0), trg_p)
        trg_p = torch.where(torch.isinf(
            trg_p), torch.full_like(trg_p, 0), trg_p)
        # print('trg_p:', trg_p.dtype)
        return trg_p

    def un_normalize(self, input_img):
        mean = self.hyperparameters['mean']
        std = self.hyperparameters['std']
        output_img = input_img * std + mean
        return output_img

    def normalize(self, input_img):
        pass
        return input_img

    def sample_seg(self, x_a, x_b, hyperparameters):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_a_cycle, x_b_cycle, x_ba1, x_ba2, x_ab1, x_ab2, x_a_seg, x_b_seg, x_ab1_seg, x_ba1_seg = [
        ], [], [], [], [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            # encode again
            c_b_recon1, s_a_recon1 = self.gen_a.encode(
                self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            c_a_recon1, s_b_recon1 = self.gen_b.encode(
                self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            # decode again (if needed)
            x_a_cycle.append(self.gen_a.decode(c_a_recon1, s_a_fake))
            x_b_cycle.append(self.gen_b.decode(c_b_recon1, s_b_fake))
            x_a_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_a(x_a[i].unsqueeze(0))[0])))
            x_b_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_b(x_b[i].unsqueeze(0))[0])))
            x_ab1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))[0])))
            x_ba1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))[0])))

            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_seg, x_b_seg = torch.cat(x_a_seg), torch.cat(x_b_seg)
        x_ab1_seg, x_ba1_seg = torch.cat(x_ab1_seg), torch.cat(x_ba1_seg)
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_a_cycle, x_b_cycle = torch.cat(x_a_cycle), torch.cat(x_b_cycle)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_seg, x_a_recon, x_a_cycle, x_ab1, x_ab1_seg, x_ab2, x_b, x_b_seg, x_b_recon, x_b_cycle, x_ba1, x_ba1_seg, x_ba2

    def sample_test_MR_ours_validate(self, input_x_a, test_display_masks_a, input_x_b, test_display_masks_b):
        self.eval()
        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        x_babs, x_bab_segs = [], []
        x_b0s, x_b0_segs = [], []
        x_b_segs = []
        x_ab0s, x_ab0_segs = [], []
        x_a_masks, x_b_masks = [], []
        x_as, x_bs = [], []
        x_b_coloraugs = []
        x_b_coloraug_segs = []
        combine_pred_b_teas = []
        combine_pred_b_teas_v2 = []
        x_b_seg_volume = []
        x_b_seg_volume_aug = []
        x_b_seg_volume_aug_v2 = []
        x_b_seg_volume_aug_v3 = []
        x_b_seg_volume_aug_v4 = []
        x_ab_volume = []
        loss_mask_ones = torch.ones_like(
            input_x_b[0].unsqueeze(0)).cuda().detach()
        for i in range(input_x_b.size(0)):
            c_a, s_a_fake = self.gen_a.encode(input_x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(input_x_b[i].unsqueeze(0))
            x_b0 = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_ba = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            # encode again    ####  cycle reconstruct result
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            # decode again (if needed)
            x_bab = self.gen_b.decode(c_b_recon, s_b0[i].unsqueeze(0))
            x_ab0 = self.gen_b.decode(c_a, s_b0[i].unsqueeze(0))

            x_bab = self.un_normalize(x_bab)
            x_b0 = self.un_normalize(x_b0)
            x_ab0 = self.un_normalize(x_ab0)

            x_b_ori = self.un_normalize(input_x_b[i]).unsqueeze(0).float()
            x_bs.append(x_b_ori)
            x_b_coloraug = test_augmentations_color(x_b_ori, loss_mask_ones)
            x_b_coloraug_img = x_b_coloraug[0]
            x_b_coloraugs.append(x_b_coloraug_img)
            x_b_coloraug_seg = self.eval_net(x_b_coloraug_img, [])
            x_b_seg = self.eval_net(x_b_ori, [])
            # x_b0_seg = self.eval_net(x_b0.float())
            x_bab_seg = self.eval_net(x_bab.float(), [])
            # x_ab0_seg = self.eval_net(x_ab0.float())
            if self.hyperparameters['seg']['segmentor'] == 'ReTri':
                prob_x_b_seg = x_b_seg[0]
                # prob_x_b0_seg = x_b0_seg[0]
                prob_x_bab_seg = x_bab_seg[0]
                # prob_x_ab0_seg = x_ab0_seg[0]
                prob_x_b_coloraug_seg = x_b_coloraug_seg[0]
            else:
                prob_x_b_seg = F.softmax(x_b_seg, dim=1)
                # prob_x_b0_seg = F.softmax(x_b0_seg, dim=1)
                prob_x_bab_seg = F.softmax(x_bab_seg, dim=1)
                # prob_x_ab0_seg = F.softmax(x_ab0_seg, dim=1)
                prob_x_b_coloraug_seg = F.softmax(x_b_coloraug_seg, dim=1)

            # mean_logits_b_tea = (
            #     prob_x_b_seg + prob_x_b0_seg + prob_x_bab_seg + prob_x_b_coloraug_seg) / 4

            # x_b_seg_volume_aug.append(self.validate_mask(mean_logits_b_tea))

            # mean_logits_b_tea_v2 = (prob_x_b_seg + prob_x_b_coloraug_seg) / 2
            # x_b_seg_volume_aug_v2.append(self.validate_mask(mean_logits_b_tea_v2))

            mean_logits_b_tea_v3 = (
                prob_x_b_seg + prob_x_b_coloraug_seg + prob_x_bab_seg) / 3
            x_b_seg_volume_aug_v3.append(
                self.validate_mask(mean_logits_b_tea_v3))

            # mean_logits_b_tea_v4 = (prob_x_b_seg + prob_x_bab_seg) / 2
            # x_b_seg_volume_aug_v4.append(self.validate_mask(mean_logits_b_tea_v4))

            x_b_seg_volume.append(self.validate_mask(prob_x_b_seg))

        # x_b_seg_volume_aug = torch.cat(x_b_seg_volume_aug)
        # x_b_seg_volume_aug_v2 = torch.cat(x_b_seg_volume_aug_v2)
        x_b_seg_volume_aug_v3 = torch.cat(x_b_seg_volume_aug_v3)
        # x_b_seg_volume_aug_v4 = torch.cat(x_b_seg_volume_aug_v4)
        x_b_seg_volume = torch.cat(x_b_seg_volume)
        self.train()
        # return x_b_seg_volume_aug_v4,x_b_seg_volume_aug_v3, x_b_seg_volume_aug_v2, x_b_seg_volume_aug, x_b_seg_volume
        return x_b_seg_volume_aug_v3, x_b_seg_volume

    def sample_test_Echo_validate(self, input_x_b, test_display_masks_b):
        """
        Validation function specifically designed for Echo dataset
        where each patient has only 2 images (ED and ES)
        """
        self.eval()
        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        
        x_b_seg_results = []
        x_b_seg_aug_results = []
        
        loss_mask_ones = torch.ones_like(
            input_x_b[0].unsqueeze(0)).cuda().detach()
        
        for i in range(input_x_b.size(0)):
            # c_a, s_a_fake = self.gen_a.encode(input_x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(input_x_b[i].unsqueeze(0))
            x_b0 = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_ba = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            # encode again    ####  cycle reconstruct result
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            # decode again (if needed)
            x_bab = self.gen_b.decode(c_b_recon, s_b0[i].unsqueeze(0))

            x_bab = self.un_normalize(x_bab)
            x_b0 = self.un_normalize(x_b0)
            x_b_ori = self.un_normalize(input_x_b[i]).unsqueeze(0).float()
            
            # Color augmentation
            x_b_coloraug = test_augmentations_color(x_b_ori, loss_mask_ones)
            x_b_coloraug_img = x_b_coloraug[0]
            
            # Segmentation predictions
            x_b_seg = self.eval_net(x_b_ori, [])
            x_b_coloraug_seg = self.eval_net(x_b_coloraug_img, [])
            x_bab_seg = self.eval_net(x_bab.float(), [])
            
            if self.hyperparameters['seg']['segmentor'] == 'ReTri':
                prob_x_b_seg = x_b_seg[0]
                prob_x_bab_seg = x_bab_seg[0]
                prob_x_b_coloraug_seg = x_b_coloraug_seg[0]
            else:
                prob_x_b_seg = F.softmax(x_b_seg, dim=1)
                prob_x_bab_seg = F.softmax(x_bab_seg, dim=1)
                prob_x_b_coloraug_seg = F.softmax(x_b_coloraug_seg, dim=1)

            # Original prediction
            x_b_seg_results.append(self.validate_mask(prob_x_b_seg))
            
            # Augmented prediction (ensemble of 3 methods)
            mean_logits_b_tea_v3 = (
                prob_x_b_seg + prob_x_b_coloraug_seg + prob_x_bab_seg) / 3
            x_b_seg_aug_results.append(
                self.validate_mask(mean_logits_b_tea_v3))

        x_b_seg_results = torch.cat(x_b_seg_results)
        x_b_seg_aug_results = torch.cat(x_b_seg_aug_results)
        
        self.train()
        return x_b_seg_aug_results, x_b_seg_results

    def sample_test_seg(self, x_a, x_b, test_display_masks_a, test_display_masks_b):
        self.eval()
        s_b0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_a0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_babs, x_bab_segs = [], []
        x_b0s, x_b0_segs = [], []
        x_b_segs = []
        x_b_segs2 = []
        x_bab_segs2 = []
        x_ab0s, x_ab0_segs = [], []
        x_a_masks, x_b_masks = [], []
        x_b0_segs2, x_ab0_segs2 = [], []
        x_as, x_bs = [], []
        x_b_coloraugs = []
        x_b_coloraug_segs = []
        loss_mask_ones = torch.ones_like(x_b[0].unsqueeze(0)).cuda().detach()
        for i in range(x_a.size(0)):

            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_b0 = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_ba = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            # encode again    ####  cycle reconstruct result
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            # decode again (if needed)
            x_bab = self.gen_b.decode(c_b_recon, s_b0[i].unsqueeze(0))
            x_ab0 = self.gen_b.decode(c_a, s_b0[i].unsqueeze(0))
            x_bab = self.un_normalize(x_bab)
            x_b0 = self.un_normalize(x_b0)
            x_ab0 = self.un_normalize(x_ab0)
            x_babs.append(x_bab)
            x_b0s.append(x_b0)
            x_ab0s.append(x_ab0)
            x_b_ori = self.un_normalize(x_b[i]).unsqueeze(0).float()
            x_b_coloraug = test_augmentations_color(x_b_ori, loss_mask_ones)
            x_b_coloraug_img = x_b_coloraug[0]
            x_b_coloraugs.append(x_b_coloraug_img)
            x_b_coloraug_seg = self.eval_net(x_b_coloraug_img,[])
            x_b_seg = self.eval_net(x_b_ori, [])
            x_b0_seg = self.eval_net(x_b0.float(), [])
            x_ab0_seg = self.eval_net(x_ab0.float(), [])
            x_bab_seg = self.eval_net(x_bab.float(), [])
            # prob_x_b_seg = F.softmax(x_b_seg, dim=1)
            # prob_x_b0_seg = F.softmax(x_b0_seg, dim=1)
            # prob_x_ab0_seg = F.softmax(x_ab0_seg, dim=1)
            # prob_x_bab_seg = F.softmax(x_bab_seg, dim=1)
            # prob_x_b_coloraug_seg = F.softmax(x_b_coloraug_seg, dim=1)
            prob_x_b_seg = x_b_seg[0]
            prob_x_b0_seg = x_b0_seg[0]
            prob_x_ab0_seg = x_ab0_seg[0]
            prob_x_bab_seg = x_bab_seg[0]
            prob_x_b_coloraug_seg = x_b_coloraug_seg[0]

            x_b_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_b_seg)))
            x_b0_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_b0_seg)))
            x_ab0_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_ab0_seg)))
            x_bab_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_bab_seg)))
            x_b_coloraug_segs.append(self.mask2color_single(
                self.validate_mask(prob_x_b_coloraug_seg)))

            # x_b_seg2 = self.seg_student(
            #     self.un_normalize(x_b[i]).unsqueeze(0).float())
            # x_b0_seg2 = self.seg_student(x_b0.float())
            # x_ab0_seg2 = self.seg_student(x_ab0.float())
            # x_bab_seg2 = self.seg_student(x_bab.float())
            # prob_x_b_seg2 = F.softmax(x_b_seg2, dim=1)
            # prob_x_b0_seg2 = F.softmax(x_b0_seg2, dim=1)
            # prob_x_ab0_seg2 = F.softmax(x_ab0_seg2, dim=1)
            # prob_x_bab_seg2 = F.softmax(x_bab_seg2, dim=1)

            # x_b_segs2.append(self.mask2color_single(
            #     self.validate_mask(prob_x_b_seg2)))
            # x_b0_segs2.append(self.mask2color_single(
            #     self.validate_mask(prob_x_b0_seg2)))
            # x_ab0_segs2.append(self.mask2color_single(
            #     self.validate_mask(prob_x_ab0_seg2)))
            # x_bab_segs2.append(self.mask2color_single(
            #     self.validate_mask(prob_x_bab_seg2)))

            x_a_mask = self.mask2color_single(
                (test_display_masks_a[i].unsqueeze(0))[:, 0, :, :])
            x_b_mask = self.mask2color_single(
                (test_display_masks_b[i].unsqueeze(0))[:, 0, :, :])
            x_a_masks.append(x_a_mask)
            x_b_masks.append(x_b_mask)
            x_as.append(self.un_normalize(x_a[i]).unsqueeze(0))
            x_bs.append(self.un_normalize(x_b[i]).unsqueeze(0))
        x_b0s, x_ab0s = torch.cat(x_b0s), torch.cat(x_ab0s)
        x_babs = torch.cat(x_babs)
        x_b0_segs, x_ab0_segs = torch.cat(x_b0_segs), torch.cat(x_ab0_segs)
        # x_b0_segs2, x_ab0_segs2 = torch.cat(x_b0_segs2), torch.cat(x_ab0_segs2)
        # x_bab_segs, x_bab_segs2 = torch.cat(x_bab_segs), torch.cat(x_bab_segs2)
        # x_b_segs, x_b_segs2 = torch.cat(x_b_segs), torch.cat(x_b_segs2)
        x_bab_segs = torch.cat(x_bab_segs)
        x_b_segs = torch.cat(x_b_segs)
        x_a_masks, x_b_masks = torch.cat(x_a_masks), torch.cat(x_b_masks)
        x_as, x_bs = torch.cat(x_as), torch.cat(x_bs)
        x_b_coloraugs = torch.cat(x_b_coloraugs)
        x_b_coloraug_segs = torch.cat(x_b_coloraug_segs)
        self.train()
        return x_as, x_ab0s, x_ab0_segs, x_a_masks, \
            x_bs, x_b_segs, x_b0s, x_b0_segs, x_babs, x_bab_segs, x_b_coloraugs, x_b_coloraug_segs, x_b_masks

    def sample_train_seg(self, x_a, x_b, test_display_masks_a, test_display_masks_b):
        self.eval()
        s_b0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_b0s, x_b0_strongs = [], []
        x_ab0s, x_ab0_strongs = [], []
        x_a_masks, x_b_masks = [], []
        mix_stus, mix_stus_segs = [], []
        x_ab0_weaks, x_b0_weaks = [], []
        x_ab0_weak_segs, x_b0_weak_segs = [], []
        mix_tea_segs = []
        for i in range(x_a.size(0)):

            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_b0 = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_ab0 = self.gen_b.decode(c_a, s_b0[i].unsqueeze(0))

            # prepare the data for semi-seg
            size_ab0 = x_ab0.size()
            size_b = x_b0.size()
            assert size_ab0 == size_b

            sample_seg_b0 = {}
            sample_seg_b0['image_size_yx'] = np.array(size_b[2::])
            # BCHW ---> HWC
            sample_seg_b0['mask_arr'] = np.full(
                size_b[2::], 255, dtype=np.uint8)
            sample_seg_b0['image_arr'] = x_b0[0].permute(
                1, 2, 0).detach().cpu().numpy()
            sample_seg_b0['image_arr'] = (
                ((sample_seg_b0['image_arr'] * 0.5) + 0.5) * 255).astype(np.uint8)

            sample_seg_ab0 = {}
            sample_seg_ab0['image_size_yx'] = np.array(size_ab0[2::])
            sample_seg_ab0['image_arr'] = x_b0[0].permute(
                1, 2, 0).detach().cpu().numpy()
            sample_seg_ab0['image_arr'] = (
                ((sample_seg_ab0['image_arr'] * 0.5) + 0.5) * 255).astype(np.uint8)
            sample_seg_ab0['labels_arr'] = np.array(test_display_masks_a[i].unsqueeze(0).detach().cpu())[
                :, 0, :, :][0].astype(np.int32)
            sample_seg_ab0['mask_arr'] = np.full(
                size_ab0[2::], 255, dtype=np.uint8)
            # weak and strong aug of the source image: sample_seg_ab0 is the weak supervised augmentation and sample_seg_ab1 is the strong augmentation for cutmix
            # final shape: BCHW
            sample_seg_ab0_paired = self.train_unsup_transforms.apply(
                sample_seg_ab0)
            # strong and weak paired aug to the target domain images
            # final shape: BCHW, type
            sample_seg_b0_paired = self.train_unsup_transforms.apply(
                sample_seg_b0)

            batch_ux0_tea = sample_seg_ab0_paired['sample0']['image']
            batch_ux0_stu = sample_seg_ab0_paired['sample1']['image']
            batch_um0 = sample_seg_ab0_paired['sample0']['mask']

            batch_ux1_tea = sample_seg_b0_paired['sample0']['image']
            batch_ux1_stu = sample_seg_b0_paired['sample1']['image']
            batch_um1 = sample_seg_b0_paired['sample0']['mask']

            x_ab0_weaks.append(batch_ux0_tea)
            x_b0_weaks.append(batch_ux1_tea)
            x_ab0_strongs.append(batch_ux0_stu)
            x_b0_strongs.append(batch_ux1_stu)

            batch_mix_masks = torch.from_numpy(self.clss_generate_mask_params(
                sample_seg_ab0_paired)).cuda()

            # Mix images with masks, cross mixed
            batch_ux_stu_mixed = batch_ux0_stu * \
                (1 - batch_mix_masks) + \
                batch_ux1_stu * batch_mix_masks
            mix_stus.append(batch_ux_stu_mixed)

            logits_u0_tea = self.seg_teacher(
                batch_ux0_tea, []).detach()
            x_ab0_weak_segs.append(self.mask2color_single(
                self.validate_mask(logits_u0_tea)))

            logits_u1_tea = self.seg_teacher(
                batch_ux1_tea, []).detach()
            x_b0_weak_segs.append(self.mask2color_single(
                self.validate_mask(logits_u1_tea)))

            # Get student prediction for mixed image
            logits_cons_stu = self.seg_student(batch_ux_stu_mixed.float(), [])

            logits_cons_tea = logits_u0_tea * \
                (1 - batch_mix_masks) + \
                logits_u1_tea * batch_mix_masks

            # Logits -> probs
            prob_cons_tea = F.softmax(logits_cons_tea, dim=1)
            mix_tea_segs.append(self.mask2color_single(
                self.validate_mask(prob_cons_tea)))

            prob_cons_stu = F.softmax(logits_cons_stu, dim=1)
            mix_stus_segs.append(self.mask2color_single(
                self.validate_mask(prob_cons_stu)))

        # x_b0s, x_ab0s = torch.cat(x_b0s), torch.cat(x_ab0s)
        x_ab0_strongs, x_b0_strongs = torch.cat(
            x_ab0_strongs), torch.cat(x_b0_strongs)
        mix_stus, mix_stus_segs = torch.cat(mix_stus), torch.cat(mix_stus_segs)
        x_ab0_weaks, x_ab0_weak_segs = torch.cat(
            x_ab0_weaks), torch.cat(x_ab0_weak_segs)
        x_b0_weaks, x_b0_weak_segs = torch.cat(
            x_b0_weaks), torch.cat(x_b0_weak_segs)
        mix_tea_segs = torch.cat(mix_tea_segs)
        self.train()
        return x_ab0_strongs, x_b0_strongs, mix_stus, mix_stus_segs, x_ab0_weaks, x_ab0_weak_segs, x_b0_weaks, x_b0_weak_segs, mix_tea_segs

    def visual_train_cutout(self, input_x_b):
        self.eval()
        x_b1s = []
        combine_pred_teas = []
        weak_trans_teas = []
        strong_aug_stus = []
        cutout_masks = []
        cutout_img_stus = []
        pred_cutout_stus = []
        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())

        s_a0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        loss_mask_ones = torch.full_like(
            input_x_b[0].unsqueeze(0), 1).cuda().detach()
        for i in range(input_x_b.size(0)):
            # encode
            x_b1 = input_x_b[i].unsqueeze(0)
            c_b1, s_b1_prime = self.gen_b.encode(x_b1)
            b1_b_tea = x_b1.clone()
            b1_b_tea = self.un_normalize(b1_b_tea)
            b1_b0_tea = self.gen_b.decode(c_b1, s_b1[i].unsqueeze(0))
            b1_b0_tea = self.un_normalize(b1_b0_tea)
            b1_ba0_2b = self.gen_a.decode(c_b1, s_a1[i].unsqueeze(0))
            c_ba0b_rec, _ = self.gen_a.encode(b1_ba0_2b)
            b1_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b2[i].unsqueeze(0))
            b1_ba0b_tea = self.un_normalize(b1_ba0b_tea)

            x_b1s.append(b1_b_tea)
            # prepare the data for semi-seg

            x_b1_weak = unsup_augmentations_weak(b1_b_tea, loss_mask_ones)
            params_weak_b1_tea = unsup_augmentations_weak._params
            img_b1_weak = x_b1_weak[0]
            loss_mask_b1_weak = x_b1_weak[1]
            x_b1_weak_strong_stu = unsup_augmentations_strongcolor(
                img_b1_weak, loss_mask_ones)
            img_b1_weak_strong_stu = x_b1_weak_strong_stu[0]
            strong_aug_stus.append(img_b1_weak_strong_stu)

            logitsb1_b_tea = self.seg_teacher(
                b1_b_tea, []).detach()
            logitsb1_b0_tea = self.seg_teacher(
                b1_b0_tea, []).detach()
            logitsb1_ba0b_tea = self.seg_teacher(
                b1_ba0b_tea, []).detach()
            mean_logits_b1_tea = (
                logitsb1_b_tea + logitsb1_b0_tea + logitsb1_ba0b_tea) / 3
            logits_b1_tea_weak = unsup_augmentations_weak(
                mean_logits_b1_tea, loss_mask_ones, params=params_weak_b1_tea)
            logits_cons_tea_cutout0 = logits_b1_tea_weak[0]
            prob_mean_logits_b1_tea = F.softmax(mean_logits_b1_tea, dim=1)
            combine_pred_teas.append(self.mask2color_single(
                self.validate_mask(prob_mean_logits_b1_tea)))
            prob_cons_tea_cutout0 = F.softmax(logits_cons_tea_cutout0, dim=1)
            weak_trans_teas.append(self.mask2color_single(
                self.validate_mask(prob_cons_tea_cutout0)))

            batch_cut_masks0 = torch.from_numpy(
                self.clss_generate_cutout_mask_params(b1_b_tea)).cuda()
            # Cut image with mask (mask regions to zero)
            cutout_masks.append(batch_cut_masks0)
            batch_ux0_stu_cut = img_b1_weak_strong_stu * batch_cut_masks0
            cutout_img_stus.append(batch_ux0_stu_cut)
            # Get student prediction for cut image
            logits_cons_stu_cutout0 = self.seg_student(
                batch_ux0_stu_cut.float().contiguous(), [])

            prob_cons_stu_cutout0 = F.softmax(logits_cons_stu_cutout0, dim=1)
            pred_cutout_stus.append(self.mask2color_single(
                self.validate_mask(prob_cons_stu_cutout0)))
        x_b1s = torch.cat(x_b1s)
        combine_pred_teas = torch.cat(combine_pred_teas)
        weak_trans_teas = torch.cat(weak_trans_teas)
        strong_aug_stus = torch.cat(strong_aug_stus)
        cutout_masks = torch.cat(cutout_masks)
        cutout_img_stus = torch.cat(cutout_img_stus)
        pred_cutout_stus = torch.cat(pred_cutout_stus)
        self.train()
        return x_b1s, combine_pred_teas, weak_trans_teas, strong_aug_stus,\
            cutout_masks, cutout_img_stus, pred_cutout_stus

    def visual_train_quadtree(self, input_x_b, hyperparameters):
        self.eval()
        x_b2s = []
        combine_pred_teas = []
        weak_trans_teas = []
        strong_aug_stus = []
        quad_masks = []
        quad_img_stus = []
        pred_quad_stus = []
        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())

        s_a0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        loss_mask_ones = torch.full_like(
            input_x_b[0].unsqueeze(0), 1).cuda().detach()
        for i in range(input_x_b.size(0)):
            # encode
            x_b2 = input_x_b[i].unsqueeze(0)
            c_b2, s_b1_prime = self.gen_b.encode(x_b2)
            b2_b_tea = x_b2.clone()
            b2_b_tea = self.un_normalize(b2_b_tea)
            b2_b0_tea = self.gen_b.decode(c_b2, s_b1[i].unsqueeze(0))
            b2_b0_tea = self.un_normalize(b2_b0_tea)
            b2_ba0_2b = self.gen_a.decode(c_b2, s_a1[i].unsqueeze(0))
            c_ba0b_rec, _ = self.gen_a.encode(b2_ba0_2b)
            b2_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b2[i].unsqueeze(0))
            b2_ba0b_tea = self.un_normalize(b2_ba0b_tea)

            x_b2s.append(b2_b_tea)
            # prepare the data for semi-seg

            x_b2_weak_quad = unsup_augmentations_weak(
                b2_b_tea, loss_mask_ones)
            params_weak_b2_tea_quad = unsup_augmentations_weak._params
            img_b2_weak_quad = x_b2_weak_quad[0]
            loss_mask_b2_weak_quad = x_b2_weak_quad[1]
            x_b2_weak_strong_stu_quad = unsup_augmentations_strongcolor(
                img_b2_weak_quad, loss_mask_ones)
            img_b2_weak_strong_stu_quad = x_b2_weak_strong_stu_quad[0]
            strong_aug_stus.append(img_b2_weak_strong_stu_quad)

            logitsb2_b_tea = self.seg_teacher(
                b2_b_tea, []).detach()
            logitsb2_b0_tea = self.seg_teacher(
                b2_b0_tea, []).detach()
            logitsb2_ba0b_tea = self.seg_teacher(
                b2_ba0b_tea, []).detach()
            mean_logits_b2_tea_quad = (
                logitsb2_b_tea + logitsb2_b0_tea + logitsb2_ba0b_tea) / 3
            logits_b2_tea_weak_quad = unsup_augmentations_weak(
                mean_logits_b2_tea_quad, loss_mask_ones, params=params_weak_b2_tea_quad)
            logits_cons_tea_b2 = logits_b2_tea_weak_quad[0]

            prob_mean_logits_b2_tea = F.softmax(mean_logits_b2_tea_quad, dim=1)
            combine_pred_teas.append(self.mask2color_single(
                self.validate_mask(prob_mean_logits_b2_tea)))
            prob_cons_tea_b2 = F.softmax(logits_cons_tea_b2, dim=1)
            weak_trans_teas.append(self.mask2color_single(
                self.validate_mask(prob_cons_tea_b2)))

            qt_mix_b2 = QTree(
                0.0005, 2, img_b2_weak_strong_stu_quad.squeeze(0))
            qt_mix_b2.subdivide()
            quad_mask_stu = qt_mix_b2.gen_quadaug_mask(
                ratio=hyperparameters['strong_aug_para']['quad_ratio'])
            quad_mask_stu = quad_mask_stu.unsqueeze(0)
            # print(quad_mask_stu.shape)
            quad_masks.append(quad_mask_stu)
            b2_stu = quad_mask_stu * img_b2_weak_strong_stu_quad
            quad_img_stus.append(b2_stu)
            # Get student prediction for quadcut image
            logits_cons_stu_b2 = self.seg_student(
                b2_stu.float().contiguous(), [])

            # Logits -> probs
            prob_cons_stu_b2 = F.softmax(
                logits_cons_stu_b2, dim=1)
            pred_quad_stus.append(self.mask2color_single(
                self.validate_mask(prob_cons_stu_b2)))

        x_b2s = torch.cat(x_b2s)
        combine_pred_teas = torch.cat(combine_pred_teas)
        weak_trans_teas = torch.cat(weak_trans_teas)
        strong_aug_stus = torch.cat(strong_aug_stus)
        quad_masks = torch.cat(quad_masks)
        quad_img_stus = torch.cat(quad_img_stus)
        pred_quad_stus = torch.cat(pred_quad_stus)
        self.train()
        return x_b2s, combine_pred_teas, weak_trans_teas, strong_aug_stus,\
            quad_masks, quad_img_stus, pred_quad_stus

    def visual_train_cutmix_ab_b(self, input_x_a, input_x_b, hyperparameters):
        self.eval()
        x_as = []
        x_ab0s = []
        strong_aug_ab0_stus = []
        strong_aug_b_stus = []
        cutmix_masks = []
        cutmix_img_stus = []
        quad_masks = []
        quad_img_stus = []
        pred_stus = []
        x_bs = []
        pred_b_teas = []
        x_b0s = []
        pred_b0_teas = []
        x_cycs = []
        pred_cycs_teas = []
        combine_pred_b_teas = []
        weak_trans_b_teas = []
        pred_ab0_teas = []
        weak_trans_ab0_teas = []
        cutmix_pred_teas = []
        s_b0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())

        s_a0 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())
        s_a1 = Variable(torch.randn(input_x_b.size(0),
                        self.style_dim, 1, 1).cuda())

        for i in range(input_x_b.size(0)):
            # encode
            x_a = input_x_a[i].unsqueeze(0)
            x_as.append(self.un_normalize(x_a))
            x_b = input_x_b[i].unsqueeze(0)
            x_bs.append(self.un_normalize(x_b))
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # decode (within domain) ####self reconstruction#####
            # [-1,1] [1,3,224,256]
            x_b_tea = x_b.clone()
            x_b_tea = self.un_normalize(x_b_tea)
            x_b0_tea = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_b0_tea = self.un_normalize(x_b0_tea)
            x_b0s.append(x_b0_tea)
            x_ba0_2b = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            c_ba0b_rec, _ = self.gen_a.encode(x_ba0_2b)
            x_ba0b_tea = self.gen_b.decode(c_ba0b_rec, s_b1[i].unsqueeze(0))
            x_ba0b_tea = self.un_normalize(x_ba0b_tea)
            x_cycs.append(x_ba0b_tea)
            x_ab0_tea = self.gen_b.decode(c_a, s_b2[i].unsqueeze(0))
            x_ab0_tea = self.un_normalize(x_ab0_tea)
            x_ab0s.append(x_ab0_tea)
            loss_mask_ones = torch.full_like(
                x_b_tea, 1).cuda().detach()

            x_ab0_weak = unsup_augmentations_weak(
                x_ab0_tea, loss_mask_ones)
            params_weak_ab0_tea = unsup_augmentations_weak._params
            img_ab0_weak = x_ab0_weak[0]
            loss_mask_ab0_weak = x_ab0_weak[1]
            x_ab0_weak_strong_stu = unsup_augmentations_strongcolor(
                img_ab0_weak, loss_mask_ones)
            img_ab0_weak_strong_stu = x_ab0_weak_strong_stu[0]
            strong_aug_ab0_stus.append(img_ab0_weak_strong_stu)

            logits_ab0_tea = self.seg_teacher(
                x_ab0_tea, []).detach()
            # print('logits_ab0_tea.shape:', logits_ab0_tea.shape)
            logits_ab0_tea_weak = unsup_augmentations_weak(
                logits_ab0_tea, loss_mask_ones, params=params_weak_ab0_tea)
            logits_ab0_tea_weak = logits_ab0_tea_weak[0]
            prob_ab0_tea = F.softmax(logits_ab0_tea, dim=1)
            pred_ab0_teas.append(self.mask2color_single(
                self.validate_mask(prob_ab0_tea)))
            prob_ab0_tea_weak = F.softmax(logits_ab0_tea_weak, dim=1)
            weak_trans_ab0_teas.append(self.mask2color_single(
                self.validate_mask(prob_ab0_tea_weak)))

            x_b_weak = unsup_augmentations_weak(x_b_tea, loss_mask_ones)
            params_weak_b_tea = unsup_augmentations_weak._params
            img_b_weak = x_b_weak[0]
            loss_mask_b_weak = x_b_weak[1]
            x_b_weak_strong_stu = unsup_augmentations_strongcolor(
                img_b_weak, loss_mask_ones)
            img_b_weak_strong_stu = x_b_weak_strong_stu[0]
            strong_aug_b_stus.append(img_b_weak_strong_stu)
            with torch.no_grad():
                logits_b_tea = self.seg_teacher(
                    x_b_tea, []).detach()
                pred_b_teas.append(self.mask2color_single(
                    self.validate_mask(logits_b_tea)))
                logits_b0_tea = self.seg_teacher(
                    x_b0_tea, []).detach()
                pred_b0_teas.append(self.mask2color_single(
                    self.validate_mask(logits_b0_tea)))
                logits_ba0b_tea = self.seg_teacher(
                    x_ba0b_tea, []).detach()
                pred_cycs_teas.append(self.mask2color_single(
                    self.validate_mask(logits_ba0b_tea)))
                mean_logits_b_tea = (
                    logits_b_tea + logits_b0_tea + logits_ba0b_tea) / 3
                combine_pred_b_teas.append(self.mask2color_single(
                    self.validate_mask(mean_logits_b_tea)))

                logits_b_tea_weak = unsup_augmentations_weak(
                    mean_logits_b_tea, loss_mask_ones, params=params_weak_b_tea)
                logits_b_tea_weak = logits_b_tea_weak[0]
                weak_trans_b_teas.append(self.mask2color_single(
                    self.validate_mask(logits_b_tea_weak)))
            batch_mix_masks = torch.from_numpy(
                self.clss_generate_mask_params(x_ab0_tea)).cuda()
            # print('batch_mix_masks.shape', batch_mix_masks.shape)
            cutmix_masks.append(batch_mix_masks)

            ab0_b_stu_mixed = img_ab0_weak_strong_stu * \
                (1 - batch_mix_masks) + \
                img_b_weak_strong_stu * batch_mix_masks
            ab0_b_stu_mixed = ab0_b_stu_mixed.float().contiguous()
            cutmix_img_stus.append(ab0_b_stu_mixed)

            qt_mix = QTree(0.0005, 2, ab0_b_stu_mixed.squeeze(0))
            qt_mix.subdivide()
            quad_mask_stu = qt_mix.gen_quadaug_mask(
                ratio=hyperparameters['strong_aug_para']['quad_ratio'])
            quad_mask_stu = quad_mask_stu.unsqueeze(0)
            # print(quad_mask_stu.shape)
            quad_masks.append(quad_mask_stu)
            ab0_b_stu_mixed = quad_mask_stu * ab0_b_stu_mixed
            quad_img_stus.append(ab0_b_stu_mixed)
            # Get student prediction for mixed image
            logits_cons_stu = self.seg_student(
                ab0_b_stu_mixed.float().contiguous(), [])

            logits_cons_tea = logits_ab0_tea_weak * \
                (1 - batch_mix_masks) + \
                logits_b_tea_weak * batch_mix_masks

            prob_cons_tea = F.softmax(logits_cons_tea, dim=1)
            prob_cons_stu = F.softmax(logits_cons_stu, dim=1)
            pred_stus.append(self.mask2color_single(
                self.validate_mask(prob_cons_stu)))
            cutmix_pred_teas.append(self.mask2color_single(
                self.validate_mask(prob_cons_tea)))
        x_as = torch.cat(x_as)
        x_ab0s = torch.cat(x_ab0s)
        strong_aug_ab0_stus = torch.cat(strong_aug_ab0_stus)
        strong_aug_b_stus = torch.cat(strong_aug_b_stus)
        cutmix_masks = torch.cat(cutmix_masks)
        cutmix_img_stus = torch.cat(cutmix_img_stus)
        quad_masks = torch.cat(quad_masks)
        quad_img_stus = torch.cat(quad_img_stus)
        pred_stus = torch.cat(pred_stus)
        x_bs = torch.cat(x_bs)
        pred_b_teas = torch.cat(pred_b_teas)
        x_b0s = torch.cat(x_b0s)
        pred_b0_teas = torch.cat(pred_b0_teas)
        x_cycs = torch.cat(x_cycs)
        pred_cycs_teas = torch.cat(pred_cycs_teas)
        combine_pred_b_teas = torch.cat(combine_pred_b_teas)
        weak_trans_b_teas = torch.cat(weak_trans_b_teas)
        pred_ab0_teas = torch.cat(pred_ab0_teas)
        weak_trans_ab0_teas = torch.cat(weak_trans_ab0_teas)
        cutmix_pred_teas = torch.cat(cutmix_pred_teas)
        self.train()
        return x_as, x_ab0s, strong_aug_ab0_stus, strong_aug_b_stus,\
            cutmix_masks, cutmix_img_stus, quad_masks, quad_img_stus,\
            pred_stus, x_bs, pred_b_teas, x_b0s, pred_b0_teas, x_cycs,\
            pred_cycs_teas, combine_pred_b_teas, weak_trans_b_teas,\
            pred_ab0_teas, weak_trans_ab0_teas, cutmix_pred_teas

    def sample_train_seg_our(self, x_a, x_b):
        self.eval()
        s_b0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_b4 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        s_a0 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        x_b0_stus_b, x_ab0b_stus_a = [], []
        x_ba0b_teas_b, x_ab0_teas_a = [], []
        mix_stus, mix_stus_segs = [], []
        x_ab0_tea_segs, x_b0_tea_segs = [], []
        mix_tea_segs = []
        mix_masks = []
        cutout_masks_a = []
        cutout_masks_b = []
        cutout_stu_a = []
        cutout_stu_b = []
        cutout_stuas_segs = []
        cutout_stubs_segs = []
        for i in range(x_a.size(0)):

            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_b0_stu_b = self.gen_b.decode(c_b, s_b0[i].unsqueeze(0))
            x_ab0_2b = self.gen_b.decode(c_a, s_b1[i].unsqueeze(0))
            c_ab0b_rec, _ = self.gen_b.encode(x_ab0_2b)
            x_ab0b_stu_a = self.gen_b.decode(c_ab0b_rec, s_b2[i].unsqueeze(0))

            x_ba0_2b = self.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
            c_ba0b_rec, _ = self.gen_a.encode(x_ba0_2b)
            x_ba0b_tea_b = self.gen_b.decode(c_ba0b_rec, s_b3[i].unsqueeze(0))
            x_ab0_tea_a = self.gen_b.decode(c_a, s_b4[i].unsqueeze(0))

            x_b0_stus_b.append(x_b0_stu_b)
            x_ab0b_stus_a.append(x_ab0b_stu_a)
            x_ba0b_teas_b.append(x_ba0b_tea_b)
            x_ab0_teas_a.append(x_ab0_tea_a)
            # prepare the data for semi-seg

            batch_mix_masks = torch.from_numpy(
                self.clss_generate_mask_params(x_b0_stu_b)).cuda()
            mix_masks.append(batch_mix_masks)
            batch_cutout_masks_a = torch.from_numpy(
                self.clss_generate_cutout_mask_params(x_b0_stu_b)).cuda()
            cutout_masks_a.append(batch_cutout_masks_a)
            batch_cutout_masks_b = torch.from_numpy(
                self.clss_generate_cutout_mask_params(x_b0_stu_b)).cuda()
            cutout_masks_b.append(batch_cutout_masks_b)
            batch_uxa_stu_cut = x_ab0b_stu_a * batch_cutout_masks_a
            batch_uxb_stu_cut = x_b0_stu_b * batch_cutout_masks_b
            cutout_stu_a.append(batch_uxa_stu_cut)
            cutout_stu_b.append(batch_uxb_stu_cut)
            logits_cutout_stu_a = self.seg_student(batch_uxa_stu_cut.float(), [])
            logits_cutout_stu_b = self.seg_student(batch_uxb_stu_cut.float(), [])
            prob_cutout_stu_a = F.softmax(logits_cutout_stu_a, dim=1)
            cutout_stuas_segs.append(self.mask2color_single(
                self.validate_mask(prob_cutout_stu_a)))
            prob_cutout_stu_b = F.softmax(logits_cutout_stu_b, dim=1)
            cutout_stubs_segs.append(self.mask2color_single(
                self.validate_mask(prob_cutout_stu_b)))
            # Mix images with masks, cross mixed
            batch_ux_stu_mixed = x_ab0b_stu_a * (1 - batch_mix_masks) + \
                x_b0_stu_b * batch_mix_masks
            mix_stus.append(batch_ux_stu_mixed)

            logits_u0_tea = self.seg_teacher(x_ab0_tea_a, []).detach()

            x_ab0_tea_segs.append(self.mask2color_single(
                self.validate_mask(logits_u0_tea)))

            logits_u1_tea = self.seg_teacher(x_ba0b_tea_b, []).detach()

            x_b0_tea_segs.append(self.mask2color_single(
                self.validate_mask(logits_u1_tea)))

            # Get student prediction for mixed image
            logits_cons_stu = self.seg_student(batch_ux_stu_mixed.float(), [])

            logits_cons_tea = logits_u0_tea * (1 - batch_mix_masks) + \
                logits_u1_tea * batch_mix_masks

            # Logits -> probs
            prob_cons_tea = F.softmax(logits_cons_tea, dim=1)
            mix_tea_segs.append(self.mask2color_single(
                self.validate_mask(prob_cons_tea)))

            prob_cons_stu = F.softmax(logits_cons_stu, dim=1)
            mix_stus_segs.append(self.mask2color_single(
                self.validate_mask(prob_cons_stu)))

        x_b0_stus_b, x_ab0b_stus_a = torch.cat(
            x_b0_stus_b), torch.cat(x_ab0b_stus_a)
        x_ba0b_teas_b, x_ab0_teas_a = torch.cat(
            x_ba0b_teas_b), torch.cat(x_ab0_teas_a)
        mix_stus, mix_stus_segs = torch.cat(mix_stus), torch.cat(mix_stus_segs)
        x_b0_tea_segs, x_ab0_tea_segs = torch.cat(
            x_b0_tea_segs), torch.cat(x_ab0_tea_segs)
        mix_tea_segs = torch.cat(mix_tea_segs)
        mix_masks = torch.cat(mix_masks)
        cutout_masks_a = torch.cat(cutout_masks_a)
        cutout_masks_b = torch.cat(cutout_masks_b)
        cutout_stu_a = torch.cat(cutout_stu_a)
        cutout_stu_b = torch.cat(cutout_stu_b)
        cutout_stuas_segs = torch.cat(cutout_stuas_segs)
        cutout_stubs_segs = torch.cat(cutout_stubs_segs)
        self.train()
        return x_a, x_ab0b_stus_a, x_b, x_b0_stus_b, mix_masks, mix_stus, mix_stus_segs,\
            x_ab0_teas_a, x_ab0_tea_segs, x_ba0b_teas_b, x_b0_tea_segs, mix_tea_segs,\
            x_a, x_ab0b_stus_a, cutout_masks_a, cutout_stu_a, cutout_stuas_segs, x_ab0_tea_segs,\
            x_b, x_b0_stus_b, cutout_masks_b, cutout_stu_b, cutout_stubs_segs, x_b0_tea_segs

    def sample_TTA_CTA_seg(self, x_a, x_b, test_display_masks_a, test_display_masks_b):
        self.eval()
        # cross_aug
        s_a1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b1 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # self_aug
        s_a3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b3 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_a4 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b4 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        x_a1, x_a2,  x_ab1, x_ab2 = [], [], [], []
        x_b1, x_b2,  x_ba1, x_ba2 = [], [], [], []
        x_a1_seg, x_a2_seg,  x_ab1_seg, x_ab2_seg = [], [], [], []
        x_b1_seg, x_b2_seg,  x_ba1_seg, x_ba2_seg = [], [], [], []
        x_a_seg, x_b_seg, x_a_mask, x_b_mask = [], [], [], []
        aug_a, aug_b, self_seg_pred_a, self_seg_pred_b, a_final, b_final = [], [], [], [], [], []
        x_a_recon, x_b_recon, x_a_cycle, x_b_cycle = [], [], [], []
        x_a_recon_seg, x_b_recon_seg = [], []
        x_a_cycle_seg, x_b_cycle_seg = [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))

            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))

            x_a_recon_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_a(x_a_recon[-1])[0])))
            x_b_recon_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_b(x_b_recon[-1])[0])))

            # encode again
            c_b_recon1, s_a_recon1 = self.gen_a.encode(
                self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            c_a_recon1, s_b_recon1 = self.gen_b.encode(
                self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            # decode again (if needed)
            x_a_cycle.append(self.gen_a.decode(c_a_recon1, s_a_fake))
            x_b_cycle.append(self.gen_b.decode(c_b_recon1, s_b_fake))

            x_a_cycle_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_a(x_a_cycle[-1])[0])))
            x_b_cycle_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_b(x_b_cycle[-1])[0])))

            x_a_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_a(x_a[i].unsqueeze(0))[0])))
            x_b_seg.append(self.mask2color_single(
                self.validate_mask(self.seg_b(x_b[i].unsqueeze(0))[0])))
            self_seg_pred_a.append(self.validate_mask(
                self.seg_a(x_a[i].unsqueeze(0))[0]))
            self_seg_pred_b.append(self.validate_mask(
                self.seg_b(x_b[i].unsqueeze(0))[0]))

            # cross_aug
            x_ab1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))[0])))
            x_ab2_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))[0])))
            x_ba1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))[0])))
            x_ba2_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))[0])))
            # self_aug
            x_a1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_a, s_a3[i].unsqueeze(0)))[0])))
            x_a2_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_a(self.gen_a.decode(c_a, s_a4[i].unsqueeze(0)))[0])))
            x_b1_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))[0])))
            x_b2_seg.append(self.mask2color_single(self.validate_mask(
                self.seg_b(self.gen_b.decode(c_b, s_b4[i].unsqueeze(0)))[0])))

            x_a_mask.append(self.mask2color_single(
                (test_display_masks_a[i].unsqueeze(0))[:, 0, :, :]))
            x_b_mask.append(self.mask2color_single(
                (test_display_masks_b[i].unsqueeze(0))[:, 0, :, :]))

            x_a1.append(self.gen_a.decode(c_a, s_a3[i].unsqueeze(0)))
            x_a2.append(self.gen_a.decode(c_a, s_a4[i].unsqueeze(0)))
            x_b1.append(self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))
            x_b2.append(self.gen_b.decode(c_b, s_b4[i].unsqueeze(0)))

            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))

            seg_activate0 = self.seg_a(x_a[i].unsqueeze(0))[0]
            seg_activate1 = self.seg_b(
                self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))[0]
            seg_activate2 = self.seg_b(
                self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))[0]
            seg_activate3 = self.seg_a(
                self.gen_a.decode(c_a, s_a3[i].unsqueeze(0)))[0]
            seg_activate4 = self.seg_a(
                self.gen_a.decode(c_a, s_a4[i].unsqueeze(0)))[0]
            seg_activate_a = seg_activate0 + seg_activate1 + \
                seg_activate2 + seg_activate3 + seg_activate4
            aug_a.append(self.mask2color_single(
                self.validate_mask(seg_activate_a)))
            a_final.append(self.validate_mask(seg_activate_a))

            seg_activate0 = self.seg_b(x_b[i].unsqueeze(0))[0]
            seg_activate1 = self.seg_b(
                self.gen_b.decode(c_b, s_b3[i].unsqueeze(0)))[0]
            seg_activate2 = self.seg_b(
                self.gen_b.decode(c_b, s_b4[i].unsqueeze(0)))[0]
            seg_activate3 = self.seg_a(
                self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))[0]
            seg_activate4 = self.seg_a(
                self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))[0]
            seg_activate_b = seg_activate0 + seg_activate1 + \
                seg_activate2 + seg_activate3 + seg_activate4
            aug_b.append(self.mask2color_single(
                self.validate_mask(seg_activate_b)))
            b_final.append(self.validate_mask(seg_activate_b))

        x_a_mask, x_b_mask = torch.cat(x_a_mask), torch.cat(x_b_mask)
        x_a_seg, x_b_seg = torch.cat(x_a_seg), torch.cat(x_b_seg)
        x_ab1_seg, x_ba1_seg = torch.cat(x_ab1_seg), torch.cat(x_ba1_seg)
        x_ab2_seg, x_ba2_seg = torch.cat(x_ab2_seg), torch.cat(x_ba2_seg)
        x_a1_seg, x_b1_seg = torch.cat(x_a1_seg), torch.cat(x_b1_seg)
        x_a2_seg, x_b2_seg = torch.cat(x_a2_seg), torch.cat(x_b2_seg)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        x_a1, x_a2 = torch.cat(x_a1), torch.cat(x_a2)
        x_b1, x_b2 = torch.cat(x_b1), torch.cat(x_b2)
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_a_cycle, x_b_cycle = torch.cat(x_a_cycle), torch.cat(x_b_cycle)
        x_a_cycle_seg, x_b_cycle_seg = torch.cat(
            x_a_cycle_seg), torch.cat(x_b_cycle_seg)
        x_a_recon_seg, x_b_recon_seg = torch.cat(
            x_a_recon_seg), torch.cat(x_b_recon_seg)
        self_seg_pred_a, self_seg_pred_b, a_final, b_final = torch.cat(self_seg_pred_a),\
            torch.cat(self_seg_pred_b), torch.cat(a_final), torch.cat(b_final)
        aug_a, aug_b = torch.cat(aug_a), torch.cat(aug_b)

        self.train()
        return x_a, x_a_mask, x_a_seg, x_a_recon, x_a_recon_seg, x_a_cycle, x_a_cycle_seg, x_a1, x_a1_seg, x_a2, x_a2_seg, x_ab1, x_ab1_seg, x_ab2, x_ab2_seg, aug_a,\
            x_b, x_b_mask, x_b_seg, x_b_recon, x_b_recon_seg, x_b_cycle, x_b_cycle_seg, x_b1, x_b1_seg, x_b2, x_b2_seg, x_ba1, x_ba1_seg, x_ba2, x_ba2_seg, aug_b,\
            self_seg_pred_a, self_seg_pred_b, a_final, b_final

    def sample_seg_nosyn(self, x_a, mask_a, segnet):
        segnet.eval()
        x_a_seg, x_a_seg_predict = [], []
        for i in range(x_a.size(0)):
            x_a_seg.append(self.mask2color_single(
                (mask_a[i][0, :, :]).unsqueeze(0)))
            x_a_seg_predict.append(self.mask2color_single(
                self.validate_mask(segnet(x_a[i].unsqueeze(0))[0])))
        x_a_seg, x_a_seg_predict = torch.cat(
            x_a_seg), torch.cat(x_a_seg_predict)
        segnet.train()
        return x_a, x_a_seg, x_a_seg_predict

    def sample(self, x_a, x_b, hyperparameters):
        self.eval()
        # it means that self.training = False
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_a_cycle, x_b_cycle, x_ba1, x_ba2, x_ab1, x_ab2, x_a_seg, x_b_seg = [
        ], [], [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            # encode again
            c_b_recon1, s_a_recon1 = self.gen_a.encode(
                self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            c_a_recon1, s_b_recon1 = self.gen_b.encode(
                self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            # decode again (if needed)
            x_a_cycle.append(self.gen_a.decode(c_a_recon1, s_a_fake))
            x_b_cycle.append(self.gen_b.decode(c_b_recon1, s_b_fake))

            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_a_cycle, x_b_cycle = torch.cat(x_a_cycle), torch.cat(x_b_cycle)
        # print('x_a_cycle.shape', x_a_cycle.shape)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        # it means that self.training = True
        return x_a, x_a_recon, x_a_cycle, x_ab1, x_ab2, x_b, x_b_recon, x_b_cycle, x_ba1, x_ba2

    # def sample(self, x_a, x_b, config):
    #     self.eval()
    #     shape_a = x_a.shape
    #     shape_b = x_b.shape
    #     s_a1 = Variable(self.s_a)
    #     s_b1 = Variable(self.s_b)
    #     s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
    #     s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
    #     x_a_recon, x_b_recon, x_a_cycle, x_b_cycle, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], [], [], []
    #     content_visual_a,content_visual_b, phase_a_before, phase_b_before,gaussian_before, gaussian_after = [], [], [], [], [], []
    #     with torch.no_grad():
    #         for i in range(x_a.size(0)):
    #             c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
    #             c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
    #             phase_a_before.append(self.phase_content(x_a[i].unsqueeze(0),config))
    #             phase_b_before.append(self.phase_content(x_b[i].unsqueeze(0),config))
    #             pa = self.phase_content(x_a[i].unsqueeze(0),config)
    #             # print(c_a)
    #             # print(c_a.max())
    #             # print(c_a.min())
    #             # print(c_a.mean())
    #             # print(pa.max())
    #             # print(pa.min())
    #             # print(pa.mean())
    #             # visual content
    #             layer_a = random.randint(0, c_a.shape[1] - 1)
    #             layer_b = random.randint(0, c_b.shape[1] - 1)
    #             input_a = c_a[0, layer_a,:,:].unsqueeze(0).unsqueeze(1)
    #             input_b = c_b[0, layer_b,:,:].unsqueeze(0).unsqueeze(1)

    #             input_a = F.interpolate(input_a, scale_factor = 4, mode='bicubic', align_corners=True)
    #             input_b = F.interpolate(input_b, scale_factor = 4, mode='bicubic', align_corners=True)
    #             # print(input_a.shape)
    #             # print(input_a.max())
    #             # print(input_b.max())
    #             # print(input_a.min())
    #             # print(input_b.min())
    #             content_visual_a.append(input_a/input_a.max())
    #             content_visual_b.append(input_b/input_b.max())
    #             x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
    #             x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
    #             # encode again
    #             c_b_recon1, s_a_recon1 = self.gen_a.encode(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
    #             c_a_recon1, s_b_recon1 = self.gen_b.encode(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
    #             # decode again (if needed)
    #             x_a_cycle.append(self.gen_a.decode(c_a_recon1, s_a_fake))
    #             x_b_cycle.append(self.gen_b.decode(c_b_recon1, s_b_fake))
    #             x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
    #             x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
    #             x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
    #             x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
    #         phase_a_before, phase_b_before = torch.cat(phase_a_before), torch.cat(phase_b_before)
    #         content_visual_a, content_visual_b = torch.cat(content_visual_a), torch.cat(content_visual_b)
    #         x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
    #         x_a_cycle, x_b_cycle = torch.cat(x_a_cycle), torch.cat(x_b_cycle)
    #         x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
    #         x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
    #     self.train()
    #     return x_a,phase_a_before, x_a_recon,x_a_cycle,content_visual_a, x_ab1, x_ab2, x_b, phase_b_before, x_b_recon,x_b_cycle,content_visual_b, x_ba1, x_ba2

    def mask_img_tocolor(self, imgpath, savepath):
        showimga = cv2.imread(imgpath)
        # print(showimga.shape)
        # showimga = cv2.cvtColor(showimga,cv2.COLOR_BGR2RGB)
        # print('lalalala',np.unique(showimga[:,:,0]))
        showimga = np.around(np.array((showimga/255)*5))
        # print(showimga.shape)
        showimga = showimga[:, :, 0]
        # print(np.unique(showimga))

        img2show = np.zeros(
            (showimga.shape[0], showimga.shape[1], 3)).astype(np.int)
        img2show[showimga == 0] = [0, 0, 0]
        img2show[showimga == 1] = [0, 255, 255]
        img2show[showimga == 2] = [255, 255, 0]
        img2show[showimga == 3] = [0, 255, 0]
        img2show[showimga == 4] = [250, 58, 196]
        img2show[showimga == 5] = [0, 0, 255]
        # img_RGB = cv2.cvtColor(img2show, cv2.COLOR_BGR2RGB)
        # cv2 saves image as a BGR format by default, for example, here [0,0,255] is red !!!!!!
        cv2.imwrite(savepath, img2show)

    def validate_mask(self, output):
        mask = torch.argmax(output, 1)
        # print(mask.shape)
        # print('torch.unique(mask)',torch.unique(mask))
        return mask

    def mask2color_single(self, mask):

        maskarray = np.array(mask.cpu().squeeze())  # hw
        # print(maskarray.shape)
        color_dict = {0: [0, 0, 0], 1: [255, 255, 0], 2: [0, 255, 255], 3: [0, 255, 0],
                      4: [196, 58, 250], 5: [255, 0, 0]}
        mask2color = np.zeros(
            (maskarray.shape[0], maskarray.shape[1], 3)).astype(int)
        for k, v in color_dict.items():
            mask2color[maskarray == k] = v
        # maskarray = np.expand_dims(maskarray,-1).repeat(3,-1)
        masktensor = torch.from_numpy(mask2color)
        # the image value ranges within [-1,1], we should also made the min in mask to -1 to visualize normally
        # masktensor = masktensor.permute(2, 0, 1).unsqueeze(
        #     0).cuda().float().div(255. / 2).sub(1)
        masktensor = masktensor.permute(2, 0, 1).unsqueeze(
            0).cuda().float().div(255.)

        return masktensor

    def sample_mask(self, x_a, x_b):
        # draw image purely, so I didn't set self.eval and self.train
        x_a_recon, x_b_recon, x_a_cycle, x_b_cycle, x_ba1, x_ba2, x_ab1, x_ab2 = [
        ], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):

            img2showa = x_a[i].unsqueeze(0)
            # print('label_unique:',torch.unique(img2showa[0,:,:]))
            img2showb = x_b[i].unsqueeze(0)
            x_a_recon.append(img2showa)
            x_b_recon.append(img2showb)
            x_a_cycle.append(img2showa)
            x_b_cycle.append(img2showb)
            x_ba1.append(img2showb)
            x_ba2.append(img2showb)
            x_ab1.append(img2showa)
            x_ab2.append(img2showa)
            # content_visual_a.append(img2showa)
            # content_visual_b.append(img2showb)
            # phase_a_before.append(img2showa)
            # phase_b_before.append(img2showb)
        # print(x_a_recon[0].dtype)
        # phase_a_before, phase_b_before = torch.cat(phase_a_before), torch.cat(phase_b_before)
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_a_cycle, x_b_cycle = torch.cat(x_a_cycle), torch.cat(x_b_cycle)
        # print('x_a_cycle.shape', x_a_cycle.shape) [12,3,224,192]
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        # content_visual_a, content_visual_b = torch.cat(content_visual_a), torch.cat(content_visual_b)
        return x_a_recon, x_a_recon, x_a_cycle, x_ab1, x_ab2, x_b_recon, x_b_recon, x_b_cycle, x_ba1, x_ba2

    def sample_mask_seg(self, x_a, x_b):

        x_a_recon, x_b_recon, x_a_cycle, x_b_cycle, x_ba1, x_ba2, x_ab1, x_ab2 = [
        ], [], [], [], [], [], [], []
        # content_visual_a,content_visual_b, phase_a_before, phase_b_before,gaussian_before, gaussian_after = [], [], [], [], [], []

        for i in range(x_a.size(0)):
            # shape BCHW
            img2showa = x_a[i].unsqueeze(0)
            # print('label_unique:',torch.unique(img2showa[0,:,:]))
            img2showb = x_b[i].unsqueeze(0)
            x_a_recon.append(img2showa)
            x_b_recon.append(img2showb)
            x_a_cycle.append(img2showa)
            x_b_cycle.append(img2showb)
            x_ba1.append(img2showb)
            x_ba2.append(img2showb)
            x_ab1.append(img2showa)
            x_ab2.append(img2showa)
            # content_visual_a.append(img2showa)
            # content_visual_b.append(img2showb)
            # phase_a_before.append(img2showa)
            # phase_b_before.append(img2showb)
        # print(x_a_recon[0].dtype)
        # phase_a_before, phase_b_before = torch.cat(phase_a_before), torch.cat(phase_b_before)
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_a_cycle, x_b_cycle = torch.cat(x_a_cycle), torch.cat(x_b_cycle)
        # print('x_a_cycle.shape', x_a_cycle.shape)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        # content_visual_a, content_visual_b = torch.cat(content_visual_a), torch.cat(content_visual_b)
        return x_a_recon, x_a_recon, x_a_recon, x_a_cycle, x_ab1, x_ab1, x_ab2, x_b_recon, x_b_recon, x_b_recon, x_b_cycle, x_ba1, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # print('self.training dis_update:', self.training)
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)  # [1,3,256,256]
        x_ab = self.gen_b.decode(c_a, s_b)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
            hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.seg_scheduler is not None:
            self.seg_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        resume_iter = hyperparameters['resume_iter']
        last_model_name = get_model_list(
            checkpoint_dir, "gen", hyperparameters['snapshot_save_iter'], resume_iter)
        state_dict = torch.load(last_model_name, weights_only=False)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        print(last_model_name)
        iterations = int(last_model_name[-8:-3])
        # Load discriminators
        last_model_name = get_model_list(
            checkpoint_dir, "dis", hyperparameters['snapshot_save_iter'], resume_iter)
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def resume_with_seg(self, checkpoint_dir, hyperparameters):
        # Load generators
        resume_iter = hyperparameters['resume_iter']
        last_model_name = get_model_list(
            checkpoint_dir, "gen", hyperparameters['snapshot_save_iter'], resume_iter)
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        print(last_model_name)
        iterations = int(last_model_name[-8:-3])
        # Load discriminators
        last_model_name = get_model_list(
            checkpoint_dir, "dis", hyperparameters['snapshot_save_iter'], resume_iter)
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, hyperparameters['step_size'], iterations)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, hyperparameters['step_size'], iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def test_load_model_cutmix(self, checkpoint_dir, numpt=-1, netname='seg', netkind='seg'):
        # Load generators or segmentors
        last_model_name = get_test_model_list(
            checkpoint_dir, netname, numpt=numpt)
        print(last_model_name)
        state_dict = torch.load(last_model_name, weights_only=False)
        if netkind == 'gen':
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])
        if netkind == 'seg':
            self.eval_net = torch.load(last_model_name, weights_only=False)
            print('load seg model:%s' % last_model_name)
        basename = os.path.basename(last_model_name)
        # iterations = int(basename[-20:-15])
        iterations = basename
        print('Test from iteration %s' % iterations)
        print('Test model name:', last_model_name)
        return iterations, last_model_name

    def test_load_model_cutmix_student(self, checkpoint_dir, numpt=-1, netname='seg', netkind='seg'):
        # Load generators or segmentors
        last_model_name = get_test_model_list(
            checkpoint_dir, netname, numpt=numpt)
        print(last_model_name)
        # 首先加载整个字典
        checkpoint = torch.load(last_model_name)
        if netkind == 'gen':
            self.gen_a.load_state_dict(checkpoint['a'])
            self.gen_b.load_state_dict(checkpoint['b'])
        if netkind == 'seg':
            self.seg_student.load_state_dict(checkpoint['studentnet'])
            print('load seg model:%s' % last_model_name)
        basename = os.path.basename(last_model_name)
        # iterations = int(basename[-20:-15])
        iterations = basename
        print('Test from iteration %s' % iterations)
        print('Test model name:', last_model_name)
        return iterations
    
    def get_gaussian_blur(self, x, k, stride=1, padding=0):
        res = []
        x = F.pad(x, (padding, padding, padding, padding),
                  mode='constant', value=0)
        for xx in x.split(1, 1):
            res.append(F.conv2d(xx, k, stride=stride, padding=0))
        return torch.cat(res, 1)

    def get_low_freq(self, im, gauss_kernel):
        padding = (gauss_kernel.shape[-1] - 1) // 2
        low_freq = self.get_gaussian_blur(im, gauss_kernel, padding=padding)
        return low_freq

    def gaussian_blur(self, x, k, stride=1, padding=0):
        res = []
        x = F.pad(x, (padding, padding, padding, padding),
                  mode='constant', value=0)
        for xx in x.split(1, 1):
            res.append(F.conv2d(xx, k, stride=stride, padding=0))
        return torch.cat(res, 1)

    def get_gaussian_kernel(self, size=3):
        kernel = cv2.getGaussianKernel(size, 0).dot(
            cv2.getGaussianKernel(size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def find_fake_freq(self, im, gauss_kernel, index=None):
        padding = (gauss_kernel.shape[-1] - 1) // 2
        low_freq = self. gaussian_blur(im, gauss_kernel, padding=padding)
        im_gray = im[:, 0, ...] * 0.299 + \
            im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
        im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
        low_gray = self.gaussian_blur(im_gray, gauss_kernel, padding=padding)
        return im_gray - low_gray

    def save(self, snapshot_dir, iterations, config):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        seg_name = os.path.join(snapshot_dir, 'seg_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')

        if (iterations + 1) < config['pre_train_before_a_seg']:
            torch.save({'a': self.gen_a.state_dict(),
                       'b': self.gen_b.state_dict()}, gen_name)
            torch.save({'a': self.dis_a.state_dict(),
                       'b': self.dis_b.state_dict()}, dis_name)
            torch.save({'gen': self.gen_opt.state_dict(),
                       'dis': self.dis_opt.state_dict()}, opt_name)
        else:
            if (iterations + 1) < config['pre_train_before_b_seg']:
                torch.save({'a': self.gen_a.state_dict(),
                           'b': self.gen_b.state_dict()}, gen_name)
                torch.save({'a': self.dis_a.state_dict(),
                           'b': self.dis_b.state_dict()}, dis_name)
                torch.save({'a': self.seg_a.state_dict()}, seg_name)
                torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(
                ), 'seg_a': self.seg_opt_a.state_dict()}, opt_name)

            else:
                torch.save({'a': self.gen_a.state_dict(),
                           'b': self.gen_b.state_dict()}, gen_name)
                torch.save({'a': self.dis_a.state_dict(),
                           'b': self.dis_b.state_dict()}, dis_name)
                torch.save({'a': self.seg_a.state_dict(),
                           'b': self.seg_b.state_dict()}, seg_name)
                torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(
                ), 'seg_a': self.seg_opt_a.state_dict(), 'seg_b': self.seg_opt_b.state_dict()}, opt_name)

    def save_cp(self, snapshot_dir, iterations, config):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        seg_name = os.path.join(snapshot_dir, 'seg_%08d.pt' % (iterations + 1))
        seg_name2 = os.path.join(
            snapshot_dir, 'segv2_%08d.pth' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')

        if (iterations + 1) < config['pre_train_before_seg']:
            torch.save({'a': self.gen_a.state_dict(),
                       'b': self.gen_b.state_dict()}, gen_name)
            torch.save({'a': self.dis_a.state_dict(),
                       'b': self.dis_b.state_dict()}, dis_name)
            torch.save({'gen': self.gen_opt.state_dict(),
                       'dis': self.dis_opt.state_dict()}, opt_name)
        else:

            torch.save({'a': self.gen_a.state_dict(),
                        'b': self.gen_b.state_dict()}, gen_name)
            torch.save({'a': self.dis_a.state_dict(),
                        'b': self.dis_b.state_dict()}, dis_name)
            torch.save({'evalnet': self.eval_net.state_dict(),
                        'studentnet': self.seg_student.state_dict()}, seg_name)
            torch.save(self.eval_net, seg_name2)
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(
            ), 'seg_stu': self.student_opt.state_dict(), 'seg_tea': self.teacher_opt}, opt_name)

    def save_dice_checkpoint(self, iterations, best_dice, snapshot_dir, logger=None):
        """
        Args:

            best_dice (bool):
            snapshot_dir (string): directory where the checkpoint are to be saved
        """

        def log_info(message):
            if logger is not None:
                logger.info(message)

        if not os.path.exists(snapshot_dir):
            os.mkdir(snapshot_dir)

        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(
            snapshot_dir, 'gen_%08d_best%.4f.pt' % (iterations + 1, best_dice))
        dis_name = os.path.join(
            snapshot_dir, 'dis_%08d_best%.4f.pt' % (iterations + 1, best_dice))
        seg_name = os.path.join(
            snapshot_dir, 'seg_%08d_best%.4f.pt' % (iterations + 1, best_dice))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(),
                   'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(),
                   'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'a': self.seg_a.state_dict(),
                   'b': self.seg_b.state_dict()}, seg_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(
        ), 'seg_a': self.seg_opt_a.state_dict(), 'seg_b': self.seg_opt_b.state_dict()}, opt_name)

    # def save_dice_checkpoint_cutmix(self, iterations, best_dice, snapshot_dir, post_str, logger=None):
    #     """
    #     Args:

    #         best_dice (bool):
    #         snapshot_dir (string): directory where the checkpoint are to be saved
    #     """

    #     def log_info(message):
    #         if logger is not None:
    #             logger.info(message)

    #     if not os.path.exists(snapshot_dir):
    #         os.mkdir(snapshot_dir)

    #     # Save generators, discriminators, and optimizers
    #     gen_name = os.path.join(
    #         snapshot_dir, '%s_gen_%08d_best%.4f.pt' % (post_str, iterations + 1, best_dice))
    #     dis_name = os.path.join(
    #         snapshot_dir, '%s_dis_%08d_best%.4f.pt' % (post_str, iterations + 1, best_dice))
    #     seg_name = os.path.join(
    #         snapshot_dir, '%s_seg_%08d_best%.4f.pt' % (post_str, iterations + 1, best_dice))
    #     seg_namev2 = os.path.join(
    #         snapshot_dir, '%s_segv2_%08d_best%.4f.pth' % (post_str, iterations + 1, best_dice))
    #     opt_name = os.path.join(snapshot_dir, '%s_optimizer.pt' % post_str)
    #     torch.save({'a': self.gen_a.state_dict(),
    #                'b': self.gen_b.state_dict()}, gen_name)
    #     torch.save({'a': self.dis_a.state_dict(),
    #                'b': self.dis_b.state_dict()}, dis_name)
    #     torch.save({'evalnet': self.eval_net.state_dict(),
    #                 'studentnet': self.seg_student.state_dict()}, seg_name)
    #     torch.save(self.eval_net, seg_namev2)
    #     # torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(
    #     # ), 'seg_stu': self.student_opt.state_dict(), 'seg_tea': self.teacher_opt}, opt_name)

    def save_dice_checkpoint_cutmix(self, iterations, best_dice, snapshot_dir, post_str, logger=None):
        """
        Args:

            best_dice (bool):
            snapshot_dir (string): directory where the checkpoint are to be saved
        """

        def log_info(message):
            if logger is not None:
                logger.info(message)

        if not os.path.exists(snapshot_dir):
            os.mkdir(snapshot_dir)

        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(
            snapshot_dir, '%s_gen_%08d_best%.4f.pt' % (post_str, iterations + 1, best_dice))
        # dis_name = os.path.join(
        #     snapshot_dir, '%s_dis_%08d_best%.4f.pt' % (post_str, iterations + 1, best_dice))
        seg_name = os.path.join(
            snapshot_dir, '%s_seg_%08d_best%.4f.pt' % (post_str, iterations + 1, best_dice))
        seg_namev2 = os.path.join(
            snapshot_dir, '%s_segv2_%08d_best%.4f.pth' % (post_str, iterations + 1, best_dice))
        seg_name_stu = os.path.join(
            snapshot_dir, '%s_segstu_%08d_best%.4f.pth' % (post_str, iterations + 1, best_dice))

        # opt_name = os.path.join(snapshot_dir, '%s_optimizer.pt' % post_str)
        torch.save({'a': self.gen_a.state_dict(),
                   'b': self.gen_b.state_dict()}, gen_name)
        # torch.save({'a': self.dis_a.state_dict(),
        #            'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'evalnet': self.eval_net.state_dict(),
                    'studentnet': self.seg_student.state_dict()}, seg_name)
        torch.save(self.eval_net, seg_namev2)
        torch.save(self.seg_student, seg_name_stu)
        files = glob.glob(snapshot_dir + '/*')
        files.sort(key=lambda x: os.path.getmtime(x))
        if len(files) > 30:
            for f in files[:-30]:
                os.remove(f)
        # torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(
        # ), 'seg_stu': self.student_opt.state_dict(), 'seg_tea': self.teacher_opt}, opt_name)


# QuadTree Definition
# # CHW
class Node():
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self):
        return self.points

    def get_points(self, img):
        return img[:, self.x0:self.x0 + self.get_width(), self.y0:self.y0+self.get_height()]

    def get_error(self, img):
        pixels = self.get_points(img)
        r_avg = torch.mean(pixels[0, :, :])
        r_mse = torch.square(torch.sub(pixels[0, :, :], r_avg)).mean()

        g_avg = torch.mean(pixels[1, :, :])
        g_mse = torch.square(torch.sub(pixels[1, :, :], g_avg)).mean()

        b_avg = torch.mean(pixels[2, :, :])
        b_mse = torch.square(torch.sub(pixels[2, :, :], b_avg)).mean()
        # The human eye is most sensitive to green and least sensitive to blue
        e = r_mse * 0.2989 + g_mse * 0.5870 + b_mse * 0.1140
        # the variance
        return e


class QTree():
    def __init__(self, stdThreshold, minPixelSize, img):
        self.threshold = stdThreshold
        self.min_size = minPixelSize
        self.minPixelSize = minPixelSize
        self.img = img
        self.root = Node(0, 0, img.shape[1], img.shape[2])

    def subdivide(self):
        self.recursive_subdivide(self.root, self.threshold,
                                 self.minPixelSize, self.img)

    def gen_quadaug_mask(self, ratio1=0.1, ratio2=0.2):
        # CHWi
        mask_ones1 = torch.ones_like(self.img)
        mask_ones2 = torch.ones_like(self.img)
        # print(mask_ones.dtype)

        c = self.find_children(self.root)
        # print("Number of segments: %d" % len(c))
        select_num1 = math.floor(len(c) * ratio1)
        select_num2 = math.floor(len(c) * ratio2)
        random_select1 = random.sample(c, select_num1)
        random_select2 = random.sample(c, select_num2)
        # print('select nodes:', select_num)
        for n in random_select1:
            mask_ones1[:, n.x0:n.x0 + n.get_width(), n.y0:n.y0 +
                       n.get_height()] = 0
        for n in random_select2:
            mask_ones2[:, n.x0:n.x0 + n.get_width(), n.y0:n.y0 +
                       n.get_height()] = 0
        # print(mask_ones.shape)
        return mask_ones1, mask_ones2

    def recursive_subdivide(self, node, k, minPixelSize, img):

        if node.get_error(img) <= k:
            return
        w_1 = int(math.floor(node.width/2))
        w_2 = int(math.ceil(node.width/2))
        h_1 = int(math.floor(node.height/2))
        h_2 = int(math.ceil(node.height/2))

        if w_1 <= minPixelSize or h_1 <= minPixelSize:
            return
        x1 = Node(node.x0, node.y0, w_1, h_1)  # top left
        self.recursive_subdivide(x1, k, minPixelSize, img)

        x2 = Node(node.x0, node.y0 + h_1, w_1, h_2)  # btm left
        self.recursive_subdivide(x2, k, minPixelSize, img)

        x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)  # top right
        self.recursive_subdivide(x3, k, minPixelSize, img)

        x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2)  # btm right
        self.recursive_subdivide(x4, k, minPixelSize, img)

        node.children = [x1, x2, x3, x4]

    def find_children(self, node):
        # find all leaf nodes
        if not node.children:
            return [node]
        else:
            children = []
            for child in node.children:
                children += (self.find_children(child))
        return children
