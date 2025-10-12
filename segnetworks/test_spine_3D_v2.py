"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
import job_helper
import torchvision
import click
import os
import numpy as np
from medpy.io import load, save


def dice_all_class(prediction, target, class_num=20, eps=1e-10):
    '''

    :param prediction: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param target: numpy array with shape of [D, H, W], [H, W], or [H, W, D]
    :param class_num:
    :param eps:
    :return:
    '''
    dices = []
    for i in range(1, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0)
        prediction_per_class = np.where(prediction == i, 1, 0)
        dice = dice_per_class(prediction_per_class, target_per_class)
        dices.append(dice)
    return np.mean(dices)


def dice_per_class(prediction, target, eps=1e-10):
    '''

    :param prediction: numpy array
    :param target: numpy array
    :param eps:
    :return:
    '''
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    return (2. * intersect / (np.sum(prediction) + np.sum(target) + eps))


def prepare_test_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory_a = os.path.join(output_directory, 're_3D_a')
    checkpoint_directory_b = os.path.join(output_directory, 're_3D_b')
    if not os.path.exists(checkpoint_directory_a):
        print("Creating directory: {}".format(checkpoint_directory_a))
        os.makedirs(checkpoint_directory_a)
    if not os.path.exists(checkpoint_directory_b):
        print("Creating directory: {}".format(checkpoint_directory_b))
        os.makedirs(checkpoint_directory_b)
    return checkpoint_directory_a, checkpoint_directory_b, image_directory


@job_helper.job('test_spine_3D_v2', enumerate_job_names=False)
def test_spine_3D_v2(submit_config: job_helper.SubmitConfig, dataset, model, arch, freeze_bn,
                     opt_type, sgd_momentum, sgd_nesterov, sgd_weight_decay,
                     learning_rate, lr_sched, lr_step_epochs, lr_step_gamma, lr_poly_power,
                     teacher_alpha, bin_fill_holes,
                     crop_size, aug_hflip, aug_vflip, aug_hvflip, aug_scale_hung, aug_max_scale, aug_scale_non_uniform, aug_rot_mag,
                     aug_strong_colour, aug_colour_brightness, aug_colour_contrast, aug_colour_saturation, aug_colour_hue,
                     aug_colour_prob, aug_colour_greyscale_prob,
                     mask_mode, mask_prop_range,
                     boxmask_n_boxes, boxmask_fixed_aspect_ratio, boxmask_by_size, boxmask_outside_bounds, boxmask_no_invert,
                     cons_loss_fn, cons_weight, conf_thresh, conf_per_pixel, rampup, unsup_batch_ratio,
                     num_epochs, iters_per_epoch, batch_size,
                     n_sup, n_unsup, n_val, split_seed, split_path, val_seed, save_preds, save_model, num_workers):
    settings = locals().copy()
    del settings['submit_config']

    import os
    import math
    import time
    import itertools
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as tvt
    from architectures import network_architectures
    import torch.utils.data
    from datapipe import datasets
    from datapipe import seg_data, seg_transforms, seg_transforms_cv
    import evaluation
    import optim_weight_ema
    import mask_gen
    import lr_schedules

    if crop_size == '':
        crop_size = None
    else:
        crop_size = [int(x.strip()) for x in crop_size.split(',')]

    torch_device = torch.device('cuda:0')

    # model_name = os.path.splitext(os.path.basename(opts.config))[0]
    day_hour_minute = time.strftime("%d%H%M", time.localtime())
    model_name = 'spine_cutmix' + day_hour_minute
    output_directory = os.path.join(
        "/data1/xr/Spine_MUNIT/outputs", model_name)
    re_3D_directory_a, re_3D_directory_b, image_directory = prepare_test_sub_folder(
        output_directory)

    #
    # Load data sets

    ds_dict = datasets.load_dataset_spine(dataset, n_unsup, split_seed)

    ds_src = ds_dict['ds_src']
    ds_tgt = ds_dict['ds_tgt']
    tgt_val_ndx = ds_dict['val_ndx_tgt']
    src_val_ndx = ds_dict['val_ndx_src'] if ds_src is not ds_tgt else None
    test_ndx = ds_dict['test_ndx_tgt']
    sup_ndx = ds_dict['sup_ndx']
    unsup_ndx = ds_dict['unsup_ndx']

    n_classes = ds_src.num_classes
    root_n_classes = math.sqrt(n_classes)

    if bin_fill_holes and n_classes != 2:
        print('Binary hole filling can only be used with binary (2-class) segmentation datasets')
        return

    print('Loaded data')

    # Build network
    NetClass = network_architectures.seg.get(arch)

    student_net = NetClass(ds_src.num_classes).to(torch_device)

    if model == 'mean_teacher':
        teacher_net = NetClass(ds_src.num_classes).to(torch_device)

        for p in teacher_net.parameters():
            p.requires_grad = False

        teacher_optim = optim_weight_ema.EMAWeightOptimizer(
            teacher_net, student_net, teacher_alpha)
        eval_net = teacher_net
    elif model == 'pi':
        teacher_net = student_net
        teacher_optim = None
        eval_net = student_net
    else:
        print('Unknown model type {}'.format(model))
        return

    # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    # teacher_net.load_state_dict(torch.load(
    #     "/data1/xr/Spine_MUNIT/cutmix-semisup-seg-master/results/train_seg_semisup_cutmix_spine/spine_denseunet_1000_lr0.1_wd5e-4_sclrot_cutmix_cw1.0_semisup_50_run01/preds/cp_models/model_3_0.785.pth"))

    teacher_net = torch.load(
        "/data1/xr/Spine_MUNIT/cutmix-semisup-seg-master/results/train_seg_semisup_cutmix_spine/spine_denseunet_1000_lr0.1_wd5e-4_sclrot_cutmix_cw1.0_semisup_50_run01/preds/cp_models/model_3_0.785.pth")
    eval_net = teacher_net
    BLOCK_SIZE = student_net.BLOCK_SIZE
    NET_MEAN, NET_STD = seg_transforms.get_mean_std(ds_tgt, student_net)

    print('Built network')

    eval_transforms = []

    eval_transforms.append(
        seg_transforms_cv.SegCVTransformNormalizeToTensor(NET_MEAN, NET_STD))

    collate_fn = seg_data.SegCollate(BLOCK_SIZE)

    # Eval pipeline
    src_val_loader, tgt_val_loader, test_loader = datasets.eval_data_pipeline(
        ds_src, ds_tgt, src_val_ndx, tgt_val_ndx, test_ndx, batch_size, collate_fn, NET_MEAN, NET_STD, num_workers)

    # Report setttings
    print('Settings:')
    print(', '.join(['{}={}'.format(key, settings[key])
          for key in sorted(list(settings.keys()))]))

    # Report dataset size
    print('Dataset:')
    print('len(sup_ndx)={}'.format(len(sup_ndx)))
    print('len(unsup_ndx)={}'.format(len(unsup_ndx)))
    if ds_src is not ds_tgt:
        print('len(src_val_ndx)={}'.format(len(tgt_val_ndx)))
        print('len(tgt_val_ndx)={}'.format(len(tgt_val_ndx)))
    else:
        print('len(val_ndx)={}'.format(len(tgt_val_ndx)))
    if test_ndx is not None:
        print('len(test_ndx)={}'.format(len(test_ndx)))

    # if n_sup != -1:
    #     print('sup_ndx={}'.format(sup_ndx.tolist()))

    best_dice_eval = 0

    src_val_iter = iter(
        src_val_loader) if src_val_loader is not None else None
    tgt_val_iter = iter(
        tgt_val_loader) if tgt_val_loader is not None else None
# Start testing
    eval_net.eval()
    tgt_iou_eval = evaluation.EvaluatorIoU(
        ds_tgt.num_classes, bin_fill_holes)
    tgt_dice_eval = evaluation.EvaluatorDice(
        ds_tgt.num_classes)

    dices_3D_b = []
    b_3D_img_slices = []
    b_3D_mask_slices = []
    b_3D_seg_slices = []
    dice_list = []
    for i, batch in enumerate(tgt_val_loader):

        # This is the mean dice of all classes WITHOUT the background 0 of current test data
        if (i + 1) % 45 == 0:
            #  ## draw the image for one patient
            num = int((i + 1) / 45)
            display_index = [j for j in range(45 * (num - 1), 45 * (num))]

            test_display_images_b_one = torch.stack(
                [tgt_val_loader.dataset[i]['image'] for i in display_index]).cuda()
            test_display_masks_b_one = torch.stack(
                [tgt_val_loader.dataset[i]['labels'] for i in display_index]).cuda()
            test_display_index_b_one = torch.stack(
                [tgt_val_loader.dataset[i]['index'] for i in display_index]).cuda()

            namesample = ds_tgt.samplename(test_display_index_b_one[-1])
            x_a1_seg, x_a2_seg, x_a3_seg, x_ab1_seg, x_ab2_seg = [], [], [], [], []
            x_b1_seg, x_b2_seg, x_b3_seg, x_ba1_seg, x_ba2_seg = [], [], [], [], []

            for i in range(test_display_images_b_one.size(0)):
                batch_x = test_display_images_b_one[i].to(
                    torch_device).unsqueeze()
                # torch.Size([1, 3, 224, 224])
                masks_b = test_display_masks_b_one[i].numpy()
                batch_ndx = test_display_index_b_one[i]
                # This is the result after softmax and argmax
                # torch.Size([1, 1, 224, 224])
                logits = eval_net(batch_x)
                # torch.Size([1, 2, 224, 224])
                prob_eval = F.softmax(logits, dim=1)
                outputb = torch.argmax(prob_eval, dim=1).detach().cpu().numpy()
                # (1, 224, 224)

                # masks_b = np.array(masks_b[:, 0, :, :])

                x_b1_seg.append(outputb.unsqueeze())

                x_b_mask.append(masks_b)

            x_b_mask = torch.cat(x_b_mask)
            x_b1_seg = torch.cat(x_b1_seg)

            b_3D_img = test_display_images_b_one[:, 0, :, :]
            b_3D_mask = x_b_mask[:, 0, :, :]
            b1_seg_3D = x_b1_seg[:, 0, :, :]

            #  # we obtain the y and z orientation, here is h w, and we may not need to transpose
            #  ## h w corresponds to z and y, respectively.

            b_3D_img = np.transpose(b_3D_img.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]

            b_3D_mask = np.transpose(b_3D_mask.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]

            b1_seg_3D = np.transpose(b1_seg_3D.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]        # [x, y, z]

            spath_b = os.path.join(
                re_3D_directory_b, 'b_%s' % (namesample[8:11]))
            if not os.path.exists(spath_b):
                os.mkdir(spath_b)

            save(b_3D_img, "%s/b_3D_img.nii.gz" % (spath_b))
            save(b1_seg_3D, "%s/b1_3D.nii.gz" % (spath_b))
            save(b_3D_mask, "%s/b_3D_mask.nii.gz" % (spath_b))

            # compute Dice
            #  dice_3D_a = dice_all_class(x_a_seg_3D,a_3D_mask,config['seg']['n_classes'])
            dice_3D_b = dice_all_class(
                b1_seg_3D, b_3D_mask, 6)
            #  dices_3D_a.append(dice_3D_a)
            dices_3D_b.append(dice_3D_b)

            #  with open('{}/test_a_3d_dice.txt'.format(output_directory), 'a') as f:
            #             f.writelines('patient:{}, 3d_dice:{}\n'.
            #                         format(idx_patient_a, dice_3D_a))
            with open('{}/test_b_3d_dice.txt'.format(output_directory), 'a') as f:
                f.writelines('patient:{}, 3d_dice:{}\n'.
                             format(namesample[8:11], dice_3D_b))

            print('finish one 3D reconstruct!')

    # a_validate_dice_mean_3d = np.mean(np.array(dices_3D_a))
    # a_validate_dice_std_3d = np.std(np.array(dices_3D_a))
    # print('Validate a 3d, mean Dice = %.4f, std Dice = %.4f' % (a_validate_dice_mean_3d, a_validate_dice_std_3d))

    b_validate_dice_mean_3d = np.mean(np.array(dices_3D_b))
    b_validate_dice_std_3d = np.std(np.array(dices_3D_b))
    print('Validate b 3d, mean Dice = %.4f, std Dice = %.4f' %
          (b_validate_dice_mean_3d, b_validate_dice_std_3d))

    print('finish!')


@click.command()
@click.option('--job_desc', type=str, default='')
@click.option('--dataset', type=click.Choice(['camvid', 'cityscapes', 'pascal', 'pascal_aug', 'isic2017', 'spine']),
              default='spine')
@click.option('--model', type=click.Choice(['mean_teacher', 'pi']), default='mean_teacher')
@click.option('--arch', type=str, default='resnet101_deeplab_imagenet')
@click.option('--freeze_bn', is_flag=True, default=False)
@click.option('--opt_type', type=click.Choice(['adam', 'sgd']), default='adam')
@click.option('--sgd_momentum', type=float, default=0.9)
@click.option('--sgd_nesterov', is_flag=True, default=False)
@click.option('--sgd_weight_decay', type=float, default=5e-4)
@click.option('--learning_rate', type=float, default=1e-4)
@click.option('--lr_sched', type=click.Choice(['none', 'stepped', 'cosine', 'poly']), default='none')
@click.option('--lr_step_epochs', type=str, default='')
@click.option('--lr_step_gamma', type=float, default=0.1)
@click.option('--lr_poly_power', type=float, default=0.9)
@click.option('--teacher_alpha', type=float, default=0.99)
@click.option('--bin_fill_holes', is_flag=True, default=False)
@click.option('--crop_size', type=str, default='321,321')
@click.option('--aug_hflip', is_flag=True, default=False)
@click.option('--aug_vflip', is_flag=True, default=False)
@click.option('--aug_hvflip', is_flag=True, default=False)
@click.option('--aug_scale_hung', is_flag=True, default=False)
@click.option('--aug_max_scale', type=float, default=1.0)
@click.option('--aug_scale_non_uniform', is_flag=True, default=False)
@click.option('--aug_rot_mag', type=float, default=0.0)
@click.option('--aug_strong_colour', is_flag=True, default=False)
@click.option('--aug_colour_brightness', type=float, default=0.4)
@click.option('--aug_colour_contrast', type=float, default=0.4)
@click.option('--aug_colour_saturation', type=float, default=0.4)
@click.option('--aug_colour_hue', type=float, default=0.1)
@click.option('--aug_colour_prob', type=float, default=0.8)
@click.option('--aug_colour_greyscale_prob', type=float, default=0.2)
@click.option('--mask_mode', type=click.Choice(['zero', 'mix']), default='mix')
@click.option('--mask_prop_range', type=str, default='0.5')
@click.option('--boxmask_n_boxes', type=int, default=1)
@click.option('--boxmask_fixed_aspect_ratio', is_flag=True, default=False)
@click.option('--boxmask_by_size', is_flag=True, default=False)
@click.option('--boxmask_outside_bounds', is_flag=True, default=False)
@click.option('--boxmask_no_invert', is_flag=True, default=False)
@click.option('--cons_loss_fn', type=click.Choice(['var', 'bce', 'kld', 'logits_var', 'logits_smoothl1']), default='var')
@click.option('--cons_weight', type=float, default=1.0)
@click.option('--conf_thresh', type=float, default=0.97)
@click.option('--conf_per_pixel', is_flag=True, default=False)
@click.option('--rampup', type=int, default=-1)
@click.option('--unsup_batch_ratio', type=int, default=1)
@click.option('--num_epochs', type=int, default=300)
@click.option('--iters_per_epoch', type=int, default=-1)
@click.option('--batch_size', type=int, default=1)
@click.option('--n_sup', type=int, default=100)
@click.option('--n_unsup', type=int, default=-1)
@click.option('--n_val', type=int, default=-1)
@click.option('--split_seed', type=int, default=12345)
@click.option('--split_path', type=click.Path(readable=True, exists=True))
@click.option('--val_seed', type=int, default=131)
@click.option('--save_preds', is_flag=True, default=True)
@click.option('--save_model', is_flag=True, default=True)
@click.option('--num_workers', type=int, default=0)
def experiment(job_desc, dataset, model, arch, freeze_bn,
               opt_type, sgd_momentum, sgd_nesterov, sgd_weight_decay,
               learning_rate, lr_sched, lr_step_epochs, lr_step_gamma, lr_poly_power,
               teacher_alpha, bin_fill_holes,
               crop_size, aug_hflip, aug_vflip, aug_hvflip, aug_scale_hung, aug_max_scale, aug_scale_non_uniform, aug_rot_mag,
               aug_strong_colour, aug_colour_brightness, aug_colour_contrast, aug_colour_saturation, aug_colour_hue,
               aug_colour_prob, aug_colour_greyscale_prob,
               mask_mode, mask_prop_range,
               boxmask_n_boxes, boxmask_fixed_aspect_ratio, boxmask_by_size, boxmask_outside_bounds, boxmask_no_invert,
               cons_loss_fn, cons_weight, conf_thresh, conf_per_pixel, rampup, unsup_batch_ratio,
               num_epochs, iters_per_epoch, batch_size,
               n_sup, n_unsup, n_val, split_seed, split_path, val_seed, save_preds, save_model, num_workers):
    params = locals().copy()

    test_spine_3D_v2.submit(**params)


if __name__ == '__main__':
    experiment()
    sys.exit('Finish testing!')
