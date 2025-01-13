import numpy as np
import cv2
import glob
from segnetworks.utils import dice_all_class,mean_assds_all_class
import dataset_final
import time
import shutil
import tensorboardX
import sys
import argparse
from utils_new import get_all_data_loaders, prepare_sub_folder, prepare_test_sub_folder, write_html, write_loss, get_config, write_2images, write_2images_TTA,write_2images_single, Timer

from torch.autograd import Variable
from medpy.io import load, save
from trainer_strongbase_origin_cutmix import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# checkpoint directory path
# cp_base_cutmix = '/media/gdp/date/gxr/spine_MUNIT/outputs/NEW_baseline_origin_cutmix051037/b_best_checkpoints/'
# cp_base_cutmix_pai = '/media/gdp/date/gxr/spine_MUNIT/outputs/NEW_baseline_origin_cutmix_pai051033/b_best_checkpoints/'
# cp_base_cutmix_pai_quad = '/media/gdp/date/gxr/spine_MUNIT/outputs/NEW_baseline_origin_cutmix_pai_quad051028/b_best_checkpoints/'
# cp_base_cutmix_pai_quad_mean = '/media/gdp/date/gxr/spine_MUNIT/outputs/NEW_baseline_origin_cutmix_pai_quad_mean061239/b_best_checkpoints/'
# cp_base_cutmix_pai_mean = '/media/gdp/date/gxr/spine_MUNIT/outputs/NEW_baseline_origin_cutmix_pai_bottleneck512/b_best_checkpoints/'
CYCMIS_model = '/media/gdp/date/gxr/spine_MUNIT/outputs/CyCMIS041816/b_best_checkpoints/'
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='./configs/config_new_base.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="outputs path")
parser.add_argument("--checkpoint_directory", type=str,
                    default=CYCMIS_model)
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
resume_munit = config['resume_munit']
resume_dir = config['resume_dir']
config['vgg_model_path'] = opts.output_path
random_range = config['random_range']
# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)

else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()

# please remember that glob.glob may shuffle the file index
train_A_path = sorted(glob.glob(os.path.join(
    config['data_root'], config['train_A_dir'], '*')))
train_B_path = sorted(glob.glob(os.path.join(
    config['data_root'], config['train_B_dir'], '*')))
test_A_path = sorted(glob.glob(os.path.join(
    config['data_root'], config['test_A_dir'], '*')))
test_B_path = sorted(glob.glob(os.path.join(
    config['data_root'], config['test_B_dir'], '*')))

print(os.path.basename(train_B_path[0]).split('_'))
print(train_B_path[:12])
train_A_path.sort(key = lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[3])))
train_B_path.sort(key = lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[3])))
test_A_path.sort(key = lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[3])))
test_B_path.sort(key = lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[3])))
print(train_A_path[:12])
# print(len(test_B_path))
train_num_A = len(train_A_path)
train_num_B = len(train_B_path)
print('train_num_A:', train_num_A)
print('train_num_B:', train_num_B)
train_patient_num_A = config['train_patient_num_A']
train_patient_num_B = config['train_patient_num_B']
test_patient_num_B = config['test_patient_num_B']
patient_slicesA = config['patient_slicesA']
patient_slicesB = config['patient_slicesB']
# maxnum = max(train_num_A, train_num_B)
# train_pair_num = maxnum
if config['target'] == "B":
    train_pair_num = train_num_B
    print('train_pair_num:', train_pair_num)
    test_pair_num = len(test_B_path)
    print('test_pair_num:', test_pair_num)
else:
    train_pair_num = train_num_A
    test_pair_num = len(test_A_path)
v_my_sampler = dataset_final.ValidateRandomSampler(test_pair_num)
v_my_batch_sampler = torch.utils.data.BatchSampler(
    v_my_sampler, batch_size=config['batch_size'], drop_last=False)
validate_dataset = dataset_final.SpineDataSet(
    config, test_A_path, test_B_path, train_patient_num_A, test_patient_num_B, random_range, True, phase='test')
validate_dataloader = torch.utils.data.DataLoader(
    validate_dataset, batch_sampler=v_my_batch_sampler)

# Setup logger and output folders
# model_name = os.path.splitext(os.path.basename(opts.config))[0]
day_hour_minute = time.strftime("%d%H%M", time.localtime())
model_name = config['test_model_name'] + day_hour_minute
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
re_3D_directory_a, re_3D_directory_b, image_directory = prepare_test_sub_folder(
    output_directory)
# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))
gauss_kernel = trainer.get_gaussian_kernel(3).cuda()
with open('{}/train_glob_data_B.txt'.format(output_directory), 'a') as f:
    f.writelines('patient:{}\n'.format(train_B_path))
with open('{}/test_glob_data_B.txt'.format(output_directory), 'a') as f:
    f.writelines('patient:{}\n'.format(test_B_path))
# Start testing
print(opts.checkpoint_directory)
test_net_str = 'ori_'
# test_net_str = 'mean_coloraug_'
iterations = trainer.test_load_model_cutmix(
    opts.checkpoint_directory, numpt=-1, netname= test_net_str + 'segv2', netkind='seg')
iterations = trainer.test_load_model_cutmix(
    opts.checkpoint_directory, numpt=-1, netname= test_net_str + 'gen', netkind='gen')

with torch.no_grad():

    dices_3D_b = []
    assds_3D_b = []
    dices_3D_b_aug = []
    assds_3D_b_aug = []
    dices_3D_b_augv2 = []
    assds_3D_b_augv2 = []
    dices_3D_b_augv3 = []
    assds_3D_b_augv3 = []
    next_start_index = 0
    for j, one_b_path in enumerate(test_B_path):
        print(one_b_path)
        basename_B = os.path.basename(one_b_path)
        split_basename = basename_B.split('_')
        slice_num = int(split_basename[3])
        patient_num = int(split_basename[1])
        total_slice_num_current = int(split_basename[5][:-4])
        if slice_num == total_slice_num_current:
            threed_dir_path = one_b_path.replace('2D', '3D')
            split_3d_path = threed_dir_path.split('/')
            idx_patient_b = validate_dataloader.dataset[j][-3]
            idx_patient_a = validate_dataloader.dataset[j][-4]
            threed_file_path = os.path.join(split_3d_path[0],split_3d_path[1],split_3d_path[2],split_3d_path[3],'Case%s.nii.gz'%patient_num)
            print(threed_file_path)
            img_3D,h = load(threed_file_path)
            end_index = test_B_path.index(one_b_path)
            current_patient_paths = test_B_path[next_start_index:end_index + 1]
            
            display_index = [j for j in range(next_start_index, end_index + 1)]
            print(display_index)
            
            test_display_images_a_one = torch.stack(
                [validate_dataloader.dataset[i][0] for i in display_index]).cuda()
            test_display_masks_a_one = torch.stack(
                [validate_dataloader.dataset[i][2] for i in display_index]).cuda()

            test_display_images_b_one = torch.stack(
                [validate_dataloader.dataset[i][3] for i in display_index]).cuda()
            test_display_masks_b_one = torch.stack(
                [validate_dataloader.dataset[i][5] for i in display_index]).cuda()

            test_image_outputs_one = trainer.sample_test_MR_ours(test_display_images_a_one, test_display_masks_a_one, test_display_images_b_one, test_display_masks_b_one)
            write_2images_single(test_image_outputs_one[4:-4], len(display_index), image_directory, 'test_%s' % (idx_patient_b), 'img')
            write_2images_single(test_image_outputs_one[4:10], len(display_index), image_directory, 'test_%s_simple' % (idx_patient_b), 'img')
            write_2images_single(test_image_outputs_one[0:4], len(display_index), image_directory, 'test_%s_a' % (idx_patient_a), 'img')

            test_masks_outputs = trainer.sample_mask_seg(test_display_masks_a_one, test_display_masks_b_one)
            write_2images_single(test_masks_outputs[0:4], len(display_index), image_directory, 'test_mask_a_%s' % (idx_patient_a),'mask')
            write_2images_single(test_masks_outputs[4:10], len(display_index), image_directory, 'test_mask_b_%s' % (idx_patient_b),'mask')

            bottoma = cv2.imread('%s/%s.png' % (image_directory, 'test_%s_a' % (idx_patient_a)))
            uppera = cv2.imread('%s/%s.png' % (image_directory, 'test_mask_a_%s' % (idx_patient_a)))
            overlappinga = cv2.addWeighted(bottoma,1,uppera,0.4,0)
            cv2.imwrite('%s/%s.png' % (image_directory, 'maskover_a_%s' % (idx_patient_a)), overlappinga)

            bottomb = cv2.imread('%s/%s.png' % (image_directory, 'test_%s_simple' % (idx_patient_b)))
            upperb = cv2.imread('%s/%s.png' % (image_directory, 'test_mask_b_%s' % (idx_patient_b)))
            overlappingb = cv2.addWeighted(bottomb,1,upperb,0.4,0)
            cv2.imwrite('%s/%s.png' % (image_directory, 'maskover_b_%s' % (idx_patient_b)), overlappingb)

            a_3D_img = test_image_outputs_one[0][:, 0, :, :]
            b_3D_img = test_image_outputs_one[4][:, 0, :, :]
            b_3D_mask = test_display_masks_b_one[:, 0, :, :]
            ab_3D_img = test_image_outputs_one[1][:, 0, :, :]
            b_seg_3D = test_image_outputs_one[-1]
            x_b_segaug_3D = test_image_outputs_one[-2]
            x_b_segaug_3D_v2 = test_image_outputs_one[-3]
            x_b_segaug_3D_v3 = test_image_outputs_one[-4]
            #  # we obtain the y and z orientation, here is h w, and we may not need to transpose
            #  ## h w corresponds to z and y, respectively.

            b_3D_img = np.transpose(b_3D_img.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]
            a_3D_img = np.transpose(a_3D_img.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]
            ab_3D_img = np.transpose(ab_3D_img.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]
            b_3D_mask = np.transpose(b_3D_mask.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]

            b_seg_3D = np.transpose(b_seg_3D.cpu().float().numpy(), (0, 2, 1))[
                :, :, ::-1]        # [x, y, z]

            x_b_segaug_3D = np.transpose(
                x_b_segaug_3D.cpu().float().numpy(), (0, 2, 1))[:, :, ::-1]

            x_b_segaug_3D_v2 = np.transpose(
                x_b_segaug_3D_v2.cpu().float().numpy(), (0, 2, 1))[:, :, ::-1]

            x_b_segaug_3D_v3 = np.transpose(
                x_b_segaug_3D_v3.cpu().float().numpy(), (0, 2, 1))[:, :, ::-1]

            spath_b = os.path.join(re_3D_directory_b, 'b_%s' % (idx_patient_b))
            if not os.path.exists(spath_b):
                os.mkdir(spath_b)
            save(a_3D_img, "%s/a_3D_img.nii.gz" % (spath_b),h)
            save(ab_3D_img, "%s/ab_3D_img.nii.gz" % (spath_b),h)
            save(b_3D_img, "%s/b_3D_img.nii.gz" % (spath_b),h)
            save(b_seg_3D, "%s/b_seg_3D.nii.gz" % (spath_b),h)
            save(b_3D_mask, "%s/b_3D_mask.nii.gz" % (spath_b),h)
            save(x_b_segaug_3D, "%s/x_b_segaug_3D.nii.gz" % (spath_b),h)
            # save(x_b_segaug_3D_v2, "%s/x_b_segaug_3D_v2.nii.gz" % (spath_b),h)


            dice_3D_b = dice_all_class(
                b_seg_3D, b_3D_mask, config['seg']['n_classes'])
            dices_3D_b.append(dice_3D_b)
            
            assd_3D_b = mean_assds_all_class(b_seg_3D, b_3D_mask, class_num=config['seg']['n_classes'], voxel_size=h.spacing)
            assds_3D_b.append(assd_3D_b)

            dice_3D_b_aug = dice_all_class(
                x_b_segaug_3D, b_3D_mask, config['seg']['n_classes'])
            dices_3D_b_aug.append(dice_3D_b_aug)

            assd_3D_b_aug = mean_assds_all_class(x_b_segaug_3D, b_3D_mask, class_num=config['seg']['n_classes'], voxel_size=h.spacing)
            assds_3D_b_aug.append(assd_3D_b_aug)

            dice_3D_b_augv2 = dice_all_class(
                x_b_segaug_3D_v2, b_3D_mask, config['seg']['n_classes'])
            dices_3D_b_augv2.append(dice_3D_b_augv2)

            assd_3D_b_augv2 = mean_assds_all_class(x_b_segaug_3D_v2, b_3D_mask, class_num=config['seg']['n_classes'], voxel_size=h.spacing)
            assds_3D_b_augv2.append(assd_3D_b_augv2)

            dice_3D_b_augv3 = dice_all_class(
                x_b_segaug_3D_v3, b_3D_mask, config['seg']['n_classes'])
            dices_3D_b_augv3.append(dice_3D_b_augv3)

            assd_3D_b_augv3 = mean_assds_all_class(x_b_segaug_3D_v3, b_3D_mask, class_num=config['seg']['n_classes'], voxel_size=h.spacing)
            assds_3D_b_augv3.append(assd_3D_b_augv3)


            with open('{}/test_b_3d_dice.txt'.format(output_directory), 'a') as f:
                f.writelines('patient:{}, 3d_Dice:{}, 3d_ASSD:{}\n'.
                            format(idx_patient_b, dice_3D_b, assd_3D_b))
            
            with open('{}/test_b_3d_dice_aug.txt'.format(output_directory), 'a') as f:
                f.writelines('patient:{}, 3d_dice:{}, 3d_ASSD:{}\n'.
                            format(idx_patient_b, dice_3D_b_aug, assd_3D_b_aug))
            
            with open('{}/test_b_3d_dice_aug_v2.txt'.format(output_directory), 'a') as f:
                f.writelines('patient:{}, 3d_dice:{}, 3d_ASSD:{}\n'.
                            format(idx_patient_b, dice_3D_b_augv2, assd_3D_b_augv2))
            
            with open('{}/test_b_3d_dice_aug_v3.txt'.format(output_directory), 'a') as f:
                f.writelines('patient:{}, 3d_dice:{}, 3d_ASSD:{}\n'.
                            format(idx_patient_b, dice_3D_b_augv3, assd_3D_b_augv3))
            
            print('finish one 3D reconstruct!')
            next_start_index = end_index + 1
    b_validate_dice_mean_3d = np.mean(np.array(dices_3D_b))
    b_validate_dice_std_3d = np.std(np.array(dices_3D_b))
    print('Validate b 3d, mean Dice = %.4f, std Dice = %.4f' %
          (b_validate_dice_mean_3d, b_validate_dice_std_3d))
    b_validate_assd_mean_3d = np.mean(np.array(assds_3D_b))
    b_validate_assd_std_3d = np.std(np.array(assds_3D_b))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Dice mean:{}, std:{}\n'.
                             format(b_validate_dice_mean_3d, b_validate_dice_std_3d))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('ASSD mean:{}, std:{}\n'.
                             format(b_validate_assd_mean_3d, b_validate_assd_std_3d))


    b_validate_dice_mean_3d_aug = np.mean(np.array(dices_3D_b_aug))
    b_validate_dice_std_3d_aug = np.std(np.array(dices_3D_b_aug))
    print('Validate b 3d_aug, mean Dice = %.4f, std Dice = %.4f' %
          (b_validate_dice_mean_3d_aug, b_validate_dice_std_3d_aug))
    b_validate_assd_mean_3d_aug = np.mean(np.array(assds_3D_b_aug))
    b_validate_assd_std_3d_aug = np.std(np.array(assds_3D_b_aug))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Aug Dice mean:{}, std:{}\n'.
                            format(b_validate_dice_mean_3d_aug, b_validate_dice_std_3d_aug))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Aug ASSD mean:{}, std:{}\n'.
                            format(b_validate_assd_mean_3d_aug, b_validate_assd_std_3d_aug))

    b_validate_dice_mean_3d_augv2 = np.mean(np.array(dices_3D_b_augv2))
    b_validate_dice_std_3d_augv2 = np.std(np.array(dices_3D_b_augv2))
    print('Validate b 3d_aug, mean Dice = %.4f, std Dice = %.4f' %
        (b_validate_dice_mean_3d_augv2, b_validate_dice_std_3d_augv2))
    b_validate_assd_mean_3d_augv2 = np.mean(np.array(assds_3D_b_augv2))
    b_validate_assd_std_3d_augv2 = np.std(np.array(assds_3D_b_augv2))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Aug v2 Dice mean:{}, std:{}\n'.
                            format(b_validate_dice_mean_3d_augv2, b_validate_dice_std_3d_augv2))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Aug v2 ASSD mean:{}, std:{}\n'.
                            format(b_validate_assd_mean_3d_augv2, b_validate_assd_std_3d_augv2))

    b_validate_dice_mean_3d_augv3 = np.mean(np.array(dices_3D_b_augv3))
    b_validate_dice_std_3d_augv3 = np.std(np.array(dices_3D_b_augv3))
    print('Validate b 3d_aug, mean Dice = %.4f, std Dice = %.4f' %
        (b_validate_dice_mean_3d_augv3, b_validate_dice_std_3d_augv3))
    b_validate_assd_mean_3d_augv3 = np.mean(np.array(assds_3D_b_augv3))
    b_validate_assd_std_3d_augv3 = np.std(np.array(assds_3D_b_augv3))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Aug v3 Dice mean:{}, std:{}\n'.
                            format(b_validate_dice_mean_3d_augv3, b_validate_dice_std_3d_augv3))
    with open('{}/test_result.txt'.format(output_directory), 'a') as f:
                f.writelines('Aug v3 ASSD mean:{}, std:{}\n'.
                            format(b_validate_assd_mean_3d_augv3, b_validate_assd_std_3d_augv3))


print('finish!')
