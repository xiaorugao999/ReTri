import swanlab
import cv2
import glob
import dataset_RUN2BMC
import time
import shutil
import psutil

import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from medpy.io import load, save
from utils_new import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, write_2images_single, Timer
from segnetworks.utils import dice_all_class_gpu
import argparse
from torch.autograd import Variable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn.functional as F


from trainer_ReTri import MUNIT_Trainer

def upsample_images_and_masks(images, masks, target_size=384):
    upsampled_images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    if masks.dim() == 3:
        upsampled_masks = F.interpolate(masks.unsqueeze(1).float(), size=(target_size, target_size), mode='nearest')
        upsampled_masks = upsampled_masks.squeeze(1).long()
    else:
        upsampled_masks = F.interpolate(masks.float(), size=(target_size, target_size), mode='nearest')
        upsampled_masks = upsampled_masks.long()
    
    return upsampled_images, upsampled_masks

def downsample_predictions(predictions, original_size):
    if predictions.dim() == 3:
        predictions_float = predictions.unsqueeze(1).float()
        downsampled = F.interpolate(predictions_float, size=(original_size, original_size), mode='nearest')
        downsampled = downsampled.squeeze(1).long()
    else:
        downsampled = F.interpolate(predictions.float(), size=(original_size, original_size), mode='bilinear', align_corners=False)
        if predictions.dtype == torch.long or predictions.dtype == torch.int:
            downsampled = downsampled.long()
    
    return downsampled

try:
    from itertools import izip as zip
except ImportError:
    pass
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str,
                    default='./configs/config_Prostate1_RUN2BMC_Ours.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="outputs path")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--save_name', type=str, default='Prostate1_RUN2BMC_Ours', help="MUNIT|UNIT")
opts = parser.parse_args()
cudnn.benchmark = True

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

swanlab.init(project="Prostate_RUN_BMC_UDA",experiment_name ="RUN2BMC")

train_A_path = sorted([f for f in glob.glob(os.path.join(
    config['data_root'], config['train_A_dir'], '*/*')) if not f.endswith('_segmentation.png')])
train_B_path = sorted([f for f in glob.glob(os.path.join(
    config['data_root'], config['train_B_dir'], '*/*')) if not f.endswith('_segmentation.png')]) 
test_A_path = sorted([f for f in glob.glob(os.path.join(
    config['data_root'], config['test_A_dir'], '*/*')) if not f.endswith('_segmentation.png')])
test_B_path = sorted([f for f in glob.glob(os.path.join(
    config['data_root'], config['test_B_dir'], '*/*')) if not f.endswith('_segmentation.png')])

print(os.path.basename(train_B_path[0]).split('_'))
print(train_B_path[:12])
train_A_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
train_B_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
test_A_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
test_B_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
print(train_A_path[:18])
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
v_my_sampler = dataset_RUN2BMC.ValidateRandomSampler(test_pair_num)
v_my_batch_sampler = torch.utils.data.BatchSampler(
    v_my_sampler, batch_size=config['batch_size'], drop_last=False)
validate_dataset = dataset_RUN2BMC.RUN2BMC_DataSet(
    config, test_A_path, test_B_path, train_patient_num_A, test_patient_num_B, random_range, True, phase='test')
validate_dataloader = torch.utils.data.DataLoader(
    validate_dataset, batch_sampler=v_my_batch_sampler)
test_display = [validate_dataloader.dataset[i] for i in range(display_size)]
test_display_images_a = torch.stack([i[0] for i in test_display]).cuda()
test_display_masks_a = torch.stack([i[2] for i in test_display]).cuda()
test_display_images_b = torch.stack([i[3] for i in test_display]).cuda()
test_display_masks_b = torch.stack([i[5] for i in test_display]).cuda()

test_display_images_a_up, test_display_masks_a_up = upsample_images_and_masks(test_display_images_a, test_display_masks_a)
test_display_images_b_up, test_display_masks_b_up = upsample_images_and_masks(test_display_images_b, test_display_masks_b)

day_hour_minute = time.strftime("%d%H%M", time.localtime())
model_name = opts.save_name + day_hour_minute
output_directory = os.path.join(opts.output_path + "/Prostate_outputs", model_name)
checkpoint_directory, image_directory, best_checkpoint_directory_a, best_checkpoint_directory_b = prepare_sub_folder(
    output_directory)


# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))
current_script = __file__
if os.path.exists(current_script):
    script_name = os.path.basename(current_script)
    shutil.copy(current_script, os.path.join(output_directory, script_name))
    print(f"Training script copied: {script_name}")
trainer_script = 'trainer_ReTri.py'
if os.path.exists(trainer_script):
    shutil.copy(trainer_script, os.path.join(output_directory, trainer_script))
    print(f"Trainer script copied: {trainer_script}")
else:
    print(f"Warning: Trainer script not found: {trainer_script}")



with open('{}/train_glob_data_B.txt'.format(output_directory), 'a') as f:
    f.writelines('patient:{}\n'.format(train_B_path))
with open('{}/test_glob_data_B.txt'.format(output_directory), 'a') as f:
    f.writelines('patient:{}\n'.format(test_B_path))

# Start training
print(resume_munit)
iterations = trainer.resume(
    resume_dir, hyperparameters=config) if resume_munit else 0

print('!!! The begin iterations:', iterations)

training_start_time = time.time()
iteration_times = []
gpu_memory_usage = []

while True:
    my_sampler = dataset_RUN2BMC.TrainRandomSampler(train_pair_num)
    my_batch_sampler = torch.utils.data.BatchSampler(
        my_sampler, batch_size=config['batch_size'], drop_last=False)
    train_dataset = dataset_RUN2BMC.RUN2BMC_DataSet(
        config, train_A_path, train_B_path, train_patient_num_A, train_patient_num_B, random_range)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=my_batch_sampler, num_workers=config['num_workers'])
    train_display = [train_dataloader.dataset[i] for i in range(display_size)]
    train_display_images_a = torch.stack([i[1] for i in train_display]).cuda()
    train_display_images_b = torch.stack([i[4] for i in train_display]).cuda()
    train_display_masks_a = torch.stack([i[2] for i in train_display]).cuda()
    train_display_masks_b = torch.stack([i[5] for i in train_display]).cuda()
    train_display_images_b1 = torch.stack([i[7] for i in train_display]).cuda()
    train_display_images_b2 = torch.stack([i[10] for i in train_display]).cuda()
    
    train_display_images_a_up, train_display_masks_a_up = upsample_images_and_masks(train_display_images_a, train_display_masks_a)
    train_display_images_b_up, train_display_masks_b_up = upsample_images_and_masks(train_display_images_b, train_display_masks_b)
    for i, data in enumerate(train_dataloader):
        # Record iteration start time
        iter_start_time = time.time()
        
        images_a, images_a_forseg, masks_a, images_b, images_b_forseg, masks_b, images_b1, images_b1_forseg, masks_b1, images_b2, images_b2_forseg, masks_b2 = data[
            0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]
        # [1,3,384,384]BCHW
        # make the mask shape to BHW
        # masks_a = masks_a[:, 0, :, :].cuda().detach()
        masks_a = masks_a.cuda().detach()
        # print(torch.unique(masks_a))
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        images_b1_forseg = images_b1_forseg.cuda().detach()
        images_b2_forseg = images_b2_forseg.cuda().detach()
        images_a_forseg, images_b_forseg = images_a_forseg.cuda(
        ).detach(), images_b_forseg.cuda().detach()

        original_size = images_a.shape[-1]
        
        images_a_up, _ = upsample_images_and_masks(images_a, masks_a)
        images_b_up, _ = upsample_images_and_masks(images_b, masks_a)
        
        images_a_forseg_up, masks_a_up = upsample_images_and_masks(images_a_forseg, masks_a)
        images_b_forseg_up, _ = upsample_images_and_masks(images_b_forseg, masks_a)
        images_b1_forseg_up, _ = upsample_images_and_masks(images_b1_forseg, masks_a)
        images_b2_forseg_up, _ = upsample_images_and_masks(images_b2_forseg, masks_a)

        with Timer("Elapsed time in update: %f"):
            if (iterations + 1) < config['pre_train_before_seg']:
                print(config['pre_train_before_seg'])
                trainer.dis_update(images_a_up, images_b_up, config)
                trainer.gen_update(images_a_up, images_b_up, config, gauss_kernel)
                
                if (iterations + 1) % config.get('log_iter', 10) == 0:
                    
                    # Image Reconstruction Losses  
                    swanlab.log({"RDIA/self_recon_loss_s": trainer.loss_gen_recon_x_a})
                    swanlab.log({"RDIA/self_recon_loss_t": trainer.loss_gen_recon_x_b})
                    swanlab.log({"RDIA/cyclic_recon_loss_s": trainer.loss_gen_cycrecon_x_a})
                    swanlab.log({"RDIA/cyclic_recon_loss_t": trainer.loss_gen_cycrecon_x_b})
                    swanlab.log({"RDIA/total_image_recon_loss": trainer.loss_gen_recon_x_a + trainer.loss_gen_recon_x_b + 
                                                              trainer.loss_gen_cycrecon_x_a + trainer.loss_gen_cycrecon_x_b})
                    
                    # Latent Reconstruction Losses
                    swanlab.log({"RDIA/content_recon_loss_s": trainer.loss_gen_recon_c_a})
                    swanlab.log({"RDIA/content_recon_loss_t": trainer.loss_gen_recon_c_b})
                    swanlab.log({"RDIA/style_recon_loss_s": trainer.loss_gen_recon_s_a})
                    swanlab.log({"RDIA/style_recon_loss_t": trainer.loss_gen_recon_s_b})
                    swanlab.log({"RDIA/total_latent_recon_loss": trainer.loss_gen_recon_c_a + trainer.loss_gen_recon_c_b +
                                                               trainer.loss_gen_recon_s_a + trainer.loss_gen_recon_s_b})
                    swanlab.log({"Training/gen_lr": trainer.gen_scheduler.get_last_lr()[0]})
                    swanlab.log({"Training/dis_lr": trainer.dis_scheduler.get_last_lr()[0]})
                    
            else:
                trainer.dis_update(images_a_up, images_b_up, config)
                trainer.gen_update(images_a_up, images_b_up, config, gauss_kernel)
                print('updating segmentation!')
                trainer.seg_update_BSFE(
                    images_a_forseg_up, masks_a_up, images_b_forseg_up, config)
                trainer.seg_update_Multiview_Duallevel(
                    images_a_forseg_up, images_b_forseg_up, images_b1_forseg_up, images_b2_forseg_up, config)
                
                if (iterations + 1) % config.get('log_iter', 10) == 0:
                    
                    # Image Reconstruction Losses
                    swanlab.log({"RDIA/self_recon_loss_s": trainer.loss_gen_recon_x_a})
                    swanlab.log({"RDIA/self_recon_loss_t": trainer.loss_gen_recon_x_b})
                    swanlab.log({"RDIA/cyclic_recon_loss_s": trainer.loss_gen_cycrecon_x_a})
                    swanlab.log({"RDIA/cyclic_recon_loss_t": trainer.loss_gen_cycrecon_x_b})
                    swanlab.log({"RDIA/total_image_recon_loss": trainer.loss_gen_recon_x_a + trainer.loss_gen_recon_x_b + 
                                                              trainer.loss_gen_cycrecon_x_a + trainer.loss_gen_cycrecon_x_b})
                    
                    # Latent Reconstruction Losses
                    swanlab.log({"RDIA/content_recon_loss_s": trainer.loss_gen_recon_c_a})
                    swanlab.log({"RDIA/content_recon_loss_t": trainer.loss_gen_recon_c_b})
                    swanlab.log({"RDIA/style_recon_loss_s": trainer.loss_gen_recon_s_a})
                    swanlab.log({"RDIA/style_recon_loss_t": trainer.loss_gen_recon_s_b})
                    swanlab.log({"RDIA/total_latent_recon_loss": trainer.loss_gen_recon_c_a + trainer.loss_gen_recon_c_b +
                                                               trainer.loss_gen_recon_s_a + trainer.loss_gen_recon_s_b})
                    
                    swanlab.log({"Segmentation/total_supervised_loss": trainer.loss_supervise})
                    swanlab.log({"Segmentation/CE_loss": trainer.celoss})
                    swanlab.log({"Segmentation/Dice_loss": trainer.diceloss})
                    swanlab.log({"Segmentation/CE_DSC_loss": trainer.loss_supervise1})
                    
                    # === TCFA Module Losses ===
                    lambda1 = config.get('recon_x_w', 10)
                    lambda2 = config.get('recon_c_w', 1) + config.get('recon_s_w', 1)
                    lambda3 = config.get('attn_consistency_w', 1)
                    lambda4 = config.get('cons_weight_multi_feacut', 1)
                    lambda5 = config.get('cons_weight_multi_transcut', 1)
                    
                    total_rdia_loss = (trainer.loss_gen_adv_a + trainer.loss_gen_adv_b + 
                                     lambda1 * (trainer.loss_gen_recon_x_a + trainer.loss_gen_recon_x_b + 
                                               trainer.loss_gen_cycrecon_x_a + trainer.loss_gen_cycrecon_x_b) +
                                     lambda2 * (trainer.loss_gen_recon_c_a + trainer.loss_gen_recon_c_b +
                                               trainer.loss_gen_recon_s_a + trainer.loss_gen_recon_s_b))
                    
                    total_tcfa_loss = (trainer.loss_supervise + 
                                     lambda4 * trainer.multi_feacut + 
                                     lambda5 * trainer.multi_transcut)
                    
                    swanlab.log({"Overall/weighted_RDIA_loss": total_rdia_loss})
                    swanlab.log({"Overall/weighted_TCFA_loss": total_tcfa_loss})
                    
                    swanlab.log({"Training/gen_lr": trainer.gen_scheduler.get_last_lr()[0]})
                    swanlab.log({"Training/dis_lr": trainer.dis_scheduler.get_last_lr()[0]})
                    swanlab.log({"Training/seg_lr": trainer.seg_scheduler.get_last_lr()[0]})

            torch.cuda.synchronize()

        trainer.update_learning_rate()

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save_cp(checkpoint_directory, iterations, config)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
