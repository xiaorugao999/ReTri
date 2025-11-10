"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import glob
import dataset_RUN2BMC
import time
import shutil
import kornia as K
import os
import torch
import torch.backends.cudnn as cudnn
import sys
from medpy.metric.binary import assd, dc

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ASSD and Dice are suitable for 3D Prostate images
from trainer_ReTri import MUNIT_Trainer
import numpy as np
from medpy.io import load, save
import nibabel as nib
from utils_new import get_all_data_loaders, prepare_sub_folder, prepare_test_sub_folder, write_html, write_loss, get_config, write_2images, write_2images_TTA,write_2images_single, Timer
from segnetworks.utils import dice_all_class_gpu
import argparse
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass


NET_STD = np.array((0.5, 0.5, 0.5))
NET_MEAN = np.array((0.5, 0.5, 0.5))
def un_normalize(input_img):
    mean = 0.5
    std = 0.5
    output_img = input_img * std + mean
    return output_img

def load_original_3d_info(patient_id, domain='BMC'):
    """
    Load original 3D NIfTI file information including spacing and affine matrix
    Args:
        patient_id: Patient ID (e.g., 'Case00')
        domain: 'BMC' or 'RUN'
    Returns:
        spacing, affine, original_data_shape, origin
    """
    try:
        if domain == 'BMC':
            nii_path = f"datasets/Prostate_3D_nii_dir/ISBI_BMC/{patient_id}.nii.gz"
        elif domain == 'RUN':  # RUN
            nii_path = f"datasets/Prostate_3D_nii_dir/ISBI_RUNMC/{patient_id}.nii.gz"
        elif domain == 'UCL':
            nii_path = f"datasets/Prostate_3D_nii_dir/UCL/{patient_id}.nii.gz"
        elif domain == 'HK':
            nii_path = f"datasets/Prostate_3D_nii_dir/HK/{patient_id}.nii.gz"
        elif domain == 'I2CVB':
            nii_path = f"datasets/Prostate_3D_nii_dir/I2CVB/{patient_id}.nii.gz"
        elif domain == 'BIDMC':
            nii_path = f"datasets/Prostate_3D_nii_dir/BIDMC/{patient_id}.nii.gz"
        
        if os.path.exists(nii_path):
            # Use nibabel to load NIfTI file
            nii_img = nib.load(nii_path)
            
            # Get spacing (voxel size) from header
            spacing = nii_img.header.get_zooms()[:3]  # (x, y, z) spacing
            
            # Get affine matrix (includes origin and orientation)
            affine = nii_img.affine
            
            # Get original data shape
            original_shape = nii_img.shape
            
            # Extract origin from affine matrix
            origin = affine[:3, 3]
            
            print(f"Loaded 3D info for {patient_id} ({domain}): shape={original_shape}, spacing={spacing}")
            
            return spacing, affine, original_shape, origin
        else:
            print(f"Warning: Original 3D file not found: {nii_path}")
            return (1.0, 1.0, 1.0), np.eye(4), None, (0.0, 0.0, 0.0)
    except Exception as e:
        print(f"Error loading 3D info for {patient_id}: {e}")
        return (1.0, 1.0, 1.0), np.eye(4), None, (0.0, 0.0, 0.0)

def mean_assds_all_class(prediction, target, class_num=20, eps=1e-10, voxel_size=(1,1,1), empty_value=-1.0, connectivity=1):
    """
    Calculate ASSD for all classes in 3D volumes
    """
    assds = empty_value * np.ones((class_num - 1), dtype=np.float32)
    for i in range(1, class_num):
        if i not in target:
            continue
        if i not in prediction:
            continue
        pred_class = (prediction == i).astype(np.uint8)
        target_class = (target == i).astype(np.uint8)
        if np.sum(pred_class) == 0 or np.sum(target_class) == 0:
            continue
        try:
            assds[i-1] = assd(pred_class, target_class, voxelspacing=voxel_size, connectivity=connectivity)
        except Exception as e:
            print(f"ASSD calculation failed for class {i}: {e}")
            assds[i-1] = empty_value
    return assds

def upsample_images_and_masks(images, masks, target_size=256):
    """
    Upsample images and masks from original size to target_size x target_size
    Args:
        images: tensor of shape [B, C, H, W]
        masks: tensor of shape [B, H, W] or [B, C, H, W]
        target_size: target size (default: 256)
    Returns:
        upsampled_images: tensor of shape [B, C, target_size, target_size]
        upsampled_masks: tensor of shape [B, target_size, target_size] or [B, C, target_size, target_size]
    """
    # Upsample images using bilinear interpolation
    upsampled_images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    # Upsample masks using nearest interpolation to preserve label values
    if masks.dim() == 3:  # [B, H, W]
        upsampled_masks = F.interpolate(masks.unsqueeze(1).float(), size=(target_size, target_size), mode='nearest')
        upsampled_masks = upsampled_masks.squeeze(1).long()
    else:  # [B, C, H, W]
        upsampled_masks = F.interpolate(masks.float(), size=(target_size, target_size), mode='nearest')
        upsampled_masks = upsampled_masks.long()
    
    return upsampled_images, upsampled_masks


def jaccard_all_class_gpu(prediction, target, class_num=20):
    '''
    Calculate Jaccard coefficient (IoU) for all valid classes
    :param prediction: tensor with shape of [B, H, W] or [H, W]
    :param target: tensor with shape of [B, H, W] or [H, W]
    :param class_num: maximum number of classes
    :return: mean Jaccard coefficient across valid classes
    '''
    jaccards = []
    # Find unique classes that actually exist in the target
    unique_classes = torch.unique(target)
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background class
            continue
        if class_id >= class_num:  # Skip classes beyond our range
            continue
            
        target_per_class = torch.where(target == class_id, 1, 0)
        prediction_per_class = torch.where(prediction == class_id, 1, 0)
        
        # Calculate metric if class exists in target mask
        if torch.sum(target_per_class) > 0:
            jaccard = jaccard_per_class_gpu(prediction_per_class, target_per_class)
            jaccards.append(jaccard)
    
    if len(jaccards) > 0:
        return torch.mean(torch.stack(jaccards))
    else:
        return torch.tensor(0.0).cuda()

def jaccard_per_class_gpu(prediction, target):
    '''
    Calculate Jaccard coefficient (IoU) for a single class
    :param prediction: tensor
    :param target: tensor
    :return: Jaccard coefficient
    '''
    eps = torch.tensor(1e-10).cuda()
    prediction = prediction.float()
    target = target.float()
    
    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target) - intersection
    return intersection / (union + eps)

def dice_all_class_gpu(prediction, target, class_num=20):
    '''
    Calculate Dice coefficient for all valid classes
    :param prediction: tensor with shape of [B, H, W] or [H, W]
    :param target: tensor with shape of [B, H, W] or [H, W]
    :param class_num: maximum number of classes
    :return: mean Dice coefficient across valid classes
    '''
    dices = []
    # Find unique classes that actually exist in the target
    unique_classes = torch.unique(target)
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background class
            continue
        if class_id >= class_num:  # Skip classes beyond our range
            continue
            
        target_per_class = torch.where(target == class_id, 1, 0)
        prediction_per_class = torch.where(prediction == class_id, 1, 0)
        
        # Calculate metric if class exists in target mask
        if torch.sum(target_per_class) > 0:
            dice = dice_per_class_gpu(prediction_per_class, target_per_class)
            dices.append(dice)
    
    if len(dices) > 0:
        return torch.mean(torch.stack(dices))
    else:
        return torch.tensor(0.0).cuda()

def dice_per_class_gpu(prediction, target):
    '''
    Calculate Dice coefficient for a single class
    :param prediction: tensor
    :param target: tensor
    :return: Dice coefficient
    '''
    eps=torch.tensor(1e-10).cuda()
    prediction = prediction.float()
    target = target.float()
    intersect = torch.sum(prediction * target)
    return (2. * intersect / (torch.sum(prediction) + torch.sum(target) + eps))

def save_3d_nifti_with_spacing(data, filepath, spacing, affine, origin=None):
    """
    Save 3D data as NIfTI file with correct spacing and affine matrix
    Args:
        data: 3D numpy array in format [H, W, Slices]
        filepath: output file path
        spacing: voxel spacing (x, y, z)
        affine: 4x4 affine matrix
        origin: origin coordinates (optional)
    """
    try:
        # Create NIfTI image with proper spacing and affine
        nii_img = nib.Nifti1Image(data, affine)
        
        # Set voxel spacing in header
        nii_img.header.set_zooms(spacing)
        
        # Save the NIfTI file
        nib.save(nii_img, filepath)
        print(f"Saved 3D NIfTI with spacing {spacing}: {filepath}")
        
    except Exception as e:
        print(f"Error saving 3D NIfTI {filepath}: {e}")
        # Fallback to medpy save
        save(data, filepath)

def mask2color_prostate(mask):
    """Color mapping for Prostate segmentation masks"""
    maskarray = np.array(mask.cpu().squeeze())  # hw
    # Color mapping for prostate classes: background, prostate gland
    color_dict = {0: [0, 0, 0], 1: [255, 255, 0]}  # Background: black, Prostate: yellow
    mask2color = np.zeros(
        (maskarray.shape[0], maskarray.shape[1], 3)).astype(int)
    for k, v in color_dict.items():
        mask2color[maskarray == k] = v
    masktensor = torch.from_numpy(mask2color)
    return masktensor

def sample_test_Prostate_ours(input_x_a, test_display_masks_a, input_x_b, test_display_masks_b, trainer, imgdir, slice_num=None):
    """
    Test function for our method on Prostate dataset with domain adaptation
    """
    trainer.eval()
    s_b0 = Variable(torch.randn(input_x_b.size(0), 8, 1, 1).cuda())
    s_a0 = Variable(torch.randn(input_x_b.size(0), 8, 1, 1).cuda())
    x_ab0s, x_ab0_segs = [], []
    x_as, x_bs = [], []
    x_b_seg_volume = []
    loss_mask_ones = torch.ones_like(input_x_b[0].unsqueeze(0)).cuda().detach()
    
    for i in range(input_x_b.size(0)):
        current_slice_num = slice_num if slice_num is not None else i + 1
        
        # Domain adaptation: A(source) -> B(target)
        c_a, s_a_fake = trainer.gen_a.encode(input_x_a[i].unsqueeze(0))
        c_b, s_b_fake = trainer.gen_b.encode(input_x_b[i].unsqueeze(0))
        x_ba = trainer.gen_a.decode(c_b, s_a0[i].unsqueeze(0))
        
        # Cycle reconstruction
        c_b_recon, s_a_recon = trainer.gen_a.encode(x_ba)
        x_bab = trainer.gen_b.decode(c_b_recon, s_b0[i].unsqueeze(0))
        x_ab0 = trainer.gen_b.decode(c_a, s_b0[i].unsqueeze(0))
        
        # Unnormalize images
        x_bab = trainer.un_normalize(x_bab)
        x_ab0 = trainer.un_normalize(x_ab0)
        x_ab0s.append(x_ab0)
        x_b_ori = trainer.un_normalize(input_x_b[i]).unsqueeze(0).float()
        x_bs.append(x_b_ori)
        
  
        # Segmentation inference using our trainer's eval_net
        
        x_b_seg = trainer.eval_net(x_b_ori, [])
        
        
        # Extract predictions based on model type (consistent with training code)
        if trainer.hyperparameters['seg']['segmentor'] == 'ResUNet2D_Featurelevel_cutmix':
            prob_x_b_seg = x_b_seg[0]
        else:
            prob_x_b_seg = F.softmax(x_b_seg, dim=1)
    
        out_seg = trainer.validate_mask(prob_x_b_seg)
       
        x_b_seg_volume.append(out_seg)
        x_as.append(trainer.un_normalize(input_x_a[i]).unsqueeze(0))
        
        # Save visualization images
        # Handle mask dimensions properly for Prostate dataset
        if test_display_masks_b[i].dim() == 3:
            if test_display_masks_b[i].shape[0] > 1:
                # Multi-channel mask, take first channel
                mask_for_vis = test_display_masks_b[i][0, :, :].unsqueeze(0)
            else:
                mask_for_vis = test_display_masks_b[i].squeeze(0).unsqueeze(0)
        else:
            mask_for_vis = test_display_masks_b[i].unsqueeze(0)
            
        mask_png = mask2color_prostate(mask_for_vis)
        predict_png = mask2color_prostate(out_seg)
        inver_norm_img = x_b_ori.cpu().numpy() * 255
        inver_norm_img = np.transpose(inver_norm_img, [0, 2, 3, 1])
        
        imgpng = Image.fromarray(np.uint8(inver_norm_img[0, :, :, :]))
        imgpng.save(os.path.join(imgdir, f'img_slice{current_slice_num:03d}.png'))
        maskpng = Image.fromarray(np.uint8(mask_png))
        maskpng.save(os.path.join(imgdir, f'mask_slice{current_slice_num:03d}.png'))
        predictpng = Image.fromarray(np.uint8(predict_png))
        predictpng.save(os.path.join(imgdir, f'predict_slice{current_slice_num:03d}.png'))
    
    x_ab0s = torch.cat(x_ab0s)  
    x_as = torch.cat(x_as)
    x_bs = torch.cat(x_bs)
    x_b_seg_volume = torch.cat(x_b_seg_volume)
    
    return x_as, x_ab0s, x_bs, x_b_seg_volume


# Prostate RUN2BMC model checkpoint path for our method
RUN2BMC_model = 'Prostate_outputs/Prostate_RUN2BMC_Ours/checkpoints'


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='./configs/config_Prostate_RUN2BMC_Ours.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="outputs path")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument("--checkpoint_directory", type=str,
                    default=RUN2BMC_model)

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

# RUN2BMC dataset uses subdirectory structure, exclude segmentation files
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

# RUN2BMC dataset sorting: sort by patient directory and then by filename
train_A_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
train_B_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
test_A_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
test_B_path.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))
print(train_A_path[:18])
print(train_B_path[:12])
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

# model_name = os.path.splitext(os.path.basename(opts.config))[0]
day_hour_minute = time.strftime("%d%H%M", time.localtime())
model_name = config['Test_Name'] + day_hour_minute
output_directory = os.path.join(opts.output_path + "/Test_Prostate_outputs", model_name)
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
iterations, last_model_name_seg = trainer.test_load_model_cutmix(
    opts.checkpoint_directory, numpt=-1, netname= test_net_str + 'seg', netkind='seg')
iterations, last_model_name_gen = trainer.test_load_model_cutmix(
    opts.checkpoint_directory, numpt=-1, netname= test_net_str + 'gen', netkind='gen')


trainer.eval()  ## important when evaluation


print(f"\n=== Starting Test ===")
print(f"Test script: {os.path.abspath(__file__)}")
print(f"Model checkpoint directory: {opts.checkpoint_directory}")
print(f"Config file: {opts.config}")
print(f"Output directory: {output_directory}")
print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


with open('{}/test_progress.txt'.format(output_directory), 'w') as f:
    f.write("=== Test Progress Log ===\n")
    f.write(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

with torch.no_grad():
    # Initialize metric storage lists for 3D Prostate RUN2BMC dataset
    dices_3D_ori = []  # 3D volumes original prediction
    assds_3D_ori = []  # 3D volumes original prediction ASSD
    
    print(f"Starting 3D Prostate RUN2BMC testing with {len(test_B_path)} test slices")
    
    # Group test paths by patient for RUN2BMC dataset
    patient_groups = {}
    for i, one_b_path in enumerate(test_B_path):
        # Extract patient ID from path: datasets/Prostate_RUN2BMC/test_BMC_target_10/Case00/Case00_slice01_all11.png
        patient_id = os.path.basename(os.path.dirname(one_b_path))  # Case00
        
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(i)

    print(f"Found {len(patient_groups)} patients for RUN2BMC testing")
    
    # Process each patient's slices for 3D evaluation
    for patient_id, slice_indices in patient_groups.items():
        print(f"Processing patient {patient_id} with {len(slice_indices)} slices")
        
        # Store 3D predictions and masks for this patient
        patient_predictions_ori = []
        patient_masks = []
        
        # Create patient directories for 3D volumes and 2D images
        spath_3d = os.path.join(re_3D_directory_b, patient_id, '3D_volumes')
        spath_2d = os.path.join(re_3D_directory_b, patient_id, '2D_images')
        if not os.path.exists(spath_3d):
            os.makedirs(spath_3d)
        if not os.path.exists(spath_2d):
            os.makedirs(spath_2d)
        
        # Get source domain patient ID from the first slice (all slices of the same target patient should have the same source patient)
        first_slice_idx = slice_indices[0]
        first_test_data = validate_dataloader.dataset[first_slice_idx]
        source_patient_id = first_test_data[12]  # idx_patient_a is at index 12
        
        # Extract actual patient ID from the filename (remove file extension)
        source_patient_id = os.path.splitext(source_patient_id)[0]
        # If it contains slice information, extract just the patient ID
        if '_slice' in source_patient_id:
            source_patient_id = source_patient_id.split('_slice')[0]
        
        print(f"Source patient ID for target patient {patient_id}: {source_patient_id}")
        

        target_domain = config['target_domain_name']  
        source_domain = config['source_domain_name']   
        
        # Load spacing and affine info from original 3D data
        target_spacing, target_affine, target_orig_shape, target_origin = load_original_3d_info(patient_id, target_domain)
        source_spacing, source_affine, source_orig_shape, source_origin = load_original_3d_info(source_patient_id, source_domain)
        
        # Store 3D images and masks for this patient
        patient_images_b = []  # Target domain images
        patient_images_a = []  # Source domain images (for reference)
        patient_images_ab = []  # Translated images (A->B)
        
        for slice_order, slice_idx in enumerate(slice_indices):
            test_data = validate_dataloader.dataset[slice_idx]
            
            test_image_a = test_data[0].unsqueeze(0).cuda()
            test_mask_a = test_data[2].unsqueeze(0).cuda()
            test_image_b = test_data[3].unsqueeze(0).cuda()
            test_mask_b = test_data[5].unsqueeze(0).cuda()

            test_image_a_up, test_mask_a_up = upsample_images_and_masks(test_image_a, test_mask_a, target_size=384)
            test_image_b_up, test_mask_b_up = upsample_images_and_masks(test_image_b, test_mask_b, target_size=384)

            test_outputs = sample_test_Prostate_ours(
                test_image_a_up, test_mask_a_up, 
                test_image_b_up, test_mask_b_up, trainer, spath_2d, slice_num=slice_order+1)
            
            # Extract images from test outputs
            # test_outputs format: [x_as, x_ab0s, x_bs, x_b_seg_volume]
            x_a = test_outputs[0]  # Source domain image
            x_ab = test_outputs[1] 
            x_b = test_outputs[2]  # Target domain image
            
            # Handle mask dimensions properly
            if test_mask_b_up.dim() == 4 and test_mask_b_up.shape[1] == 1:
                b_mask = test_mask_b_up.squeeze(1)
            elif test_mask_b_up.dim() == 4:
                b_mask = test_mask_b_up[:, 0, :, :]
            else:
                b_mask = test_mask_b_up

            b_seg_ori = test_outputs[-1]  # Original prediction
            
            patient_predictions_ori.append(b_seg_ori.float().cpu().numpy())
            patient_masks.append(b_mask.float().cpu().numpy())
            
            patient_images_a.append(x_a.cpu().numpy())
            patient_images_ab.append(x_ab.cpu().numpy())
            patient_images_b.append(x_b.cpu().numpy())
        
        patient_predictions_ori_3d = np.stack(patient_predictions_ori, axis=0)
        
        patient_masks_3d = np.stack(patient_masks, axis=0)
        
        patient_images_a_3d = np.stack(patient_images_a, axis=0)
        patient_images_ab_3d = np.stack(patient_images_ab, axis=0)  
        patient_images_b_3d = np.stack(patient_images_b, axis=0)
        
     
        patient_predictions_ori_3d_tensor = torch.from_numpy(patient_predictions_ori_3d).cuda().squeeze(1)
        patient_masks_3d_tensor = torch.from_numpy(patient_masks_3d).cuda().squeeze(1)
        
        print(f'Patient {patient_id} 3D tensor shapes:')
        print(f'  Predictions ori: {patient_predictions_ori_3d_tensor.shape}')
        print(f'  Masks: {patient_masks_3d_tensor.shape}')
        
   
        dice_ori_3d = dice_all_class_gpu(
            patient_predictions_ori_3d_tensor, patient_masks_3d_tensor, config['seg']['n_classes'])
       
        dices_3D_ori.append(dice_ori_3d)
      
        
        # For segmentation masks and predictions - fix rotation issue
        # Current data: [Slices, H, W] -> should become proper medical orientation
        patient_predictions_ori_3d_fixed = np.transpose(patient_predictions_ori_3d.squeeze(1), (2, 1, 0))  # [W, H, Slices] -> [H, W, Slices]
        patient_masks_3d_fixed = np.transpose(patient_masks_3d.squeeze(1), (2, 1, 0))
        
        # For ASSD calculation, use the corrected orientation
        patient_predictions_ori_3d_transposed = patient_predictions_ori_3d_fixed 
        patient_masks_3d_transposed = patient_masks_3d_fixed
        
        try:
            # Calculate ASSD for original prediction
            assd_3D_ori_class = mean_assds_all_class(
                patient_predictions_ori_3d_transposed, patient_masks_3d_transposed, 
                class_num=config['seg']['n_classes'], voxel_size=target_spacing)
            valid_assds_ori = assd_3D_ori_class[assd_3D_ori_class != -1.0]
            assd_3D_ori = np.mean(valid_assds_ori) if len(valid_assds_ori) > 0 else np.nan
            assds_3D_ori.append(assd_3D_ori)
            
      
        except Exception as e:
            print(f"Warning: ASSD calculation failed for patient {patient_id}: {e}")
            assds_3D_ori.append(np.nan)
            assd_3D_ori = np.nan
           
        
        if patient_images_a_3d.ndim == 5:  # [Slices, 1, C, H, W]
            patient_images_a_3d_squeezed = patient_images_a_3d.squeeze(1)  # [Slices, C, H, W]
        else:
            patient_images_a_3d_squeezed = patient_images_a_3d
            
        if patient_images_ab_3d.ndim == 5:
            patient_images_ab_3d_squeezed = patient_images_ab_3d.squeeze(1)
        else:
            patient_images_ab_3d_squeezed = patient_images_ab_3d
            
        if patient_images_b_3d.ndim == 5:
            patient_images_b_3d_squeezed = patient_images_b_3d.squeeze(1)
        else:
            patient_images_b_3d_squeezed = patient_images_b_3d
        
        # For grayscale images, take first channel and fix orientation: [Slices, C, H, W] -> [H, W, Slices]
        if patient_images_a_3d_squeezed.shape[1] > 1:  # Multi-channel
            patient_images_a_3d_fixed = np.transpose(patient_images_a_3d_squeezed[:, 0, :, :], (2, 1, 0))  # Fix rotation
            patient_images_ab_3d_fixed = np.transpose(patient_images_ab_3d_squeezed[:, 0, :, :], (2, 1, 0))
            patient_images_b_3d_fixed = np.transpose(patient_images_b_3d_squeezed[:, 0, :, :], (2, 1, 0))
        else:  # Single channel
            patient_images_a_3d_fixed = np.transpose(patient_images_a_3d_squeezed.squeeze(1), (2, 1, 0))  # Fix rotation
            patient_images_ab_3d_fixed = np.transpose(patient_images_ab_3d_squeezed.squeeze(1), (2, 1, 0))
            patient_images_b_3d_fixed = np.transpose(patient_images_b_3d_squeezed.squeeze(1), (2, 1, 0))
        
        print(f"Saving 3D volumes for patient {patient_id}...")
        
        # Save segmentation results with target domain spacing (RUN)
        save_3d_nifti_with_spacing(patient_predictions_ori_3d_transposed, 
                                   f"{spath_3d}/seg_prediction_original.nii.gz", 
                                   target_spacing, target_affine, target_origin)
        save_3d_nifti_with_spacing(patient_masks_3d_transposed, 
                                   f"{spath_3d}/seg_groundtruth.nii.gz", 
                                   target_spacing, target_affine, target_origin)
        
        # Save images with appropriate spacing
        save_3d_nifti_with_spacing(patient_images_a_3d_fixed, 
                                   f"{spath_3d}/image_source_domain.nii.gz", 
                                   source_spacing, source_affine, source_origin)
        save_3d_nifti_with_spacing(patient_images_ab_3d_fixed, 
                                   f"{spath_3d}/image_translated_A2B.nii.gz", 
                                   target_spacing, target_affine, target_origin)
        save_3d_nifti_with_spacing(patient_images_b_3d_fixed, 
                                   f"{spath_3d}/image_target_domain.nii.gz", 
                                   target_spacing, target_affine, target_origin)
        
        print(f"  - Segmentation prediction (original): seg_prediction_original.nii.gz")
        print(f"  - Segmentation ground truth: seg_groundtruth.nii.gz")
        print(f"  - Source domain image: image_source_domain.nii.gz")
        print(f"  - Translated image (A->B): image_translated_A2B.nii.gz")
        print(f"  - Target domain image: image_target_domain.nii.gz")
        
        
        # Save individual results
        with open(f'{output_directory}/test_b_3d_results.txt', 'a') as f:
            f.writelines(f'Target_Patient:{patient_id}, Source_Patient:{source_patient_id}, '
                       f'3D_Dice_ori:{dice_ori_3d:.4f}, 3D_ASSD_ori:{assd_3D_ori:.4f},)
        
        print(f'Finished 3D evaluation for target patient {patient_id} (source: {source_patient_id}): '
              f'Dice_ori={dice_ori_3d:.4f}, ASSD_ori={assd_3D_ori:.4f},)
        
        # Update progress log
        with open(f'{output_directory}/test_progress.txt', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Completed 3D volume for patient {patient_id}\n")

    # Calculate statistics for 3D Prostate RUN2BMC dataset
    print(f"\n=== Prostate RUN2BMC 3D Test Results (Our Method) - Evaluated at 384x384 ===")
    
    # Calculate 3D statistics
    if dices_3D_ori:
        dice_3D_ori_mean = torch.mean(torch.stack(dices_3D_ori))
        dice_3D_ori_std = torch.std(torch.stack(dices_3D_ori))
        print(f'3D Original - Dice: mean={dice_3D_ori_mean:.4f}, std={dice_3D_ori_std:.4f}')

    if assds_3D_ori:
        valid_assds_ori = [x for x in assds_3D_ori if not np.isnan(x)]
        if valid_assds_ori:
            assd_3D_ori_mean = np.mean(valid_assds_ori)
            assd_3D_ori_std = np.std(valid_assds_ori)
            print(f'3D Original - ASSD: mean={assd_3D_ori_mean:.4f}, std={assd_3D_ori_std:.4f}')
    

    with open('{}/prostate_3d_test_results_summary.txt'.format(output_directory), 'w') as f:
        f.write("=== Prostate RUN2BMC 3D Test Results Summary (Our Method) ===\n\n")
        f.write("Note: All evaluations are performed at 384x384 resolution (matching training pipeline).\n")
        f.write("3D Dice coefficient and ASSD are reported for 3D Prostate volumes.\n\n")
        
        if dices_3D_ori:
            f.write(f"3D Original - Dice: mean={dice_3D_ori_mean:.4f}, std={dice_3D_ori_std:.4f}\n")
        
     
        if assds_3D_ori and valid_assds_ori:
            f.write(f"3D Original - ASSD: mean={assd_3D_ori_mean:.4f}, std={assd_3D_ori_std:.4f}\n")
        
    
        f.write(f"Total 3D volumes tested: {len(dices_3D_ori)}\n")
        f.write(f"Total 2D slices processed: {len(test_B_path)}\n")



print('finish!')