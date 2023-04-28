import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


class CTDataset(Dataset):
    def __init__(self,data_root,mode='train',transform=None,augment_dict=None,patches=(False,10,64)) -> None:
        super().__init__()
        assert mode in ['train','test','val'], "mode is 'train', 'test' or 'val' "
        
        self.transform = transform
        self.augment_dict = augment_dict
        self.is_patch = patches[0]
        self.patch_num = patches[1]
        self.patch_size = patches[2]
        
        subfolder = {'train':'TrainSet',
                     'test':'TestSet',
                     'val':'ValSet'}[mode]
        self.data_root = os.path.join(data_root,subfolder)
        
        self.full_image_list = []
        self.quarter_image_list = []
        
        patient_no = os.listdir(os.path.join(self.data_root,'Full_Dose'))
        
        for no in patient_no:
            full_dose_list = sorted(os.listdir(os.path.join(self.data_root,'Full_Dose',no)))
            quarter_dose_list = sorted(os.listdir(os.path.join(self.data_root,'Quarter_Dose',no)))
            
            full_dose_list = [os.path.join(self.data_root,'Full_Dose',no,x) for x in full_dose_list]
            quarter_dose_list = [os.path.join(self.data_root,'Quarter_Dose',no,x) for x in quarter_dose_list]
            
            self.full_image_list.extend(full_dose_list)
            self.quarter_image_list.extend(quarter_dose_list)
        
        assert len(self.full_image_list) == len(self.quarter_image_list), 'Full_Dose image should match with Quarter_Dose image'

    def __getitem__(self, index):
        
        quarter_image = pil_loader(self.quarter_image_list[index])
        full_image = pil_loader(self.full_image_list[index])
        
        # Apply the same augmentation to image and target
        quarter_image, full_image = self._augment(quarter_image, full_image)
        
        if self.transform is not None:
            quarter_image = self.transform(quarter_image)
            full_image = self.transform(full_image)
            
        if self.is_patch:
            quarter_image,full_image = self._get_patches(quarter_image, full_image)
        
        return quarter_image,full_image
    
    def __len__(self):
        return len(self.full_image_list)
    
    def _get_patches(self,full_input_img,full_target_img):
        
        assert full_input_img.shape == full_target_img.shape
        patch_input_imgs = []
        patch_target_imgs = []
        h, w = full_input_img.shape[-2:]
        new_h, new_w = self.patch_size,self.patch_size
        for _ in range(self.patch_num):
            top = np.random.randint(0, h-new_h)
            left = np.random.randint(0, w-new_w)
            patch_input_img = full_input_img[:,top:top+new_h, left:left+new_w]
            patch_target_img = full_target_img[:,top:top+new_h, left:left+new_w]
            patch_input_imgs.append(patch_input_img.unsqueeze(0))
            patch_target_imgs.append(patch_target_img.unsqueeze(0))
        return torch.cat(patch_input_imgs,dim=0),torch.cat(patch_target_imgs,dim=0)
    
    def _augment(self,quarter_img,full_img):
        
        if self.augment_dict is None:
            return quarter_img,full_img
        
        if 'crop' in self.augment_dict:
            image_width, image_height = quarter_img.size
            crop_size = self.augment_dict['crop']
            if isinstance(crop_size,list):
                assert 0 < max(crop_size) < image_height , 'crop size is invalid'
                crop_size = np.random.randint(crop_size[0],crop_size[1])
            elif isinstance(crop_size,tuple):
                assert 0 < max(crop_size) < image_height , 'crop size is invalid'
                crop_size = np.random.choice(crop_size)
            elif isinstance(crop_size,float):
                assert 0 < crop_size < 1 , 'crop size is invalid'
                crop_size = int(image_height*crop_size)
            elif isinstance(crop_size,int):
                assert 0 < crop_size < image_height , 'crop size is invalid'
                crop_size = crop_size
            

            left = random.randint(0, image_width - crop_size)
            top = random.randint(0, image_height - crop_size)
            right = left + crop_size
            bottom = top + crop_size
            
            quarter_img = quarter_img.crop((left, top, right, bottom))
            full_img = full_img.crop((left,top,right,bottom))
        
        if 'flip' in self.augment_dict:
            p = self.augment_dict['flip']
            if random.random() < 0.5:
                quarter_img = quarter_img.transpose(Image.FLIP_LEFT_RIGHT)
                full_img = full_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                quarter_img = quarter_img.transpose(Image.FLIP_TOP_BOTTOM)
                full_img = full_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        return quarter_img,full_img


class InferenceDataset(CTDataset):
    def __init__(self, data_root, mode='train', transform=None) -> None:
        super().__init__(data_root=data_root, 
                         mode=mode, 
                         transform=transform, 
                         augment_dict=None, 
                         patches=(False,0,0))
        
    def __getitem__(self, index):
        
        quarter_file = os.path.basename(self.quarter_image_list[index])
        full_file = os.path.basename(self.full_image_list[index])
        quarter_image = pil_loader(self.quarter_image_list[index])
        full_image = pil_loader(self.full_image_list[index])
        
        if self.transform is not None:
            quarter_image = self.transform(quarter_image)
            full_image = self.transform(full_image)
        
        return quarter_image,full_image,quarter_file,full_file
    
    def __len__(self):
        return super().__len__()