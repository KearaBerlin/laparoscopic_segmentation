import os, sys
from torch._C import dtype
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as vutils
import torch
import numpy as np
import math
import cv2
import albumentations
import multiprocessing

class CobotLoaderBinary(Dataset):

    def __init__(self, root_dir, label, num_labels, transform, image_size=None, id=-1):
        self.root_dir = root_dir
        self.images = []
        self.labels = []

        self.label = label

        self.transform = transform

        self.num_pixels = 0
        self.num_bg_pixels = 0
        self.files = []
        self.a_masks = []

        self.id = id
    
        for file in os.listdir(root_dir):
            if "png" in file and file[0:5] == "image": 
                file = os.path.join(root_dir, file)
                img = cv2.imread(file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.files.append(file)
   
                if image_size is not None:
                    img = cv2.resize(img,image_size)

                mask_file = file.replace("image", "mask")
                mask_orig = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
                
                if image_size is not None:
                    mask_orig = cv2.resize(mask_orig,image_size, interpolation=cv2.INTER_NEAREST)
                
                self.num_bg_pixels += np.sum(mask_orig == 0)
                self.num_pixels += mask_orig.shape[0]*mask_orig.shape[1]

                mask = (mask_orig > 0)*label

                self.images.append(img)
                self.labels.append(mask)

    def get_frequency(self):
        return self.num_bg_pixels, self.num_pixels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]
            
        if not self.transform is None:
            transformed = self.transform(image=img, mask=mask)

            img = transformed["image"]
            mask = transformed["mask"]
      
        return img, mask
        