import itertools
import os, sys
import random

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

from seg_gen import SegGen


class CobotLoaderBinary(Dataset):

    def __add_file(self, img, mask_orig):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)

        if self.image_size is not None:
            mask_orig = cv2.resize(mask_orig, self.image_size, interpolation=cv2.INTER_NEAREST)

        self.num_bg_pixels += np.sum(mask_orig == 0)
        self.num_pixels += mask_orig.shape[0] * mask_orig.shape[1]

        mask = (mask_orig > 0) * 1

        self.images.append(img)
        self.labels.append(mask)

    def __generate_aug(self, k, seed=False):
        assert 0 <= k <= 1.0
        print("Generate_Augs/k: ",k)
        num = len(self.files)
        k_num = num * k
        perms = list(itertools.permutations(range(num), 2))
        pairs = random.sample(perms, int(k_num * k_num))
        if seed:
            rng = random.Random(42)
            pairs = rng.sample(perms, int(k_num * k_num))
        gens = dict()
        print("Generate_Augs/perms: ",perms)
        print("Generate_Augs/gens: ",gens)

        for p in pairs:
            gen = gens.get(p[0])
            img_pair1 = self.files[p[0]]
            img_pair2 = self.files[p[1]]
            if gen is None:
                gen = SegGen(img_pair1[0], img_pair1[1])
                gens[p[0]] = gen
            img = gen.generate(img_pair2[0], img_pair2[1])

            mask_orig = cv2.imread(img_pair1[1], cv2.IMREAD_GRAYSCALE)
            self.__add_file(img, mask_orig)
            
    def __generate_aug_single(self, idx, weights=None):
        #weights is an optional dictionary of (idx,weight) to set pair selection. None defaults to uniform, global selection
        
        img_pair1 = self.files[idx]
        idx2 = np.random.choice(len(self.files),p=weights)
        img_pair2 = self.files[idx2]
        
        gen = self.aug_gens.get(idx)
        if gen is None:
            gen = SegGen(img_pair1[0], img_pair1[1])
            self.aug_gens[idx] = gen
        img = gen.generate(img_pair2[0], img_pair2[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)
        #mask_orig = cv2.imread(img_pair1[1], cv2.IMREAD_GRAYSCALE)
        #self.__add_file(img, mask_orig)
        return img

    def __init__(self, root_dir, label, num_labels, transform, 
                 image_size=None, id=-1, create_negative_labels=False,
                 aug_method="none", k_aug=0.0, seed=False,batch_size=1):

        self.root_dir = root_dir
        self.images = []
        self.labels = []

        self.label = label

        self.transform = transform
        self.create_negative_labels = create_negative_labels

        self.num_pixels = 0
        self.num_bg_pixels = 0
        self.files = []
        self.a_masks = []

        self.id = id
        self.num_labels = num_labels
        self.image_size = image_size
        
        self.aug_gens=dict()
        self.aug_method=aug_method
        self.k_aug=k_aug
        
        if seed:
            np.random.seed(seed)
        
        #Batchwise not tested! -- This is only for mutual augmentation within batch.
        if 'batchwise' in self.aug_method and batch_size >10:
            print('Dataloader:__init__  Using Batchwise augmentation pairing')
            self.batch_size=batch_size
            self.batch_idx=0
            self.batch_current=[None]*batch_size
        else:
            self.batch_size=1
            print('Dataloader:__init__  Using Global augmentation pairing')

        self.files = []
        for file in os.listdir(self.root_dir):
            if "png" in file and file[0:5] == "image":
                file = os.path.join(self.root_dir, file)
                self.files.append((file, file.replace("image", "mask")))

        for pair in self.files:
            file, mask_file = pair
            img = cv2.imread(file)
            mask_orig = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            self.__add_file(img, mask_orig)

    def get_frequency(self):
        return self.num_bg_pixels, self.num_pixels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]
        do_aug=False
        if self.k_aug > 0:
            #Choose augmentation with probability k for each sample
            do_aug=np.random.choice((False,True),self.batch_size,(1-self.k_aug,self.k_aug))
        
        
        if do_aug:
            if 'rand_pair' in self.aug_method:
                img=self.__generate_aug_single(idx)

        if self.create_negative_labels:
            masks = []
            mask_dict = {}
            mask = self.labels[idx]
            mask_dict["mask"] = mask

            neg_mask = mask - 1   

            for i in range(self.num_labels):
                str = "mask%d" % i
                if i == self.label - 1:
                    c_mask = mask
                else:
                    c_mask = neg_mask
                mask_dict[str] = c_mask

            transformed = self.transform(image=self.images[idx], **mask_dict)

            for i in range(self.num_labels):
                str = "mask%d" % i
                masks.append(transformed[str])
            
            mask = np.stack(masks, axis=0)
            img = transformed["image"]
            mask_orig = transformed["mask"]

            return img, mask, self.label, mask_orig
            
        if not self.transform is None:
            transformed = self.transform(image=img, mask=mask)

            img = transformed["image"]
            mask = transformed["mask"]
            
        print('Dataloader::Getitem/Index,Aug',idx,do_aug)
        return img, mask, self.label
        
