import itertools
import os, sys
import random
from pathlib import Path

from torch._C import dtype
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as vutils
import torch
import numpy as np
import pandas as pd
import math
import cv2
import albumentations
import multiprocessing

from seg_gen2 import SegGen2


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

        num = len(self.organ_ii)
        k_num = num * k
        perms = list(itertools.permutations(range(num), 2))
        pairs = random.sample(perms, int(k_num * k_num))
        if seed:
            rng = random.Random(42)
            pairs = rng.sample(perms, int(k_num * k_num))
        gens = dict()

        for p in pairs:
            i_0 = self.organ_ii[p[0]]
            i_1 = self.organ_ii[p[1]]

            gen = gens.get(i_0)
            img_pair1 = self.files[i_0]
            img_pair2 = self.files[i_1]
            
            if gen is None:
                gen = SegGen2(img_pair1[0], img_pair1[1])
                gens[p[0]] = gen
            img = gen.generate(img_pair2[0], img_pair2[1])

            mask_orig = cv2.imread(img_pair1[1], cv2.IMREAD_GRAYSCALE)
            self.__add_file(img, mask_orig)

    def add_file_and_mask(self, file):
        
        mask_filename = file.replace("image", "mask")
        self.files.append((file, mask_filename))

    def add_neg_imgs(self):
        root = Path(self.root_dir)
        data_dir = root.parent.parent.absolute()
        print(f"data dir: {data_dir}")

        subfolders = [f for f in os.scandir(data_dir) if f.is_dir()]

        for folder in subfolders:
            if folder.name in ["multilabel",self.organ_name]:
                continue

            id_subfolder = os.path.join(data_dir,folder.name,root.name)
            if os.path.exists(id_subfolder):
                files = [f for f in os.scandir(id_subfolder) if f.is_file()]

                df = pd.read_csv(os.path.join(id_subfolder, 'weak_labels.csv'), header=None)
                neg_img_df = df[df.iloc[:, self.organ_id+1] == 0]
                neg_imgs = list(neg_img_df.iloc[:,0])
                # the file names are different for some reason, so fix them
                neg_imgs = [fn.replace("images0","image") for fn in neg_imgs]

                for file in files:
                    rn = random.uniform(0,1)
                    if file.name in neg_imgs and rn > self.p_neg_img:
                        self.add_file_and_mask(os.path.join(id_subfolder, file.name))


    def __init__(self, root_dir, label, num_labels, transform, 
                 image_size=None, id=-1, create_negative_labels=False,
                 organ_id=None, organ_name=None, p_neg_img=0.1,
                 aug_method="none", k_aug=0.0, seed=False):

        self.root_dir = root_dir
        self.images = []
        self.labels = []

        self.label = label
        self.organ_name = organ_name
        self.organ_id = organ_id

        self.transform = transform
        self.create_negative_labels = create_negative_labels
        self.p_neg_img = p_neg_img 	# percent of images with all-0 mask to include

        self.num_pixels = 0
        self.num_bg_pixels = 0
        self.files = []
        self.a_masks = []

        self.id = id
        self.num_labels = num_labels
        self.image_size = image_size

        self.files = []
        self.organ_ii = []
        i = 0
        for file in os.listdir(self.root_dir):
            if "png" in file and file[0:5] == "image":
                file = os.path.join(self.root_dir, file)
                self.add_file_and_mask(file)
                self.organ_ii.append(i)
                i += 1

        if self.p_neg_img > 0:
            self.add_neg_imgs()
            
        for i, (file, mask_file) in enumerate(self.files):
            img = cv2.imread(file)
            mask_orig = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            self.__add_file(img, mask_orig)

        if k_aug > 0 and aug_method == "rand_pair":
            self.__generate_aug(k_aug, seed)

        print(f"files: {len(self.files)}")

    def get_frequency(self):
        return self.num_bg_pixels, self.num_pixels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]

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
      
        return img, mask, self.label
        
