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
from math import inf
import cv2
import albumentations
import multiprocessing

from seg_gen import SegGen
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
        #print("AddFile::Mask ",mask.shape,mask_manual)
        _, mask_image=cv2.threshold(mask_orig, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow("orig_mask",mask_orig)
        #cv2.imshow("thrsh_mask",mask)
        #cv2.imshow("trsh_manual",mask_manual)
        #cv2.waitKey()
        #sys.exit()

        self.images.append(img)
        self.labels.append(mask)
        #self.masks.append(mask_image)

    def __generate_aug(self, k, seed=False):
        assert 0 <= k <= 1.0

        #print("Generate_Augs/k: ",k)
        num = len(self.organ_ii)
        #print("Generate_Augs/num: ",num)

        k_num = num * k
        perms = list(itertools.permutations(range(num), 2))
        #print("Generate_Augs/perms: ",perms)
        pairs = random.sample(perms, int((k_num) * (k_num)))
        if seed:
            rng = random.Random(42)
            pairs = rng.sample(perms, int((k_num) * (k_num)))
        gens = dict()
        
        #print("Generate_Augs/gens: ",gens)

        for p in pairs:
            #With Similar Pair enabled, skip pass if items are dissimilar.
            S=self.__get_item_pair_similarity_fast(p[0],p[1])
            #print("Generate_Augs/S: ",S)
            if "sim_pair" in self.aug_method and S<self.sim_score:
                continue
                
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

    """
    def __get_item_pair_similarity(self,idx1,idx2):
        #gray = cv2.cvtColor(self.labels[idx1], cv2.COLOR_BGR2GRAY)
        #_, thresh = cv2.threshold(self.labels[idx1], 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(self.masks[idx1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #ref_gray = cv2.cvtColor(self.labels[idx2], cv2.COLOR_BGR2GRAY)
        #_, ref_thresh = cv2.threshold(self.labels[idx2], 127, 255, cv2.THRESH_BINARY_INV)
        ref_contours, _ = cv2.findContours(self.masks[idx2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #ref_contour = max(ref_contours, key=cv2.contourArea)
        similarity = cv2.matchShapes(ref_contours[0], contours[0], cv2.CONTOURS_MATCH_I1,0.0)
        #print(f"Dataloader::Similarity({self.files[idx1][1]},{self.files[idx2][1]})={similarity}")

        return similarity
    """
    def __get_item_pair_similarity_fast(self,idx1,idx2):
        dist=np.bitwise_xor(self.labels[idx1],self.labels[idx2])
        similarity=np.sum(dist)/(self.image_size[0]*self.image_size[1])
        #print(f"SimilarityFast: S({self.files[idx1][1]},{self.files[idx2][1]})={similarity}")
        return similarity
    """    
    def __generate_aug_single(self, idx, similarity=-1):
        #weights is an optional dictionary of (idx,weight) to set pair selection. None defaults to uniform, global selection
        img_pair1 = self.files[idx]
        idx2 = self.rng.choice(len(self.files))
        img_pair2 = self.files[idx2]
        #print("Generate_Augs:: similarity,sim_score: ",similarity,self.sim_score)
        iter_lim=0
        while iter_lim < 10 and similarity>=np.abs(self.sim_score):
            #idx2 = np.random.choice(len(self.files))
            idx2=self.rng.choice(len(self.files))
            img_pair2 = self.files[idx2]
            similarity=self.__get_item_pair_similarity(idx,idx2)
            #print(f"Generate_Augs::PairSimilarity S={similarity}")
            iter_lim+=1
        if (iter_lim>=10):
            print("Generate_Augs::PairSimilarity Too many tries to meet similarity threshold")
        
        gen = self.aug_gens.get(idx)
        if gen is None:
            gen = SegGen2(img_pair1[0], img_pair1[1])
            self.aug_gens[idx] = gen
        img = gen.generate(img_pair2[0], img_pair2[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)

        return img
        """
    def add_file_and_mask(self, file):
        
        mask_filename = file.replace("image", "mask")
        self.files.append((file, mask_filename))

    def add_neg_imgs(self):
        root = Path(self.root_dir)
        data_dir = root.parent.parent.absolute()
        #print(f"data dir: {data_dir}")

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
                 aug_method="none", k_aug=0.0, seed=False, batch_size=1,sim_score=None):

        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.masks = []

        self.label = label
        self.organ_name = organ_name
        self.organ_id = organ_id

        self.transform = transform
        self.create_negative_labels = create_negative_labels
        self.p_neg_img = p_neg_img 	# percent of images with all-0 mask to include

        self.num_pixels = 0
        self.num_bg_pixels = 0
        self.a_masks = []

        self.id = id
        self.num_labels = num_labels
        self.image_size = image_size
        
        
        """
        self.rng = np.random.default_rng()
        #Batchwise not tested! -- This is only for mutual augmentation within batch.
        if 'batchwise' in self.aug_method and batch_size >10:
            #print('Dataloader:__init__  Using Batchwise augmentation pairing')
            self.batch_size=batch_size
            self.batch_idx=0
            self.batch_current=[None]*batch_size
        else:
            self.batch_size=1
            #print('Dataloader:__init__  Using Global augmentation pairing')
        """
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
            
        self.N=len(self.files)
        self.aug_gens=dict()
        self.aug_method=aug_method
        self.k_aug=k_aug

        if sim_score is not None:
            self.sim_score=sim_score
        else:
            self.sim_score=0
            
        if "none" not in self.aug_method:
            self.__generate_aug(k_aug,seed)

    def get_frequency(self):
        return self.num_bg_pixels, self.num_pixels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]
        
        """
        do_aug=False
        k=self.k_aug
        if k > 0:
            #Choose augmentation with probability k for each sample
            n=self.N
            q=(n*k)/(n+(n*k)) #Calculate P of choosing augmented sample from augmentation fraction k
            assert(q<1)
            p=1-q
            #do_aug=np.random.Generator.choice((False,True),size=self.batch_size,p=(p,q))
            do_aug=self.rng.choice((False,True),size=self.batch_size,p=(p,q))[0]
        
        if do_aug == True:
            #print("Getitem:AugMethod ",self.aug_method)
            if "rand_pair" in self.aug_method:
                #print("Getitem/aug_method -- Detected Global Rand Pair")
                self.sim_score=inf
                img=self.__generate_aug_single(idx)
            if "sim_pair" in self.aug_method and self.sim_score is not None:
                #print("Getitem/aug_method -- Detected Global Similar Pair")
                img=self.__generate_aug_single(idx,self.sim_score)
        """        

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
            
        #print('Dataloader::Getitem/Index,Aug',idx,do_aug)
        return img, mask, self.label
        
