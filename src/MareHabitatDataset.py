from __future__ import print_function, division
import os
import pickle
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MareHabitatDataset(Dataset):
    """PyTorch Dataset for MARE's Oceana videos, to be represented as frames
       includes habitat class of frames"""

    def __init__(self, id_lst_pth, pickle_home, image_home, transforms, to_crop7575=True, habs_or_hbts='habs'):
        self.image_home = image_home
        self.pickle_home = pickle_home
        self.transforms = transforms
        self.to_crop7575 = to_crop7575
        self.type = habs_or_hbts
        self.habitats = self.prepare_habitats(pickle_home)
        print('ordered {} are : {}'.format(self.type, self.habitats))
        with open(id_lst_pth, 'r') as f:
            lines = f.readlines()
            id_lst = [l.strip() for l in lines]
        self.id_lst = id_lst
        return
 
    def __getitem__(self, idx):
        ide = self.id_lst[idx]
        orig_id, frame = ide.split('_')
        with open(os.path.join(self.pickle_home, orig_id + '_{}.p'.format(self.type)), 'rb') as f:
            hannos = pickle.load(f)
        habs = hannos[int(frame)]
        # was getting some No data around here
        if len(habs) < 1: print('no {} for {}'.format(self.type, ide))

        habs = list(habs)
        if 'Off Transect' in habs:
            habs.remove('Off Transect')
        if 'No data' in habs:
            habs.remove('No data')
        inds = [self.habitats.index(h) for h in habs]
        one_hot = np.zeros(len(self.habitats))
        one_hot[inds] = 1

        arr = np.load(os.path.join(self.image_home, ide + '.npy'))
        h,w,c = arr.shape
        # crop out top and sides
        arr_7575 = arr[int(0.25*h):, int(0.125*w):int(0.875*w),:]
        img = Image.fromarray(arr_7575)
        # use transforms
        tens = self.transforms(img)

        sample = {'image': tens, 'label': one_hot, 'rgb_crop': arr_7575, 'id': ide}
        return sample

    def __len__(self):
        return len(self.id_lst)

    def prepare_habitats(self, pth):
        ''' read all the pickles and come up with all habitats in the dataset
            so that we can count number of classes and one hot encode
        '''
        all_habs = set()
        for fname in os.listdir(pth):
            if self.type not in fname: continue
            if '.py' in fname: continue
            with open(os.path.join(pth, fname), 'rb') as f:
                hannos = pickle.load(f)
            for subs in hannos.substrates:
                all_habs = all_habs.union(subs)

        habs_lst = []
        habs_dict = {}
        for i,hab in enumerate(all_habs):
            habs_lst.append(hab)
            habs_dict[i] = hab

        if 'Off Transect' in habs_lst:
            habs_lst.remove('Off Transect')
        habs_lst.sort()
        return habs_lst


class MareHabitatDatasetInference(Dataset):
    """PyTorch Dataset for MARE's Oceana videos, to be represented as frames
       includes habitat class of frames"""

    def __init__(self, id_lst_path, image_home, transforms, to_crop7575=True):
        
        self.image_home = image_home
        self.transforms = transforms
        self.to_crop7575 = to_crop7575
        
        with open(id_lst_path, 'r') as f:
            lines = f.readlines()
            id_lst = [l.strip() for l in lines]
        self.id_lst = id_lst
        
        return
 
    def __getitem__(self, idx):
        ide = self.id_lst[idx]

        arr = np.load(os.path.join(self.image_home, ide + '.npy'))
        h,w,c = arr.shape
        # crop out top and sides
        arr_7575 = arr[int(0.25*h):, int(0.125*w):int(0.875*w),:]
        img = Image.fromarray(arr_7575)
        # use transforms
        tens = self.transforms(img)

        sample = {'image': tens, 'rgb_crop': arr_7575, 'id': ide}
        return sample

    def __len__(self):
        return len(self.id_lst)
