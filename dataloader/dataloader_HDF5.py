#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: dataloader_hd5.py
# Created Date: Monday January 11th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 16th November 2022 12:34:03 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import torch
import random
import h5py
from torch.utils import data
from torchvision import transforms as T


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.length = len(self.loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            self.lr, self.hr,self.lr_2 = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.lr, self.hr,self.lr_2 = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.hr = self.hr.cuda(non_blocking=True)
            self.lr = self.lr.cuda(non_blocking=True)
            self.lr_2 = self.lr_2.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.__preload__()
        return self.lr, self.hr, self.lr_2

    def __len__(self):
        """Return the number of images."""
        return self.length


class HDF5Dataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                 h5_path,
                 data_transform=None,
                 seed=1234):
        """Initialize and preprocess the lmdb dataset."""
        self.h5_path = h5_path
        self.h5file = h5py.File(h5_path, 'r')
        if not self.h5file.__contains__("__len__"):
           print("Error")
        self.keys = self.h5file["__len__"][()]  # 86366
        # self.keys   = self.keys[0]
        self.length = self.keys
        self.data_transform = data_transform
        self.keys = [str(k) for k in range(self.keys)]
        random.seed(seed)
        random.shuffle(self.keys)

    def __getitem__(self, index):
        """Return low-resolution frames and its corresponding high-resolution."""
        iii = self.keys[index]
        hr = self.h5file[iii + "hr"][()]
        lr = self.h5file[iii + "lr"][()]
        lr_2 = self.h5file[iii + "lr_from_hr"][()]

        if self.data_transform is not None:
            hr = self.data_transform(hr)
            lr = self.data_transform(lr)
            lr_2 = self.data_transform(lr_2)

        return lr, hr,lr_2

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.h5_path + ')'


def GetLoader(hdf5_dir,
              batch_size=16,
              random_seed=1234,
              num_workers=8):
    """Build and return a data loader."""

    c_transforms = []

    c_transforms.append(T.RandomHorizontalFlip())

    c_transforms.append(T.RandomVerticalFlip())

    c_transforms = T.Compose(c_transforms)

    c_transforms = None

    content_dataset = HDF5Dataset(hdf5_dir, c_transforms, random_seed)
    content_data_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size,
                                          drop_last=True, shuffle=True, num_workers=num_workers, pin_memory=True)
    content_data_loader = DataPrefetcher(content_data_loader)
    return content_data_loader

def GetLoader2(hdf5_dir,
              batch_size=16,
              random_seed=1234,
              num_workers=8):
    """Build and return a data loader."""

    c_transforms = []

    c_transforms.append(T.RandomHorizontalFlip())

    c_transforms.append(T.RandomVerticalFlip())

    c_transforms = T.Compose(c_transforms)

    c_transforms = None

    content_dataset = HDF5Dataset(hdf5_dir, c_transforms, random_seed)
    content_data_loader = data.DataLoader(dataset=content_dataset, batch_size=batch_size,
                                          drop_last=True, shuffle=True, num_workers=num_workers, pin_memory=True)
    content_data_loader = DataPrefetcher(content_data_loader)
    return content_data_loader


if __name__ == "__main__":

    dataset_path = "/public/001/suqingguo/RainNet_HR2LR/RainNet_radio_06_08/test/Test_RainNet_Patches.hdf5"
    s_transforms = []

    s_transforms.append(T.RandomHorizontalFlip())

    s_transforms.append(T.RandomVerticalFlip())
    s_transforms = T.Compose(s_transforms)
    s_transforms = None

    hdf5_dataloader = GetLoader(dataset_path, 1)
    print(len(hdf5_dataloader))
    # hdf5_dataloader = iter(hdf5_dataloader)
    import time
    import datetime

    start_time = time.time()
    for i in range(1):
        lr, _,lr2 = hdf5_dataloader.next()
        # lr,hr = next(hdf5_dataloader)
        data1 = lr
        data2 = lr2
        print("year: ", data1[0, 1, 0, 0])
        print("year: ", data1[0, 1, 0, 0])
        print("data2.shape: ", data2.shape)
        print(data2.shape)
        # hr = hr +1
    elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Elapsed [{}]".format(elapsed))
