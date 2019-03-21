import os
import random
import pathlib

import torch
import torch.utils.data as data
import torchvision
import numpy as np
import pandas as pd
import PIL
import PIL.Image


class SiamesePairDataset(data.Dataset):
    def __init__(self, root, ext='jpg', transform=None, pair_transform=None, target_transform=None):
        super(SiamesePairDataset, self).__init__()
        self.transform = transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        self.root = root

        self.base_path = pathlib.Path(root)
        self.files = sorted(list(self.base_path.glob("*/*." + ext)))
        self.files_map = self._files_mapping()
        self.pair_files = self._pair_files()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        (imp1, imp2), sim = self.pair_files[idx]
        im1 = PIL.Image.open(imp1)
        im2 = PIL.Image.open(imp2)

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        if self.pair_transform:
            im1, im2 = self.transform_pair(im1, im2)

        if self.target_transform:
            sim = self.target_transform(sim)
        return im1, im2, sim

    def _files_mapping(self):
        dirname = []
        filename = []
        dct = {}
        for f in self.files:
            spl = str(f).split('/')
            dirname = spl[-2]
            filename = spl[-1]

            if dirname not in dct.keys():
                dct.update({dirname: []})
            else:
                dct[dirname].append(filename)
                dct[dirname] = sorted(dct[dirname])
        return dct

    def _similar_pair(self):
        fmap = self.files_map
        atp = {}
        c = 0
        for key in fmap.keys():
            atp.update({key: []})
            n = len(fmap[key])
            ctp = ((n - 1) * n) + n
            for i in range(n):
                for j in range(n):
                    fp = os.path.join(key, fmap[key][i])
                    fo = os.path.join(key, fmap[key][j])
                    atp[key].append(((fp, fo), 0))
        return atp

    def _len_similar_pair(self):
        fmap = self.files_map
        dct = {}
        spair = self._similar_pair()
        for key in fmap.keys():
            dd = {key: len(spair[key])}
            dct.update(dd)
        return dct

    def _diff_pair_dircomp(self):
        fmap = self.files_map
        dirname = list(fmap.keys())
        pair_dircomp = []
        for idx in range(len(dirname)):
            dirtmp = dirname.copy()
            dirtmp.pop(idx)
            odir = dirtmp
            pdir = dirname[idx]
            pdc = (pdir, odir)
            pair_dircomp.append(pdc)
        return pair_dircomp

    def _different_pair(self):
        fmap = self.files_map
        pair_sampled = {}
        pair_dircomp = self._diff_pair_dircomp()
        len_spair = self._len_similar_pair()
        for idx, (kp, kvo) in enumerate(pair_dircomp):
            val_pri = fmap[kp]
            num_sample = len(val_pri) // 4

            pair_sampled.update({kp: []})
            for vp in val_pri:
                # get filename file primary
                fp = os.path.join(kp, vp)
                for ko in kvo:
                    vov = fmap[ko]
                    pair = []
                    for vo in vov:
                        fo = os.path.join(ko, vo)
                        pair.append(((fp, fo), 1))
                    mout = random.sample(pair, num_sample)
                    pair_sampled[kp].append(mout)

        for key in pair_sampled.keys():
            val = pair_sampled[key]
            num_sample = len_spair[key]
            tmp_val = []
            for va in val:
                for v in va:
                    tmp_val.append(v)
            pair_sampled[key] = random.sample(tmp_val, num_sample)

        return pair_sampled

    def _pair_files(self):
        fmap = self.files_map
        base_path = self.root
        sim_pair = self._similar_pair()
        diff_pair = self._different_pair()
        files_list = []
        for key in fmap.keys():
            spair = sim_pair[key]
            dpair = diff_pair[key]
            n = len(spair)
            for i in range(n):
                spair_p = os.path.join(base_path, spair[i][0][0])
                spair_o = os.path.join(base_path, spair[i][0][1])
                spair[i] = ((spair_p, spair_o), 0)

                dpair_p = os.path.join(base_path, dpair[i][0][0])
                dpair_o = os.path.join(base_path, dpair[i][0][1])
                dpair[i] = ((dpair_p, dpair_o), 1)

                files_list.append(spair[i])
                files_list.append(dpair[i])

        return files_list

