import os
import random
import pathlib
from bisect import insort_right
from collections import defaultdict

import torch.utils.data as data
import PIL
import PIL.Image
from torchwisdom.vision.transforms import transforms as ptransforms


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
            im1, im2 = self.pair_transform(im1, im2)

        if self.target_transform:
            sim = self.target_transform(sim)
        return im1, im2, sim

    def _files_mapping(self):
        dct = defaultdict(list)
        for f in self.files:
            dirname = f.parent.name
            filename = f.name
            insort_right(dct[dirname], filename)
        return dct

    def _similar_pair(self):
        fmap = self.files_map
        atp = defaultdict(list)
        for _dir in fmap.keys():
            n = len(fmap[_dir])
            for i in range(n):
                for j in range(n):
                    fp = os.path.join(_dir, fmap[_dir][i])
                    fo = os.path.join(_dir, fmap[_dir][j])
                    atp[_dir].append(((fp,fo),0))
        return atp

    def _len_similar_pair(self):
        spair = self._similar_pair()
        return {key: len(spair[key]) for key in spair}

    def _diff_pair_dircomp(self):
        fmap = self.files_map
        return [(_class, list(filter(lambda other_class: other_class is not _class, fmap))) for _class in fmap]

    def _different_pair(self):
        fmap = self.files_map
        pair_sampled = defaultdict(list)
        pair_dircomp = self._diff_pair_dircomp()
        len_spair = self._len_similar_pair()
        for idx, (kp, kvo) in enumerate(pair_dircomp):
            val_pri = fmap[kp]
            num_sample = len(val_pri) // 4 if len(val_pri) >= 4 else len(val_pri)
            for vp in val_pri:
                # get filename file primary
                fp = os.path.join(kp, vp)
                for ko in kvo:
                    vov = fmap[ko]
                    pair = []
                    for vo in vov:
                        fo = os.path.join(ko, vo)
                        pair.append(((fp, fo), 1))
                    if len(pair)>num_sample:
                        mout = random.sample(pair, num_sample)
                    else:
                        mout = pair
                    pair_sampled[kp].append(mout)

        for key in pair_sampled.keys():
            val = pair_sampled[key]
            num_sample = len_spair[key]
            tmp_val = []
            for va in val:
                for v in va:
                    tmp_val.append(v)

            if len(tmp_val) > num_sample:
                pair_sampled[key] = random.sample(tmp_val, num_sample)
            else:
                pair_sampled[key] = tmp_val
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



if __name__ == '__main__':
    train_tmft = ptransforms.PairCompose([
        ptransforms.PairResize((220)),
        ptransforms.PairRandomRotation(20),
        ptransforms.PairToTensor(),
    ])
    root = '/data/att_faces_new/valid'
    sd = SiamesePairDataset(root, ext="pgm", pair_transform=train_tmft)
    # loader = data.DataLoader(sd, batch_size=32, shuffle=True)
    print(sd.__getitem__(0))