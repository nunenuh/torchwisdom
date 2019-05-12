import os
import random
import pathlib
import torch.utils.data as data
import PIL
import PIL.Image
from torchwisdom.vision.transforms import transforms as ptransforms
from torchvision.datasets import ImageFolder

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
        dct = {}
        for f in self.files:
            spl = str(f).split('/')
            dirname = spl[-2]
            filename = spl[-1]
            if dirname not in dct.keys():
                dct.update({dirname: [filename]})
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
            if len(val_pri)>=4:
                num_sample = len(val_pri) // 4
            else:
                num_sample = len(val_pri)

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


            if len(tmp_val)>num_sample:
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


class AutoEncoderDataset(data.Dataset):
    def __init__(self, root, feature_dirname=None, target_dirname=None,
                 feature_transform=None, pair_transform=None, target_transform=None, **kwargs):
        super(AutoEncoderDataset, self).__init__()
        self.root = pathlib.Path(root)
        self.feature_dirname = feature_dirname
        self.target_dirname = target_dirname
        self.feature_path: pathlib.Path = None
        self.target_path: pathlib.Path = None
        self.feature_files = None
        self.target_files = None
        self.feature_transform = feature_transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        self.kwargs = kwargs
        self._build_path()
        self._build_files()
        self._build_usage()

    def _build_usage(self):
        size = self.kwargs.get("size", 1)
        total = int(self.__len__() * size)
        self.feature_files = self.feature_files[0:total]
        self.target_files = self.target_files[0:total]

    def _build_path(self):
        if self.feature_dirname is None:
            self.feature_path: pathlib.Path = self.root.joinpath('feature')
        else:
            self.feature_path: pathlib.Path = self.root.joinpath(self.feature_dirname)

        if self.target_dirname is None:
            self.target_path = self.root.joinpath('target')
        else:
            self.target_path = self.root.joinpath(self.target_dirname)

    def _build_files(self):
        self.feature_files = sorted(list(self.feature_path.glob("*")))
        self.target_files = sorted(list(self.target_path.glob("*")))
        feat_len = len(self.feature_files)
        targ_len = len(self.target_files)
        assert feat_len == targ_len, "Total files from feature dir and target " \
                                     "dir is not equal, expected equal number"

    def __len__(self):
        feat_len = len(list(self.feature_files))
        return feat_len

    def __getitem__(self, idx: int):
        feature_path = self.feature_files[idx]
        target_path = self.target_files[idx]

        feature = PIL.Image.open(feature_path)
        target = PIL.Image.open(target_path)

        if self.feature_transform:
            feature = self.feature_transform(feature)

        if self.pair_transform:
            feature, target = self.pair_transform(feature, target)

        if self.target_transform:
            target = self.target_transform(target)

        return feature, target




if __name__ == '__main__':
    # train_tmft = ptransforms.PairCompose([
    #     ptransforms.PairResize((220)),
    #     ptransforms.PairRandomRotation(20),
    #     ptransforms.PairToTensor(),
    # ])
    # root = '/data/att_faces_new/valid'
    # sd = SiamesePairDataset(root, ext="pgm", pair_transform=train_tmft)
    # # loader = data.DataLoader(sd, batch_size=32, shuffle=True)
    # print(sd.__getitem__(0))
    ...