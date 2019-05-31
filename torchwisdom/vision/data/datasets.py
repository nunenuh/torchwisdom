import os
import random
import pathlib
from typing import List, Any
from tqdm import tqdm

import torch.utils.data as data
import PIL
import PIL.Image
from torchwisdom.vision.transforms import pair as pair_transforms
import torchvision.transforms as transforms
from torchvision import datasets
from typing import *
import torch


class ImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolder, self).__init__(root, transform, target_transform)

    def sample(self, num, shuffle=False, use_classes=True):
        samples = self.samples
        if shuffle: random.shuffle(samples)
        data = samples[0:num]
        samples, targets = [], []
        for idx, (path, target) in enumerate(data):
            sample = self.loader(path)
            samples.append(sample)
            targets.append(target)
        if use_classes:
            targets = self.target_classes(targets)
        return samples, targets

    def target_classes(self, targets):
        out = []
        for t in targets:
            out.append(self.classes[t])
        return out


class SiamesePairDataset(data.Dataset):
    def __init__(self, root, ext: str = 'jpg', glob_pattern: str = "*/*.",
                 similar_factor: float = 1., different_factor: float = 0.38,
                 micro_factor: float = 0.38,
                 transform: transforms.Compose = None,
                 pair_transform: pair_transforms.PairCompose = None,
                 target_transform: transforms.Compose = None):
        super(SiamesePairDataset, self).__init__()
        self._init_seed()
        self.similar_factor = similar_factor
        self.micro_factor = micro_factor
        self.different_factor = different_factor

        self.transform: transforms.Compose = transform
        self.pair_transform: pair_transforms.PairCompose = pair_transform
        self.target_transform: transforms.Compose = target_transform
        self.root: str = root

        # print(f"Files Mapping from {self.root}, please wait...")
        self.base_path = pathlib.Path(root)
        self.files = sorted(list(self.base_path.glob(glob_pattern + ext)))
        self.files_map = self._files_mapping()
        self.classes = list(self.files_map.keys())
        self.similar_pair = self._similar_pair()
        self.different_pair = self._different_pair()
        self.pair_files = self._pair_files()

    @staticmethod
    def _init_seed():
        random.seed(1261)

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
        return (im1, im2), sim

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

    def _similar_pair(self) -> List:
        # print("Generating Similar Pair, please wait...")
        fmap = self.files_map
        # atp = {}
        similar = []
        text = f'generating similar pair from\t {self.root}'
        bar = tqdm(fmap.keys(), desc=text)
        for key in bar:
            n = len(fmap[key])
            for (idz, idj, sim) in self._similar_sampling_generator(n):
                fz = os.path.join(self.base_path, key, fmap[key][idz])
                fj = os.path.join(self.base_path, key, fmap[key][idj])
                # atp[key].append(((fz, fj), 0))
                similar.append(((fz, fj), 0))
        num_sample = int(len(similar) * self.similar_factor)
        similar = random.sample(similar, num_sample)
        return similar

    def _similar_sampling_generator(self, n):
        num_sample = int(n * n * self.micro_factor)
        list_similar = [(i, j, 0) for i in range(n) for j in range(n)]
        sampled = random.sample(list_similar, num_sample)
        return sampled

    def _len_similar_pair(self):
        dct = {}
        for key in self.files_map.keys():
            dd = {key: len(self.similar_pair[key])}
            dct.update(dd)
        return dct

    def _pair_class_to_other(self) -> List:
        # print("Generating pair class to other class, please wait...")
        num = len(self.classes)
        list_idx = [i for i in range(num)]
        pair = []
        for idz in range(num):
            other_list = list_idx.copy()
            other_list.pop(idz)
            for idj in other_list:
                pair.append((idz, idj))
        num_sample = int(len(pair) * self.different_factor)
        pair = random.sample(pair, num_sample)
        # print("Generating pair class to other class, finished...")
        return pair

    def _diff_sampling_generator(self):
        # print("Creating diff sampling generator, please wait...")
        list_sampled: List[Any] = []
        for idx, (cidx, oidx) in enumerate(self._pair_class_to_other()):
            cname, cother = self.classes[cidx], self.classes[oidx]
            num_cname, num_cother = len(self.files_map[cname]), len(self.files_map[cother])
            num_sample = int(num_cname * num_cother * self.micro_factor)
            list_diff = [((cidx, i), (oidx, j), 1) for i in range(num_cname) for j in range(num_cother)]
            sampled = random.sample(list_diff, num_sample)
            list_sampled += sampled
        # print("Creating diff sampling generator, finished...")
        return list_sampled

    def _different_pair(self):
        # print("Generating Different Pair, please wait...")
        diff = []
        text = f'generating different pair from\t {self.root}'
        bar = tqdm(self._diff_sampling_generator(), desc=text)
        for z, j, sim in bar:
            zname, idz = self.classes[z[0]], z[1]
            jname, idj = self.classes[j[0]], j[1]
            zfile = self.files_map[zname][idz]
            jfile = self.files_map[jname][idj]
            fz = os.path.join(self.base_path, zname, zfile)
            fj = os.path.join(self.base_path, jname, jfile)
            diff.append(((fz, fj), 1))

        sim_len = len(self.similar_pair)
        if len(diff) > sim_len:
            diff = random.sample(diff, sim_len)
        # print("Generating Different Pair, finished...")
        return diff

    def _pair_files(self):
        sim_pair = self.similar_pair
        diff_pair = self.different_pair
        all_pair = sim_pair + diff_pair
        return all_pair


class AutoEncoderDataset(data.Dataset):
    def __init__(self, root, feature_dir: str = None, target_dir: str = None,
                 feature_transform: transforms.Compose = None,
                 pair_transform: pair_transforms.PairCompose = None,
                 target_transform: transforms.Compose = None,
                 limit_size: Union[int, float] = 0, limit_type: Union[int, float] = int,
                 split_dataset: bool = False, mode: str = 'train', valid_size: float = 0.2):
        super(AutoEncoderDataset, self).__init__()
        self.root = pathlib.Path(root)
        self.feature_dir = feature_dir
        self.target_dir = target_dir
        self.split_dataset = split_dataset
        self.mode = mode
        self.valid_size = valid_size

        self.feature_path = self.root.joinpath("feature")
        if feature_dir is not None:
            self.feature_path: pathlib.Path = self.root.joinpath(feature_dir)

        self.target_path = self.root.joinpath("target")
        if target_dir is not None:
            self.target_path: pathlib.Path = self.root.joinpath(target_dir)

        self.feature_transform = feature_transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        self.limit_size = limit_size
        self.limit_type = limit_type

        self.feature_files = None
        self.target_files = None
        self.train_feature_files = None
        self.train_target_files = None
        self.valid_feature_files = None
        self.valid_target_files = None

        self._build_files()
        self._build_usage()
        if split_dataset:
            self._split_dataset()

    def _build_files(self):
        self.feature_files = sorted(list(self.feature_path.glob("*")))
        self.target_files = sorted(list(self.target_path.glob("*")))
        # print(self.feature_files)
        feat_len = len(self.feature_files)
        targ_len = len(self.target_files)
        assert feat_len == targ_len, f"Total files from feature dir and target " \
            f"dir is not equal ({feat_len}!={targ_len}), expected equal number"

    def _build_usage(self):
        if self.limit_type == float:
            total = int(self.__len__() * self.limit_size)
        elif self.limit_type == int:
            if self.limit_size == 0:
                total = len(self.feature_files)
            else:
                total = self.limit_size
        else:
            total = self.limit_size
        self.feature_files = self.feature_files[0:total]
        self.target_files = self.target_files[0:total]

    def _split_dataset(self):
        random.seed(1261)
        valid_index = []
        size = 0
        len_files = len(self.feature_files)
        list_index = list(range(len_files))
        if self.valid_size > 0:
            size = int(self.valid_size * self.__len__())
        valid_index += random.sample(list_index, size)
        train_index = list(set(list_index) - set(valid_index))
        train_index, valid_index = sorted(train_index), sorted(valid_index)
        # print(train_index)
        self.train_feature_files = [self.feature_files[i] for i in train_index]
        self.train_target_files = [self.target_files[i] for i in train_index]
        self.valid_feature_files = [self.feature_files[i] for i in valid_index]
        self.valid_target_files = [self.target_files[i] for i in valid_index]

        if self.mode == 'train':
            self.feature_files = self.train_feature_files
            self.target_files = self.train_target_files
        else:
            self.feature_files = self.valid_feature_files
            self.target_files = self.valid_target_files

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
