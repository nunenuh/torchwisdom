import pathlib
import random
import os
import shutil

class TrainValidSplit(object):
    def __init__(self, src_path, val_size=0.2):
        self.val_size = val_size
        self.src_path = pathlib.Path(src_path)
        self.src_files = sorted(list(self.src_path.glob("*/*.*")))
        self.src_files_map = self._files_map()


    def _files_map(self) -> object:
        files_map = {}
        for f in self.src_files:
            key = str(f).split('/')[-2]
            if key in files_map.keys():
                files_map[key].append(f)
            else:
                files_map.update({key :[f]})
        return files_map

    def _split_map(self):
        bfiles_map = self.src_files_map
        vfiles = {}
        tfiles = {}
        for key in bfiles_map.keys():
            files = bfiles_map[key]
            n = len(bfiles_map[key])
            nd = [i for i in range(n)]
            sampling_size = int( n *self.val_size)
            valid_idx = random.sample(nd, sampling_size)

            set_all = set(nd)
            set_valid = set(valid_idx)
            set_train  = set_all - set_valid
            train_idx = list(set_train)

            for vidx in valid_idx:
                if key in vfiles.keys():
                    vfiles[key].append(files[vidx])
                else:
                    vfiles.update({key :[files[vidx]]})

            for tidx in train_idx:
                if key in tfiles.keys():
                    tfiles[key].append(files[tidx])
                else:
                    tfiles.update({key :[files[tidx]]})
        return tfiles, vfiles

    def execute(self, dst_path:str, mode:str='cp'):
        self.train_files_map, self.valid_files_map = self._split_map()
        if dst_path:
            self.dst_path = pathlib.Path(dst_path)
            self.dst_path.mkdir(parents=True, exist_ok=True)

            train_path = self.dst_path.joinpath('train')
            train_path.mkdir(parents=True, exist_ok=True)

            valid_path = self.dst_path.joinpath('valid')
            valid_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("dst_path cannot be None, it must has value for destination directory")

        for key in self.src_files_map.keys():
            train_path.joinpath(key).mkdir(parents=True, exist_ok=True)
            valid_path.joinpath(key).mkdir(parents=True, exist_ok=True)

        TrainValidSplit.copy_move(train_path, self.train_files_map, mode=mode)
        TrainValidSplit.copy_move(valid_path, self.valid_files_map, mode=mode)

    @staticmethod
    def copy_move(path: object, files_map: object, mode: object = 'cp'):
        for key in files_map.keys():
            files = files_map[key]
            for f in files:
                src_file = str(f)
                filename = str(f).split('/')[-1]

                dst_file = path.joinpath(key).joinpath(filename)
                dst_file = str(dst_file)

                if mode is 'cp':
                    shutil.copyfile(src_file, dst_file)
                elif mode is 'mv':
                    shutil.copyfile(src_file, dst_file)
                    f.unlink()
                else:
                    raise ValueError('mode only accept value "cp" or "mv"')


class TrainValidTestSplit(TrainValidSplit):
    def __init__(self, src_path, val_size=0.2, test_size=0.1):
        super(TrainValidTestSplit, self).__init__(src_path, val_size=val_size)
        self.test_size = test_size

    def _split_map(self):
        bfiles_map = self.src_files_map
        valid_files = {}
        train_files = {}
        test_files = {}

        for key in bfiles_map.keys():
            files = bfiles_map[key]
            n = len(bfiles_map[key])
            nd = [i for i in range(n)]
            test_sampling_size = int( n * self.test_size)
            valid_sampling_size = int((n-test_sampling_size) * self.val_size)
            test_set = set(random.sample(nd, test_sampling_size))
            train_val_set = set(nd) - test_set
            valid_set = set(random.sample(list(train_val_set), valid_sampling_size))
            train_set = train_val_set - valid_set

            for vidx in list(valid_set):
                if key in valid_files.keys():
                    valid_files[key].append(files[vidx])
                else:
                    valid_files.update({key :[files[vidx]]})

            for tidx in list(train_set):
                if key in train_files.keys():
                    train_files[key].append(files[tidx])
                else:
                    train_files.update({key :[files[tidx]]})

            for tidx in list(test_set):
                if key in test_files.keys():
                    test_files[key].append(files[tidx])
                else:
                    test_files.update({key :[files[tidx]]})


        return train_files, valid_files, test_files


    def execute(self, dst_path:str, mode:str='cp'):
        self.train_files_map, self.valid_files_map, self.test_files_map = self._split_map()
        if dst_path:
            self.dst_path = pathlib.Path(dst_path)
            self.dst_path.mkdir(parents=True, exist_ok=True)

            train_path = self.dst_path.joinpath('train')
            train_path.mkdir(parents=True, exist_ok=True)

            valid_path = self.dst_path.joinpath('valid')
            valid_path.mkdir(parents=True, exist_ok=True)

            test_path = self.dst_path.joinpath('test')
            test_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("dst_path cannot be None, it must has value for destination directory")

        for key in self.src_files_map.keys():
            train_path.joinpath(key).mkdir(parents=True, exist_ok=True)
            valid_path.joinpath(key).mkdir(parents=True, exist_ok=True)
            test_path.joinpath(key).mkdir(parents=True, exist_ok=True)


        TrainValidSplit.copy_move(train_path, self.train_files_map, mode=mode)
        TrainValidSplit.copy_move(valid_path, self.valid_files_map, mode=mode)
        TrainValidSplit.copy_move(test_path, self.test_files_map, mode=mode)



if __name__ == '__main__':
    src = '/data/att_faces'
    dst = '/data/att_faces_new'
    rs = TrainValidTestSplit(src_path=src, val_size=0.3, test_size=0.1)
    rs.execute(dst, mode='cp')
