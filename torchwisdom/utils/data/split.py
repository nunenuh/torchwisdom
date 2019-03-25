import pathlib
import random
import os
import shutil

class RandomSampleSplit(object):
    def __init__(self, src_path, val_size=0.2):
        self.val_size = val_size
        self.src_path = pathlib.Path(src_path)
        self.src_files = sorted(list(self.src_path.glob("*/*.*")))
        self.src_files_map = self._files_map()
        self.train_files_map, self.valid_files_map = self._split_map()

    def _files_map(self):
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

    def execute(self, dst_path, mode='cp'):
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

        self._copy_move(train_path, self.train_files_map, mode=mode)
        self._copy_move(valid_path, self.valid_files_map, mode=mode)


    def _copy_move(self, path, files_map, mode='cp'):
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

if __name__ == '__main__':
    src = '/data/att_faces'
    dst = '/data/att_faces_new'
    rs = RandomSampleSplit(src_path=src, val_size=0.3)
    rs.execute(dst, mode='cp')
