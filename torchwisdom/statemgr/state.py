import torch
from typing import *
import random
import hashlib
import pandas as pd
import os
from pathlib import Path
import datetime


__all__ = ['Store', 'Logger', 'StateManager']


wisdom_home = Path.home().joinpath('.torchwisdom')
state_home = wisdom_home.joinpath('state')
logger_file = state_home.joinpath('main.csv')


class Store(object):
    def __init__(self, data={}):
        self.state = State()

    def _is_exist(self, id, mode='curr'):
        p = state_home.joinpath(id)
        if p.is_dir():
            curr_filename = id + f"_{mode}.pt"
            if p.joinpath(curr_filename).is_file():
                return True
        return False

    def _is_dir(self, id):
        p = state_home.joinpath(id)
        if p.exists() and p.is_dir() :
            return True
        else:
            return False

    def _get_previous_saved_file(self, id):
        curr_filename = id + f"_curr.pt"
        curr = state_home.joinpath(id).joinpath(curr_filename)

        last_filename = id + f"_last.pt"
        last = state_home.joinpath(id).joinpath(last_filename)

        ct = curr.stat().st_mtime
        lt = last.stat().st_mtime
        if ct > lt:
            return last
        if lt > ct:
            return curr

    def _get_last_saved_file(self, id):
        curr_filename = id + f"_curr.pt"
        curr = state_home.joinpath(id).joinpath(curr_filename)

        last_filename = id + f"_last.pt"
        last = state_home.joinpath(id).joinpath(last_filename)

        ct = curr.stat().st_atime
        lt = last.stat().st_atime
        if ct > lt:
            return curr
        if lt > ct:
            return last

    def _test_load(self, id, mode,  map_location: Any = None):
        filename = id + f"_{mode}.pt"
        path = state_home.joinpath(id).joinpath(filename)
        try:
            loaded = torch.load(str(path), map_location=map_location)
            loaded.keys()
            return True
        except:
            return False


    def save(self, id, state):
        # check is if curr and last file is exist
        # check which file is newer
        # update the file with lowest status
        if self._is_dir(id):
            if self._is_exist(id, 'curr') and self._is_exist(id, 'last'):
                path = self._get_previous_saved_file(id)
                torch.save(state, str(path))
                # print(f'File saved {str(path)}')
            elif self._is_exist(id, 'curr') and not self._is_exist(id, 'last'):
                curr_filename = id + f"_last.pt"
                path = state_home.joinpath(id).joinpath(curr_filename)
                torch.save(state, str(path))
                # print(f'File saved {str(path)}')
            else:
                curr_filename = id + f"_curr.pt"
                path: Path = state_home.joinpath(id).joinpath(curr_filename)
                torch.save(state, str(path))
                # print(f'File saved {str(path)}')

        else:
            #if dir does not exist than create dir
            path: Path = state_home.joinpath(id)
            os.makedirs(str(path), exist_ok=True)

    def load(self, id,  map_location: Any = None) -> Any:
        curr_test = self._test_load(id, 'curr')
        last_test = self._test_load(id, 'last')
        if curr_test and last_test:
            path = self._get_last_saved_file(id)
            self.state = torch.load(str(path), map_location=map_location)
            print(f"File {str(path)} is loaded!")
            return self.state
        elif curr_test and not last_test:
            curr_filename = id + f"_curr.pt"
            path: Path = state_home.joinpath(id).joinpath(curr_filename)
            self.state = torch.load(str(path), map_location=map_location)
            print(f"File {str(path)} is loaded!")
            return self.state
        elif not curr_test and last_test:
            last_filename = id + f"_last.pt"
            path: Path = state_home.joinpath(id).joinpath(last_filename)
            self.state = torch.load(str(path), map_location=map_location)
            print(f"File {str(path)} is loaded!")
            return self.state
        else:
            raise FileNotFoundError(f"File is not found")


        # path = state_home.joinpath(id).joinpath(filename)
        # if path.exists() and path.is_file():
        #     self.state = torch.load(str(path), map_location=map_location)
        #     return self.state
        # else:
        #     raise FileNotFoundError(f"File {str(path)} is not found")


class Logger(object):
    def __init__(self):
        super(Logger, self).__init__()
        self._dataframe = self._get_dataframe()

    def dataframe(self):
        return self._dataframe

    def add(self, id, name='', desc=''):
        file = logger_file
        row = {'id': id, 'name': name, 'desc': desc, 'created': str(datetime.datetime.now()), "modified": ""}
        self._dataframe = self._dataframe.append(row, ignore_index=True)
        self._dataframe.to_csv(str(file), index=None, header=True)

    def get(self, idx) -> pd.DataFrame:
        return self._dataframe.iloc[idx]

    def last(self) -> pd.DataFrame:
        return self._dataframe.iloc[-1]

    def find(self, id):
        df = self.dataframe()
        search = df[df['id'] == id]
        return search

    def is_exist(self, id):
        if len(self.find(id))>0:
            return True
        else:
            return False

    def find_by(self, field='name', value=""):
        pass

    def _get_dataframe(self) -> pd.DataFrame:
        file = logger_file
        path = state_home

        if not file.exists():
            os.makedirs(str(path), exist_ok=True)
            df = self._build_csv()
            df.to_csv(str(file), index=None, header=True)
            print(f"build for the first time in {str(file)}")
            return df
        else:
            return pd.read_csv(file)

    def _build_csv(self):
        data = {'id': [], 'name': [], 'desc': [], 'created': [], "modified": []}
        return pd.DataFrame(data, columns=['id', 'name', 'desc', 'created', 'modified'])

class State(dict):
    def __init__(self, *args, **kwargs):
        super(State, self).__init__(*args, **kwargs)

    def set(self, key, val):
        self.__setitem__(key, val)


class StateManager(object):
    def __init__(self):
        super(StateManager, self).__init__()
        self._state: State = State()
        self._store = Store()
        self._logger = Logger()
        self.id: str = random_generator()
        self.name: str = ''
        self.desc: str = ''
        self._build_defaults_state()

    def dataframe(self) -> pd.DataFrame:
        return self._logger.dataframe()

    def logger(self) -> Logger:
        return self._logger

    def store(self) -> Store:
        return self._store

    @property
    def state(self) -> Dict:
        return self._state

    def load(self, id):
        self._state = self.store().load(id)
        self.id = id
        return self._state

    def load_last(self):
        id = self.logger().last()['id']
        self._state = self.store().load(id)
        self.id = id

    def save(self, name='', desc=''):
        self.store().save(self.id, self._state)
        if not self.logger().is_exist(self.id):
            self.logger().add(self.id, name=name, desc=desc)



    def _build_defaults_state(self):
        self._state = {
            "data": {
                "dataset": {"trainset": {}, "validset": {}, "testset": {}},
                "dataloader": {"train_loader": {}, "valid_loader": {}},
                "num_workers":0,
                "shuffle": True,
            },
            "model": {
                "state_dict": None,
                "arch": None,
                "object": None,
            },
            "metric": {
                "train": {'loss': {'val': [], 'mean': [], 'std': [], 'epoch': []}},
                "valid": {'loss': {'val': [], 'mean': [], 'std': [], 'epoch': []}},
                "test": {'loss': {'val': [], 'mean': [], 'std': [], 'epoch': []}}
            },
            'criterion': None,
            "optimizer": {'defaults': None, 'state_dict': None, 'classname': None, 'object': None},
            "scheduler": {},
            "trainer": {
                "id": "",
                "name": "",
                "desc": "",
                "epoch": {'start': 0, 'curr': 0, 'num': 0, 'time': [], 'time_start': 0, 'time_end': 0, 'remain': []},
                "lr": 0.01,
                "created": "",
                "modified": ""
            },
            "callbacks": {}
        }


def random_generator():
    h = hashlib.new('ripemd160')
    id_encode = str(random.randint(111111111111, 999999999999)).encode()
    h.update(id_encode)
    hid = h.hexdigest()
    return hid





if __name__ == '__main__':
    s = State()
    s.set("k", {})
    print(s)
    #
    # # ds1 = DatasetState('trainset')
    # # ds2 = DatasetState('validset')
    # # print(ds1.data['name'], ds2.data['name'])
    # sm = StateManager()
    #
    # # sm.add_state([ds1, ds2])
    # sm.set_id('4d95e6fcc0bf0b02badcd41370d53ff2e4e1d0e9')
    # # print(sm.states)
    # # print(sm.state_dict())
    # sm.save()
    #
    # sm.load()
    # print(sm.states)
    # print(sm.state_dict())
    # print(sm.history.head())
    #
    # # sm.dframe_add('4d95e6fcc0bf0b02badcd41370d53ff2e4e1d0e9', 'MyProject')
    #
