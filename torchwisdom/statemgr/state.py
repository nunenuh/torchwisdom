import torch
from typing import *
import random
import hashlib
import pandas as pd
import os
from pathlib import Path


__all__ = ['Store', 'Logger', 'StateManager']


wisdom_home = Path.home().joinpath('.torchwisdom')
state_home = wisdom_home.joinpath('state')
logger_file = state_home.joinpath('main.csv')


class Store(object):
    def __init__(self, data={}):
        self.state = data

    def save(self, filename, state):
        path = state_home.joinpath(filename)
        torch.save(state, str(path))

    def load(self, filename,  map_location: Any = None) -> Any:
        path = state_home.joinpath(filename)
        if path.exists() and path.is_file():
            self.state = torch.load(str(path), map_location=map_location)
            return self.state
        else:
            raise FileNotFoundError(f"File {str(path)} is not found")


class Logger(object):
    def __init__(self):
        super(Logger, self).__init__()
        self._dataframe = self._get_dataframe()

    def dataframe(self):
        return self._dataframe

    def add(self, id, name='', desc=''):
        file = logger_file
        row = {'id': id, 'name': name, 'desc': desc, 'datetime': str(datetime.datetime.now())}
        self._dataframe = self._dataframe.append(row, ignore_index=True)
        self._dataframe.to_csv(str(file), index=None, header=True)

    def get(self, idx) -> pd.DataFrame:
        return self._dataframe.iloc[idx]

    def last(self) -> pd.DataFrame:
        return self._dataframe.iloc[-1]

    def find(self, id):
        pass

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
        data = {'id': [], 'name': [], 'desc': [], 'datetime': []}
        return pd.DataFrame(data, columns=['id', 'name', 'desc', 'datetime'])


class StateManager(object):
    def __init__(self):
        super(StateManager, self).__init__()
        self._state: Dict = None
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
        filename = id + ".pt"
        self._state = self.store().load(filename)
        self.id = id
        return self._state

    def load_last(self):
        id = self.logger().last()['id']
        filename = id+".pt"
        self._state = self.store().load(filename)
        self.id = id

    def save(self, name='', desc=''):
        filename = self.id + ".pt"
        self.store().save(filename, self._state)
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
            "optimizer": {},
            "scheduler": {},
            "trainer": {
                "id": "",
                "name": "",
                "desc": "",
                "epoch": {'start': 0, 'curr': 0, 'num': 0, 'time': [], 'time_start': 0, 'time_end': 0, 'remain': []},
                "created": "",
                "modified": ""
            }
        }


def random_generator():
    h = hashlib.new('ripemd160')
    id_encode = str(random.randint(111111111111, 999999999999)).encode()
    h.update(id_encode)
    hid = h.hexdigest()
    return hid


if __name__ == '__main__':

    # ds1 = DatasetState('trainset')
    # ds2 = DatasetState('validset')
    # print(ds1.data['name'], ds2.data['name'])
    sm = StateManager()

    # sm.add_state([ds1, ds2])
    sm.set_id('4d95e6fcc0bf0b02badcd41370d53ff2e4e1d0e9')
    # print(sm.states)
    # print(sm.state_dict())
    sm.save()

    sm.load()
    print(sm.states)
    print(sm.state_dict())
    print(sm.history.head())

    # sm.dframe_add('4d95e6fcc0bf0b02badcd41370d53ff2e4e1d0e9', 'MyProject')

