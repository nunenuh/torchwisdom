import torch
from typing import *
import random
import hashlib
import pandas as pd
from pathlib import Path
import os
import datetime


class StateBase(object):

    def current_data(self):
        raise NotImplementedError()

    def set_data(self, data: Any):
        raise NotImplementedError()

    def get_data(self) -> Any:
        raise NotImplementedError()

    def set_property(self, prop: str, value: Any):
        raise NotImplementedError()

    def get_property(self, prop: str) -> Any:
        raise NotImplementedError()


class StateStore(object):

    def __init__(self, data={}):
        self.data = data
        self.name: str = ''
        self.manager_id: str = ''
        self.base_path: Path = None

    def save(self):
        if self.base_path:
            path = self.base_path.joinpath(self.name+'.pt')
            print(f'save to {self.data["name"]} {str(path)}')
            torch.save(self.data, str(path))
        else:
            raise ValueError("base_path is None, expected to have base_path from StateManager")

    def load(self, base_path: Any = None,  map_location: Any = None) -> Any:
        if base_path:
            self.base_path = base_path

        if self.base_path:
            path = self.base_path.joinpath(self.name + '.pt')
            if path.exists() and path.is_file():
                self.data = torch.load(str(path), map_location=map_location)
                return self.data
            else:
                raise FileNotFoundError("File {str(path)} is not found")
        else:
            raise ValueError("base_path is None, expected to have base_path from StateManager")


    def set_base_path(self, base_path):
        self.data['base_path'] = str(base_path)
        self.base_path = base_path

    def set_manager_id(self, id):
        self.data['manager_id'] = id
        self.manager_id = id


class State(StateBase, StateStore):
    """ State Recorder """

    def __init__(self):
        super(State, self).__init__()
        self.data: Any = {}
        self.name: str = ''
        self.manager_id: str = ''
        self.base_path: Path = None

    def current_data(self):
        return self.data

    def get_data(self)->Any:
        return self.data

    def set_data(self, data):
        self.data = data

    def get_property(self, prop):
        if prop in self.data.keys():
            return self.data[prop]
        else:
            raise ValueError(f"prop {prop} not found!")

    def set_property(self, prop, value):
        self.data[prop] = value

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class StateDataFrame(object):

    def __init__(self):
        super(StateDataFrame, self).__init__()
        self.data_frame = self._get_dataframe()

    def dframe_add(self, id, name='', desc=''):
        file = self._get_file_path()
        row = {'id':id, 'name':name, 'desc':desc, 'datetime':str(datetime.datetime.now())}
        self.data_frame = self.data_frame.append(row, ignore_index=True)
        self.data_frame.to_csv(str(file), index=None, header=True)

    def dframe_get(self, idx):
        return self.data_frame.iloc[idx]

    def dframe_last(self):
        return self.data_frame.iloc[-1]

    def _get_base_path(self):
        self.state_path = Path.home().joinpath('.torchwisdom').joinpath('state')
        return self.state_path

    def _get_file_path(self):
        self.state_file = self._get_base_path().joinpath('main.csv')
        return self.state_file

    def _get_random_id(self):
        h = hashlib.new('ripemd160')
        id_encode = str(random.randint(111111111111, 999999999999)).encode()
        h.update(id_encode)
        hid = h.hexdigest()
        return hid

    def _get_dataframe(self)->pd.DataFrame:
        file = self._get_file_path()
        path = self._get_base_path()

        if not file.exists():
            os.makedirs(str(path), exist_ok=True)
            df = self._build_csv()
            df.to_csv(str(file), index=None, header=True)
            print(f"build for the first time in {str(file)}")
            return df

        else:
            return pd.read_csv(self.state_file)

    def _build_csv(self):
        data = { 'id': [], 'name':[], 'desc':[], 'datetime': [] }
        return pd.DataFrame(data, columns=['id', 'name', 'desc', 'datetime'])


class StateManager(StateDataFrame):

    def __init__(self, name: str='', desc: str='', states: Dict = {}):
        super(StateManager, self).__init__()
        self.name = name
        self.desc = desc
        self.states: Dict[State] = states or {}
        self.id = self._get_random_id()
        self.history = self._get_dataframe()


    def add_state(self, state: Union[State, List[State]]):
        if type(state) is list:
            for st in state:
                st.set_manager_id(self.id)
                st.set_base_path(self._get_base_path().joinpath(self.id))
                self.states[st.get_name()] = st
        else:
            state.set_manager_id(self.id)
            state.set_base_path(self._get_base_path().joinpath(self.id))
            self.states[state.get_name()] = state

    def update_state(self, state: Union[State, List[State]]):
        self.add_state(state)

    def reset_state(self):
        self.states = {}

    def set_id(self, id):
        self.id = id
        for key, state in self.states.items():
            state.set_manager_id(id)
            state.set_base_path(self._get_base_path().joinpath(self.id))

    def get_state(self, name):
        if name in self.states.keys():
            return self.states[name]
        else:
            return None

    def remove_state(self, name):
        if name in self.states.keys():
            self.states.pop(name)

    def state_dict(self):
        wrap = {}
        for key, state in self.states.items():
            wrap[key] = state.data
        return wrap

    def save(self):

        os.makedirs(self._get_base_path().joinpath(self.id), exist_ok=True)
        for key, state in self.states.items():
            print(key, state.data['name'])
            state.save()

    def load(self):
        self.reset_state()
        path = self._get_base_path().joinpath(self.id)
        state_objs = []
        if path.exists() and path.is_dir():
            files = list(path.glob("*.pt"))
            for file in files:
                state_dict = torch.load(str(file))
                if 'classname' in state_dict.keys():
                    classname = state_dict['classname']
                    obj = class_map(classname)
                    if obj:
                        obj.set_name(state_dict['name'])
                        obj.set_data(state_dict)
                        state_objs.append(obj)
            self.add_state(state_objs)
        else:
            raise NotADirectoryError('Directory {str(path)} not found!')


class DatasetState(State):
    """ Dataset State for storing information base on dataset """

    def __init__(self, name: str = 'dataset'):
        super(DatasetState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class DataLoaderState(State):
    """ Dataset State for storing information base on dataset """

    def __init__(self, name: str = 'dataset'):
        super(DataLoaderState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class DataCollectorState(State):
    """ Dataset State for storing information base on dataset """

    def __init__(self, name: str = 'datacollector'):
        super(DataCollectorState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class ModelState(State):
    def __init__(self, name: str = 'model'):
        super(ModelState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class OptimizerState(State):
    def __init__(self, name: str = 'optimizer'):
        super(OptimizerState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class SchedulerState(State):
    def __init__(self, name: str = 'optimizer'):
        super(SchedulerState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class MetricState(State):
    def __init__(self, name: str = 'metric'):
        super(MetricState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


class TrainerState(State):
    def __init__(self, name: str = 'trainer'):
        super(DatasetState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())


def class_map(classname: str):
    # classname = classname.strip()
    if classname == 'DatasetState': return DatasetState()
    elif classname == "DataLoaderState": return DataLoaderState()
    elif classname == "DataCollectorState": return DataCollectorState()
    elif classname == "ModelState": return ModelState()
    elif classname == "OptimizerState": return OptimizerState()
    elif classname == "SchedulerState": return SchedulerState()
    elif classname == "MetricState": return MetricState()
    elif classname == "TrainerState": return TrainerState()
    else: return None


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

