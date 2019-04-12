import torch
from typing import *
from pathlib import Path
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
            # print(f'save to {self.data["name"]} {str(path)}')
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
        self.data: Dict = {}
        self.name: str = ''
        self.manager_id: str = ''
        self.base_path: Path = None

    def current_data(self):
        return self.data

    def get_data(self)->Any:
        return self.data

    def set_data(self, data):
        self.data = data

    def get_property(self, prop) -> Union[Any, Dict, List]:
        if prop in self.data.keys():
            return self.data[prop]
        else:
            raise ValueError(f"prop {prop} not found!")

    def propery_exist(self, prop):
        if prop in self.data.keys():
            return True
        else:
            return False

    def set_property(self, prop, value):
        self.data[prop] = value

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


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

    def __init__(self, name: str = 'datacoll'):
        super(DataCollectorState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())
        self.data['train'] = {
            'dataset': {'path': '', 'len': 0},
            'loader': {}
        }
        self.data['valid'] = {
            'dataset': {'path': '', 'len': 0},
            'loader': {}
        }
        self.data['test'] = {
            'dataset': {'path': '', 'len': 0},
            'loader': {}
        }
        self.data['batch_size'] = 0
        self.data['num_worker'] = 0
        self.data['shuffle'] = True



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
        self.data['train'] = {'loss': {'val': [], 'mean': [], 'std': [], 'epoch': []}}
        self.data['valid'] = {'loss': {'val': [], 'mean': [], 'std': [], 'epoch': []}}


class TrainerState(State):
    def __init__(self, name: str = 'trainer'):
        super(TrainerState, self).__init__()
        self.name = name
        self._init_data()

    def _init_data(self):
        self.data['classname'] = self.__class__.__name__
        self.data['name'] = self.get_name()
        self.data['created'] = str(datetime.datetime.now())
        self.data['epoch'] = {'start': 0, 'curr': 0, 'num': 0, 'time': [], 'time_start': 0, 'time_end': 0, 'remain':[]}



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

