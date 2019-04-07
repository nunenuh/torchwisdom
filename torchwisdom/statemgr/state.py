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





