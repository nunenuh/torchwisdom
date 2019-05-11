from typing import *
from torchwisdom.core.callback import Callback
import time
from torchwisdom.core.statemgr.state import StateManager


class StateManagerCallback(Callback):
    def __init__(self):
        super(StateManagerCallback, self).__init__()
        self.statemgr: StateManager = None

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        epoch_num = kwargs.get('epoch_num')
        trainer_state: Dict = self.statemgr.state.get('trainer')
        epoch_state: Dict = trainer_state.get('epoch')
        epoch_state['start'] = 0
        epoch_state['curr'] = 0
        epoch_state['num'] = epoch_num

        dcoll_state: Dict = self.statemgr.state.get('data')
        dcoll_state['batch_size'] = self.trainer.data.batch_size
        dcoll_state['num_workers']= self.trainer.data.num_workers
        dcoll_state['shuffle'] = self.trainer.data.shuffle
        dcoll_state.get('dataset').get('trainset')['len'] = self.trainer.data.trainset.__len__()
        dcoll_state.get('dataset').get('validset')['len'] = self.trainer.data.validset.__len__()

        self.statemgr.state["criterion"] = self.trainer.criterion
        self.statemgr.state["callbacks"] = self.trainer.handler.callbacks_odict

    def on_resume_begin(self, *args: Any, **kwargs: Any) -> None:
        epoch_num = kwargs.get('epoch_num')
        trainer_state: Dict = self.statemgr.state.get('trainer')
        epoch_state: Dict = trainer_state.get('epoch')
        epoch_state['num'] = epoch_num

        dcoll_state: Dict = self.statemgr.state.get('data')
        dcoll_state['batch_size'] = self.trainer.data.batch_size
        dcoll_state['num_workers'] = self.trainer.data.num_workers
        dcoll_state['shuffle'] = self.trainer.data.shuffle
        dcoll_state.get('dataset').get('trainset')['len'] = self.trainer.data.trainset.__len__()
        dcoll_state.get('dataset').get('validset')['len'] = self.trainer.data.validset.__len__()

        self.statemgr.state["criterion"] = self.trainer.criterion
        self.statemgr.state["callbacks"] = self.trainer.handler.callbacks_odict


    def on_fit_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
        epoch_state: Dict = self.statemgr.state.get('trainer').get('epoch')
        epoch_state['time_start'] = time.time()

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        # print("StateManagerCallback: on_epoch_end")
        trainer_state: Dict = self.statemgr.state.get('trainer')
        epoch_state: Dict = trainer_state.get('epoch')

        start = epoch_state.get('time_start')
        end = time.time()
        delta_time = end - start
        epoch_state['time_end'] = end
        epoch_state['time'].append(delta_time)

        epoch_num = epoch_state['num']
        elapsed_time = sum(epoch_state['time'])
        tremain = (delta_time * epoch_num) - elapsed_time
        if tremain <= 0: tremain = 0
        epoch_state['remain'].append(tremain)
        epoch_state['curr'] += 1
        if self.trainer.log_state:
            self.statemgr.save()



def build_callback_state(trainer: object, statemgr: StateManager):
    callbacks_state = statemgr.state.get("callbacks")
    callback_list: List[Callback] = trainer.handler.callbacks
    for cb in callback_list:
        callbacks_state[cb.__class__.__name__] = cb

