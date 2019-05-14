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
        trainer_state['id'] = self.statemgr.id
        best_state = trainer_state.get("best")
        best_state["best_state"] = self.trainer.save_best_state
        best_state["best_mode"] = self.trainer.save_best_mode
        best_state["best_metric"] = self.trainer.save_best_metric

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

        self.statemgr.id = trainer_state['id']


        best_state = trainer_state.get("best")
        self.trainer.save_best_state = best_state["best_state"]
        self.trainer.save_best_mode = best_state["best_mode"]
        self.trainer.save_best_metric = best_state["best_metric"]

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
            if self.trainer.save_best_state:
               self._execute_best_save()
            else:
                self.statemgr.save()

    def _execute_best_save(self):
        best_mode: str = self.trainer.save_best_mode
        best_metric: str = self.trainer.save_best_metric
        trainer_state: Dict = self.statemgr.state.get('trainer')
        progress_state: Dict = trainer_state.get("progress", {})
        best_state: Dict = trainer_state.get("best")
        if trainer_state.get("epoch").get("curr")>1:
            if best_metric in progress_state:
                best_metric_list: List = progress_state.get(best_metric)
                last_best = best_state.get("last_best")
                last_value = best_metric_list[-1]
                if last_best is not None:
                    if best_mode == "min":
                        logic = last_best > last_value
                    else:
                        logic = last_best < last_value

                    if logic:
                        best_state["last_best"] = last_value
                        best_state["epoch_saved"] = trainer_state.get("epoch").get("curr")
                        self.statemgr.save(is_best=True)
                    else:
                        self.statemgr.save()
                else:
                    best_state["last_best"] = last_value
                    self.statemgr.save()
            else:
                self.statemgr.save()
        else:
            self.statemgr.save()


def build_callback_state(trainer: object, statemgr: StateManager):
    callbacks_state = statemgr.state.get("callbacks")
    callback_list: List[Callback] = trainer.handler.callbacks
    for cb in callback_list:
        callbacks_state[cb.__class__.__name__] = cb

