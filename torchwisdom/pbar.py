from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import isnotebook
from torchwisdom.callback import Callback
from typing import *
from torchwisdom.statemgr.manager import StateManager
from torchwisdom.statemgr.state import MetricState, TrainerState, DataCollectorState
from datetime import timedelta
from torchwisdom.core import *



class ProgressTable(object):
    def __init__(self):
        pass

def time_formatter(sec, last_cut=-4)->str:
    return str(timedelta(seconds=sec))[:last_cut]

def time_delta_remain(epoch_state):
    delta_last = epoch_state.get('time')[-1]
    delta = time_formatter(delta_last)
    remain_last = epoch_state.get('remain')[-1]
    remain = time_formatter(remain_last)
    return delta, remain

def line_builder(metric_state: MetricState, epoch, tdelta, tremain):
    train: Dict = metric_state.get_property('train')
    valid: Dict = metric_state.get_property('valid')

    line = [epoch]
    for key in train.keys():
        line.append(f"{train[key]['mean'][-1]:.6f}")
        line.append(f"{valid[key]['mean'][-1]:.6f}")
    line.append(f'{tdelta}')
    line.append(f'{tremain}')

    if isnotebook():
        return line
    else:
        return '\t'.join(line)

def line_head_builder(metric_state: MetricState):
    train: Dict = metric_state.get_property('train')

    line = ['epoch']
    for val in train.keys():
        line.append(f'trn_{val}')
        line.append(f'val_{val}')
    line.append('time')
    line.append('remain')

    if isnotebook():
        return line
    else:
        return '\t'.join(line)


def graph_builder(metric_state: MetricState, trainer_state: TrainerState):
    train: Dict = metric_state.get_property('train')
    valid: Dict = metric_state.get_property('valid')
    epoch_curr = trainer_state.get_property('epoch')['curr']

    train_loss = train.get('loss').get('epoch')
    valid_loss = valid.get('loss').get('epoch')
    if epoch_curr == 1:
        x = [1]
    else:
        x = list(range(1, epoch_curr+1))
    graph = [[x, train_loss], [x, valid_loss]]
    return graph



class ProgressBarCallback(Callback):
    def __init__(self):
        super(ProgressBarCallback, self).__init__()
        self.statemgr: StateManager = None

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.mbar: master_bar = kwargs.get('master_bar')

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        mbar: master_bar = kwargs.get('master_bar')
        epoch: int = kwargs.get("epoch")
        trainer_state: TrainerState = self.statemgr.get_state('trainer')
        metric_state: MetricState = self.statemgr.get_state('metric')

        epoch_state: Dict = trainer_state.get_property('epoch')
        tdelta, tremain = time_delta_remain(epoch_state)

        line = line_builder(metric_state, epoch, tdelta, tremain)
        if isnotebook():
            mbar.write(line, table=True)
        else:
            mbar.write(line, table=False)

        epoch_curr = trainer_state.get_property('epoch')['curr']
        if epoch_curr > 1:
            graph = graph_builder(metric_state, trainer_state)
            mbar.names = ['trn_loss', 'val_loss']
            if isnotebook():
                mbar.update_graph(graph)

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        mbar: master_bar = kwargs.get('master_bar')
        mbar.child.comment = f'Train'

    def on_validate_batch_end(self, *args: Any, **kwargs: Any) -> None:
        mbar: master_bar = kwargs.get('master_bar')
        mbar.child.comment = f'Validate'

    def on_validate_end(self, *args: Any, **kwargs: Any) -> None:
        mbar: master_bar = kwargs.get('master_bar')
        trainer_state: TrainerState = self.statemgr.get_state('trainer')
        epoch_curr = trainer_state.get_property('epoch')['curr']
        if epoch_curr == 0: # show header for first time
            metric_state: MetricState = self.statemgr.get_state('metric')
            line = line_head_builder(metric_state)
            if isnotebook():
                mbar.write(line, table=True)
            else:
                mbar.write(line, table=False)
