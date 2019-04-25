from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import isnotebook
from torchwisdom.callback import Callback
from typing import *
from torchwisdom.statemgr.state import StateManager
from datetime import timedelta
from torchwisdom.core import *

__all__ = []

class ProgressTable(object):
    def __init__(self):
        pass

def time_formatter(sec, last_cut=-4)->str:
    return str(timedelta(seconds=sec))[:last_cut]


def format_text(text, empty_space=15):
    ltext=len(text)
    if empty_space>ltext:
        len_iter = empty_space-ltext
        space = "".join([" " for i in range(len_iter)])
        out = space+text
    else:
        out = " "+text+" "
    return out


def build_line_console(line, use_tab=False):
    str_build = ""
    for ln in line:
        text = format_text(ln)
        str_build+=text
        if use_tab: str_build+="\t"
    return str_build


def time_delta_remain(epoch_state):
    delta_last = epoch_state.get('time')[-1]
    delta = time_formatter(delta_last)
    remain_last = epoch_state.get('remain')[-1]
    remain = time_formatter(remain_last)
    return delta, remain

def time_delta_remain_resume(epoch_state, epoch):
    delta_last = epoch_state.get('time')[epoch]
    delta = time_formatter(delta_last)
    remain_last = epoch_state.get('remain')[epoch]
    remain = time_formatter(remain_last)
    return delta, remain

def line_builder(metric_state: Dict, epoch, tdelta, tremain):
    train: Dict = metric_state.get('train')
    valid: Dict = metric_state.get('valid')

    line = [f'{epoch}']
    for key in train.keys():
        line.append(f"{train[key]['mean'][-1]:.6f}")
        line.append(f"{valid[key]['mean'][-1]:.6f}")
    line.append(f'{tdelta}')
    line.append(f'{tremain}')

    if isnotebook():
        return line
    else:
        return build_line_console(line)

def line_builder_resume(metric_state: Dict, epoch, tdelta, tremain):
    train: Dict = metric_state.get('train')
    valid: Dict = metric_state.get('valid')

    line = [f'{epoch+1}']
    for key in train.keys():
        line.append(f"{train[key]['epoch'][epoch]:.6f}")
        line.append(f"{valid[key]['epoch'][epoch]:.6f}")
    line.append(f'{tdelta}')
    line.append(f'{tremain}')

    if isnotebook():
        return line
    else:
        return build_line_console(line)


def line_head_builder(metric_state: Dict):
    train: Dict = metric_state.get('train')

    line = ['epoch']
    for val in train.keys():
        line.append(f'trn_{val}')
        line.append(f'val_{val}')
    line.append('time')
    line.append('remain')

    if isnotebook():
        return line
    else:
        return build_line_console(line)


def graph_builder(metric_state: Dict, trainer_state: Dict):
    train: Dict = metric_state.get('train')
    valid: Dict = metric_state.get('valid')
    epoch_curr = trainer_state.get('epoch')['curr']

    train_loss = train.get('loss').get('epoch')
    valid_loss = valid.get('loss').get('epoch')

    if epoch_curr == 1:
        x = [1]
    else:
        x = list(range(1, len(train_loss)+1))
    graph = [[x, train_loss], [x, valid_loss]]
    # print(graph)
    return graph


def clean_up_metric_resume(metric_state: Dict, epoch_curr):
    train: Dict = metric_state.get('train')
    valid: Dict = metric_state.get("valid")

    for key in train.keys():
        # print("train epoch len", len(train[key]['epoch']))
        # print(key, train[key]['epoch'])
        if len(train[key]['epoch']) != epoch_curr-1:
            train[key]['epoch'].pop()

        # print("valid epoch len", len(valid[key]['epoch']))
        # print(key, valid[key]['epoch'])
        if len(valid[key]['epoch']) != epoch_curr-1:
            valid[key]['epoch'].pop()









class ProgressBarCallback(Callback):
    def __init__(self):
        super(ProgressBarCallback, self).__init__()
        self.statemgr: StateManager = None

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.mbar: master_bar = kwargs.get('master_bar')

    def on_resume_begin(self, *args: Any, **kwargs: Any) -> None:
        self.mbar: master_bar = kwargs.get('master_bar')

        mbar: master_bar = kwargs.get('master_bar')
        trainer_state: Dict = self.statemgr.state.get('trainer')
        epoch_curr = trainer_state.get('epoch')['curr']
        trainer_state.get('epoch')['curr'] = epoch_curr - 1

        metric_state: Dict = self.statemgr.state.get('metric')
        # clean up metric
        clean_up_metric_resume(metric_state, epoch_curr)


        # header write
        line = line_head_builder(metric_state)
        if isnotebook():
            mbar.write(line, table=True)
        else:
            mbar.write(line, table=False)

        for epoch in range(epoch_curr-1):
            metric_state: Dict = self.statemgr.state.get('metric')
            epoch_state: Dict = trainer_state.get('epoch')
            tdelta = time_formatter(epoch_state.get("time")[epoch])
            tremain = time_formatter(epoch_state.get("remain")[epoch])
            line = line_builder_resume(metric_state, epoch, tdelta, tremain)
            if isnotebook():
                mbar.write(line, table=True)
            else:
                mbar.write(line, table=False)


    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        mbar: master_bar = kwargs.get('master_bar')
        epoch: int = kwargs.get("epoch")
        trainer_state: Dict = self.statemgr.state.get('trainer')
        metric_state: Dict = self.statemgr.state.get('metric')

        epoch_state: Dict = trainer_state.get('epoch')
        tdelta, tremain = time_delta_remain(epoch_state)

        line = line_builder(metric_state, epoch, tdelta, tremain)
        if isnotebook():
            mbar.write(line, table=True)
        else:
            mbar.write(line, table=False)

        epoch_curr = trainer_state.get('epoch')['curr']
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
        trainer_state: Dict = self.statemgr.state.get('trainer')
        epoch_curr = trainer_state.get('epoch')['curr']
        if epoch_curr == 0: # show header for first time
            metric_state: Dict = self.statemgr.state.get('metric')
            line = line_head_builder(metric_state)
            if isnotebook():
                mbar.write(line, table=True)
            else:
                mbar.write(line, table=False)
