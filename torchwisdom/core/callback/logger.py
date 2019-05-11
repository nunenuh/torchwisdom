from torchwisdom.core.callback import Callback
from ..progress import *
from typing import *


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



class CSVLoggerCallback(Callback):
    def __init__(self, filepath, separator=',', replace=False):
        super(CSVLoggerCallback, self).__init__()
        self.filepath = filepath
        self.separator = separator
        self.replace = replace

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_fit_end(self, *args: Any, **kwargs: Any) -> None:
        pass

