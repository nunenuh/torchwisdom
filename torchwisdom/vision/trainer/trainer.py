from torchwisdom.optim.callback import *
from torchwisdom.metrics import *
from torchwisdom.callback import *
from torchwisdom.utils.data.collector import *
from torch.optim.optimizer import Optimizer
from torchwisdom.trainer import *
from torchwisdom.envi import *

from fastprogress import master_bar, progress_bar

__all__ = []

shell = python_shell()
if shell == 'shell': from tqdm import tqdm, tnrange
elif shell == 'ipython': from tqdm import tqdm, tnrange
elif shell == 'ipython-notebook': from tqdm import tqdm_notebook as tqdm, tnrange
elif shell == 'jupter-notebook': from tqdm import tqdm_notebook as tqdm, tnrange
else: from tqdm import tqdm, tnrange


class ConvTrainer(Trainer):
    def __init__(self, data: DatasetCollector, model: nn.Module, criterion: nn.Module,
                 metrics: Collection[Callback]=None, callbacks: Collection[Callback]=None,
                 optimizer: Optimizer = None, device='cpu'):
        '''
        :param data:
        :param model:
        :param optimizer:
        :param criterion:
        :param metrics:
        :param device:

        '''
        super(ConvTrainer, self).__init__(data=data, model=model, criterion=criterion, metrics=metrics,
                                          optimizer=optimizer, callbacks=callbacks, device=device)

        self.data = data
        self.bunch = self.data.bunch()
        self._set_device()



    def _set_device(self):
        self.model = self.model.to(device=self.device)
        # self.optimizer = self.optimizer.to(device=self.device)

    def build_optimizer(self, lr=0.001, mmt=0.9, wd=0.1):
        if self.optimizer is 'sgd':
            self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=mmt, weight_decay=wd)

    def train(self, epoch, mbar: master_bar):
        self.cb_handler.on_train_begin( master_bar=mbar)
        self.model.train()
        train_loader = self.bunch['train']
        trainbar = progress_bar(train_loader, parent=mbar)
        for idx, (img, label) in enumerate(trainbar):
            self.cb_handler.on_train_batch_begin(batch_curr=idx, master_bar=mbar)

            img = img.to(device=self.device)
            label = label.to(device=self.device)

            self.cb_handler.on_train_forward_begin(feature=img, target=label)
            out = self.model(img)
            loss = self.criterion(out, label)
            self.cb_handler.on_train_forward_end(loss=loss, y_pred=out, y_true=label, )

            self.cb_handler.on_train_backward_begin()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.cb_handler.on_train_backward_end()

            self.cb_handler.on_train_batch_end(master_bar=mbar)
        self.cb_handler.on_train_end()

    def validate(self, epoch, mbar: master_bar):
        self.cb_handler.on_validate_begin(master_bar=mbar)
        self.model.eval()
        valid_loader = self.bunch['valid']
        progbar = progress_bar(valid_loader, parent=mbar)
        with torch.no_grad():
            for idx, (img, label) in enumerate(progbar):
                self.cb_handler.on_validate_batch_begin(batch_curr=idx, master_bar=mbar)
                img = img.to(device=self.device)
                label = label.to(device=self.device)

                self.cb_handler.on_validate_forward_begin(feature=img, target=label, )
                out = self.model(img)
                loss = self.criterion(out, label)
                self.cb_handler.on_validate_forward_end(loss=loss, y_pred=out, y_true=label, )

                self.cb_handler.on_validate_batch_end( master_bar=mbar)
        self.cb_handler.on_validate_end(epoch=epoch,  master_bar=mbar)

    def fit(self, epoch_num, lr=0.01, wd=0, verbose=False, callbacks=None, **kwargs):
        self._build_optimizer(lr, weight_decay=wd, **kwargs)
        self._build_state_manager()
        self._build_callback_handler()  # CallbackHandler need to be the last to build

        mbar = master_bar(range(epoch_num))
        self.cb_handler.on_fit_begin(epoch_num=epoch_num, master_bar=mbar)
        for epoch in mbar:
            self.cb_handler.on_epoch_begin(epoch=epoch, master_bar=mbar)
            epoch = epoch + 1
            self.train(epoch, mbar)
            self.validate(epoch, mbar)
            self.cb_handler.on_epoch_end(epoch=epoch, master_bar=mbar)
        self.cb_handler.on_fit_end(epoch=epoch, master_bar=mbar)


    def resume(self):
        self.cb_handler.on_epoch_resume()

        self.cb_handler.on_resume_end()


    def freeze(self):
        pass

    def unfreeze(self):
        pass

