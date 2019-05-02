from torchwisdom.core.callback import *
from torch.optim.optimizer import Optimizer
from torchwisdom.core.trainer import *
from typing import *
from fastprogress import master_bar, progress_bar
import torch.nn as nn
import torch
from torchwisdom.vision.predictor import ConvPredictor

__all__ = ['ConvTrainer']

# shell = python_shell()
# if shell == 'shell': from tqdm import tqdm, tnrange
# elif shell == 'ipython': from tqdm import tqdm, tnrange
# elif shell == 'ipython-notebook': from tqdm import tqdm_notebook as tqdm, tnrange
# elif shell == 'jupter-notebook': from tqdm import tqdm_notebook as tqdm, tnrange
# else: from tqdm import tqdm, tnrange


class ConvTrainer(ClassifierTrainer):
    def __init__(self, data: DatasetCollector, model: nn.Module,
                 criterion: nn.Module = None, optimizer: Optimizer = None,
                 metrics: Collection[Callback] = None, callbacks: Collection[Callback] = None):
        '''
        :param data:
        :param model:
        :param optimizer:
        :param criterion:
        :param metrics:
        :param device:

        '''
        super(ConvTrainer, self).__init__(data=data, model=model,
                                          criterion=criterion, optimizer=optimizer,
                                          metrics=metrics, callbacks=callbacks)

        self.data = data
        self.bunch = self.data.bunch()
        self.predictor: ConvPredictor = None

        self._set_device()
        self._build_predictor()

    def _build_predictor(self):
        self.predictor: ConvPredictor = ConvPredictor(self.model, self.data)

    def _data_loss_check_clean(self, pred, target):
        name = self.criterion.__class__.__name__
        if name == 'BCELoss' or name == 'BCEWithLogitsLoss':
            pred = pred.float()
            target = target.unsqueeze(dim=1).float()
        return pred, target

    def train(self, epoch, mbar: master_bar):
        self.cb_handler.on_train_begin( master_bar=mbar)
        self.model.train()
        train_loader = self.bunch['train']
        trainbar = progress_bar(train_loader, parent=mbar)
        for idx, (feature, target) in enumerate(trainbar):
            self.cb_handler.on_train_batch_begin(batch_curr=idx, master_bar=mbar)

            feature = feature.to(device=self.device)
            target = target.to(device=self.device)

            self.cb_handler.on_train_forward_begin(feature=feature, target=target)
            out = self.model(feature)
            out, target = self._data_loss_check_clean(out, target)
            loss = self.criterion(out, target)
            self.cb_handler.on_train_forward_end(loss=loss, y_pred=out, y_true=target, )

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
            for idx, (feature, target) in enumerate(progbar):
                self.cb_handler.on_validate_batch_begin(batch_curr=idx, master_bar=mbar)
                feature = feature.to(device=self.device)
                target = target.to(device=self.device)

                self.cb_handler.on_validate_forward_begin(feature=feature, target=target, )
                out = self.model(feature)
                out, target = self._data_loss_check_clean(out, target)
                loss = self.criterion(out, target)
                self.cb_handler.on_validate_forward_end(loss=loss, y_pred=out, y_true=target, )

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


    def resume(self, from_last=True, id=None, **kwargs):
        self._build_state_manager()
        if id is not None:
            self.state_manager.load(id)
        if from_last:
            self.state_manager.load_last()


        self.optimizer = self.state_manager.state.get('optimizer').get("object")
        self.model = self.state_manager.state.get('model').get("object")
        self.criterion = self.state_manager.state.get('criterion')
        trainer_state: Dict = self.state_manager.state.get('trainer')
        lr = trainer_state.get("lr")
        epoch_curr= trainer_state.get("epoch").get("curr")
        epoch_num = trainer_state.get("epoch").get("num")
        self._build_optimizer(lr)
        self._build_callback_handler_resume()# CallbackHandler need to be the last to build

        #
        mbar = master_bar(range(epoch_curr-1, epoch_num))

        self.cb_handler.on_resume_begin(epoch_num=epoch_num, master_bar=mbar)

        # self.cb_handler.on_fit_begin(epoch_num=epoch_num, master_bar=mbar)
        for epoch in mbar:
            self.cb_handler.on_epoch_begin(epoch=epoch, master_bar=mbar)
            epoch = epoch + 1
            self.train(epoch, mbar)
            self.validate(epoch, mbar)
            self.cb_handler.on_epoch_end(epoch=epoch, master_bar=mbar)
        # self.cb_handler.on_fit_end(epoch=epoch, master_bar=mbar)

        self.cb_handler.on_resume_end(epoch=epoch, master_bar=mbar)


    def freeze(self, last_from: int = -1, last_to: int = None):
        params = list(self.model.parameters())
        if last_to == None:
            for param in params[:last_from]:
                param.requires_grad = False

            for param in params[last_from:]:
                param.requires_grad = True
        else:
            for param in params[:last_to]:
                param.requires_grad = False

            for param in params[last_from:last_to]:
                param.requires_grad = True


    def unfreeze(self):
        params = self.model.parameters()
        for param in params:
            param.requires_grad = True

    def predict(self, data: Union[AnyStr, torch.Tensor], use_topk=False, kval=5, transform=None):
        self.predictor.transform = transform
        result = self.predictor.predict(data, use_topk=use_topk, kval=kval)
        return result
