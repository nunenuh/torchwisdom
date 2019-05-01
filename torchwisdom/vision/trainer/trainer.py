from torchwisdom.optim.callback import *
from torchwisdom.metrics import *
from torchwisdom.callback import *
from torchwisdom.utils.data.collector import *
from torch.optim.optimizer import Optimizer
from torchwisdom.trainer import *
from torchwisdom.envi import *
from typing import *
from fastprogress import master_bar, progress_bar
import torch.nn as nn
import torch
import torch.optim as optim
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchwisdom.vision.predictor import ConvPredictor

__all__ = []

shell = python_shell()
if shell == 'shell': from tqdm import tqdm, tnrange
elif shell == 'ipython': from tqdm import tqdm, tnrange
elif shell == 'ipython-notebook': from tqdm import tqdm_notebook as tqdm, tnrange
elif shell == 'jupter-notebook': from tqdm import tqdm_notebook as tqdm, tnrange
else: from tqdm import tqdm, tnrange


class ConvTrainer(Trainer):
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



    def _set_device(self):
        self.model = self.model.to(device=self.device)
        # self.optimizer = self.optimizer.to(device=self.device)

    def build_optimizer(self, lr=0.001, mmt=0.9, wd=0.1):
        if self.optimizer is 'sgd':
            self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=mmt, weight_decay=wd)

    # def _loss_fn(self, pred, target):
    #     name = self.criterion.__class__.__name__
    #     if name=='BCELoss' or name=='BCEWithLogitsLoss':
    #         pred = pred.float()
    #         target = target.unsqueeze(dim=1).float()
    #         return self.criterion(pred, target)
    #     return self.criterion(pred, target)

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
        return  result
    #     if use_dataset_transform:
    #         tmft_image = self._get_transformed_image(data)
    #     else:
    #         if len(data.size()) ==3:
    #             tmft_image = data.unsqueeze(dim=0)
    #         else:
    #             tmft_image = data
    #     tmft_image = tmft_image.to(self.device)
    #
    #     self.model.to(self.device)
    #     self.model.eval()
    #     output = self.model(tmft_image)
    #     return output
    #
    # def predict_single(self, data, readable=False, topk=None, show_graph=False):
    #     pred = self.predict(data)
    #     if len(pred.size())==4:
    #         pred = pred.squeeze()
    #
    #     if readable:
    #         argmx = torch.argmax(pred, dim=1).item()
    #         itc = idx_to_class(self.data.validset.class_to_idx)
    #         return itc[argmx]
    #     else:
    #         return pred



    # def _get_transformed_image(self, data: Union[AnyStr, torch.Tensor]):
    #     if type(data) is str:
    #         path = Path(data)
    #         if path.exists() and path.is_file():
    #             image = Image.open(data)
    #             if image:
    #                 tmft = self._get_transform()
    #                 img_tmft = tmft(image)
    #                 if len(img_tmft.size())==3:
    #                     img_tmft = img_tmft.unsqueeze(dim=0)
    #                 return img_tmft
    #             else:
    #                 raise ValueError("Image is not exist!")
    #         else:
    #             raise ValueError("File not found!")
    #     elif type(data) is torch.Tensor:
    #         tmft = self._get_transform()
    #         img_tmft = tmft(data)
    #         if len(img_tmft.size()) == 3:
    #             img_tmft = img_tmft.unsqueeze(dim=0)
    #         return img_tmft
    #     else:
    #         raise ValueError("data must type string or Tensor!")
    #
    #
    # def _get_transform(self, from_pil=True):
    #     if hasattr(self.data.validset, 'transform'):
    #         tmft = self.data.validset.transform
    #     elif hasattr(self.data.trainset, 'transform'):
    #         tmft = self.data.trainset.transform
    #     else:
    #         if from_pil:
    #             tmft = transforms.Compose([transforms.ToTensor()])
    #         else:
    #             tmft = transforms.Compose([transforms.ToPILImage(),
    #                                        transforms.ToTensor()])
    #
    #     return tmft

