import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class ImagenetteTrainingModule(LightningModule):
    """
    Example of LightningModule for Teacher model

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        image_size: int = 244,
        num_classes: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        pretrained=False,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # model
        self.model = net

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.criterion(preds, labels).mean()

        # log train metrics
        acc = self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": labels}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(-1) == labels).float()

        # log val metrics
        acc = self.val_acc(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": labels}

    def test_step(self, batch, batch_idx):
        """
        No test step defined
        """
        pass

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
