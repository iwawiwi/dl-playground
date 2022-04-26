from turtle import forward
import torch
import torch.nn as nn

from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.cnnmodel_factory import CnnModelFactory

from pytorch_lightning import LightningModule


class StudentTrainingModule(LightningModule):
    """
    Example of LightningModule

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
        student_model: str = "densenet121",
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
        self.model = self.load_model(self.hparams.student_model)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def load_model(self, model_name):
        model_factory = CnnModelFactory()
        return model_factory.build_model(
            model_name=model_name, 
            image_size=self.hparams.image_size,
            num_classes=self.hparams.num_classes,
            pretrained=self.hparams.pretrained
        )

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
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay,
        )


class DistilledTrainingModule(StudentTrainingModule):
    def __init__(
        self, 
        student_model, 
        teacher_model,
        image_size: int = 244, 
        num_classes: int = 10, 
        lr: float = 0.001, 
        weight_decay: float = 0.0005,
        pretrained=False,
    ):
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        super().__init__(student_model=student_model, image_size=image_size, num_classes=num_classes, lr=lr, weight_decay=weight_decay, pretrained=pretrained)
        self.distill_loss = nn.MSELoss()
        self.teacher_model = self.load_model(self.hparams.teacher_model)
        # load saved weights
        self.teacher_model.load_model()
        # freeze teacher weights
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        student_pred = self.forward(images)                     # student prediction
        teacher_pred = self.teacher_model.forward(images)       # teacher prediction

        loss = self.distill_loss(student_pred, teacher_pred)    # loss based on mse loss between student_pred and teacher_pred 

        # log train metrics
        acc = self.train_acc(student_pred, labels)              # accuracy of student based on true target label
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": student_pred, "targets": labels}
