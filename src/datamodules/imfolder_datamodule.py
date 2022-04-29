import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class ImageFolderDataModule(LightningDataModule):
    """
    Example of LightningDataModule for ImageFolder dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/imagenette2",
        image_size: int = 224,
        num_class: int = 10,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = None
        self.test_transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # define transforms for training, validation and testing
        if not self.train_transforms and not self.test_transforms:
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomResizedCrop(
                        (self.hparams.image_size, self.hparams.image_size)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.test_transforms = transforms.Compose(
                [
                    transforms.Resize((self.hparams.image_size, self.hparams.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # load training samples
            self.data_train = ImageFolder(
                root=os.path.join(self.hparams.data_dir, "train"), transform=self.train_transforms
            )
            # load validation samples
            self.data_val = ImageFolder(
                root=os.path.join(self.hparams.data_dir, "val"), transform=self.test_transforms
            )
            # load test samples, if available
            self.data_test = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        """
        No test set for this example.
        """
        pass
