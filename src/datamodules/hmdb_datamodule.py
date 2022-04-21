from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import HMDB51
from torchvision.transforms import transforms

from src.utils import hmbd_utils as T


class Hmdb51DataModule(LightningDataModule):
    """
    Example of LightningDataModule for HMDB51 dataset.

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
        data_dir: str = "data/hmdb/video_data",
        annotation_path: str = "data/hmdb/test_train_splits",
        val_split: float = 0.1,
        frames_per_clip: int = 16,
        step_between_clips: int = 50,
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
        return 51

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
                    T.ToFloatTensorInZeroOne(),
                    T.Resize((128, 171)),
                    T.RandomHorizontalFlip(),
                    T.Normalize(
                        mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                    ),
                    T.RandomCrop((112, 112)),
                ]
            )
            self.test_transforms = transforms.Compose(
                [
                    T.ToFloatTensorInZeroOne(),
                    T.Resize((128, 171)),
                    T.Normalize(
                        mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                    ),
                    T.CenterCrop((112, 112)),
                ]
            )

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # load all training samples
            train_all = HMDB51(
                self.hparams.data_dir,
                self.hparams.annotation_path,
                self.hparams.frames_per_clip,
                self.hparams.step_between_clips,
                train=True,
                transform=self.train_transforms,
            )
            # split all training into train and val based on validation split
            total_train_samples = len(train_all)
            total_val_samples = round(self.hparams.val_split * total_train_samples)
            self.data_train, self.data_val = random_split(
                train_all, [total_train_samples - total_val_samples, total_val_samples]
            )
            # load test samples
            self.data_test = HMDB51(
                self.hparams.data_dir,
                self.hparams.annotation_path,
                self.hparams.frames_per_clip,
                self.hparams.step_between_clips,
                train=False,
                transform=self.test_transforms,
            )

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
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
