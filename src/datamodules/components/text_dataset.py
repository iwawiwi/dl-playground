import os
import random
from string import ascii_letters

import torch
from torch.utils.data import Dataset
from unidecode import unidecode


class NameDataset(Dataset):
    CHAR2IDX = {char: idx for idx, char in enumerate(ascii_letters + " .,:;-'")}
    LANG2LABEL = {}

    def __init__(self, path: str):
        self.path = path

        # populate label tensor to class variable
        NameDataset.LANG2LABEL = {
            file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
            for i, file_name in enumerate(os.listdir(path))
        }

        self.tensor_names = []
        self.target_langs = []
        self.read_names()

    def read_names(self):
        """
        Read names from files
        """
        for file in os.listdir(self.path):
            with open(os.path.join(self.path, file), "r", encoding="utf-8") as f:
                lang = file.split(".")[0]
                names = [unidecode(line.rstrip()) for line in f]
                for name in names:
                    # print(len(self.tensor_names))
                    try:
                        self.tensor_names.append(
                            NameDataset.name2tensor(name)
                        )  # convert name to tensor
                        self.target_langs.append(self.LANG2LABEL[lang])  # check target_langs
                    except KeyError:
                        pass
                # print("finisherd reading {}".format(lang))

    def __getitem__(self, index):
        """
        Get sample from dataset
        """
        return self.tensor_names[index], self.target_langs[index]

    def __len__(self):
        """
        Get dataset length
        """
        return len(self.tensor_names)

    @staticmethod
    def name2tensor(name):
        """
        Convert a name to a tensor.
        """
        tensor = torch.zeros(len(name), 1, len(NameDataset.CHAR2IDX))
        for i, char in enumerate(name):
            tensor[i, 0, NameDataset.CHAR2IDX[char]] = 1
        return tensor
