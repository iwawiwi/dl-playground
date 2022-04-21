import torch
import torch.utils.data as data

class XORDataset(data.Dataset):
    def __init__(self, size: int, std: float=0.1) -> None:
        """
        Inputs:
            size - banyaknya data yang akan digenerate
            std - standar deviasi untuk noise (lihat fungsi generate_continuous_xor)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.data, self.label = self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Setiap data dalam XORDataset memilikii dua variabel, x dan y, yang nilainya yaitu 0 atau 1
        # Target label adalah kombinasi dari hasul XOR antara dua variabel tersebut, i.e. 
        # target bernilai 1 jika hanya x atau hanya y saja yang bernilai 1 sedangkan nilai yang lainnya adalah 0.
        # Jika x=y, the target label adalah 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).long() # nilai 1 pada xor jika x = 0 dan y = 1 ATAU x = 1 dan y = 0 (jumlahan x dan y adalah 1)
        # beri noise untuk membuat data menjadi lebih menantang, kita menambahkan gaussian noise
        data += torch.randn(data.shape) * self.std

        return data, label

    def __len__(self):
        # Jumlah data yang ada dalam XORDataset
        return self.size

    def __getitem__(self, idx):
        # Mengembalikan data dan label dari index ke-idx
        return self.data[idx], self.label[idx]