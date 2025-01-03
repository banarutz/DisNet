import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seed=42):
        """
        Args:
            root_dir (str): Directorul care conține fișierele .npy.
            split (str): 'train' sau 'val', în funcție de setul dorit.
            transform (callable, optional): Transformări aplicate pe imagini (ex. normalizare).
            seed (int): Seed pentru reproducibilitate.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.seed = seed

        # Obține lista cu fișiere GT
        self.gt_files = self._get_gt_files()

    def _get_gt_files(self):
        """
        Creează o listă cu căile către fișierele GT din directorul specificat.
        """
        gt_files = []
        prefix = f"{self.split}_gt_patch_"

        for file_name in os.listdir(self.root_dir):
            if file_name.startswith(prefix):
                gt_files.append(os.path.join(self.root_dir, file_name))

        # Sortează fișierele pentru a menține ordinea consistentă
        gt_files.sort()
        return gt_files

    def __len__(self):
        """
        Returnează numărul de patch-uri (bazat pe numărul de fișiere GT).
        """
        return len(self.gt_files)

    def __getitem__(self, idx):
        """
        Returnează perechea [gt_patch, noisy_patch] pentru un anumit index.
        """
        gt_file_path = self.gt_files[idx]
        noisy_file_path = gt_file_path.replace('_gt_patch_', '_noisy_patch_')

        # Încarcă patch-urile GT și NOISY din fișierele .npy
        gt_patch = np.load(gt_file_path)
        noisy_patch = np.load(noisy_file_path)

        # Aplică transformările, dacă sunt specificate
        if self.transform:
            gt_patch = self.transform(gt_patch)
            noisy_patch = self.transform(noisy_patch)

        # Convertește la tensori Torch
        return torch.tensor(gt_patch, dtype=torch.float32), torch.tensor(noisy_patch, dtype=torch.float32)
