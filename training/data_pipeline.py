import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seed=42):
        """
        Args:
            root_dir (str): Directorul care conține folderele 'train' și 'val'.
            split (str): 'train' sau 'val', în funcție de setul dorit.
            transform (callable, optional): Transformări aplicate pe imagini (ex. normalizare).
            seed (int): Seed pentru reproducibilitate.
        """
        self.root_dir = os.path.join(root_dir, split)  # Setează calea către folderul split-ului
        self.transform = transform
        self.seed = seed
        self.split = split

        # Obține lista cu fișiere GT și NOISY
        self.gt_files, self.noisy_files = self._get_patch_pairs()

    def _get_patch_pairs(self):
        """
        Creează liste cu căile către fișierele GT și NOISY din directorul specificat.
        Returnează două liste sortate: una pentru GT și una pentru NOISY.
        """
        gt_files = []
        noisy_files = []

        for file_name in os.listdir(self.root_dir):
            if file_name.startswith(f"{self.split}_gt_") and file_name.endswith(".npy"):
                gt_files.append(os.path.join(self.root_dir, file_name))
            elif file_name.startswith(f"{self.split}_noisy_") and file_name.endswith(".npy"):
                noisy_files.append(os.path.join(self.root_dir, file_name))

        # Sortează fișierele pentru a menține ordinea consistentă
        gt_files.sort()
        noisy_files.sort()

        # Verifică că numărul de fișiere GT și NOISY este egal
        assert len(gt_files) == len(noisy_files), \
            f"Numărul de fișiere GT ({len(gt_files)}) și NOISY ({len(noisy_files)}) nu corespunde."

        return gt_files, noisy_files

    def __len__(self):
        """
        Returnează numărul de perechi de patch-uri.
        """
        return len(self.gt_files)
    
    def __getitem__(self, idx):
        """
        Returnează perechea [gt_patch, noisy_patch] pentru un anumit index.
        """
        gt_file_path = self.gt_files[idx]
        noisy_file_path = self.noisy_files[idx]

        gt_patch = np.load(gt_file_path)
        noisy_patch = np.load(noisy_file_path)
        
        if self.transform:
            gt_patch = self.transform(gt_patch)
            noisy_patch = self.transform(noisy_patch)

        return gt_patch, noisy_patch