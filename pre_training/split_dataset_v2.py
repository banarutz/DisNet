import os
import numpy as np
from PIL import Image

def extract_patches(image, patch_size=50):
    """Taie imaginea în patch-uri de dimensiune fixă."""
    patches = []
    h, w = image.shape[:2]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[:2] == (patch_size, patch_size):  # Verifică dimensiunea
                patches.append(patch)
    return patches

def process_dataset(input_dir, output_dir, patch_size=50):
    """Procesează dataset-ul și generează patch-uri GT și NOISY."""
    for split in ["train", "val"]:
        split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        idx = 0  # Index global pentru patch-uri
        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"Skipping {folder_path} as it is not a directory.")
                continue

            # Identifică imaginile GT și NOISY
            gt_image_path = None
            noisy_image_path = None
            for file in os.listdir(folder_path):
                if "GT" in file and file.endswith((".PNG", ".jpg")):
                    gt_image_path = os.path.join(folder_path, file)
                elif "NOISY" in file and file.endswith((".PNG", ".jpg")):
                    noisy_image_path = os.path.join(folder_path, file)
                

            if gt_image_path and noisy_image_path:
                # Încarcă imaginile și convertește la numpy array
                gt_image = np.array(Image.open(gt_image_path).convert("RGB"))
                noisy_image = np.array(Image.open(noisy_image_path).convert("RGB"))

                # Taie imaginile în patch-uri
                gt_patches = extract_patches(gt_image, patch_size=patch_size)
                noisy_patches = extract_patches(noisy_image, patch_size=patch_size)

                # Verifică că numărul de patch-uri corespunde
                assert len(gt_patches) == len(noisy_patches), \
                    f"Numărul de patch-uri nu corespunde pentru {folder_path}"

                # Salvează patch-urile în folderul de destinație
                for gt_patch, noisy_patch in zip(gt_patches, noisy_patches):
                    gt_patch_path = os.path.join(output_split_dir, f"{split}_gt_{patch_size}x{patch_size}_patch_{idx}.npy")
                    noisy_patch_path = os.path.join(output_split_dir, f"{split}_noisy_{patch_size}x{patch_size}_patch_{idx}.npy")

                    np.save(gt_patch_path, gt_patch)
                    np.save(noisy_patch_path, noisy_patch)
                    idx += 1
        print(f"Procesare completă pentru {split}!")

# Exemplu de utilizare
input_dir = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_Small_sRGB_Only/Data/"
output_dir = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_small_50x50_patches_v2/"
process_dataset(input_dir, output_dir)
