import os
import numpy as np
import matplotlib.pyplot as plt

def get_sorted_patch_lists(folder_path):
    """Găsește și sortează patch-urile GT și noisy dintr-un folder."""
    gt_patches = []
    noisy_patches = []

    # Parcurge toate fișierele din folder
    for file in os.listdir(folder_path):
        if "val_gt_patch" in file and file.endswith(".npy"):
            gt_patches.append(os.path.join(folder_path, file))
        elif "val_noisy_patch" in file and file.endswith(".npy"):
            noisy_patches.append(os.path.join(folder_path, file))

    # Sortează listele
    gt_patches.sort()
    noisy_patches.sort()

    return gt_patches, noisy_patches

def load_patches(patch_paths):
    """Încarcă toate patch-urile dintr-o listă de căi."""
    return [np.load(patch).transpose(1, 2, 0) for patch in patch_paths]  # Transpune la format (H, W, C)

def plot_and_save_patches(gt_patches, noisy_patches, save_path="comparison_plot.png"):
    """
    Plotează ultimele 50 de patch-uri GT și noisy și salvează figura.
    GT în stânga, noisy în dreapta.
    """
    num_patches = min(50, len(gt_patches), len(noisy_patches))  # Asigură-te că sunt maximum 50
    fig, axes = plt.subplots(nrows=num_patches, ncols=2, figsize=(10, num_patches * 2))
    
    for i in range(num_patches):
        # GT patch (stânga)
        axes[i, 0].imshow(gt_patches[-num_patches + i])  # Selectează ultimele patch-uri
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("GT Patch", fontsize=12)

        # Noisy patch (dreapta)
        axes[i, 1].imshow(noisy_patches[-num_patches + i])  # Selectează ultimele patch-uri
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title("Noisy Patch", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figura a fost salvată la {save_path}")

def main():
    # Setează folderul cu patch-uri
    folder_path = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_medium_50x50_patches"  # Înlocuiește cu calea ta

    # Obține listele sortate de patch-uri
    gt_patch_paths, noisy_patch_paths = get_sorted_patch_lists(folder_path)

    # Încarcă patch-urile
    gt_patches = load_patches(gt_patch_paths)
    noisy_patches = load_patches(noisy_patch_paths)

    # Plotează și salvează figura
    plot_and_save_patches(gt_patches, noisy_patches, save_path="comparison_plot.png")

if __name__ == "__main__":
    main()