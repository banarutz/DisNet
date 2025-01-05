import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule

# Define the model class (DnCNN) if not already imported
from training.net_architecture import DnCNN

# Load the model checkpoint
def load_model(ckpt_path):
    model = DnCNN.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

# Load and preprocess image
def load_image(file_path):
    return np.load(file_path)

# Perform inference
def infer(model, noisy_image):
    noisy_image_tensor = torch.from_numpy(noisy_image).unsqueeze(0).float()
    model = model.to('cpu')
    with torch.no_grad():
        predicted_noise = model(noisy_image_tensor).squeeze().numpy()  # Remove batch and channel dimensions
    denoised_image = noisy_image - predicted_noise
    return denoised_image, predicted_noise

# Plot results
def plot_images(gt_patch, denoised_patch):
    plt.figure(figsize=(10, 5))
    
    # Plot GT patch
    plt.subplot(1, 2, 1)
    plt.imshow(gt_patch.transpose(1, 2, 0))  # Transpune pentru a pune canalele la sfârșit (HWC)
    plt.title('GT Patch')
    plt.axis('off')
    
    # Plot denoised patch
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_patch.transpose(1, 2, 0))  # Transpune pentru a pune canalele la sfârșit (HWC)
    plt.title('Denoised Patch')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("./comparing.png")

def main():
    # Paths to files
    ckpt_path = "/home/smbanaru/Desktop/DisNet/saved_models/best_model-v1.ckpt"  # Replace with your model checkpoint path
    noisy_patch_path = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_medium_50x50_patches/val_noisy_patch_998783.npy"  # Replace with noisy patch file path
    gt_patch_path = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_medium_50x50_patches/val_gt_patch_998783.npy"  # Replace with GT patch file path

    # Load model
    model = load_model(ckpt_path)
    print("Model loaded successfully.")

    # Load noisy image
    noisy_patch = load_image(noisy_patch_path)
    print("Noisy image loaded successfully.")

    # Perform inference
    denoised_patch, predicted_noise = infer(model, noisy_patch)
    print("Inference completed.")

    # Load GT patch
    gt_patch = load_image(gt_patch_path)
    print("GT patch loaded successfully.")
    
    # Plot GT patch and denoised patch
    plot_images(gt_patch, denoised_patch)
    print("Results displayed.")

if __name__ == "__main__":
    main()
