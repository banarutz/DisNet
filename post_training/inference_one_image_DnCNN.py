import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw
import os
from pytorch_lightning import LightningModule
from training.net_architecture import DnCNN

def split_into_patches(image, patch_size):
    """Împarte o imagine în patch-uri de dimensiune fixă."""
    h, w, c = image.shape
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches, (h, w)

def reconstruct_from_patches(patches, image_shape, patch_size):
    """Recompune imaginea originală din patch-uri."""
    h, w = image_shape
    reconstructed = np.zeros((h, w, 3))
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch_h, patch_w, _ = patches[idx].shape
            reconstructed[i:i+patch_h, j:j+patch_w] = patches[idx]
            idx += 1
    return reconstructed

def add_zoom_patch(image, zoom_region, zoom_factor=4):
    """Adaugă un patch zoom-uit pe imagine."""
    x1, y1, x2, y2 = zoom_region
    patch = image.crop((x1, y1, x2, y2))
    patch = patch.resize((patch.width * zoom_factor, patch.height * zoom_factor), Image.NEAREST)
    
    # Crează un canvas pentru imaginea cu patch-ul zoom-uit
    canvas = Image.new("RGB", (image.width, image.height + patch.height + 10), (255, 255, 255))
    canvas.paste(image, (0, 0))
    canvas.paste(patch, (10, image.height + 10))
    
    # Desenează un pătrat pentru patch-ul original
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return canvas

def denoise_image(model, image_path, gt_path, patch_size, output_path, zoom_region):
    """Procesează imaginea pentru a elimina zgomotul și o compară cu GT."""
    # Încarcă imaginile
    noisy_image = np.array(Image.open(image_path).convert("RGB"))
    gt_image = np.array(Image.open(gt_path).convert("RGB"))

    # Împarte imaginea în patch-uri
    patches, image_shape = split_into_patches(noisy_image, patch_size)

    # Transformă patch-urile în tensori și procesează prin model
    model.eval()
    model.to('cpu')
    denoised_patches = []
    with torch.no_grad():
        for patch in patches:
            patch_tensor = ToTensor()(patch).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            denoised_patch = model(patch_tensor).squeeze(0)  # [1, C, H, W] -> [C, H, W]
            denoised_patches.append(ToPILImage()(denoised_patch))

    # Reconstruiește imaginea denoisată
    denoised_image = reconstruct_from_patches(
        [np.array(patch) for patch in denoised_patches], image_shape, patch_size
    )
    
    print(np.max(denoised_image), np.min(denoised_image))  
    

    # Creează imaginea side-by-side
    # denoised_pil = Image.fromarray(denoised_image.astype(np.uint8))
    denoised_pil = Image.fromarray((noisy_image - denoised_image).clip(0, 255).astype(np.uint8))   
    gt_pil = Image.fromarray(gt_image)
    side_by_side = Image.new("RGB", (denoised_pil.width + gt_pil.width, denoised_pil.height))
    side_by_side.paste(denoised_pil, (0, 0))
    side_by_side.paste(gt_pil, (denoised_pil.width, 0))

    # Adaugă patch-uri zoom-uite
    side_by_side_with_zoom = add_zoom_patch(side_by_side, zoom_region)

    # Salvează imaginea rezultată
    side_by_side_with_zoom.save(output_path)
    print(f"Imaginea side-by-side cu patch zoom-uit a fost salvată la: {output_path}")

# Exemplu de utilizare
if __name__ == "__main__":
    model = DnCNN.load_from_checkpoint("/home/smbanaru/Desktop/DisNet/saved_models/best_model-v2.ckpt")  # Încarcă modelul antrenat
    denoise_image(
        model=model,
        image_path="/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_Small_sRGB_Only/Data/val/0194_009_IP_01600_04000_3200_N/NOISY_SRGB_010.PNG",
        gt_path="/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_Small_sRGB_Only/Data/val/0194_009_IP_01600_04000_3200_N/GT_SRGB_010.PNG",
        patch_size=50,
        output_path="experiment_1_dncnn_small.png",
        zoom_region=(10, 10, 60, 60)  # Coordonate pentru patch-ul zoom-uit (x1, y1, x2, y2)
    )
