import os
import numpy as np
from PIL import Image

def save_npy_as_png(source_dir, destination_dir):
    """
    Citește fișierele .npy din directorul sursă, le convertește în imagini .png și le salvează în directorul destinație.
    
    Args:
    - source_dir (str): Calea către directorul sursă cu fișierele .npy.
    - destination_dir (str): Calea către directorul destinație pentru fișierele .png.
    """
    # Verifică dacă directorul sursă există
    if not os.path.exists(source_dir):
        print(f"Directorul sursă {source_dir} nu există!")
        return
    
    # Creează directorul destinație dacă nu există
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterează prin fișierele din directorul sursă
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".npy"):
            # Calea completă a fișierului sursă
            source_file = os.path.join(source_dir, file_name)
            
            # Citește fișierul .npy
            image_data = np.load(source_file)

            # Verifică dacă dimensiunea imaginii este corectă (50, 50, 3)
            if image_data.shape[-1] == 3:
                # Transformați în imagini RGB folosind PIL
                image = Image.fromarray(image_data.astype(np.uint8))
                
                # Calea fișierului destinație (.png)
                destination_file = os.path.join(destination_dir, file_name.replace(".npy", ".png"))
                
                # Salvează imaginea în format .png
                image.save(destination_file)
                print(f"Fișierul {file_name} a fost salvat ca {destination_file}")
            else:
                print(f"Fișierul {file_name} nu are forma corectă pentru o imagine RGB.")

# Exemplu de utilizare:
source_path = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_small_50x50_patches_v2/val/"
destination_path = "/media/smbanaru/9bf28602-242e-4916-960b-479e4c5d241e/datasets/SIDD_small_50x50_patches_v2/val_images/"

save_npy_as_png(source_path, destination_path)
