import os
import random
import shutil


def split_dataset(input_dir, output_dir, split_ratio=0.8):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            # Create corresponding directories in the output directory
            input_subdir = os.path.join(root, dir_name)
            output_subdir_train = os.path.join(output_dir, "train", dir_name)
            output_subdir_val = os.path.join(output_dir, "val", dir_name)
            os.makedirs(output_subdir_train, exist_ok=True)
            os.makedirs(output_subdir_val, exist_ok=True)

            # Get list of image files in the current subdirectory
            images = [f for f in os.listdir(input_subdir) if f.endswith(".jpg")]
            # Shuffle the images randomly
            random.shuffle(images)
            # Split the images into train and validation sets
            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            val_images = images[split_index:]

            # Copy train images to train subdirectory
            for image in train_images:
                src = os.path.join(input_subdir, image)
                dst = os.path.join(output_subdir_train, image)
                shutil.copyfile(src, dst)

            # Copy validation images to validation subdirectory
            for image in val_images:
                src = os.path.join(input_subdir, image)
                dst = os.path.join(output_subdir_val, image)
                shutil.copyfile(src, dst)


# Example usage
input_dir = "/home/sbanaru/Desktop/DisNet/dataset/prepared_ds/"
output_dir = "/home/sbanaru/Desktop/DisNet/dataset/split_ds"
split_dataset(input_dir, output_dir)
