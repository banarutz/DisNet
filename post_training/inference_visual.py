import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from main_training import AnimalClassifier


def validate(model, val_data, num_samples=5):
    
    val_loader = DataLoader(val_data, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(val_loader))

    # Make predictions
    with torch.no_grad():
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs, 1)

    # Get class labels
    class_labels = val_data.classes

    # Display predictions on images
    for i in range(num_samples):
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        inv_tensor = invTrans(images[i])
        image = inv_tensor.permute(1, 2, 0).numpy()
        image = np.uint8(image * 255)
        image_pil = Image.fromarray(image)  

        # Add predicted label and true label to image
        predicted_label = class_labels[predicted[i]]
        true_label = class_labels[labels[i]]
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()

        draw.text((10, 10), f'Predicted: {predicted_label}', fill=(255, 255, 255), font=font)
        draw.text((10, 30), f'True: {true_label}', fill=(255, 255, 255), font=font)

        image_pil.show()


def main(data_dir, checkpoint_path, num_samples=5):
    # Load the model
    model = torch.load(checkpoint_path)
    
    # Load the validation dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_data = ImageFolder(root=data_dir, transform=transform)
    val_loader = DataLoader(val_data, batch_size=num_samples, shuffle=True)
    
    # Validate the model
    validate(model, val_data, num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained model')
    parser.add_argument('--data_dir', 
                        type=str, 
                        default="/home/sbanaru/Desktop/DisNet/dataset/split_ds/val", 
                        help='Path to the data directory')
    parser.add_argument('--checkpoint_path', 
                        type=str, 
                        default="/home/sbanaru/Desktop/DisNet/saved_models/best_model.pt",
                        help='Path to the .ckpt model file')
    parser.add_argument('--num_samples', 
                        type=int, 
                        default=5,
                        help='Number of samples to validate')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.checkpoint_path, args.num_samples)
