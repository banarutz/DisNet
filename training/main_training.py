import os
from pathlib import Path
import yaml
import hydra
import mlflow
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import subprocess

from net_architecture import DnCNN
from data_pipeline import CustomImageDataset


@hydra.main(config_path="../configs", config_name="DnCNN_SIDD_medium_50x50_experiment_1")
def train_model(cfg):
    seed_everything(cfg.training.seed)

    # Define transformations
    transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

    # Create data loaders
    train_dataset = CustomImageDataset(root_dir=cfg.data.root_dir, split="train", transform=transform, seed=cfg.training.seed)
    val_dataset = CustomImageDataset(root_dir=cfg.data.root_dir, split="val", transform=transform, seed=cfg.training.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

    # Initialize the model
    model = DnCNN(cfg.model)

    # Initialize MLFlow logger
    try:
        subprocess.Popen(["mlflow", "server", "--host", "127.0.0.1", "--port", "8081"])
        mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow.tracking_uri)
        mlflow_logger.log_hyperparams(cfg)
    except Exception as e:
        print(f"Failed to start MLFlow server: {e}")
    # mlflow_logger = None

    # Configure model checkpoint callback to save only the best model
    checkpoint_callback = ModelCheckpoint(
        filename='best_model',
        dirpath=os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint.path),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    # Initialize LR monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    torch.set_float32_matmul_precision("medium")
    print("----------------- Starting training -----------------") 
    # Train the model
    trainer = Trainer(
        logger=mlflow_logger,
        max_epochs=cfg.training.max_epochs,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu",
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(model, train_loader, val_loader)

    # # Load the best model and save as .pt
    # best_model_path = os.path.join("/home/sbanaru/Desktop/DisNet/saved_models/best_model.ckpt")
    # best_model = AnimalClassifier.load_from_checkpoint(
    #     best_model_path,
    #     )
    # model_save_path = os.path.join("/home/sbanaru/Desktop/DisNet/saved_models/", 'best_model.pt')
    # torch.save(best_model, model_save_path)

# @hydra.main(config_path="../configs", config_name="experiment_1")
# def save_best_model(cfg):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     train_data = ImageFolder(root=cfg.data.train_dir, transform=transform)
#     model = AnimalClassifier(cfg.model, num_classes=len(train_data.classes))
#     best_model_path = os.path.join("/home/sbanaru/Desktop/DisNet/saved_models/best_model.ckpt")
#     best_model = model.load_from_checkpoint(best_model_path)
#     model_save_path = os.path.join("/home/sbanaru/Desktop/DisNet/saved_models/", 'best_model.pt')
#     torch.save(best_model.state_dict(), model_save_path)

if __name__ == "__main__":
    train_model()
    
