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

from net_architecture import DnCNN, EncoderDecoderDenoising
from data_pipeline import CustomImageDataset 
from utils import *


@hydra.main(config_path="../configs", config_name="DeVit_SIDD_small_50x50_experiment_1")
def train_model(cfg):
    seed_everything(cfg.training.seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data loaders
    train_dataset = CustomImageDataset(root_dir=cfg.data.root_dir, split="train", transform=transform, seed=cfg.training.seed)
    val_dataset = CustomImageDataset(root_dir=cfg.data.root_dir, split="val", transform=transform, seed=cfg.training.seed)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers)

    # Initialize the model
    model = model_picker(cfg)

    # Initialize MLFlow logger
    if is_port_in_use(cfg.mlflow.host, cfg.mlflow.port):
        print("MLFlow server is already running.")
        mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow.tracking_uri)
    else:
        print("Starting MLFlow server.")
        try:
            subprocess.Popen(["/home/smbanaru/Desktop/DisNet/venv/bin/mlflow", "server", "--host", "127.0.0.1", "--port", "8083"])
            mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow.tracking_uri)
            mlflow_logger.log_hyperparams(cfg)
        except Exception as e:
            print(f"Failed to start MLFlow server: {e}")
            mlflow_logger = None

    # Configure model checkpoint callback to save only the best model
    checkpoint_callback = ModelCheckpoint(
        filename='best_model',
        dirpath=os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint.path),
        save_top_k=1,
        verbose=True,
        monitor='val_psnr',
        mode='max',
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
        num_sanity_val_steps=0
    )
    trainer.fit(model, train_loader, val_loader)
    
    print("----------------- Training finished -----------------")
    print("----------------- Exporting model -----------------")
    
    model = DnCNN.load_from_checkpoint(os.path.join(cfg.checkpoint.path, cfg.checkpoint.filename))
    torch.save(model.state_dict(), os.path.join(cfg.checkpoint.path, cfg.checkpoint.filename).split(".")[0] + ".pt")
    model.eval()
    dummy_input = torch.randn(1, 3, 50, 50)
    torch.onnx.export(
        model, 
        dummy_input, 
        os.path.join(cfg.checkpoint.path + cfg.checkpoint.filename).split(".")[0] + ".onnx", 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Opțional: suport pentru batch-uri dinamice
        opset_version=11  # Ajustează versiunea dacă e necesar
    )
    
    print("done.")
    
if __name__ == "__main__":
    train_model()
