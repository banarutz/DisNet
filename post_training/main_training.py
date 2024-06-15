import os
from pathlib import Path
import yaml
import hydra
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img, label = self.image_folder[idx]
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label


class AnimalClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model = self.model.features 
        self.PredHead = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 37)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.PredHead(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        preds = torch.argmax(y_hat, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        self.log('val_acc', correct / total, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_acc'
        }


@hydra.main(config_path="../configs", config_name="experiment_1")
def train_model(cfg):
    seed_everything(cfg.training.seed)

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.RandomCrop(width=224, height=224, p=0.5),
            A.Rotate(limit=40, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Create data loaders
    train_data = ImageFolder(root=cfg.data.train_dir)
    val_data = ImageFolder(root=cfg.data.val_dir)
    
    train_dataset = CustomDataset(image_folder=train_data, transform=train_transform)
    val_dataset = CustomDataset(image_folder=val_data, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    # Initialize the model
    model = AnimalClassifier(cfg.model)

    # Initialize MLFlow logger
    mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow.tracking_uri)
    mlflow_logger.log_hyperparams(cfg)

    # Configure model checkpoint callback to save only the best model
    checkpoint_callback = ModelCheckpoint(
        filename='best_model',
        dirpath=os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint.path),
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max',
    )

    # Initialize LR monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Train the model
    trainer = Trainer(
        logger=mlflow_logger,
        max_epochs=cfg.training.max_epochs,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu",
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(model, train_loader, val_loader)

    # Load the best model and save as .pt
    best_model_path = os.path.join("/home/sbanaru/Desktop/DisNet/saved_models/best_model.ckpt")
    best_model = AnimalClassifier.load_from_checkpoint(best_model_path)
    model_save_path = os.path.join("/home/sbanaru/Desktop/DisNet/saved_models/", 'best_model.pt')
    torch.save(best_model.state_dict(), model_save_path)


if __name__ == "__main__":
    train_model()
