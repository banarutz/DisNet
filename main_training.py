# main.py
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
from pytorch_lightning.callbacks import ModelCheckpoint


class AnimalClassifier(LightningModule):
    def __init__(self, hparams, num_classes):
        super().__init__()
        self.hparams = hparams
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


@hydra.main(config_path="configs", config_name="config")
def train_model(cfg):
    seed_everything(cfg.seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create data loaders
    train_data = ImageFolder(root=cfg.data.train_dir, transform=transform)
    val_data = ImageFolder(root=cfg.data.val_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size)

    # Initialize the model
    model = AnimalClassifier(cfg.model, num_classes=len(train_data.classes))

    # Initialize MLFlow logger
    mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow.tracking_uri)

    # Configure model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
    filename='{epoch}-{val_loss:.2f}',
    dirpath=os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint.path),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
    )

    # Train the model
    trainer = Trainer(
        logger=mlflow_logger,
        max_epochs=cfg.training.max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_model()
