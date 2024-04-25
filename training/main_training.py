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


class AnimalClassifier(LightningModule):
    def __init__(self, hparams, num_classes):
        super().__init__()
        self.save_hyperparameters(hparams)  # Save hyperparameters
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
        preds = torch.argmax(y_hat, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        self.log('val_acc', correct / total, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_acc'
        }


@hydra.main(config_path="../configs", config_name="experiment_1")
def train_model(cfg):
    seed_everything(cfg.training.seed)

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
    mlflow_logger.log_hyperparams(cfg)

    # Configure model checkpoint callback to save only the best model
    checkpoint_callback = ModelCheckpoint(
        filename='best_model',
        dirpath=os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint.path),
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='min',
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
    
