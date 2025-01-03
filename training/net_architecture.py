import torch
from torch import nn
from pytorch_lightning import LightningModule

class DnCNN(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Arhitectura DnCNN
        layers = []
        # Primul strat convoluțional
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Straturi convoluționale intermediare cu normalizare batch
        for _ in range(self.hparams.num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        
        # Ultimul strat convoluțional (fără activare sau normalizare)
        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Rețea reziduală: input + output
        return x - self.model(x)

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = nn.MSELoss()(denoised, clean)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = nn.MSELoss()(denoised, clean)
        self.log('val_loss', loss)
        psnr = 10 * torch.log10(1 / loss)  # Calculul PSNR
        self.log('val_psnr', psnr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
