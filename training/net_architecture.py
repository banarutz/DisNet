import torch
from torch import nn
import torchmetrics
from pytorch_lightning import LightningModule

class DnCNN(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # Arhitectura DnCNN
        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(self.hparams.num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = nn.MSELoss()(denoised, clean)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = noisy - self(noisy)
        loss = nn.MSELoss()(denoised, clean)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        psnr = 10 * torch.log10(1 / loss)
        self.log('val_psnr', psnr, prog_bar=True, logger=True)
        ssim = self.ssim(denoised, clean)
        self.log('val_ssim', ssim, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, eta_min=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

class EncoderDecoderDenoising(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        
        mse_loss = nn.MSELoss()(denoised, clean)
        ssim_value = self.ssim(denoised, clean)
        
        epsilon = 1e-6  # pentru a evita împărțirea la 0
        loss = mse_loss * (1 / (ssim_value + epsilon))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        
        mse_loss = nn.MSELoss()(denoised, clean)
        ssim_value = self.ssim(denoised, clean)
        
        epsilon = 1e-6  # pentru stabilitate
        loss = mse_loss * (1 / (ssim_value + epsilon))

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        psnr = 10 * torch.log10(1 / mse_loss)
        self.log('val_psnr', psnr, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_value, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
