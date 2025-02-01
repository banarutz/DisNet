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
        loss = nn.L1Loss()(denoised, clean)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = noisy - self(noisy)
        loss = nn.L1Loss()(denoised, clean)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        psnr = 10 * torch.log10(1 / loss)  # Calculul PSNR
        self.log('val_psnr', psnr, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_max=32, eta_min=1e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
        

class EncoderDecoderDenoising(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Dimensiune păstrată
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Down-sampling
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Down-sampling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # Up-sampling
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Up-sampling
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Dimensiune păstrată
        )

    def forward(self, x):
        # Encoder
        latent = self.encoder(x)

        # Decoder
        reconstructed = self.decoder(latent)

        # Rețea reziduală
        return reconstructed  # Asigurăm aceeași dimensiune între x și reconstructed

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = nn.MSELoss()(denoised, clean)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = nn.MSELoss()(denoised, clean)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        psnr = 10 * torch.log10(1 / loss)  # Calculul PSNR
        self.log('val_psnr', psnr, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
