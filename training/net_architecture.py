import torch
from torch import nn
import torchmetrics
from torchvision.models import vit_b_16
from pytorch_lightning import LightningModule
import torch.nn.functional as F

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


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class TransformerEncoderDecoderDenoising(LightningModule):
    def __init__(self, hparams):
        super(TransformerEncoderDecoderDenoising, self).__init__()
        self.save_hyperparameters(hparams)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

        # Encoder: ViT-b-16
        
        self.encoder = vit_b_16(pretrained=True)
        self.encoder.heads = nn.Identity()  # Remove classification head
        for param in self.encoder.parameters():
            param.requires_grad = False 
        
        # Decoder: reconstruiește 50x50
        self.decoder = nn.Sequential(
        nn.Linear(768, 128 * 7 * 7),  # Transformă 768 într-un feature map de 128x7x7
        nn.ReLU(),
        
        # Reshape în (batch_size, 128, 7, 7) înainte de convoluții
        View((-1, 128, 7, 7)),  
        
        nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1),  # 7x7 -> 14x14
        nn.ReLU(),
        nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2),  # 7x7 -> 14x14
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),  # 14x14 -> 28x28
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),  # 28x28 -> 50x50
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, output_padding=1),
        # nn.UpsamplingBilinear2d(224), 
)

        self.loss_fn = nn.MSELoss()
        self.lr = self.hparams.learning_rate

    def forward(self, x):
        batch_size = x.shape[0]
        # x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Encoder
        x = self.encoder(x)  # Output: [batch, 197, 768]
        # x = x[:, 1:, :]  # Eliminăm tokenul de clasă -> [batch, 196, 768]
        x = x.view(batch_size, 768)  # Reshape în format spațial
  
        x = self.decoder(x)  # Output: [batch, 3, 50, 50]
        return x
    
    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self.forward(noisy)
        loss = self.loss_fn(denoised, clean)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self.forward(noisy)
        loss = self.loss_fn(denoised, clean)
        psnr = 10 * torch.log10(1 / loss)
        ssim_value = self.ssim(denoised, clean)
        self.log("val_loss", loss, prog_bar=True)
        self.log('val_psnr', psnr, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_value, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
