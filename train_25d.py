import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import deepinv as dinv
from mri_25d_dataset import MRI25DDataset
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Custom Losses ---

class CentralSliceLoss(nn.Module):
    """
    Computes loss only on the central slice (channel 1) of the output.
    Assumes input is (B, 3, H, W) or (B, C, H, W) where center is C//2.
    For 3 channels, center is index 1.
    """
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
    
    def forward(self, x_net, x, *args, **kwargs):
        # x_net, x are (B, 3, H, W)
        # Select central slice and keep dim: (B, 1, H, W)
        x_net_c = x_net[:, 1:2, :, :]
        x_c = x[:, 1:2, :, :]
        # Check if base_loss accepts physics/model args (dinv losses might)
        # We pass *args which usually contains (physics, model)
        try:
            return self.base_loss(x_net_c, x_c, *args, **kwargs)
        except TypeError:
            return self.base_loss(x_net_c, x_c)

class SSIMLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ssim_metric = dinv.loss.metric.SSIM()
        self.to(device) # Move self to device
        
    def forward(self, x_hat, x, *args, **kwargs):
        # standard SSIM returns value in [0, 1] (higher is better)
        # We want to minimize 1 - SSIM
        return 1 - self.ssim_metric(x_hat, x)

class WeightedSumLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights
        assert len(losses) == len(weights)
        
    def forward(self, *args, **kwargs):
        total_loss = 0
        for loss, w in zip(self.losses, self.weights):
            total_loss += w * loss(*args, **kwargs)
        return total_loss

# Global state for interleaved loss
class ExperimentState:
    def __init__(self):
        self.epoch = 0

state = ExperimentState()

class InterleavedLoss(nn.Module):
    """
    Switches between loss1 and loss2 every 'interval' epochs.
    """
    def __init__(self, loss1, loss2, interval=1):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.interval = interval
    
    def forward(self, *args, **kwargs):
        # Use global state
        if (state.epoch // self.interval) % 2 == 0:
            return self.loss1(*args, **kwargs)
        else:
            return self.loss2(*args, **kwargs)

class CentralSureLoss(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x_net, x, y, physics, model, **kwargs):
        # x_net: (B,3,H,W) - model output
        # y: (B,3,H,W) - noisy input
        
        # 1. MSE term on center slice: ||x_net_c - y_c||^2
        x_net_c = x_net[:, 1:2, ...]
        y_c = y[:, 1:2, ...]
        mse_term = (x_net_c - y_c).pow(2).mean()
        
        # 2. Divergence term (MC-SURE)
        # We estimate trace of Jacobian only for the central slice mapping: d(out_c)/d(in_c)
        # We start with random probe b
        b = torch.randn_like(y)
        # Zero out perturbation on neighbor slices to isolate center-to-center dependency
        b[:, 0, ...] = 0
        b[:, 2, ...] = 0
        
        eps = 1e-3
        y_pert = y + eps * b
        
        # We need to run model again with same sigma
        # Assuming model call matches: model(input, physics)
        out_pert = model(y_pert, physics)
        
        out_pert_c = out_pert[:, 1:2, ...]
        
        diff = out_pert_c - x_net_c
        b_c = b[:, 1:2, ...]
        
        # Divergence estimate
        # Scale: Sum / (N*eps) because MSE is Mean
        div = (b_c * diff).sum() * (1.0 / eps)
        num_elements = x_net_c.numel()
        
        div_term = 2 * (self.sigma ** 2) * div / num_elements
        
        return mse_term + div_term

    def adapt_model(self, model, **kwargs):
        # Allow trainer to call adapt_model
        return model

# --- Custom Trainer for Plotting ---

class CustomTrainer(dinv.Trainer):
    def plot(self, epoch, physics, x, y, x_net, train=True):
        # We only plot validation comparisons (eval epoch)
        # But Trainer calls plot for both train and eval
        # User asked for "Show ploted images... every 5 epochs".
        # We'll plot if it's an evaluation step and condition met
        
        # Check interval
        if (epoch + 1) % self.plot_interval != 0:
            return
            
        # We can plot training samples too if desired, but validation is safer for "results"
        # Let's plot both or just val? User said "ploted images... every 5 epochs".
        # If we plot train, we see training progress. If val, generalization.
        # Let's plot validation only to match standard practice, or both?
        # Trainer.plot is called with train=True or False.
        if train:
             return # Skip training plots to reduce clutter

        # Check if vectors are present
        if x is None or y is None or x_net is None:
            return

        # Create plots folder
        plot_dir = Path(self.save_path) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot first sample in batch
        # x, y, x_net are batches.
        # (B, 3, H, W). We take 0th element.
        
        # Extract central slices
        clean = x[0, 1].cpu().numpy()
        noisy = y[0, 1].cpu().numpy()
        denoised = x_net[0, 1].detach().cpu().numpy()
        
        # Scale to 0-1 for display
        clean = np.clip(clean, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        denoised = np.clip(denoised, 0, 1)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(clean, cmap='gray'); ax[0].set_title("Clean (Center)")
        ax[1].imshow(noisy, cmap='gray'); ax[1].set_title("Noisy (Center)")
        ax[2].imshow(denoised, cmap='gray'); ax[2].set_title(f"Denoised (Epoch {epoch})")
        for a in ax: a.axis('off')
        
        # Save
        # self.save_path is generic.
        phase = "train" if train else "val"
        plt.savefig(plot_dir / f"epoch_{epoch}_{phase}.png")
        plt.close()

# --- Main Training Function ---

def train(epochs=100, batch_size=4, lr=1e-4, loss_mode='l1', sigma=0.5, data_dir=None):
    # Configuration
    if data_dir is None:
        data_dir = r"d:\Diego trabalho\DICOM" # Default
        
    save_dir = r"d:\Diego trabalho\deepinv-main\checkpoints_25d"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transforms: Resize to 256x256
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    # Datasets with limit=500 and return_stats=False (default)
    # Note: MRI25DDataset needs to be importable
    train_dataset = MRI25DDataset(root_dir=data_dir, mode='train', transform=train_transform, limit=500, return_stats=False)
    val_dataset = MRI25DDataset(root_dir=data_dir, mode='val', transform=test_transform, limit=50, return_stats=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    print(f"Training on {len(train_dataset)} samples.")

    # Physics: Gaussian Noise with fixed sigma
    noise_model = dinv.physics.GaussianNoise(sigma=sigma)
    physics = dinv.physics.Denoising(noise_model)

    # Model: DRUNet 3-channel, NO pretrained weights
    model_base = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained=None, device=device)
    
    # Wrapper to handle (x, physics) -> (x, sigma)
    class DenoiserWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x, physics, **kwargs):
            return self.model(x, physics.noise_model.sigma)
            
    model = DenoiserWrapper(model_base)

    # Losses
    # Always computed on central slice
    l1 = CentralSliceLoss(nn.L1Loss())
    ssim = CentralSliceLoss(SSIMLoss(device))
    mse = CentralSliceLoss(nn.MSELoss())
    
    if loss_mode == 'l1':
        loss_fn = l1
    elif loss_mode == 'weighted':
        # "weighted L1 (0.7) + PSNR (0.2) + SIM (0.1)"
        # Interpreted as: 0.7*L1 + 0.2*MSE + 0.1*(1-SSIM)
        # Assuming PSNR component meant MSE-based optimization
        loss_fn = WeightedSumLoss([l1, mse, ssim], [0.7, 0.2, 0.1])
    elif loss_mode == 'sure':
        loss_fn = CentralSureLoss(sigma=sigma)
    elif loss_mode == 'interleaved':
        # Interleaved L1 + SSIM every 1 epoch
        loss_fn = InterleavedLoss(l1, ssim, interval=1)
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trainer
    trainer = CustomTrainer(
        model=model,
        physics=physics,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        losses=[loss_fn], # Trainer expects list or single loss
        device=device,
        save_path=save_dir,
        verbose=True,
        show_progress_bar=True,
        ckp_interval=5, # Save checkpoint every 5 epochs
        epochs=epochs,
        plot_images=True, # Enable plotting callback
        plot_interval=5,  # Plot every 5 epochs
        online_measurements=True # Generate y from x using physics
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=500, help='Max images per epoch')
    parser.add_argument('--sigma', type=float, default=0.5, help='Noise sigma')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'weighted', 'sure', 'interleaved'], help='Loss function mode')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset')
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, loss_mode=args.loss, sigma=args.sigma, data_dir=args.data_dir)
