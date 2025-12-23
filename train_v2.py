import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pydicom
import deepinv as dinv
from pathlib import Path

# --- Dataset ---
class DicomDataset(Dataset):
    def __init__(self, root_dir, limit=None, seed=42, transform=None):
        """
        Args:
            root_dir (str): Path to folder with DICOMs.
            limit (int): Max number of images to use.
            seed (int): Seed for shuffling if limit is used.
            transform (callable): Optional transform.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Recursive search
        print(f"Scanning {root_dir} for DICOM files...")
        self.files = sorted(glob.glob(os.path.join(root_dir, "**/*.dcm"), recursive=True))
        
        if len(self.files) == 0:
            # Try no extension or other patterns if needed, but assuming .dcm
            pass
            
        print(f"Found {len(self.files)} files.")
        
        # Shuffle and limit
        if limit is not None and len(self.files) > limit:
            random.seed(seed)
            random.shuffle(self.files)
            self.files = self.files[:limit]
            print(f"Randomly selected {len(self.files)} files (Seed: {seed}).")
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            ds = pydicom.dcmread(path)
            # Handle standard DICOM pixel data
            img = ds.pixel_array.astype(np.float32)
            
            # Normalize to [0, 1]
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = np.zeros_like(img) # Flat image
            
            # Add channel dim: (H, W) -> (C, H, W) where C=1
            img = torch.from_numpy(img).unsqueeze(0)
            
            if self.transform:
                img = self.transform(img)
                
            return img
            
        except Exception as e:
            print(f"Error reading {path}: {e}")
            # Return dummy or next? safer to raise or return specific error
            # For simplicity, returning zeros (trainer handles batching)
            return torch.zeros((1, 256, 256))

# --- Custom Random Rot 90 ---
class RandomRot90:
    """Randomly rotates by 0, 90, 180, 270 degrees."""
    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, dims=[-2, -1])

# --- Model Factory ---
def get_model(model_name, device):
    # Common args
    # Pretrained: generic 'download' if supported, else None.
    # Note: DeepInv models differ in init signature.
    
    if model_name == 'drunet':
        return dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    elif model_name == 'gsdrunet':
        return dinv.models.GSDRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    elif model_name == 'scunet':
        return dinv.models.SCUNet(in_channels=1, pretrained='download', device=device)
    elif model_name == 'restormer':
        return dinv.models.Restormer(in_channels=1, out_channels=1, pretrained='download', device=device)
    elif model_name == 'ram':
        # Check RAM signature. Usually accepts in_channels.
        return dinv.models.RAM(in_channels=1, out_channels=1, pretrained='download', device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# --- Main Training Function ---
def train_v2(args):
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(256, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRot90() 
    ])
    
    # Datasets
    # Split 80/10/10
    full_dataset = DicomDataset(args.data_dir, limit=args.limit, seed=42, transform=train_transform)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.workers)
    
    # Physics
    # Rician Noise Sigma 0.1
    # Note: Rician is non-additive. DeepInv usually models physics: y = Physics(x).
    # Noise model is part of physics or separate? 
    # dinv.physics.Denoising accepts a noise_model.
    # dinv.physics.RicianNoise is a NoiseModel.
    rice_noise = dinv.physics.RicianNoise(sigma=args.sigma) # assumes scaled to [0,1]
    physics = dinv.physics.Denoising(rice_noise)
    
    # Model
    model = get_model(args.model, device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    if args.loss == 'l1':
        print("Using L1 Loss")
        loss_fn = nn.L1Loss()
    elif args.loss == 'l2':
        print("Using MSE Loss")
        loss_fn = nn.MSELoss()
    elif args.loss == 'sure':
        print(f"Using SURE Loss (Gaussian approximation for Sigma={args.sigma})")
        loss_fn = dinv.loss.SureGaussianLoss(sigma=args.sigma) 
    elif args.loss == 'unsure':
        print(f"Using UNSURE Loss (Learnable Sigma, init={args.sigma})")
        loss_fn = dinv.loss.SureGaussianLoss(sigma=args.sigma, unsure=True) 
    else:
        loss_fn = nn.L1Loss()
        
    # Trainer
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=[loss_fn],
        epochs=args.epochs,
        plot_images=args.plot_images,
        ckp_interval=args.ckp_interval,
        save_path='checkpoints_v2',
        device=device,
        online_measurements=True # Important: Generate y on fly
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to DICOM folder')
    parser.add_argument('--model', type=str, default='drunet', choices=['drunet', 'gsdrunet', 'scunet', 'restormer', 'ram'])
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2', 'sure', 'unsure'])
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ckp_interval', type=int, default=10)
    parser.add_argument('--plot_images', action='store_true', default=True)
    
    args = parser.parse_args()
    train_v2(args)
