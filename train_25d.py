import torch
from torch.utils.data import DataLoader
import deepinv as dinv
from mri_25d_dataset import MRI25DDataset
from pathlib import Path
import argparse
from torchvision import transforms

def train(epochs=100, batch_size=4, lr=1e-4):
    # Configuration
    data_dir = r"d:\Diego trabalho\DICOM"
    save_dir = r"d:\Diego trabalho\deepinv-main\checkpoints_25d"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transforms (Applied to all 3 channels identically)
    # RandomCrop(512) is good.
    train_transform = transforms.Compose([
        transforms.RandomCrop(512, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    test_transform = transforms.Compose([
        transforms.CenterCrop(512)
    ])

    # Datasets
    # Using limit=1000 for training as per previous optimization
    train_dataset = MRI25DDataset(root_dir=data_dir, mode='train', transform=train_transform, limit=1000)
    val_dataset = MRI25DDataset(root_dir=data_dir, mode='val', transform=test_transform, limit=200)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")

    # Physics: Gaussian Noise
    # We simulate noise on the "clean" input
    sigma_min = 0.05
    sigma_max = 0.3
    
    class VariableGaussianNoise(dinv.physics.GaussianNoise):
        def __init__(self, sigma_min, sigma_max, **kwargs):
            super().__init__(sigma=sigma_max, **kwargs)
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
        
        def forward(self, x):
            sigma = torch.rand(1, device=x.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
            self.update_parameters(sigma=sigma.item()) 
            return super().forward(x)

    noise_model = VariableGaussianNoise(sigma_min=sigma_min, sigma_max=sigma_max)
    physics = dinv.physics.Denoising(noise_model)

    # Model: DRUNet 3-channel
    # User requested: "load the pretrained 3 channels image provided with dinv"
    # This is the default 'download' weights for in_channels=3, out_channels=3 (color)
    model = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained='download', device=device)

    # Loss: L1 (MAE)
    loss_fn = torch.nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    # We can't use dinv.Trainer easily because we want to compute loss on 3 channels?
    # Actually dinv.Trainer works with SupLoss.
    # But let's write a simple loop to be sure we handle the 2.5D logic (though it's just 3-channel images now).
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (x, _, _) in enumerate(train_dataloader):
            x = x.to(device) # (B, 3, H, W)
            
            # Add noise
            y = physics(x)
            sigma = physics.noise_model.sigma
            
            # Forward
            # DRUNet takes sigma
            x_hat = model(y, sigma)
            
            # Loss
            loss = loss_fn(x_hat, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader):.6f}")
        
        # Validation
        if (epoch+1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, _, _ in val_dataloader:
                    x = x.to(device)
                    y = physics(x)
                    sigma = physics.noise_model.sigma
                    x_hat = model(y, sigma)
                    val_loss += loss_fn(x_hat, x).item()
            print(f"Validation Loss: {val_loss/len(val_dataloader):.6f}")
            
            # Save checkpoint
            torch.save(model.state_dict(), f"{save_dir}/drunet_25d_epoch{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    train(epochs=args.epochs)
