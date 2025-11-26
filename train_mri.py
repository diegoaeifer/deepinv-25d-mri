import torch
from torch.utils.data import DataLoader
import deepinv as dinv
from mri_dataset import MRIDataset
from pathlib import Path
import argparse

from torchvision import transforms

def train(model_name='DRUNet', epochs=100):
    # Configuration
    data_dir = r"d:\Diego trabalho\DICOM"
    save_dir = r"d:\Diego trabalho\deepinv-main\checkpoints"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Variable Noise Range
    sigma_min = 0.05
    sigma_max = 0.3
    
    batch_size = 2 # Reduced batch size for heavier models
    lr = 1e-4
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for model: {model_name}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(512, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    test_transform = transforms.Compose([
        transforms.CenterCrop(512)
    ])

    # Datasets
    # Limit training to 1000 images per epoch (effectively reducing dataset size)
    # Using random_sample to get a random subset of 1000 images if we wanted, 
    # but here we use limit=1000 which takes the first 1000 after shuffle (seed=42).
    # To get different images each run, we could use seed=None.
    # But for reproducible training, fixed seed is better.
    train_dataset = MRIDataset(root_dir=data_dir, train='train', test_split=0.1, val_split=0.1, transform=train_transform, limit=1000)
    val_dataset = MRIDataset(root_dir=data_dir, train='val', test_split=0.1, val_split=0.1, transform=test_transform, limit=200) # Also limit val for speed
    test_dataset = MRIDataset(root_dir=data_dir, train='test', test_split=0.1, val_split=0.1, transform=test_transform, limit=200)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images")

    # Physics: Gaussian Noise with variable sigma
    class VariableGaussianNoise(dinv.physics.GaussianNoise):
        def __init__(self, sigma_min, sigma_max, **kwargs):
            super().__init__(sigma=sigma_max, **kwargs) # Init with max
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
        
        def forward(self, x):
            # Sample sigma uniformly
            sigma = torch.rand(1, device=x.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
            self.update_parameters(sigma=sigma.item()) 
            return super().forward(x)
            
        def update(self, **kwargs):
            return self.update_parameters(**kwargs)

    noise_model = VariableGaussianNoise(sigma_min=sigma_min, sigma_max=sigma_max)
    physics = dinv.physics.Denoising(noise_model)

    # Wrapper to handle Trainer passing 'physics' instead of 'sigma'
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, model_name):
            super().__init__()
            self.model = model
            self.model_name = model_name
        
        def forward(self, y, physics, **kwargs):
            # Extract sigma from physics
            if hasattr(physics, 'noise_model') and hasattr(physics.noise_model, 'sigma'):
                sigma = physics.noise_model.sigma
            elif hasattr(physics, 'sigma'):
                sigma = physics.sigma
            else:
                sigma = 0.1
            
            # Forward
            if self.model_name == 'RAM':
                out = self.model(y, sigma=sigma)
            else: # DRUNet, Restormer
                out = self.model(y, sigma)
                
            return out

    # Model Selection
    if model_name == 'DRUNet':
        model = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    elif model_name == 'Restormer':
        model = dinv.models.Restormer(in_channels=1, out_channels=1, pretrained='denoising_gray', device=device)
    elif model_name == 'RAM':
        model = dinv.models.RAM(pretrained=True, device=device)
    else:
        raise ValueError(f"Unknown model {model_name}")
    
    # Wrap model
    model = ModelWrapper(model, model_name)


    # Loss: SURE with Unsure=True
    loss_fn = dinv.loss.SureGaussianLoss(sigma=(sigma_min+sigma_max)/2, unsure=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trainer
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        losses=[loss_fn],
        device=device,
        save_path=save_dir,
        epochs=epochs,
        verbose=True,
        show_progress_bar=True,
        plot_images=False,
        ckp_interval=5,
        online_measurements=True
    )

    # Train
    trainer.train()

    # Test
    print("Training complete. Testing on test set...")
    trainer.test(test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DRUNet', choices=['DRUNet', 'Restormer', 'RAM'], help='Model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    
    train(model_name=args.model, epochs=args.epochs)
