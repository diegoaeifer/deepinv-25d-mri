import torch
from torch.utils.data import DataLoader
import deepinv as dinv
from deepinv.models import DRUNet, Restormer, RAM, GSDRUNet
from deepinv.loss.metric import SSIM, MAE
from mri_dataset import MRIDataset
import pandas as pd
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

def evaluate():
    # Configuration
    data_dir = r"d:\Diego trabalho\DICOM"
    sigma = 0.1 # Noise level
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Dataset
    dataset = MRIDataset(root_dir=data_dir, train='all', limit=10) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define Physics (Gaussian Noise)
    physics = dinv.physics.GaussianNoise(sigma=sigma)

    # Define Models
    models = {}
    
    # DRUNet
    try:
        print("Loading DRUNet (Pretrained)...")
        models['DRUNet'] = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    except Exception as e:
        print(f"Failed to load DRUNet: {e}")

    # GSDRUNet
    try:
        print("Loading GSDRUNet...")
        models['GSDRUNet'] = dinv.models.GSDRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    except Exception as e:
        print(f"Failed to load GSDRUNet: {e}")

    # Restormer (Native 1 channel support)
    try:
        print("Loading Restormer...")
        models['Restormer'] = dinv.models.Restormer(in_channels=1, out_channels=1, pretrained='denoising_gray', device=device)
    except Exception as e:
        print(f"Failed to load Restormer: {e}")

    # RAM (Native 1 channel support with default args)
    try:
        print("Loading RAM...")
        models['RAM'] = dinv.models.RAM(pretrained=True, device=device)
    except Exception as e:
        print(f"Failed to load RAM: {e}")

    if not models:
        print("No models loaded. Exiting.")
        return

    # Metrics
    ssim_metric = SSIM()
    mae_metric = MAE()

    results = []
    vis_images = [] # List to store data for plotting: [{'clean': ..., 'noisy': ..., 'denoised': {model: ...}}]

    print(f"Starting evaluation on {len(dataset)} images...")

    # Evaluation Loop
    for i, x in enumerate(tqdm(dataloader)):
        x = x.to(device)
        
        # If image is all zeros (error in loading), skip
        if x.max() == 0:
            continue

        # Generate noisy measurement
        y = physics(x)

        row = {'Image_Idx': i}
        
        # Compute initial metrics
        psnr_noisy = dinv.metric.PSNR()(y, x).item()
        ssim_noisy = ssim_metric(y, x).item()
        mae_noisy = mae_metric(y, x).item()
        
        row['PSNR_Noisy'] = psnr_noisy
        row['SSIM_Noisy'] = ssim_noisy
        row['MAE_Noisy'] = mae_noisy

        # Store visualization data for first 5 images
        if i < 5:
            vis_data = {
                'clean': x.cpu().squeeze().numpy(),
                'noisy': y.cpu().squeeze().numpy(),
                'denoised': {}
            }
        
        for model_name, model in models.items():
            start_time = time.time()
            with torch.no_grad():
                try:
                    # DRUNet requires sigma. 
                    # RAM requires sigma (if used as denoiser).
                    # GSDRUNet requires sigma.
                    # Restormer is blind (ignore sigma).
                    
                    if "DRUNet" in model_name:
                        x_hat = model(y, sigma)
                    elif model_name == "RAM":
                        x_hat = model(y, sigma=sigma)
                    else:
                        x_hat = model(y)
                        
                    # Metrics
                    psnr = dinv.metric.PSNR()(x_hat, x).item()
                    ssim = ssim_metric(x_hat, x).item()
                    mae = mae_metric(x_hat, x).item()
                    inference_time = time.time() - start_time
                    
                    row[f'PSNR_{model_name}'] = psnr
                    row[f'SSIM_{model_name}'] = ssim
                    row[f'MAE_{model_name}'] = mae
                    row[f'Time_{model_name}'] = inference_time
                    
                    if i < 5:
                        vis_data['denoised'][model_name] = x_hat.cpu().squeeze().numpy()
                        
                except Exception as e:
                    print(f"Error running {model_name} on image {i}: {e}")
                    row[f'PSNR_{model_name}'] = None
                    row[f'SSIM_{model_name}'] = None
                    row[f'MAE_{model_name}'] = None
                    row[f'Time_{model_name}'] = None

        if i < 5:
            vis_images.append(vis_data)

        results.append(row)
        
        # Save intermediate results every 10 images
        if i % 10 == 0:
            df = pd.DataFrame(results)
            df.to_csv('denoising_results.csv', index=False)

    # Final Save
    df = pd.DataFrame(results)
    df.to_csv('denoising_results.csv', index=False)
    
    # Summary
    print("\nEvaluation Complete. Summary:")
    print(df.mean(numeric_only=True))
    
    # Plotting
    if vis_images:
        print("Generating comparison plot...")
        num_images = len(vis_images)
        num_models = len(models)
        num_cols = 2 + num_models # Clean, Noisy, + Models
        
        fig, axes = plt.subplots(num_images, num_cols, figsize=(4 * num_cols, 4 * num_images))
        
        # If only 1 image, axes is 1D array
        if num_images == 1:
            axes = axes.reshape(1, -1)
            
        model_names = list(models.keys())
        
        for idx, img_data in enumerate(vis_images):
            # Clean
            ax = axes[idx, 0]
            ax.imshow(img_data['clean'], cmap='gray')
            if idx == 0: ax.set_title("Clean")
            ax.axis('off')
            
            # Noisy
            ax = axes[idx, 1]
            ax.imshow(img_data['noisy'], cmap='gray')
            if idx == 0: ax.set_title(f"Noisy (sigma={sigma})")
            ax.axis('off')
            
            # Denoised
            for m_idx, m_name in enumerate(model_names):
                ax = axes[idx, 2 + m_idx]
                if m_name in img_data['denoised']:
                    ax.imshow(img_data['denoised'][m_name], cmap='gray')
                    if idx == 0: ax.set_title(m_name)
                ax.axis('off')
                
        plt.tight_layout()
        plt.savefig("denoising_comparison.png")
        print("Plot saved to denoising_comparison.png")

if __name__ == "__main__":
    evaluate()
