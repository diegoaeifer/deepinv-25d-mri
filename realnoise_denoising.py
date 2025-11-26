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
from scipy.signal import convolve2d
from scipy.ndimage import median_filter

def estimate_noise(img):
    """
    Estimate noise standard deviation using Median Absolute Deviation (MAD) 
    of the Laplacian of the image (approximation of high-frequency wavelet subband).
    Assumes img is (C, H, W) or (1, H, W) and normalized to [0, 1].
    """
    try:
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().squeeze()
        else:
            img_np = img
            
        # Laplacian Kernel (detects edges/noise)
        # This kernel sums to 0.
        # [[1, -2, 1], 
        #  [-2, 4, -2], 
        #  [1, -2, 1]]
        # The L2 norm of this kernel is sqrt(1+4+1+4+16+4+1+4+1) = sqrt(36) = 6.
        # However, for noise estimation, we usually use a specific operator.
        # A standard approximation for Gaussian noise sigma is:
        # sigma = MAD(Convolve(I, K)) / C
        # where C is a constant depending on K.
        # For the standard neighbor difference (I[i] - I[i+1]), C = sqrt(2).
        # For the Laplacian K = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], L2 norm is sqrt(20).
        
        # Let's use the standard "Immerkaer" kernel:
        # K = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]] / 6
        # But let's stick to the simplest robust one:
        # Diagonal difference?
        
        # Let's use the Scikit-Image method (if we had it):
        # sigma_est = estimate_sigma(image, average_sigmas=True)
        # It uses wavelet.
        
        # Since we don't have pywt, let's use the Laplacian convolution.
        # K = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        # This is sensitive to edges.
        
        # Better approach for MRI (Rician/Gaussian mixture):
        # Calculate std in the background (air).
        # But we don't know where background is.
        
        # Let's go back to the pseudo-residual but with a scaling factor correction.
        # The previous method: diff = img - median_filter(img, 3)
        # For pure Gaussian noise, std(diff) approx sqrt(1 + 1/9)*sigma? No.
        # It's complex.
        
        # Let's use the Laplacian kernel [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
        # This kernel strongly suppresses smooth regions.
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        
        # Convolve
        H = convolve2d(img_np, kernel, mode='valid')
        
        # MAD of the result
        # sigma_noise = MAD(H) / (0.6745 * sqrt(sum(K^2)))
        # sum(K^2) = 1+4+1+4+16+4+1+4+1 = 36. sqrt(36) = 6.
        # So sigma = MAD(H) / (0.6745 * 6) = MAD(H) / 4.047
        
        mad = np.median(np.abs(H - np.median(H)))
        sigma = mad / 4.047
        
        print(f"  -> Estimated Sigma: {sigma:.5f}")
        return float(sigma)
        
    except Exception as e:
        print(f"Estimation failed: {e}, using default sigma=0.01")
        return 0.01

def evaluate_real_noise(fixed_sigma=None):
    # Configuration
    data_dir = r"d:\Diego trabalho\DICOM"
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Dataset
    # Using original size (resize_to=None) and random sampling
    dataset = MRIDataset.random_sample(root_dir=data_dir, count=10, train='all', resize_to=None) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

    results = []
    vis_images = [] 

    print(f"Starting real-noise denoising on {len(dataset)} images...")
    if fixed_sigma is not None:
        print(f"Mode: Denoising ORIGINAL images with FIXED sigma={fixed_sigma}.")
    else:
        print("Mode: Denoising ORIGINAL images (estimating natural noise).")

    # Evaluation Loop
    for i, x in enumerate(tqdm(dataloader)):
        x = x.to(device)
        
        if x.max() == 0:
            continue

        # Treat input x as the noisy image y
        y = x
        
        # Estimate sigma for this image
        if fixed_sigma is not None:
            estimated_sigma = fixed_sigma
        else:
            estimated_sigma = estimate_noise(y)
        
        row = {
            'Image_Idx': i,
            'Estimated_Sigma': estimated_sigma
        }

        # Store visualization data
        if i < 5:
            vis_data = {
                'original': y.cpu().squeeze().numpy(),
                'denoised': {}
            }
        
        for model_name, model in models.items():
            start_time = time.time()
            with torch.no_grad():
                try:
                    # Models that need sigma get the estimated one
                    if "DRUNet" in model_name:
                        x_hat = model(y, estimated_sigma)
                    elif model_name == "RAM":
                        x_hat = model(y, sigma=estimated_sigma)
                    else:
                        # Restormer is blind
                        x_hat = model(y)
                        if model_name == "Restormer":
                            print(f"Restormer - Input: {y.shape}, Range: [{y.min():.4f}, {y.max():.4f}]")
                            print(f"Restormer - Output: {x_hat.shape}, Range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
                        
                    # Clamp output to [0, 1]
                    x_hat = torch.clamp(x_hat, 0, 1)
                        
                    inference_time = time.time() - start_time
                    
                    # We can't compute PSNR/SSIM vs Ground Truth because we don't have it.
                    # We can compute PSNR vs Input to see how much it changed?
                    # Or just log time.
                    row[f'Time_{model_name}'] = inference_time
                    
                    if i < 5:
                        vis_data['denoised'][model_name] = x_hat.cpu().squeeze().numpy()
                        
                except Exception as e:
                    print(f"Error running {model_name} on image {i}: {e}")
                    row[f'Time_{model_name}'] = None

        if i < 5:
            vis_images.append(vis_data)

        results.append(row)
        
    # Final Save
    df = pd.DataFrame(results)
    df.to_csv('realnoise_results.csv', index=False)
    
    # Summary
    print("\nProcessing Complete. Summary:")
    print(df.mean(numeric_only=True))
    
    # Plotting
    if vis_images:
        print("Generating comparison plot...")
        num_images = len(vis_images)
        num_models = len(models)
        num_cols = 1 + num_models # Original + Models
        
        fig, axes = plt.subplots(num_images, num_cols, figsize=(4 * num_cols, 4 * num_images))
        
        # If only 1 image, axes is 1D array
        if num_images == 1:
            axes = axes.reshape(1, -1)
            
        model_names = list(models.keys())
        
        for idx, img_data in enumerate(vis_images):
            # Original
            ax = axes[idx, 0]
            ax.imshow(img_data['original'], cmap='gray')
            if idx == 0: 
                ax.set_title(f"Original (Sigma={results[idx]['Estimated_Sigma']:.3f})")
            ax.axis('off')
            
            # Denoised
            for m_idx, m_name in enumerate(model_names):
                ax = axes[idx, 1 + m_idx]
                if m_name in img_data['denoised']:
                    ax.imshow(img_data['denoised'][m_name], cmap='gray')
                    if idx == 0: ax.set_title(m_name)
                ax.axis('off')
                
        plt.tight_layout()
        plt.savefig("realnoise_comparison.png")
        print("Plot saved to realnoise_comparison.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=None, help='Fixed sigma value')
    args = parser.parse_args()
    evaluate_real_noise(fixed_sigma=args.sigma)
