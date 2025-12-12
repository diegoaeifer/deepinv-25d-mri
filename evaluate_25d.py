import torch
from torch.utils.data import DataLoader
import deepinv as dinv
from mri_25d_dataset import MRI25DDataset
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pydicom
import os

def evaluate(data_dir, sigma, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset (No shuffle, all data)
    dataset = MRI25DDataset(root_dir=data_dir, mode='all', shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    print(f"Evaluating on {len(dataset)} slices from {len(dataset.volumes)} volumes.")

    # Model
    model = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained='download', device=device)
    model.eval()

    print(f"Denoising with fixed sigma = {sigma}")

    with torch.no_grad():
        for i, (x, v_min, v_max) in enumerate(tqdm(dataloader)):
            x = x.to(device) # (1, 3, H, W)
            v_min = v_min.item()
            v_max = v_max.item()
            
            # Forward
            # Pass sigma as a tensor or float? DRUNet handles float.
            x_hat = model(x, sigma)
            
            # Extract center slice
            # Input was [Prev, Curr, Next]
            # Output is [Denoised_Prev, Denoised_Curr, Denoised_Next]
            # We want Denoised_Curr (Index 1)
            
            denoised_slice = x_hat[0, 1, :, :].cpu().numpy()
            original_slice = x[0, 1, :, :].cpu().numpy()
            
            # Clamp
            denoised_slice = np.clip(denoised_slice, 0, 1)
            
            # Save
            # Retrieve metadata to determine filename and type
            sample_info = dataset.samples[i]
            
            if sample_info['type'] == 'dicom':
                vol = dataset.volumes[sample_info['vol_idx']]
                item = vol[sample_info['slice_idx']]
                path = item['path']
                original_filename = os.path.basename(path)
                # Remove extension to append suffix
                base_name = os.path.splitext(original_filename)[0]
                new_filename = f"{base_name}_denoised_{sigma}.dcm"
                
                try:
                    ds = pydicom.dcmread(path)
                    # original_arr = ds.pixel_array.astype(np.float32)
                    
                    # Rescale denoised to original range using VOLUME stats
                    if v_max > v_min:
                        denoised_scaled = denoised_slice * (v_max - v_min) + v_min
                    else:
                        denoised_scaled = denoised_slice # Should be constant if min==max
                        
                    # Handle integer types
                    # Check PixelRepresentation safely
                    pixel_rep = ds.get('PixelRepresentation', 0) # Default to 0 (unsigned)
                    
                    if pixel_rep == 0: # Unsigned
                        # Clip to valid range for safety (e.g. uint16)
                        bits_stored = ds.get('BitsStored', 16)
                        max_val = (1 << bits_stored) - 1
                        denoised_scaled = np.clip(denoised_scaled, 0, max_val)
                        
                        # Determine dtype based on bits
                        if bits_stored <= 8:
                            dtype = np.uint8
                        elif bits_stored <= 16:
                            dtype = np.uint16
                        else:
                            dtype = np.uint32 # Rare for MRI
                            
                        denoised_final = denoised_scaled.astype(dtype)
                    else: # Signed
                        # For signed 16-bit, range is -32768 to 32767
                        # We assume the data fits.
                        denoised_final = denoised_scaled.astype(np.int16)
                        
                    ds.PixelData = denoised_final.tobytes()
                    ds.save_as(os.path.join(output_dir, new_filename))
                    
                except Exception as e:
                    print(f"Error saving DICOM {original_filename}: {e}")
                    # Fallback to PNG
                    plt.imsave(f"{output_dir}/{base_name}_denoised_{sigma}.png", denoised_slice, cmap='gray')

            else:
                # NIfTI or unknown
                # Just save as PNG for now
                plt.imsave(f"{output_dir}/slice_{i:04d}_denoised.png", denoised_slice, cmap='gray')
            
            # Optional: Save comparison (PNG)
            if i < 5: # Save first 5 comparisons
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(original_slice, cmap='gray')
                ax[0].set_title("Original (Center)")
                ax[1].imshow(denoised_slice, cmap='gray')
                ax[1].set_title(f"Denoised (Sigma={sigma})")
                plt.savefig(f"{output_dir}/comparison_{i:04d}.png")
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with DICOMs/NIfTIs')
    parser.add_argument('--sigma', type=float, default=0.05, help='Noise level sigma')
    parser.add_argument('--output_dir', type=str, default='results_25d', help='Output directory')
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.sigma, args.output_dir)
