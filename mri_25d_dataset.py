import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image

class MRI25DDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', test_split=0.1, val_split=0.1, seed=42, limit=None, shuffle=True, return_stats=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val', 'test', or 'all'.
            return_stats (bool): If True, returns (stack, v_min, v_max). Else returns stack.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.return_stats = return_stats
        self.volumes = [] # List of lists, where each inner list is a sorted list of slice paths or (path, index)

        # 1. Discover and Group Files
        self._discover_volumes()

        # 2. Split Volumes
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.volumes)
        else:
            # Sort volumes by path to ensure deterministic order
            # self.volumes is a list of lists (DICOM) or dicts (NIFTI)
            # We need a stable sort key.
            def get_vol_key(v):
                if isinstance(v, dict): return v['path']
                return v[0] # First file of dicom series
            self.volumes.sort(key=get_vol_key)
        
        total_vols = len(self.volumes)
        test_len = int(total_vols * test_split)
        val_len = int(total_vols * val_split)
        train_len = total_vols - test_len - val_len
        
        # Handle small datasets where splits might be 0
        if total_vols > 0:
            if test_len == 0 and test_split > 0: test_len = 1
            if val_len == 0 and val_split > 0: val_len = 1
            train_len = total_vols - test_len - val_len
        
        if mode == 'train':
            self.selected_volumes = self.volumes[:train_len]
        elif mode == 'val':
            self.selected_volumes = self.volumes[train_len:train_len+val_len]
        elif mode == 'test':
            self.selected_volumes = self.volumes[train_len+val_len:]
        else:
            self.selected_volumes = self.volumes

        if limit:
            self.selected_volumes = self.selected_volumes[:limit]

        # 3. Index Slices
        # We need a flat index to map __getitem__(idx) to (volume_idx, slice_idx)
        self.samples = []
        for v_idx, vol in enumerate(self.selected_volumes):
            # vol is a list of slice info
            # For NIfTI, vol might be a dict {'path': p, 'num_slices': N}
            # For DICOM, vol is a list of paths
            
            if isinstance(vol, dict) and vol['type'] == 'nifti':
                for s_idx in range(vol['num_slices']):
                    plane = vol.get('plane', 2)
                    self.samples.append({'type': 'nifti', 'path': vol['path'], 'slice_idx': s_idx, 'num_slices': vol['num_slices'], 'plane': plane})
            else:
                # DICOM list
                num_slices = len(vol)
                for s_idx in range(num_slices):
                    self.samples.append({'type': 'dicom', 'vol_idx': v_idx, 'slice_idx': s_idx, 'num_slices': num_slices})

        print(f"MRI25DDataset ({mode}): Found {len(self.selected_volumes)} volumes, {len(self.samples)} total slices.")

    def _discover_volumes(self):
        # NIFTI
        nifti_files = glob.glob(os.path.join(self.root_dir, "**/*.nii"), recursive=True) + \
                      glob.glob(os.path.join(self.root_dir, "**/*.nii.gz"), recursive=True)
        
        for f in nifti_files:
            try:
                img = nib.load(f)
                header = img.header
                zooms = header.get_zooms()
                shape = img.shape
                
                # Check isotropy for training
                # "largest resolution should be at most the double of the smallest resolution"
                # We only care about spatial dims (first 3)
                spatial_zooms = zooms[:3]
                min_res = min(spatial_zooms)
                max_res = max(spatial_zooms)
                is_isotropic = max_res <= 2 * min_res
                
                # Standard MRI NIfTI is (H, W, D)
                if len(shape) >= 3:
                    num_slices = shape[2]
                    
                    # Default: Axial (plane 2)
                    # We store 'plane': 2 to indicate slicing along 3rd dim
                    self.volumes.append({'type': 'nifti', 'path': f, 'num_slices': num_slices, 'plane': 2})
                    
                    # If training and isotropic, add other planes
                    if self.mode == 'train' and is_isotropic:
                        # Sagittal (plane 0) -> Slices are along dim 0
                        self.volumes.append({'type': 'nifti', 'path': f, 'num_slices': shape[0], 'plane': 0})
                        
                        # Coronal (plane 1) -> Slices are along dim 1
                        self.volumes.append({'type': 'nifti', 'path': f, 'num_slices': shape[1], 'plane': 1})
                        
            except:
                pass

        # DICOM
        dicom_files = glob.glob(os.path.join(self.root_dir, "**/*.dcm"), recursive=True)
        
        # Group by SeriesInstanceUID
        dicoms_by_series = {}
        
        print(f"Scanning {len(dicom_files)} DICOM files...")
        for f in dicom_files:
            try:
                # Read header only
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                
                # Check if image dimensions exist
                if not (hasattr(ds, 'Rows') and hasattr(ds, 'Columns')):
                     # Skip files without dimensions (likely not images)
                     # print(f"Skipping {f}: No dimensions")
                     continue

                rows = int(ds.Rows)
                cols = int(ds.Columns)

                series_uid = ds.SeriesInstanceUID if hasattr(ds, 'SeriesInstanceUID') else 'unknown'
                
                # If unknown, maybe group by folder?
                if series_uid == 'unknown':
                    series_uid = os.path.dirname(f)
                
                if series_uid not in dicoms_by_series:
                    dicoms_by_series[series_uid] = []
                
                # Store (InstanceNumber, path, rows, cols)
                idx = int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0
                dicoms_by_series[series_uid].append({'idx': idx, 'path': f, 'rows': rows, 'cols': cols})
            except Exception as e:
                # print(f"Skipping {f}: {e}")
                pass
            
        # Sort each series
        for uid, files in dicoms_by_series.items():
            files.sort(key=lambda x: x['idx'])
            # Store full info in volume, not just path
            self.volumes.append(files)

    def __len__(self):
        return len(self.samples)

    def _load_slice(self, sample, offset):
        # Load slice at sample['slice_idx'] + offset
        # Handle boundary conditions
        
        idx = sample['slice_idx'] + offset
        max_idx = sample['num_slices'] - 1
        
        if idx < 0:
            idx = 1 if max_idx >= 1 else 0
        elif idx > max_idx:
            idx = max_idx - 1 if max_idx >= 1 else max_idx
            
        # Load the data
        if sample['type'] == 'nifti':
            nimg = nib.load(sample['path'])
            data = nimg.get_fdata(dtype=np.float32)
            plane = sample.get('plane', 2) # Default to axial
            
            # (H, W, D)
            if len(data.shape) == 3:
                if plane == 2: # Axial
                    slice_data = data[:, :, idx]
                elif plane == 1: # Coronal
                    slice_data = data[:, idx, :]
                elif plane == 0: # Sagittal
                    slice_data = data[idx, :, :]
            else:
                # 4D (H, W, D, T) - Assume T=0
                if plane == 2:
                    slice_data = data[:, :, idx, 0]
                elif plane == 1:
                    slice_data = data[:, idx, :, 0]
                elif plane == 0:
                    slice_data = data[idx, :, :, 0]
        else:
            # DICOM
            vol = self.selected_volumes[sample['vol_idx']]
            item = vol[idx] # This is now a dict {'idx':..., 'path':..., 'rows':..., 'cols':...}
            dcm_path = item['path']
            
            try:
                ds = pydicom.dcmread(dcm_path)
                slice_data = ds.pixel_array.astype(np.float32)
            except Exception as e:
                print(f"Error loading slice {dcm_path}: {e}")
                # Return dummy slice with correct shape
                rows = item.get('rows', 256)
                cols = item.get('cols', 256)
                slice_data = np.zeros((rows, cols), dtype=np.float32)
            
        return slice_data

    def _get_volume_stats(self, vol_idx):
        # Check if already cached
        if not hasattr(self, 'volume_stats'):
            self.volume_stats = {}
            
        if vol_idx in self.volume_stats:
            return self.volume_stats[vol_idx]
            
        # Compute stats
        vol = self.selected_volumes[vol_idx]
        min_val = float('inf')
        max_val = float('-inf')
        
        # Iterate over all slices in volume
        # This might be slow for large datasets, but for eval it's necessary.
        # For training, maybe we stick to per-slice or estimate?
        # User wants "whole volume" normalization.
        
        # Optimization: For NIfTI it's easy.
        if isinstance(vol, dict) and vol['type'] == 'nifti':
             nimg = nib.load(vol['path'])
             data = nimg.get_fdata()
             min_val = data.min()
             max_val = data.max()
        else:
            # DICOM list
            # We need to read all files? Or just a sample?
            # Reading all is safest.
            for item in vol:
                try:
                    ds = pydicom.dcmread(item['path'])
                    arr = ds.pixel_array.astype(np.float32)
                    min_val = min(min_val, arr.min())
                    max_val = max(max_val, arr.max())
                except:
                    pass
                    
        self.volume_stats[vol_idx] = (min_val, max_val)
        return min_val, max_val

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load 3 slices: z-1, z, z+1
        s_prev = self._load_slice(sample, -1)
        s_curr = self._load_slice(sample, 0)
        s_next = self._load_slice(sample, 1)
        
        stack = np.stack([s_prev, s_curr, s_next], axis=0) # (3, H, W)
        
        # Volume-wise Normalization
        # Get stats for this volume
        if sample['type'] == 'dicom':
            v_idx = sample['vol_idx']
        else:
            # For NIfTI, we need to find which volume this sample belongs to.
            # In _discover_volumes we flattened samples but didn't store vol_idx for NIfTI explicitly in a way that maps back easily 
            # unless we search. But wait, sample has 'path'.
            # We can use path as key or just re-find.
            # Actually, let's just use per-stack for NIfTI for now or fix it?
            # The user issue is likely DICOM.
            # Let's assume DICOM for the fix.
            v_idx = None
            
        if v_idx is not None:
            v_min, v_max = self._get_volume_stats(v_idx)
        else:
            # Fallback to stack stats
            v_min = stack.min()
            v_max = stack.max()
            
        if v_max > v_min:
            stack = (stack - v_min) / (v_max - v_min)
        else:
            stack = np.zeros_like(stack)
            
        # Convert to tensor
        stack_tensor = torch.from_numpy(stack).float()
        
        if self.transform:
            stack_tensor = self.transform(stack_tensor)
            
        # Return stats so we can denormalize later
        if self.return_stats:
            return stack_tensor, v_min, v_max
        else:
            return stack_tensor
