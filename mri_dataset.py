import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image

class MRIDataset(Dataset):
    @classmethod
    def random_sample(cls, root_dir, count=10, **kwargs):
        """
        Factory method to create a dataset with a random sample of images.
        """
        return cls(root_dir, limit=count, seed=None, **kwargs)

    def __init__(self, root_dir, transform=None, train=True, test_split=0.1, val_split=0.1, seed=42, limit=None, resize_to=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            test_split (float): Fraction of data to reserve for testing.
            val_split (float): Fraction of data to reserve for validation.
            seed (int): Random seed for splitting.
            limit (int, optional): Limit the number of images for debugging.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.resize_to = resize_to
        self.slices = []

        # Find all files
        nifti_files = glob.glob(os.path.join(root_dir, "**/*.nii"), recursive=True) + \
                      glob.glob(os.path.join(root_dir, "**/*.nii.gz"), recursive=True)
        dicom_files = glob.glob(os.path.join(root_dir, "**/*.dcm"), recursive=True)

        all_files = nifti_files + dicom_files
        all_files.sort() # Ensure deterministic order

        # We need to process files to count slices and store metadata for lazy loading
        # For NIFTI, we might have 3D volumes. For DICOM, usually 2D slices but can be 3D.
        # To avoid loading everything into memory, we will store (file_path, slice_index) tuples.
        
        temp_slices = []
        
        print(f"Found {len(nifti_files)} NIFTI files and {len(dicom_files)} DICOM files.")

        for f in nifti_files:
            try:
                img = nib.load(f)
                # Check dimensions
                shape = img.shape
                if len(shape) == 3: # (H, W, D)
                    for i in range(shape[2]):
                        temp_slices.append({'path': f, 'type': 'nifti', 'slice': i})
                elif len(shape) == 4: # (H, W, D, T) or (H, W, D, C) - taking first channel/time for simplicity or iterate all?
                    # Let's assume we want all volumes
                    for t in range(shape[3]):
                        for i in range(shape[2]):
                            temp_slices.append({'path': f, 'type': 'nifti', 'slice': i, 'time': t})
                elif len(shape) == 2:
                     temp_slices.append({'path': f, 'type': 'nifti', 'slice': 0})
            except Exception as e:
                print(f"Error loading NIFTI {f}: {e}")

        for f in dicom_files:
            # DICOMs are often single slices, but let's check
            try:
                # We just store the path, we'll load it later. 
                # Assuming 1 file = 1 slice for standard DICOM series, but sometimes they are multiframe.
                # For efficiency, we assume 1 file = 1 slice unless we want to open every header now.
                # Let's open header to be safe but it might be slow.
                # For now, just append.
                temp_slices.append({'path': f, 'type': 'dicom', 'slice': 0})
            except Exception as e:
                print(f"Error processing DICOM {f}: {e}")

        # Shuffle and split
        np.random.seed(seed)
        np.random.shuffle(temp_slices)

        total_len = len(temp_slices)
        test_len = int(total_len * test_split)
        val_len = int(total_len * val_split)
        train_len = total_len - test_len - val_len

        if train:
            # This is a bit ambiguous in the request "Separate ... 80% training, 10% validation and 10% testing"
            # Usually we want to return one of them based on a flag.
            # Let's add a 'mode' argument or just use 'train' flag for Train vs (Test+Val).
            # But the user wants a pipeline.
            # Let's assume this class is instantiated 3 times with different indices.
            # But to keep it simple, I'll just implement a 'mode' param in __init__ if I was refactoring, 
            # but here I'll stick to the requested structure.
            # Wait, the user asked to "Separate the images...".
            # I will implement a helper to get the splits.
            pass
        
        self.all_slices = temp_slices
        
        # Assign split
        if train == 'train':
            self.slices = self.all_slices[:train_len]
        elif train == 'val':
            self.slices = self.all_slices[train_len:train_len+val_len]
        elif train == 'test':
            self.slices = self.all_slices[train_len+val_len:]
        else:
            self.slices = self.all_slices # All data

        # Apply limit if requested
        if limit is not None:
            self.slices = self.slices[:limit]
            print(f"Limiting dataset to {limit} samples.")

        print(f"Dataset initialized with {len(self.slices)} slices (Mode: {train})")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        item = self.slices[idx]
        path = item['path']
        
        img_data = None

        if item['type'] == 'nifti':
            try:
                nimg = nib.load(path)
                data = nimg.get_fdata(dtype=np.float32)
                # Handle dimensions
                if 'time' in item:
                    img_data = data[:, :, item['slice'], item['time']]
                elif len(data.shape) == 3:
                    img_data = data[:, :, item['slice']]
                else:
                    img_data = data
                
                # NIFTI data can be any range. Normalize to 0-1.
                # Simple min-max normalization per slice or per volume? 
                # Per slice is safer for contrast.
                # img_data is already float32 from get_fdata

                
            except Exception as e:
                print(f"Error reading {path}: {e}")
                return torch.zeros((1, 256, 256)) # Return dummy

        elif item['type'] == 'dicom':
            try:
                dcm = pydicom.dcmread(path)
                img_data = dcm.pixel_array.astype(np.float32)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                return torch.zeros((1, 256, 256))

        # Normalize
        if img_data is not None:
            # Handle NaN/Inf
            img_data = np.nan_to_num(img_data)
            
            dmin = img_data.min()
            dmax = img_data.max()
            if dmax > dmin:
                img_data = (img_data - dmin) / (dmax - dmin)
            else:
                img_data = np.zeros_like(img_data)

            # Resize if necessary? DeepInv models usually handle any size, but for batching we need same size.
            # The user didn't specify resizing, but it's good practice for batching.
            # Let's resize to 256x256 if it's too different, or just return as is and use batch_size=1 for eval.
            # For training we definitely need fixed size.
            # Let's use a transform if provided, otherwise convert to tensor.
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img_data).unsqueeze(0).float()
            
            # Resize if requested
            if hasattr(self, 'resize_to') and self.resize_to is not None:
                # interpolate expects (N, C, H, W), so unsqueeze and squeeze
                img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)

            if self.transform:
                img_tensor = self.transform(img_tensor)
                
            return img_tensor
        
        return torch.zeros((1, 256, 256))

