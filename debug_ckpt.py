import torch
import deepinv as dinv

ckpt_path = r"d:\Diego trabalho\deepinv-main\checkpoints\25-11-26-01_15_12\ckp_best.pth.tar"
device = "cpu"

try:
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("State dict keys sample:", list(state_dict.keys())[:5])
    else:
        print("No state_dict key found. Keys:", checkpoint.keys())

    print("Instantiating model...")
    model = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, device=device)
    
    print("Loading state dict...")
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Success!")

except Exception as e:
    print(f"Error: {e}")
