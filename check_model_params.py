import torch
import deepinv as dinv
from deepinv.models import DRUNet, Restormer, RAM, GSDRUNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    models = {}
    
    print("Loading models...")
    
    try:
        models['DRUNet'] = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    except Exception as e:
        print(f"Failed to load DRUNet: {e}")

    try:
        models['GSDRUNet'] = dinv.models.GSDRUNet(in_channels=1, out_channels=1, pretrained='download', device=device)
    except Exception as e:
        print(f"Failed to load GSDRUNet: {e}")

    try:
        models['Restormer'] = dinv.models.Restormer(in_channels=1, out_channels=1, pretrained='denoising_gray', device=device)
    except Exception as e:
        print(f"Failed to load Restormer: {e}")

    try:
        models['RAM'] = dinv.models.RAM(pretrained=True, device=device)
    except Exception as e:
        print(f"Failed to load RAM: {e}")

    print("\nModel Parameter Counts:")
    print("-" * 30)
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name:<15} : {params:,} parameters")
    print("-" * 30)

if __name__ == "__main__":
    main()
