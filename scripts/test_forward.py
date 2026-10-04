import sys
import os
ROOT = r"C:\Users\Admin\Quillan-Ronin"
sys.path.insert(0, os.path.join(ROOT, "_dev"))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

def check_nan(tensor, name):
    if tensor is not None and isinstance(tensor, torch.Tensor):
        has_nan = tensor.isnan().any().item()
        print(f"[{name}] nan: {has_nan} | shape: {tensor.shape}")
        if has_nan:
            print(f"!!! NAN AT {name} !!!")
            return True
    return False

def main():
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    model = QuillanRoninSovereign(QuillanArchConfig())
    import os
    ckpt_path = None
    for p in ["quillan_merged_saturated.pt", "checkpoints/router_trained.pt", "checkpoints/quillan_fixed.pt"]:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path:
        print(f"Loading checkpoint from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    model.to(device)
    model.train()
    dataset = []
    for dset in ["code_train.pt", "GPT_5.5_Distilled.pt"]:
        dset_path = os.path.join(ROOT, "training_data", dset)
        if os.path.exists(dset_path):
            data = torch.load(dset_path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                dataset.append(data.to(torch.long))
            else:
                dataset.append(torch.tensor(data, dtype=torch.long))
    
    if dataset:
        full_data = torch.cat(dataset)
        print(f"full_data max: {full_data.max().item()}, min: {full_data.min().item()}")
        # Check if any index is out of bounds
        if full_data.max().item() >= 50257:
            print("WARNING: OUT OF BOUNDS INDEX DETECTED!")
        inp = full_data[0:128].unsqueeze(0).to(device)
    else:
        inp = torch.randint(0, 50257, (1, 128), device=device)
    
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=='cuda')):
        out = model(inp)
        logits = out['logits']
        print(f"Logits shape: {logits.shape}")
        print(f"Logits max: {logits.max().item()}")
        print(f"Logits min: {logits.min().item()}")
        print(f"Logits isnan: {logits.isnan().any().item()}")

if __name__ == "__main__":
    main()
