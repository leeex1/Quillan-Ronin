import torch
import sys
from transplant_weights import QuillanConfig

def migrate(ckpt_path="quillan_transplanted.pt", out_path="quillan_transplanted_v8.pt"):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    new_sd = {}
    
    num_experts = 34
    hidden_dim = 2048
    ffn_dim = 4096
    
    # Initialize the new parameter tensors for MoE
    w1 = torch.zeros(num_experts, hidden_dim, ffn_dim)
    wgate = torch.zeros(num_experts, hidden_dim, ffn_dim)
    w2 = torch.zeros(num_experts, ffn_dim, hidden_dim)
    
    # Initialize wgate normally since it didn't exist (using normal since Kaiming uses uniform or normal)
    torch.nn.init.kaiming_normal_(wgate.view(-1, ffn_dim))
    
    # Check old layers to map
    for k, v in sd.items():
        if k == 'txt_emb.weight':
            new_sd['ingestion.txt_emb.weight'] = v
        elif k == 'mod_emb.weight':
            new_sd['ingestion.mod_emb.weight'] = v
        elif k.startswith('experts.'):
            parts = k.split('.')
            idx = int(parts[1])
            if parts[2] == 'w1':
                # v shape: [ffn_dim, hidden_dim]
                w1[idx] = v.T
            elif parts[2] == 'w2':
                # v shape: [hidden_dim, ffn_dim]
                w2[idx] = v.T
            elif parts[2] == 'swarm':
                # e.g. experts.0.swarm.lora_A -> moe.expert_swarms.0.lora_A (if using lora_A/B in CouncilExpertSwarm)
                # Let's map it directly. Wait, in old model it was swarm.A and swarm.B?
                if parts[3] == 'A':
                    new_sd[f'moe.expert_swarms.{idx}.A'] = v
                elif parts[3] == 'B':
                    new_sd[f'moe.expert_swarms.{idx}.B'] = v
                else:
                    new_sd[f'moe.expert_swarms.{idx}.{parts[3]}'] = v
        elif k == 'router.weight':
            new_sd['moe.router.weight'] = v
        elif k == 'router.bias':
            new_sd['moe.router.bias'] = v
        elif k.startswith('diffusion.0.'):
            # diffusion.0.q_proj.weight -> diffusion_core.q_proj.weight
            new_k = k.replace('diffusion.0.', 'diffusion_core.')
            new_sd[new_k] = v
        elif k.startswith('diffusion.'):
            # Ignore diffusion layers > 0
            pass
        elif k == 'quillan_finalizer.bias':
            # Ignore bias if the new layer doesn't use it, but if it's there we keep it. 
            new_sd[k] = v
        elif k == 'quillan_finalizer.weight':
            new_sd[k] = v
        elif k == 'txt_dec.weight':
            new_sd[k] = v
        elif k.startswith('decomposition.'):
            new_sd[k] = v
        else:
            new_sd[k] = v
            
    new_sd['moe.w1'] = w1
    new_sd['moe.w2'] = w2
    new_sd['moe.wgate'] = wgate
    
    print(f"Saving to {out_path}...")
    torch.save({'model_state_dict': new_sd}, out_path)
    print("Done!")

if __name__ == "__main__":
    migrate()
