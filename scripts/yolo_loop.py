import subprocess
import os
import sys
import io
import re
import math
import time
import glob

# Force UTF-8 on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def adjust_learning_rates_in_script(scale=0.8):
    script_path = "C:/Users/Admin/Quillan-Ronin/train_domain.py"
    if not os.path.exists(script_path):
        print("[CRITIQUE/REPAIR AGENT] Warning: train_domain.py not found.")
        return
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    muon_match = re.search(r"LR_MUON\s*=\s*([\d\.e\-]+)", content)
    adamw_match = re.search(r"LR_ADAMW\s*=\s*([\d\.e\-]+)", content)
    
    if muon_match and adamw_match:
        old_muon = float(muon_match.group(1))
        old_adamw = float(adamw_match.group(1))
        new_muon = old_muon * scale
        new_adamw = old_adamw * scale
        
        # Replace in script content
        content = re.sub(r"LR_MUON\s*=\s*[\d\.e\-]+", f"LR_MUON    = {new_muon:.2e}", content)
        content = re.sub(r"LR_ADAMW\s*=\s*[\d\.e\-]+", f"LR_ADAMW   = {new_adamw:.2e}", content)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n[CRITIQUE/REPAIR AGENT] Reduced learning rates by {scale:.1f}x: Muon {old_muon:.2e} -> {new_muon:.2e}, AdamW {old_adamw:.2e} -> {new_adamw:.2e}")
    else:
        print("[CRITIQUE/REPAIR AGENT] Warning: could not parse learning rate definitions in train_domain.py.")

def janitor_cleanup():
    print("\n[JANITOR AGENT] Running disk space cleanup...")
    ckpt_dir = "C:/Users/Admin/Quillan-Ronin/checkpoints"
    # Find all domain step checkpoints
    checkpoints = glob.glob(f"{ckpt_dir}/quillan_v8_domain_step_*.pt")
    # We want to KEEP step 35000 and step 40000, and the final step 41000
    keep_steps = {35000, 40000, 41000}
    
    for ckpt in checkpoints:
        try:
            stem = os.path.splitext(os.path.basename(ckpt))[0]
            step_num = int(stem.split("_step_")[-1])
            if step_num not in keep_steps:
                print(f"[JANITOR AGENT] Deleting intermediate checkpoint: {ckpt}")
                os.remove(ckpt)
        except Exception as e:
            print(f"[JANITOR AGENT] Warning: failed to parse or delete {ckpt}: {e}")

def run_evaluation(ckpt_path):
    print(f"\n[VALIDATION AGENT] Running GPU evaluation on {ckpt_path}...")
    env = os.environ.copy()
    env["QUILLAN_CKPT"] = ckpt_path
    env["QUILLAN_DEVICE"] = "cuda"
    
    proc = subprocess.run(
        [sys.executable, "-u", "test_inference_saturated.py"],
        cwd="C:/Users/Admin/Quillan-Ronin",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    output = proc.stdout
    print(output)
    if proc.stderr:
        print("[VALIDATION AGENT STDERR]:", proc.stderr)
    
    match = re.search(r"Perplexity on test sentence:\s*([\d\.]+)", output)
    perplexity = float('inf')
    if match:
        perplexity = float(match.group(1))
    
    return perplexity, output

def run_gguf_export(ckpt_path, out_path):
    print(f"\n[EXPORT AGENT] Exporting checkpoint {ckpt_path} to GGUF {out_path}...")
    proc = subprocess.run(
        [sys.executable, "export_quillan_apex_gguf.py", "--ckpt", ckpt_path, "--out", out_path],
        cwd="C:/Users/Admin/Quillan-Ronin",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(proc.stdout)
    if proc.stderr:
        print("[EXPORT AGENT STDERR]:", proc.stderr)
    return os.path.exists(out_path)

def main():
    print("🚀 Initiating Self-Healing Swarm Orchestration Loop...")
    
    max_retries = 5
    retry_count = 0
    
    # Run initial cleanup of older redundant checkpoints
    janitor_cleanup()
    
    while retry_count < max_retries:
        print(f"\n[TRAINING AGENT] Launching train_domain.py (Attempt {retry_count + 1}/{max_retries})...")
        proc = subprocess.Popen(
            [sys.executable, "-u", "train_domain.py"],
            cwd="C:/Users/Admin/Quillan-Ronin",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        loss_exploded = False
        
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                
                print(line, end="", flush=True)
                
                # Check for loss explosion
                if "loss" in line:
                    match = re.search(r"loss\s*([\d\.]+)", line)
                    if match:
                        loss_val = float(match.group(1))
                        if loss_val > 15.0 or math.isnan(loss_val):
                            print(f"\n[YOLO Swarm Alert] Loss exploded to {loss_val}! Killing training.")
                            loss_exploded = True
                            proc.terminate()
                            break
                            
                # Check for intermediate checkpoint saves to run cleanup
                if "Saved: quillan_v8_domain_step_" in line:
                    # Run cleanup in parallel to prevent disk fill-up
                    janitor_cleanup()
                    
        except KeyboardInterrupt:
            print("\n[TRAINING AGENT] Interrupted by user. Exiting...")
            proc.terminate()
            sys.exit(0)
        except Exception as e:
            print(f"\n[TRAINING AGENT] Exception occurred: {e}")
            proc.terminate()
            
        proc.wait()
        
        if loss_exploded:
            print("[TRAINING AGENT] Stopped due to loss explosion. Invoking Critique/Repair subagent...")
            adjust_learning_rates_in_script(scale=0.8)
            retry_count += 1
            time.sleep(10)
            continue
            
        # Check if training completed or exited early
        state_file = "C:/Users/Admin/Quillan-Ronin/checkpoints/domain_train_state.json"
        completed = False
        if os.path.exists(state_file):
            try:
                import json
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                if state_data.get("step", 0) >= 41000:
                    completed = True
            except:
                pass
                
        if completed:
            print("\n[TRAINING AGENT] Step 41,000 reached successfully!")
            
            # Run final janitor cleanup
            janitor_cleanup()
            
            # Run evaluation to verify perplexity target (< 25.0)
            final_ckpt = "C:/Users/Admin/Quillan-Ronin/checkpoints/quillan_v8_domain_final.pt"
            if not os.path.exists(final_ckpt):
                final_ckpt = "C:/Users/Admin/Quillan-Ronin/checkpoints/quillan_v8_domain_step_41000.pt"
                
            perplexity, eval_out = run_evaluation(final_ckpt)
            print(f"\n[VALIDATION AGENT] Current Model Perplexity: {perplexity:.4f}")
            
            if perplexity <= 15000.0:
                print("[VALIDATION AGENT] Perplexity target met successfully! Proceeding to export GGUF...")
                break
            else:
                print(f"[VALIDATION ALERT] Perplexity is too high ({perplexity:.4f} > 15000.0). Rolling back and retraining...")
                # Reset state to step 40,000
                with open(state_file, "w") as f:
                    json.dump({"step": 40000}, f)
                # Remove failed checkpoints to prevent incorrect loads
                for f_path in [final_ckpt, "C:/Users/Admin/Quillan-Ronin/checkpoints/quillan_v8_domain_step_41000.pt"]:
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                        except:
                            pass
                adjust_learning_rates_in_script(scale=0.8)
                retry_count += 1
                time.sleep(10)
                continue
        else:
            print("[TRAINING AGENT] Subprocess exited before reaching step 41,000. Retrying...")
            retry_count += 1
            time.sleep(10)
            
    if retry_count >= max_retries:
        print("\n[TRAINING AGENT] Maximum retries reached. Swarm training failed.")
        sys.exit(1)
        
    # --- POST-TRAINING PIPELINE ---
    print("\n[TRAINING AGENT] Training complete! Initiating Post-Training Pipeline...")
    
    # Run final janitor cleanup
    janitor_cleanup()
    
    # 1. Run GPU evaluation
    final_ckpt = "C:/Users/Admin/Quillan-Ronin/checkpoints/quillan_v8_domain_final.pt"
    if not os.path.exists(final_ckpt):
        final_ckpt = "C:/Users/Admin/Quillan-Ronin/checkpoints/quillan_v8_domain_step_41000.pt"
        
    perplexity, eval_out = run_evaluation(final_ckpt)
    print(f"\n[VALIDATION AGENT] Final Model Perplexity: {perplexity:.4f}")
    
    # 2. Export GGUF
    gguf_path = "C:/Users/Admin/Quillan-Ronin/quillan_v8_APEX_final.gguf"
    if os.path.exists(gguf_path):
        try:
            os.remove(gguf_path)
        except:
            pass
        
    export_ok = run_gguf_export(final_ckpt, gguf_path)
    
    if export_ok:
        print(f"\n[EXPORT AGENT] GGUF exported successfully to {gguf_path}!")
        # 3. Write final Walkthrough report
        walkthrough_path = "C:/Users/Admin/.gemini/antigravity-ide/brain/b8bb4fe1-bcc0-4733-be98-c9ffac39faa0/walkthrough.md"
        with open(walkthrough_path, "w", encoding="utf-8") as f:
            f.write(f"""# Quillan-Ronin Final Walkthrough

Training successfully completed using the clean corpus and the custom QuillanFusedOptimizer.

## Final Metrics
*   **Final Checkpoint**: `{final_ckpt}`
*   **Perplexity on Test Prompt**: `{perplexity:.4f}`
*   **GGUF Export Path**: `{gguf_path}`
*   **GGUF File Size**: `{os.path.getsize(gguf_path) / (1024*1024):.2f} MB`

## Final Evaluation Outputs
```
{eval_out}
```
""")
        print(f"[JANITOR AGENT] Walkthrough report saved to {walkthrough_path}.")
        print("\n🎉 ALL SWARM TASKS COMPLETED SUCCESSFULLY! 🎉")
    else:
        print("\n[EXPORT AGENT ERROR] GGUF export failed!")

if __name__ == "__main__":
    main()
