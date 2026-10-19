#!/usr/bin/env python3
"""
Phase 0: Hardware Validation for Quillan-Ronin Transplant
Verifies source models, RAM, GPU, and dependencies.
"""
import os
import sys
import time
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("quillan_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HardwareCheck")

RESULTS = {}

def check_dependencies():
    """Check required Python packages."""
    logger.info("=== CHECKING DEPENDENCIES ===")
    required = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "safetensors": "SafeTensors",
    }
    optional = {
        "psutil": "psutil (Phoenix affinity)",
        "scipy": "scipy (E_ICE Monte Carlo)",
        "lancedb": "LanceDB (memory)",
        "pyarrow": "PyArrow (LanceDB)",
        "pyopencl": "PyOpenCL (OpenCL swarm)",
    }
    
    missing_req = []
    missing_opt = []
    
    for pkg, name in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.info(f"  [OK] {name}: {ver}")
        except ImportError:
            missing_req.append(name)
            logger.error(f"  [FAIL] {name}: MISSING (required)")
    
    for pkg, name in optional.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.info(f"  [OK] {name}: {ver}")
        except ImportError:
            missing_opt.append(name)
            logger.warning(f"  [WARN] {name}: MISSING (optional)")
    
    RESULTS["dependencies"] = {
        "required_missing": missing_req,
        "optional_missing": missing_opt,
    }
    
    if missing_req:
        logger.error(f"FATAL: Missing required packages: {missing_req}")
        return False
    return True


def check_ram():
    """Check available RAM."""
    logger.info("=== CHECKING RAM ===")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1e9
        avail_gb = mem.available / 1e9
        logger.info(f"  Total: {total_gb:.1f} GB")
        logger.info(f"  Available: {avail_gb:.1f} GB")
        logger.info(f"  Used: {mem.used / 1e9:.1f} GB")
        RESULTS["ram"] = {"total_gb": total_gb, "available_gb": avail_gb}
        return avail_gb > 12  # Need at least 12GB free
    except ImportError:
        logger.warning("  psutil not available, skipping RAM check")
        RESULTS["ram"] = {"status": "skipped"}
        return True


def check_gpu():
    """Check GPU availability."""
    logger.info("=== CHECKING GPU ===")
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  CUDA GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        RESULTS["gpu"] = {"name": gpu_name, "vram_gb": vram, "cuda": True}
    else:
        logger.info("  No CUDA GPU detected")
        RESULTS["gpu"] = {"cuda": False}
    
    # Check OpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for p in platforms:
            devices = p.get_devices(cl.device_type.GPU)
            for d in devices:
                logger.info(f"  OpenCL GPU: {d.name}")
        RESULTS["gpu"]["opencl"] = True
    except (ImportError, Exception) as e:
        logger.warning(f"  OpenCL not available: {e}")
        RESULTS["gpu"]["opencl"] = False
    
    return True


def check_source_models():
    """Verify all source models exist and can be loaded."""
    logger.info("=== CHECKING SOURCE MODELS ===")
    
    base = r"C:\Users\Admin\Quillan-Ronin\Quillan-v4.2-model"
    models = {
        "llama": os.path.join(base, "llama model.safetensors"),
        "qwen": os.path.join(base, "qwen model.safetensors"),
        "bitnet": os.path.join(base, "bitnet model.safetensors"),
        "bitnet_copy": os.path.join(base, "model.safetensors"),
    }
    
    from safetensors import safe_open
    
    model_info = {}
    for name, path in models.items():
        if not os.path.exists(path):
            logger.error(f"  [FAIL] {name}: NOT FOUND at {path}")
            return False
        
        size_gb = os.path.getsize(path) / 1e9
        logger.info(f"  Loading {name} ({size_gb:.2f} GB)...")
        
        try:
            with safe_open(path, framework='pt') as f:
                keys = list(f.keys())
                # Get first tensor to check shape
                first_key = keys[0]
                first_tensor = f.get_tensor(first_key)
                
                # Get embedding shape
                emb_keys = [k for k in keys if 'embed' in k.lower()]
                emb_shape = None
                if emb_keys:
                    emb_tensor = f.get_tensor(emb_keys[0])
                    emb_shape = list(emb_tensor.shape)
                
                # Get layer count
                layer_nums = set()
                for k in keys:
                    if 'layers.' in k:
                        parts = k.split('.')
                        for i, p in enumerate(parts):
                            if p == 'layers' and i+1 < len(parts):
                                try:
                                    layer_nums.add(int(parts[i+1]))
                                except ValueError:
                                    pass
                
                model_info[name] = {
                    "size_gb": size_gb,
                    "num_tensors": len(keys),
                    "num_layers": len(layer_nums),
                    "embedding_shape": emb_shape,
                    "dtype": str(first_tensor.dtype),
                }
                
                logger.info(f"  [OK] {name}: {len(keys)} tensors, {len(layer_nums)} layers, embedding={emb_shape}")
                
        except Exception as e:
            logger.error(f"  [FAIL] {name}: FAILED to load - {e}")
            return False
    
    RESULTS["source_models"] = model_info
    
    # Verify total RAM needed
    total_model_gb = sum(m["size_gb"] for m in model_info.values())
    logger.info(f"  Total model size: {total_model_gb:.2f} GB")
    RESULTS["total_model_gb"] = total_model_gb
    
    return True


def check_pytorch():
    """Check PyTorch configuration."""
    logger.info("=== CHECKING PYTORCH ===")
    import torch
    logger.info(f"  Version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  CPU threads: {torch.get_num_threads()}")
    logger.info(f"  CPU count: {torch.get_num_interop_threads()}")
    
    # Set threads for i5-7500 (4 cores)
    torch.set_num_threads(2)  # Cores 2-3 via Phoenix affinity
    torch.set_num_interop_threads(2)
    logger.info(f"  Threads set to: {torch.get_num_threads()} (Phoenix affinity)")
    
    RESULTS["pytorch"] = {
        "version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
    }
    return True


def main():
    logger.info("=" * 60)
    logger.info("QUILLAN-RONIN PHASE 0: HARDWARE VALIDATION")
    logger.info("=" * 60)
    
    start = time.time()
    
    steps = [
        ("Dependencies", check_dependencies),
        ("RAM", check_ram),
        ("GPU", check_gpu),
        ("PyTorch", check_pytorch),
        ("Source Models", check_source_models),
    ]
    
    all_pass = True
    for name, check_fn in steps:
        try:
            passed = check_fn()
            if not passed:
                all_pass = False
                logger.error(f"  FAILED: {name}")
        except Exception as e:
            all_pass = False
            logger.error(f"  ERROR in {name}: {e}")
    
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PHASE 0 COMPLETE: {'PASS' if all_pass else 'FAIL'} ({elapsed:.1f}s)")
    logger.info("=" * 60)
    
    # Save results
    RESULTS["status"] = "PASS" if all_pass else "FAIL"
    RESULTS["elapsed_seconds"] = elapsed
    
    with open("hardware_check_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    
    logger.info(f"Results saved to hardware_check_results.json")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
