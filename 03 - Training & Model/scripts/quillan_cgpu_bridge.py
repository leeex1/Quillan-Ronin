"""Quillan C+GPU Compatibility Bridge.
Router + hot top-k experts live on cuda:0 (GTX 1050, 4GB); body stays on CPU.
Pinned-memory staging, per-layer prefetch, fp32-first correctness.
No compiler needed: pure torch device-map. Heavy lifting only, never the home."""
from __future__ import annotations
import torch
import torch.nn as nn


class CGPUBridge:
    """Shard-aware executor: GPU = router + hot experts, CPU = everything else."""

    def __init__(self, gpu_index: int = 0, gpu_budget_mb: int = 3000,
                 top_k_gpu: int = 4) -> None:
        assert torch.cuda.is_available(), "CUDA torch required for C+GPU bridge"
        self.gpu = torch.device(f"cuda:{gpu_index}")
        self.cpu = torch.device("cpu")
        props = torch.cuda.get_device_properties(gpu_index)
        self.gpu_budget = min(gpu_budget_mb, int(props.total_memory // 1024 // 1024) - 512)
        self.top_k_gpu = top_k_gpu
        self._gpu_bytes = 0
        self.hot: set[str] = set()
        print(f"[cgpu] {props.name} cap={props.major}.{props.minor} "
              f"budget={self.gpu_budget}MB topk_gpu={top_k_gpu}", flush=True)

    def _bytes(self, mod: nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in mod.parameters())

    def place_router(self, router: nn.Module, name: str = "router") -> nn.Module:
        router.to(self.gpu)
        self._gpu_bytes += self._bytes(router)
        self.hot.add(name)
        return router

    def place_experts(self, experts: list[nn.Module], scores: list[float],
                      prefix: str = "experts") -> list[nn.Module]:
        """Top-scoring experts go GPU until budget; rest stay CPU (pinned)."""
        order = sorted(range(len(experts)), key=lambda i: scores[i], reverse=True)
        out = list(experts)
        for rank, i in enumerate(order):
            if rank >= self.top_k_gpu:
                break
            need = self._bytes(experts[i]) // 1024 // 1024
            if self._gpu_bytes // 1024 // 1024 + need > self.gpu_budget:
                print(f"[cgpu] budget stop at expert {i} ({need}MB)", flush=True)
                break
            out[i] = experts[i].to(self.gpu)
            self._gpu_bytes += self._bytes(experts[i])
            self.hot.add(f"{prefix}.{i}")
        print(f"[cgpu] hot={sorted(self.hot)} gpu_mb={self._gpu_bytes // 1024 // 1024}",
              flush=True)
        return out

    @staticmethod
    def stage(x: torch.Tensor, device: torch.device) -> torch.Tensor:
        if x.device == device:
            return x
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True)
        return x.to(device)

    def status(self) -> dict:
        free, total = torch.cuda.mem_get_info(0)
        return {"hot": sorted(self.hot),
                "tracked_mb": self._gpu_bytes // 1024 // 1024,
                "gpu_free_mb": free // 1024 // 1024,
                "gpu_total_mb": total // 1024 // 1024}
