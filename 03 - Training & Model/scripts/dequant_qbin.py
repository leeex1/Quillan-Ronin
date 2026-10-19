"""Dequant .qbin (export_real v1) -> fp32 state_dict .pt. Reads ONLY."""
import struct
import sys
from pathlib import Path
import torch
REPO = Path(r"C:\02_QUILLAN")
QB = REPO / "quillan.cpp" / "quillan_6l_v62.qbin"
OUT = REPO / "checkpoints" / "hf_restore" / "quillan_6l_vocab62_base.pt"
with open(QB, "rb") as f:
    hdr = struct.unpack("<IIiiiiiiiiiff", f.read(4 + 4 + 9 * 4 + 2 * 4))
    magic, ver, vocab, hidden, ffn, layers = hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
    assert magic == 0x4E4C4C51, f"bad magic {magic:#x}"
    print(f"header vocab={vocab} hidden={hidden} ffn={ffn} layers={layers}",
          flush=True)
    sd, n_t, n_q = {}, 0, 0
    while True:
        lb = f.read(4)
        if not lb:
            break
        (nl,) = struct.unpack("<I", lb)
        name = f.read(nl).decode("utf-8")
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}i", f.read(4 * nd))
        dtype, scale = struct.unpack("<Bf", f.read(5))
        (blen,) = struct.unpack("<Q", f.read(8))
        blob = f.read(blen)
        assert len(blob) == blen, name
        if dtype == 1:
            q = torch.frombuffer(bytearray(blob), dtype=torch.int8).reshape(dims)
            sd[name] = (q.float() * scale)
            n_q += 1
        else:
            sd[name] = torch.frombuffer(bytearray(blob), dtype=torch.float32).reshape(dims)
        n_t += 1
print(f"tensors={n_t} dequant={n_q} fp={n_t - n_q}", flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state_dict": sd,
            "meta": {"src": "quillan_6l_v62.qbin", "vocab": vocab,
                     "note": "dequant approx of vocab62 mapped base"}},
           OUT)
print("DEQUANT DONE " + str(OUT), flush=True)
