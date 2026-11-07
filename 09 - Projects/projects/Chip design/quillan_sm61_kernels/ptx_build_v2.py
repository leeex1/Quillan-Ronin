#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build sm61_qgemm_v2.cu -> sm61_qgemm_v2_sm61.ptx via NVRTC."""

from __future__ import annotations
import ctypes, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = Path(
    os.environ.get(
        "QUILLAN_CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    )
)
NVRTC_DLL = TOOLKIT / "bin" / "nvrtc64_120_0.dll"
CU_SRC = HERE / "sm61_qgemm_v2.cu"
PTX_OUT = HERE / "sm61_qgemm_v2_sm61.ptx"
CUT_MARKER = "// Host-side launcher."


def _nvrtc():
    lib = ctypes.WinDLL(str(NVRTC_DLL))
    lib.nvrtcVersion.restype = int
    lib.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
    lib.nvrtcCreateProgram.restype = int
    lib.nvrtcCreateProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.nvrtcCompileProgram.restype = int
    lib.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcGetProgramLogSize.restype = int
    lib.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.nvrtcGetProgramLog.restype = int
    lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetPTXSize.restype = int
    lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetPTX.restype = int
    lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcDestroyProgram.restype = int
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    return lib


def _log(lib, prog) -> str:
    n = ctypes.c_size_t(0)
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value if n.value else 1)
    lib.nvrtcGetProgramLog(prog, buf)
    return buf.value.decode("utf-8", "replace")


def main() -> int:
    src = CU_SRC.read_text(encoding="utf-8")
    device_src = src.split(CUT_MARKER)[0]
    assert "<<<" not in device_src, "launch syntax leaked into NVRTC input"
    device_src = device_src.replace("#include <cstdint>", "")
    device_src = "typedef signed char int8_t;\ntypedef int int32_t;\n" + device_src
    lib = _nvrtc()
    prog = ctypes.c_void_p(None)
    rc = lib.nvrtcCreateProgram(
        ctypes.byref(prog),
        device_src.encode("utf-8"),
        CU_SRC.name.encode("utf-8"),
        0,
        None,
        None,
    )
    if rc != 0:
        print(f"FAIL: nvrtcCreateProgram rc={rc}")
        return 1
    opts = [
        b"--gpu-architecture=compute_61",
        b"--std=c++17",
        ("-I" + str(TOOLKIT / "include")).encode("utf-8"),
    ]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    log = _log(lib, prog)
    if rc != 0:
        print(f"FAIL: nvrtcCompileProgram rc={rc}\n{log}")
        return 1
    if log.strip():
        print(f"[ptx_build_v2] nvrtc log:\n{log}")
    n = ctypes.c_size_t(0)
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetPTX(prog, buf)
    ptx = bytes(buf.raw[: n.value - 1])
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    assert b"qgemm_i8_dp4a_kernel_v2" in ptx, "kernel entry missing from PTX"
    PTX_OUT.write_bytes(ptx)
    print(f"[ptx_build_v2] wrote {PTX_OUT.name} ({len(ptx)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
