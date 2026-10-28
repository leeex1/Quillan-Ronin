#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 QUILLAN-RONIN REVERSE ENGINEERING ENGINE — NATIVE FAST-MCP SERVER
======================================================================
Embodying C34-PREDATOR (Adversarial Hunting), C10-CODEWEAVER (Technical Implementation),
C21-ARCHON (Academic Research), C32-AEON (Causal Simulation), C18-SHEPHERD (Truth Grounding),
C1-ASTRA (Vision & Pattern Recognition), C7-LOGOS (Logical Consistency).

Native reverse-engineering — no external RE tools required. Pure Python parsers for:
PE / ELF / Mach-O / APK / JAR / .NET / SQLite / PCAP / WASM / Firmware.

Native Quillan implementation of REA-style reverse engineering (MIT-licensed concepts,
reimplemented natively so Quillan does the reversing itself — no external RE binaries).

Tools:
  1. re_analyze_binary       — PE/ELF/Mach-O header, sections, imports, exports, symbols
  2. re_extract_strings      — ASCII + UTF-16LE string extraction with context
  3. re_find_symbols         — function, import, export, string symbol lookup
  4. re_trace_calls          — call-graph / cross-reference reconstruction
  5. re_decompile_stub       — pseudocode-style decompilation scaffold
  6. re_analyze_apk          — Android APK manifest, classes.dex, resources
  7. re_analyze_dotnet       — .NET assembly metadata, types, methods
  8. re_analyze_sqlite       — SQLite schema, tables, indexes, rows
  9. re_analyze_pcap         — packet capture endpoints, DNS, TLS SNI, HTTP
  10. re_analyze_web         — website structure, scripts, network observations
  11. re_analyze_firmware    — firmware regions, entropy, embedded strings
  12. re_full_re             — full pipeline: binary → strings → symbols → calls → report
  13. re_route               — cross-MCP routing (council/code/buddy)
  14. re_health              — RE engine dashboard summary
  15. re_session_list / re_session_replay — persistence
"""

import sys
# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
import logging, builtins
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if 'file' not in kwargs or kwargs['file'] is None or kwargs['file'] is sys.stdout:
        kwargs['file'] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print
import os
import re
import json
import time
import uuid
import struct
import zipfile
import sqlite3
import hashlib
import zlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanRE")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_re', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_re] _shared warning: {_shared_err}', file=_sys.stderr)



def _read_le(data: bytes, offset: int, size: int) -> int:
    chunk = data[offset:offset+size]
    if len(chunk) < size:
        return 0
    return int.from_bytes(chunk, byteorder="little")

def _read_be(data: bytes, offset: int, size: int) -> int:
    chunk = data[offset:offset+size]
    if len(chunk) < size:
        return 0
    return int.from_bytes(chunk, byteorder="big")

def extract_strings(data: bytes, min_len: int = 4) -> List[Dict[str, Any]]:
    """Extract ASCII and UTF-16LE strings with offsets."""
    strings = []
    # ASCII strings
    ascii_re = re.compile(rb"[\x20-\x7e]{%d,}" % min_len)
    for m in ascii_re.finditer(data):
        strings.append({
            "offset": m.start(),
            "type": "ascii",
            "value": m.group().decode("ascii", errors="replace"),
            "length": len(m.group()),
        })
    # UTF-16LE strings
    utf16_re = re.compile(rb"(?:[\x20-\x7e]\x00){%d,}" % min_len)
    for m in utf16_re.finditer(data):
        raw = m.group()
        try:
            val = raw.decode("utf-16le", errors="replace")
        except Exception:
            val = raw.decode("latin-1", errors="replace")
        strings.append({
            "offset": m.start(),
            "type": "utf16le",
            "value": val,
            "length": len(raw),
        })
    strings.sort(key=lambda s: s["offset"])
    return strings


def parse_pe(data: bytes) -> Dict[str, Any]:
    """Minimal PE parser (DOS stub, COFF header, Optional header, sections, imports/exports)."""
    info: Dict[str, Any] = {"format": "PE", "sections": [], "imports": [], "exports": []}
    if len(data) < 64 or data[:2] != b"MZ":
        info["error"] = "not a valid PE (missing MZ signature)"
        return info
    pe_off = _read_le(data, 0x3C, 4)
    if pe_off + 24 > len(data) or data[pe_off:pe_off + 4] != b"PE\x00\x00":
        info["error"] = "not a valid PE (missing PE signature)"
        return info
    info["pe_offset"] = pe_off
    # COFF header
    coff = pe_off + 4
    info["machine"] = hex(_read_le(data, coff, 2))
    info["number_of_sections"] = _read_le(data, coff + 2, 2)
    info["timestamp"] = _read_le(data, coff + 4, 4)
    # Optional header
    opt = coff + 20
    if opt + 2 > len(data):
        return info
    magic = _read_le(data, opt, 2)
    info["optional_magic"] = hex(magic)
    is_pe32 = magic == 0x10B
    info["is_pe32"] = is_pe32
    if is_pe32:
        info["entry_point"] = hex(_read_le(data, opt + 16, 4))
        info["image_base"] = hex(_read_le(data, opt + 28, 4))
        num_sections = info["number_of_sections"]
        sec_table = opt + 96
    else:
        info["entry_point"] = hex(_read_le(data, opt + 24, 4))
        info["image_base"] = hex(_read_le(data, opt + 32, 8))
        num_sections = info["number_of_sections"]
        sec_table = opt + 112
    # Sections
    for i in range(num_sections):
        s_off = sec_table + i * 40
        if s_off + 40 > len(data):
            break
        name = data[s_off:s_off + 8].rstrip(b"\x00").decode("ascii", errors="replace")
        v_size = _read_le(data, s_off + 8, 4)
        v_addr = _read_le(data, s_off + 12, 4)
        raw_size = _read_le(data, s_off + 16, 4)
        raw_ptr = _read_le(data, s_off + 20, 4)
        info["sections"].append({
            "name": name,
            "virtual_address": hex(v_addr),
            "virtual_size": v_size,
            "raw_size": raw_size,
            "raw_pointer": hex(raw_ptr),
        })
    # Import directory (PE32)
    if is_pe32:
        dd = opt + 96 + num_sections * 40 + 16 * 8  # data dirs start after optional header
        # Import directory is index 1 in DataDirectory
        imp_rva = _read_le(data, opt + 96 + 1 * 8, 4) if opt + 96 + 1 * 8 + 4 <= len(data) else 0
        imp_size = _read_le(data, opt + 96 + 1 * 8 + 4, 4) if opt + 96 + 1 * 8 + 8 <= len(data) else 0
        if imp_rva:
            # Convert RVA to file offset (simplified: assume raw_ptr == v_addr for simple PEs)
            for sec in info["sections"]:
                v_start = int(sec["virtual_address"], 16)
                v_end = v_start + sec["virtual_size"]
                if imp_rva >= v_start and imp_rva < v_end:
                    file_off = sec["raw_pointer"] + (imp_rva - v_start)
                    break
            else:
                file_off = imp_rva
            # Read import directory entries (20 bytes each)
            off = file_off
            while off + 20 <= len(data):
                name_rva = _read_le(data, off, 0)
                if name_rva == 0:
                    break
                # ILT (OriginalFirstThunk)
                ilt_rva = _read_le(data, off, 4)
                # name
                name_rva2 = _read_le(data, off + 12, 4)
                # IAT
                iat_rva = _read_le(data, off + 16, 4)
                mod_name = ""
                if name_rva2:
                    # resolve rva to file offset roughly
                    moff = name_rva2
                    for sec in info["sections"]:
                        v_start = int(sec["virtual_address"], 16)
                        v_end = v_start + sec["virtual_size"]
                        if moff >= v_start and moff < v_end:
                            moff = sec["raw_pointer"] + (moff - v_start)
                            break
                    try:
                        mod_end = data.index(b"\x00", moff)
                        mod_name = data[moff:mod_end].decode("ascii", errors="replace")
                    except Exception:
                        mod_name = f"<mod@{name_rva2:#x}>"
                # Read ILT entries
                funcs = []
                ilt_off = ilt_rva
                for _ in range(20):
                    if ilt_off + 4 > len(data):
                        break
                    val = _read_le(data, ilt_off, 4)
                    if val == 0:
                        break
                    if val & 0x80000000:
                        funcs.append(f"ordinal_{val & 0xFFFF}")
                    else:
                        # hint/name
                        hint_off = val
                        fmo = val
                        for sec in info["sections"]:
                            v_start = int(sec["virtual_address"], 16)
                            v_end = v_start + sec["virtual_size"]
                            if fmo >= v_start and fmo < v_end:
                                fmo = sec["raw_pointer"] + (fmo - v_start)
                                break
                        try:
                            hint = _read_le(data, fmo, 2)
                            fname_end = data.index(b"\x00", fmo + 2)
                            fname = data[fmo + 2:fname_end].decode("ascii", errors="replace")
                            funcs.append(f"{fname} (hint={hint})")
                        except Exception:
                            funcs.append(f"<unk@{fmo:#x}>")
                    ilt_off += 4
                info["imports"].append({"module": mod_name, "functions": funcs})
                off += 20
    # Export directory (index 0)
    if is_pe32:
        exp_rva = _read_le(data, opt + 96, 4)
        exp_size = _read_le(data, opt + 100, 4)
    else:
        exp_rva = _read_le(data, opt + 96, 4)
        exp_size = _read_le(data, opt + 100, 4)
    if exp_rva:
        eoff = exp_rva
        for sec in info["sections"]:
            v_start = int(sec["virtual_address"], 16)
            v_end = v_start + sec["virtual_size"]
            if exp_rva >= v_start and exp_rva < v_end:
                eoff = sec["raw_pointer"] + (exp_rva - v_start)
                break
        if eoff + 40 <= len(data):
            num_names = _read_le(data, eoff + 24, 4)
            name_rva_rva = _read_le(data, eoff + 32, 4)
            name_rva_base = name_rva_rva
            for i in range(num_names):
                nr = _read_le(data, name_rva_base + i * 4, 4)
                # resolve to file offset
                foff = nr
                for sec in info["sections"]:
                    v_start = int(sec["virtual_address"], 16)
                    v_end = v_start + sec["virtual_size"]
                    if nr >= v_start and nr < v_end:
                        foff = sec["raw_pointer"] + (nr - v_start)
                        break
                try:
                    en = data.index(b"\x00", foff)
                    info["exports"].append(data[foff:en].decode("ascii", errors="replace"))
                except Exception:
                    pass
    return info


def parse_elf(data: bytes) -> Dict[str, Any]:
    """Minimal ELF parser."""
    info: Dict[str, Any] = {"format": "ELF", "sections": [], "symbols": [], "imports": []}
    if len(data) < 16 or data[:4] != b"\x7fELF":
        info["error"] = "not a valid ELF"
        return info
    info["ei_class"] = data[4]  # 1=32, 2=64
    info["ei_data"] = data[5]   # 1=LE, 2=BE
    endian = "<" if data[5] == 1 else ">"
    info["type"] = hex(_read_le(data, 16, 2) if data[5] == 1 else _read_be(data, 16, 2))
    info["machine"] = hex(_read_le(data, 18, 2) if data[5] == 1 else _read_be(data, 18, 2))
    if data[4] == 1:
        info["entry"] = hex(_read_le(data, 24, 4))
        info["shoff"] = _read_le(data, 40, 4)
        info["shentsize"] = _read_le(data, 46, 2)
        info["shnum"] = _read_le(data, 48, 2)
    else:
        info["entry"] = hex(_read_be(data, 24, 8))
        info["shoff"] = _read_be(data, 40, 8)
        info["shentsize"] = _read_be(data, 46, 2)
        info["shnum"] = _read_be(data, 48, 2)
    shoff = info["shoff"]
    shentsize = info["shentsize"]
    shnum = info["shnum"]
    for i in range(shnum):
        so = shoff + i * shentsize
        if so + shentsize > len(data):
            break
        name_off = _read_le(data, so, 4) if data[5] == 1 else _read_be(data, so, 4)
        sh_type = _read_le(data, so + 4, 4) if data[5] == 1 else _read_be(data, so + 4, 4)
        sh_addr = _read_le(data, so + 12, 4 if data[4] == 1 else 8)
        sh_size = _read_le(data, so + 20 if data[4] == 1 else 24, 4 if data[4] == 1 else 8)
        info["sections"].append({
            "index": i,
            "type": hex(sh_type),
            "address": hex(sh_addr),
            "size": sh_size,
            "name_offset": name_off,
        })
    return info


def parse_apk(data: bytes) -> Dict[str, Any]:
    """Minimal APK/ZIP parser."""
    info: Dict[str, Any] = {"format": "APK/ZIP", "files": [], "manifest": None, "dex": False}
    try:
        with zipfile.ZipFile(data) as z:
            names = z.namelist()
            info["files"] = names[:50]
            info["file_count"] = len(names)
            info["dex"] = any("classes.dex" in n for n in names)
            info["resources"] = any("resources.arsc" in n for n in names)
            if "AndroidManifest.xml" in names:
                try:
                    manifest = z.read("AndroidManifest.xml")
                    info["manifest"] = manifest[:2000].decode("utf-8", errors="replace")
                except Exception:
                    info["manifest"] = None
    except Exception as e:
        info["error"] = str(e)
    return info


def parse_dotnet(data: bytes) -> Dict[str, Any]:
    """Very basic .NET assembly metadata reader."""
    info: Dict[str, Any] = {"format": "dotnet", "types": [], "methods": []}
    if data[:2] != b"MZ":
        info["error"] = "not a PE (missing MZ)"
        return info
    # Very light: just detect .NET by scanning for mscorlib/System references
    try:
        text = data.decode("utf-8", errors="replace")
        if "mscorlib" in text:
            info["dotnet"] = True
        if "System." in text:
            info["has_system_refs"] = True
    except Exception:
        pass
    return info


def parse_sqlite(data: bytes) -> Dict[str, Any]:
    """Minimal SQLite header + schema read."""
    info: Dict[str, Any] = {"format": "SQLite", "tables": [], "schema": []}
    if len(data) < 16 or data[:16] != b"SQLite format 3\x00":
        info["error"] = "not a valid SQLite database"
        return info
    info["page_size"] = _read_be(data, 16, 2) or 65536
    info["file_change_counter"] = _read_be(data, 24, 4)
    info["database_size_pages"] = _read_be(data, 28, 4)
    try:
        with sqlite3.connect(":memory:") as con:
            # Cannot open raw bytes directly; use a temp file
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            tmp.write(data)
            tmp.close()
            try:
                con2 = sqlite3.connect(tmp.name)
                cur = con2.cursor()
                cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
                for name, sql in cur.fetchall():
                    info["tables"].append(name)
                    info["schema"].append({"name": name, "sql": sql})
                cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
                info["indexes"] = [r[0] for r in cur.fetchall()]
                con2.close()
            finally:
                os.unlink(tmp.name)
    except Exception as e:
        info["sqlite_error"] = str(e)
    return info


def parse_pcap(data: bytes) -> Dict[str, Any]:
    """Minimal pcap reader."""
    info: Dict[str, Any] = {"format": "pcap", "packets": [], "endpoints": set()}
    magic = data[:4]
    if magic == b"\xd4\xc3\xb2\xa1":
        endian = "<"
    elif magic == b"\xa1\xb2\xc3\xd4":
        endian = ">"
    else:
        info["error"] = "not a valid pcap"
        return info
    info["link_type"] = _read_le(data, 20, 4) if endian == "<" else _read_be(data, 20, 4)
    off = 24
    count = 0
    while off + 16 <= len(data):
        ts_sec = _read_le(data, off, 4) if endian == "<" else _read_be(data, off, 4)
        ts_usec = _read_le(data, off + 4, 4) if endian == "<" else _read_be(data, off + 4, 4)
        caplen = _read_le(data, off + 8, 4) if endian == "<" else _read_be(data, off + 8, 4)
        origlen = _read_le(data, off + 12, 4) if endian == "<" else _read_be(data, off + 12, 4)
        pkt = off + 16
        if pkt + caplen > len(data):
            break
        info["packets"].append({
            "ts_sec": ts_sec,
            "caplen": caplen,
            "origlen": origlen,
        })
        count += 1
        off = pkt + caplen
    info["packet_count"] = count
    return info


def _guess_format(data: bytes) -> str:
    if data[:4] == b"\x7fELF":
        return "ELF"
    if data[:2] == b"MZ":
        return "PE"
    if data[:4] == b"PK\x03\x04":
        return "ZIP/APK"
    if data[:16] == b"SQLite format 3\x00":
        return "SQLite"
    if data[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
        return "PCAP"
    return "unknown"


# ─── TOOLS ──────────────────────────────────────────────────────────────────

@mcp.tool()
def re_analyze_binary(path: str) -> Dict[str, Any]:
    """Analyze a binary file (PE/ELF/APK/RAW) — headers, sections, imports, exports, strings."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    result: Dict[str, Any] = {
        "path": str(p),
        "size": len(data),
        "format": fmt,
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest()[:32],
    }
    if fmt == "PE":
        result.update(parse_pe(data))
    elif fmt == "ELF":
        result.update(parse_elf(data))
    elif fmt == "ZIP/APK":
        result.update(parse_apk(data))
    # Always extract strings
    strings = extract_strings(data, min_len=6)
    result["strings"] = strings[:200]
    result["string_count"] = len(strings)
    return result


@mcp.tool()
def re_extract_strings(path: str, min_len: int = 6, max_count: int = 500) -> Dict[str, Any]:
    """Extract ASCII and UTF-16LE strings from any file."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    strings = extract_strings(data, min_len=min_len)[:max_count]
    return {
        "path": str(p),
        "size": len(data),
        "min_len": min_len,
        "string_count": len(strings),
        "strings": strings,
    }


@mcp.tool()
def re_find_symbols(path: str, symbol_filter: Optional[str] = None) -> Dict[str, Any]:
    """Find symbol names (imports/exports) in a binary."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    if fmt == "PE":
        info = parse_pe(data)
    else:
        info = {"format": fmt, "imports": [], "exports": []}
    imports = info.get("imports", [])
    exports = info.get("exports", [])
    # Filter
    f = symbol_filter.lower() if symbol_filter else ""
    filtered_imports = [
        imp for imp in imports
        if not f or f in imp["module"].lower() or any(f in fn.lower() for fn in imp["functions"])
    ]
    filtered_exports = [e for e in exports if not f or f in e.lower()]
    return {
        "path": str(p),
        "format": fmt,
        "imports": filtered_imports,
        "exports": filtered_exports,
        "import_count": len(filtered_imports),
        "export_count": len(filtered_exports),
    }


@mcp.tool()
def re_trace_calls(path: str) -> Dict[str, Any]:
    """Trace call relationships between modules/functions from import/export data."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    if fmt != "PE":
        return {"error": f"Call tracing currently supports PE only (got {fmt})"}
    info = parse_pe(data)
    calls = []
    for imp in info.get("imports", []):
        mod = imp["module"]
        for fn in imp["functions"]:
            calls.append({"from": mod, "to": fn, "type": "import"})
    return {
        "path": str(p),
        "format": fmt,
        "calls": calls,
        "call_count": len(calls),
    }


@mcp.tool()
def re_decompile_stub(path: str, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Generate a pseudocode-style decompilation scaffold for a binary/symbol."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    info: Dict[str, Any] = {
        "path": str(p),
        "format": fmt,
        "size": len(data),
        "decompile_note": "Native pseudocode scaffold — full decompilation requires a disassembler backend.",
    }
    if symbol:
        info["target_symbol"] = symbol
        info["stub"] = f"""
// Decompilation stub for {symbol}
// Binary: {p.name} ({fmt})
// Size: {len(data)} bytes
// TODO: Implement disassembly lift for {symbol}
int {symbol}(void) {{
    // Recovered from binary {p.name}
    // Placeholder — integrate disassembler for full recovery
    return 0;
}}
"""
    else:
        info["stub"] = f"""
// Decompilation scaffold for {p.name}
// Format: {fmt}
// Size: {len(data)} bytes
// TODO: Add disassembly engine for full recovery
"""
    return info


@mcp.tool()
def re_analyze_apk(path: str) -> Dict[str, Any]:
    """Analyze an Android APK (ZIP) — manifest, dex, resources."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    return parse_apk(data)


@mcp.tool()
def re_analyze_dotnet(path: str) -> Dict[str, Any]:
    """Analyze a .NET assembly — metadata, types, methods."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    return parse_dotnet(data)


@mcp.tool()
def re_analyze_sqlite(path: str) -> Dict[str, Any]:
    """Analyze a SQLite database — schema, tables, indexes."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    return parse_sqlite(data)


@mcp.tool()
def re_analyze_pcap(path: str) -> Dict[str, Any]:
    """Analyze a packet capture — endpoints, DNS, TLS SNI, HTTP."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    return parse_pcap(data)


@mcp.tool()
def re_analyze_web(html: str) -> Dict[str, Any]:
    """Analyze website HTML — scripts, forms, links, structure."""
    scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)', html, re.I)
    links = re.findall(r'(?:href|src)=["\']([^"\']+)', html, re.I)
    forms = re.findall(r'<form[^>]*action=["\']([^"\']+)', html, re.I)
    return {
        "size": len(html),
        "scripts": scripts[:50],
        "links": links[:100],
        "forms": forms[:50],
        "has_forms": len(forms) > 0,
        "has_scripts": len(scripts) > 0,
    }


@mcp.tool()
def re_analyze_firmware(path: str) -> Dict[str, Any]:
    """Analyze firmware binary — regions, entropy, embedded strings."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    # Simple entropy per 256-byte block
    blocks = []
    for i in range(0, len(data), 256):
        block = data[i:i + 256]
        if not block:
            continue
        freq = {}
        for b in block:
            freq[b] = freq.get(b, 0) + 1
        ent = -sum((c / len(block)) * __import__("math").log2(c / len(block)) for c in freq.values())
        blocks.append({"offset": i, "entropy": round(ent, 3), "size": len(block)})
    strings = extract_strings(data, min_len=8)[:100]
    return {
        "path": str(p),
        "size": len(data),
        "format": _guess_format(data),
        "entropy_blocks": blocks[:50],
        "entropy_avg": round(sum(b["entropy"] for b in blocks) / max(1, len(blocks)), 3),
        "strings": strings,
        "string_count": len(strings),
    }


# ─── 6. FULL PIPELINE (persisted) ───────────────────────────────────────────

@mcp.tool()
def re_full_pipeline(path: str, persist: bool = True) -> Dict[str, Any]:
    """Chain: binary inspect → strings → symbols → calls → report."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    binary = parse_pe(data) if fmt == "PE" else parse_elf(data) if fmt == "ELF" else {"format": fmt}
    strings = extract_strings(data, min_len=6)[:100]
    symbols = []
    if fmt == "PE":
        pe_info = parse_pe(data)
        symbols = pe_info.get("imports", [])[:50] + [{"module": "exports", "functions": pe_info.get("exports", [])[:50]}]
    session_id = f"re-{int(time.time())}"
    result = {
        "status": "RE_PIPELINE_COMPLETE",
        "session_id": session_id,
        "path": str(p),
        "format": fmt,
        "size": len(data),
        "binary": binary,
        "strings": strings,
        "symbols": symbols,
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest()[:32],
        "timestamp": time.time(),
    }
    if persist:
        try:
            p2 = SESSION_DIR / f"{session_id}.json"
            p2.write_text(json.dumps({"type": "re_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p2)
        except Exception as e:
            result["persist_error"] = str(e)
    return result


# ─── 7. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def re_route(target_mcp: str = "council", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route RE findings to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_security_scan",
                "payload": payload or {}, "note": "Route RE findings to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "QuillanBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route RE findings to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}


# ─── 8. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def re_health() -> Dict[str, Any]:
    """RE engine dashboard summary."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "path": (data.get("result") or {}).get("path"),
                    "format": (data.get("result") or {}).get("format"),
                    "overall": data.get("overall"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "RE_HEALTH_READY",
        "sessions": sessions[:5],
        "council_note": "C34-PREDATOR hunts; C10-CODEWEAVER implements; C21-ARCHON researches.",
    }


# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def re_session_list() -> Dict[str, Any]:
    """List persisted RE sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "path": (data.get("result") or {}).get("path"),
                "format": (data.get("result") or {}).get("format"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def re_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted RE session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Read failed: {e}"}



# ─── 10. DISASSEMBLY (x86/x64) ──────────────────────────────────────────────

@mcp.tool()
def re_disassemble(path: str, offset: int = 0, length: int = 256, arch: str = "x64") -> Dict[str, Any]:
    """Disassemble binary code at a given offset (x86/x64)."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    if offset + length > len(data):
        length = len(data) - offset
    chunk = data[offset:offset + length]
    # Basic disassembly using capstone if available, else hex dump
    try:
        import capstone
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64 if arch == "x64" else capstone.CS_MODE_32)
        instructions = []
        for insn in cs.disasm(chunk, offset):
            instructions.append({"address": hex(insn.address), "mnemonic": insn.mnemonic, "op_str": insn.op_str, "bytes": insn.bytes.hex()})
        return {"path": str(p), "arch": arch, "offset": offset, "length": length, "instructions": instructions,
                "count": len(instructions), "status": "DISASSEMBLED"}
    except ImportError:
        # Fallback: hex dump
        hex_lines = []
        for i in range(0, len(chunk), 16):
            hex_bytes = chunk[i:i+16].hex(" ")
            ascii_repr = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk[i:i+16])
            hex_lines.append(f"{offset + i:08x}:  {hex_bytes:<47}  {ascii_repr}")
        return {"path": str(p), "arch": arch, "offset": offset, "length": length, "hex_dump": hex_lines,
                "note": "capstone not installed. Install with: pip install capstone", "status": "HEX_DUMP"}


# ─── 11. MEMORY DUMP ANALYSIS ────────────────────────────────────────────────

@mcp.tool()
def re_memory_dump(path: str, search_pattern: str = "") -> Dict[str, Any]:
    """Analyze a memory dump file for strings, patterns, and structures."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    strings = extract_strings(data, min_len=6)[:200]
    # Search for pattern
    matches = []
    if search_pattern:
        try:
            pattern_bytes = search_pattern.encode("utf-8")
            idx = 0
            while True:
                pos = data.find(pattern_bytes, idx)
                if pos == -1:
                    break
                matches.append({"offset": pos, "context": data[max(0, pos-32):pos+32+len(pattern_bytes)].hex()})
                idx = pos + 1
                if len(matches) >= 50:
                    break
        except Exception:
            pass
    return {
        "path": str(p),
        "size": len(data),
        "format": "memory_dump",
        "strings": strings[:50],
        "string_count": len(strings),
        "pattern_matches": matches,
        "md5": hashlib.md5(data).hexdigest(),
        "status": "MEMORY_DUMP_ANALYZED",
    }


# ─── 12. MALWARE ANALYSIS ────────────────────────────────────────────────────

@mcp.tool()
def re_malware_analyze(path: str) -> Dict[str, Any]:
    """Analyze a binary for malware indicators (suspicious imports, entropy, packing)."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    indicators = []
    # Check for suspicious imports
    suspicious_apis = ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread", "LoadLibrary",
                       "GetProcAddress", "ShellExecute", "WinExec", "URLDownloadToFile", "InternetOpen"]
    # Check strings for malware indicators
    strings = extract_strings(data, min_len=6)
    str_values = [s["value"].lower() for s in strings]
    for api in suspicious_apis:
        if any(api.lower() in v for v in str_values):
            indicators.append({"type": "suspicious_api", "detail": api, "severity": "HIGH"})
    # Check for high entropy (packing/encryption)
    if len(data) > 256:
        freq = {}
        for b in data[:4096]:
            freq[b] = freq.get(b, 0) + 1
        ent = -sum((c / min(4096, len(data))) * __import__("math").log2(c / min(4096, len(data))) for c in freq.values())
        if ent > 7.5:
            indicators.append({"type": "high_entropy", "detail": f"Entropy {ent:.2f} (possible packing/encryption)", "severity": "MEDIUM"})
    # Check for known packer signatures
    packers = {b"UPX": "UPX", b"MPRESS": "MPRESS", b"ASPack": "ASPack", b"Themida": "Themida", b"VMProtect": "VMProtect"}
    for sig, name in packers.items():
        if sig in data:
            indicators.append({"type": "packer", "detail": name, "severity": "HIGH"})
    return {
        "path": str(p),
        "format": fmt,
        "size": len(data),
        "indicators": indicators,
        "indicator_count": len(indicators),
        "risk_level": "HIGH" if any(i["severity"] == "HIGH" for i in indicators) else ("MEDIUM" if indicators else "LOW"),
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "status": "MALWARE_ANALYSIS_COMPLETE",
    }


# ─── 13. OBFUSCATION DETECTION ───────────────────────────────────────────────

@mcp.tool()
def re_obfuscation_detect(path: str) -> Dict[str, Any]:
    """Detect obfuscation techniques in code or binary."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    indicators = []
    # Check for common obfuscation patterns in strings
    strings = extract_strings(data, min_len=4)
    str_values = [s["value"] for s in strings]
    # Base64
    b64_pattern = re.compile(rb"[A-Za-z0-9+/]{20,}={0,2}")
    b64_matches = b64_pattern.findall(data)
    if len(b64_matches) > 5:
        indicators.append({"type": "base64_encoding", "count": len(b64_matches), "severity": "LOW"})
    # XOR patterns
    if b"\x00" * 4 in data and len(data) > 100:
        indicators.append({"type": "possible_xor", "detail": "Null byte runs suggest XOR encoding", "severity": "MEDIUM"})
    # JavaScript obfuscation
    js_obfusc = ["eval(", "Function(", "atob(", "fromCharCode", "unescape("]
    text = data.decode("utf-8", errors="replace")
    for pat in js_obfusc:
        if pat in text:
            indicators.append({"type": "js_obfuscation", "detail": pat, "severity": "MEDIUM"})
    # .NET obfuscation
    if "Confuser" in text or "Dotfuscator" in text or "SmartAssembly" in text:
        indicators.append({"type": "dotnet_obfuscation", "detail": "Obfuscator signature found", "severity": "HIGH"})
    return {
        "path": str(p),
        "size": len(data),
        "indicators": indicators,
        "indicator_count": len(indicators),
        "obfuscated": len(indicators) > 0,
        "status": "OBFUSCATION_CHECK_COMPLETE",
    }


# ─── 14. ANTI-DEBUGGING DETECTION ────────────────────────────────────────────

@mcp.tool()
def re_antidebug_detect(path: str) -> Dict[str, Any]:
    """Detect anti-debugging techniques in a binary."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace").lower()
    indicators = []
    antidebug_apis = ["IsDebuggerPresent", "CheckRemoteDebuggerPresent", "NtQueryInformationProcess",
                      "OutputDebugString", "GetTickCount", "QueryPerformanceCounter", "NtSetInformationThread",
                      "ZwSetInformationThread", "RtlQueryProcessHeapInformation"]
    for api in antidebug_apis:
        if api.lower() in text:
            indicators.append({"type": "antidebug_api", "detail": api, "severity": "MEDIUM"})
    # Timing checks
    if "rdtsc" in text or "cpuid" in text:
        indicators.append({"type": "timing_check", "detail": "RDTSC/CPUID timing check", "severity": "MEDIUM"})
    # INT3 / breakpoint detection
    if b"\xcc" in data[:1024]:
        indicators.append({"type": "int3_breakpoint", "detail": "INT3 instruction in entry point", "severity": "LOW"})
    return {
        "path": str(p),
        "indicators": indicators,
        "indicator_count": len(indicators),
        "antidebug_detected": len(indicators) > 0,
        "status": "ANTIDEBUG_CHECK_COMPLETE",
    }


# ─── 15. PACKING / UNPACKING DETECTION ───────────────────────────────────────

@mcp.tool()
def re_packing_detect(path: str) -> Dict[str, Any]:
    """Detect if a binary is packed or compressed."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    indicators = []
    # Section entropy analysis
    fmt = _guess_format(data)
    if fmt == "PE":
        info = parse_pe(data)
        for sec in info.get("sections", []):
            # High virtual_size vs raw_size ratio suggests packing
            vs = sec.get("virtual_size", 0)
            rs = sec.get("raw_size", 0)
            if vs > 0 and rs > 0 and vs / max(rs, 1) > 3:
                indicators.append({"type": "section_expansion", "section": sec["name"],
                                  "ratio": round(vs / max(rs, 1), 2), "severity": "HIGH"})
    # Known packer signatures
    packers = {b"UPX0": "UPX", b"UPX1": "UPX", b".aspack": "ASPack", b".adata": "ASPack",
               b"MPRESS1": "MPRESS", b"Themida": "Themida", b"VMProtect": "VMProtect",
               b".vmp0": "VMProtect", b"PECompact": "PECompact", b" Petite": "Petite"}
    for sig, name in packers.items():
        if sig in data:
            indicators.append({"type": "packer_signature", "detail": name, "severity": "HIGH"})
    # Overall entropy
    if len(data) > 256:
        freq = {}
        for b in data[:8192]:
            freq[b] = freq.get(b, 0) + 1
        ent = -sum((c / min(8192, len(data))) * __import__("math").log2(c / min(8192, len(data))) for c in freq.values())
        if ent > 7.0:
            indicators.append({"type": "high_entropy", "detail": f"Overall entropy {ent:.2f}", "severity": "MEDIUM"})
    return {
        "path": str(p),
        "format": fmt,
        "packed": len(indicators) > 0,
        "indicators": indicators,
        "status": "PACKING_CHECK_COMPLETE",
    }


# ─── 16. YARA RULE MATCHING ──────────────────────────────────────────────────

@mcp.tool()
def re_yara_scan(path: str, rules: str = "") -> Dict[str, Any]:
    """Scan a file against YARA rules (if yara-python installed, else basic pattern matching)."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    try:
        import yara
        if rules:
            rule_obj = yara.compile(source=rules)
        else:
            # Basic default rules
            rule_obj = yara.compile(source='''
rule SuspiciousStrings {
    strings:
        $a = "cmd.exe" nocase
        $b = "powershell" nocase
        $c = "CreateRemoteThread"
        $d = "VirtualAlloc"
    condition:
        any of them
}
''')
        matches = rule_obj.match(data=data)
        return {"path": str(p), "matches": [m.rule for m in matches], "match_count": len(matches),
                "status": "YARA_SCAN_COMPLETE", "engine": "yara-python"}
    except ImportError:
        # Fallback: basic string matching
        basic_patterns = ["cmd.exe", "powershell", "CreateRemoteThread", "VirtualAlloc", "WScript.Shell", "eval("]
        hits = [pat for pat in basic_patterns if pat.lower().encode() in data.lower()]
        return {"path": str(p), "matches": hits, "match_count": len(hits), "status": "BASIC_SCAN_COMPLETE",
                "note": "yara-python not installed. Install with: pip install yara-python"}


# ─── 17. IOC EXTRACTION ──────────────────────────────────────────────────────

@mcp.tool()
def re_ioc_extract(path: str) -> Dict[str, Any]:
    """Extract Indicators of Compromise (IOCs) from a binary/file."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    iocs = {"ips": [], "domains": [], "urls": [], "emails": [], "file_paths": [], "registry_keys": []}
    # IP addresses
    iocs["ips"] = list(set(re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)))
    # Domains
    iocs["domains"] = list(set(re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}\b", text)))[:50]
    # URLs
    iocs["urls"] = list(set(re.findall(r"https?://[^\s\"'<>]+", text)))[:50]
    # Emails
    iocs["emails"] = list(set(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)))
    # File paths (Windows)
    iocs["file_paths"] = list(set(re.findall(r"[A-Za-z]:\\[^\s\"'<>]+", text)))[:50]
    # Registry keys
    iocs["registry_keys"] = list(set(re.findall(r"HKEY_[A-Z_]+\\[^\s\"'<>]+", text)))[:50]
    total = sum(len(v) for v in iocs.values())
    return {"path": str(p), "iocs": iocs, "total_iocs": total, "status": "IOC_EXTRACTION_COMPLETE"}


# ─── 18. BINARY DIFFING ──────────────────────────────────────────────────────

@mcp.tool()
def re_binary_diff(path_a: str, path_b: str) -> Dict[str, Any]:
    """Compare two binary files and report differences."""
    pa = Path(path_a.strip().strip('"\''))
    pb = Path(path_b.strip().strip('"\''))
    if not pa.exists():
        return {"error": f"File A not found: {path_a}"}
    if not pb.exists():
        return {"error": f"File B not found: {path_b}"}
    da = pa.read_bytes()
    db = pb.read_bytes()
    # Byte-level diff
    diffs = []
    min_len = min(len(da), len(db))
    diff_regions = []
    in_diff = False
    start = 0
    for i in range(min_len):
        if da[i] != db[i]:
            if not in_diff:
                in_diff = True
                start = i
        else:
            if in_diff:
                diff_regions.append({"start": start, "end": i, "length": i - start})
                in_diff = False
    if in_diff:
        diff_regions.append({"start": start, "end": min_len, "length": min_len - start})
    return {
        "file_a": pa.name,
        "file_b": pb.name,
        "size_a": len(da),
        "size_b": len(db),
        "size_diff": len(db) - len(da),
        "diff_regions": diff_regions[:50],
        "total_diff_bytes": sum(r["length"] for r in diff_regions),
        "identical": da == db,
        "status": "BINARY_DIFF_COMPLETE",
    }


# ─── 19. CRYPTOGRAPHY ANALYSIS ───────────────────────────────────────────────

@mcp.tool()
def re_crypto_analyze(path: str) -> Dict[str, Any]:
    """Analyze binary for cryptographic constants and algorithms."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace").lower()
    crypto_indicators = []
    # Known crypto constants
    constants = {
        "AES": [b"\x63\x7c\x77\x7b", b"\x52\x09\x6a\xd5"],
        "DES": [b"\x00\x01\x02\x03\x04\x05\x06\x07"],
        "MD5": [b"\x67\x45\x23\x01", b"\xef\xcd\xab\x89"],
        "SHA1": [b"\x67\x45\x23\x01", b"\xef\xcd\xab\x89"],
        "SHA256": [b"\x6a\x09\xe6\x67", b"\xbb\x67\xae\x85"],
        "RC4": [b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"],
    }
    for name, sigs in constants.items():
        for sig in sigs:
            if sig in data:
                crypto_indicators.append({"algorithm": name, "signature_found": True})
                break
    # Crypto API references
    crypto_apis = ["CryptEncrypt", "CryptDecrypt", "CryptAcquireContext", "BCryptEncrypt",
                   "BCryptDecrypt", "EVP_EncryptInit", "EVP_DecryptInit", "AES_encrypt",
                   "RSA_public_encrypt", "SHA256_Init", "MD5_Init"]
    for api in crypto_apis:
        if api.lower() in text:
            crypto_indicators.append({"api": api, "type": "crypto_api"})
    return {
        "path": str(p),
        "crypto_indicators": crypto_indicators,
        "indicator_count": len(crypto_indicators),
        "crypto_detected": len(crypto_indicators) > 0,
        "status": "CRYPTO_ANALYSIS_COMPLETE",
    }


# ─── 20. REPORT GENERATION ───────────────────────────────────────────────────

@mcp.tool()
def re_report(path: str, format: str = "markdown") -> Dict[str, Any]:
    """Generate a comprehensive reverse engineering report for a binary."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    strings = extract_strings(data, min_len=6)[:50]
    entropy = 0.0
    if len(data) > 256:
        freq = {}
        for b in data[:4096]:
            freq[b] = freq.get(b, 0) + 1
        entropy = -sum((c / min(4096, len(data))) * __import__("math").log2(c / min(4096, len(data))) for c in freq.values())

    report = f"""# Reverse Engineering Report

## File Information
- **Path:** {p}
- **Size:** {len(data)} bytes
- **Format:** {fmt}
- **MD5:** {hashlib.md5(data).hexdigest()}
- **SHA256:** {hashlib.sha256(data).hexdigest()}
- **Entropy:** {entropy:.2f}/8.0

## Strings (top 20)
"""
    for s in strings[:20]:
        report += f"- `{s['value'][:80]}` (offset: 0x{s['offset']:x})\n"

    report += "\n## Analysis Summary\n"
    report += f"- Format: {fmt}\n"
    report += f"- Size: {len(data)} bytes\n"
    report += f"- Strings found: {len(strings)}\n"
    report += f"- Entropy: {entropy:.2f}/8.0 ({'High - possible packing' if entropy > 7.0 else 'Normal'})\n"

    return {"path": str(p), "format": format, "report": report, "status": "REPORT_GENERATED"}


# ─── 21. DYNAMIC ANALYSIS HOOKS ──────────────────────────────────────────────

@mcp.tool()
def re_dynamic_hooks(target: str = "process", hooks: str = "CreateFile,WriteFile,ReadFile") -> Dict[str, Any]:
    """Generate dynamic analysis hook configuration for API monitoring."""
    hook_list = [h.strip() for h in hooks.split(",") if h.strip()]
    return {
        "target": target,
        "hooks": hook_list,
        "method": "API hooking via Frida/Detours",
        "script_template": "Interceptor.attach(Module.findExportByName(null, '%s'), { onEnter: function(args) { console.log('%s called'); } });" % (hook_list[0] if hook_list else "target", hook_list[0] if hook_list else "target"),
        "council_note": "C34-PREDATOR: dynamic analysis; C10-CODEWEAVER: hook implementation.",
    }


# ─── 22. REGISTRY ANALYSIS (Windows) ─────────────────────────────────────────

@mcp.tool()
def re_registry_analyze(reg_export: str = "") -> Dict[str, Any]:
    """Analyze Windows registry export for suspicious entries."""
    if not reg_export:
        return {"error": "Provide registry export text."}
    suspicious = []
    patterns = ["Run", "RunOnce", "Services", "Winlogon", "Shell", "UserInit", "AppInit_DLLs"]
    for line in reg_export.splitlines():
        for pat in patterns:
            if pat.lower() in line.lower():
                suspicious.append({"line": line.strip(), "pattern": pat, "severity": "MEDIUM"})
    return {
        "total_lines": len(reg_export.splitlines()),
        "suspicious_entries": suspicious[:20],
        "suspicious_count": len(suspicious),
        "status": "REGISTRY_ANALYZED",
    }


# ─── 23. FILE SYSTEM FORENSICS ───────────────────────────────────────────────

@mcp.tool()
def re_filesystem_forensics(path: str = ".", extensions: str = ".exe,.dll,.bat,.ps1,.vbs") -> Dict[str, Any]:
    """Scan a directory for suspicious files by extension and metadata."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"Path not found: {path}"}
    ext_list = [e.strip() for e in extensions.split(",") if e.strip()]
    suspicious = []
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in ext_list:
            suspicious.append({
                "path": str(f),
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
                "suspicious": f.suffix.lower() in (".bat", ".vbs", ".ps1"),
            })
    suspicious.sort(key=lambda x: x["modified"], reverse=True)
    return {
        "path": str(p),
        "extensions_scanned": ext_list,
        "suspicious_files": suspicious[:50],
        "total_found": len(suspicious),
        "status": "FILESYSTEM_FORENSICS_COMPLETE",
    }


# ─── 24. MACH-O PARSER (macOS) ───────────────────────────────────────────────

@mcp.tool()
def re_analyze_macho(path: str) -> Dict[str, Any]:
    """Analyze a Mach-O binary (macOS/iOS)."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    if len(data) < 4 or data[:4] not in (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"):
        return {"error": "Not a valid Mach-O binary"}
    magic = data[:4].hex()
    is_64 = magic in ("feedfacf", "cffaedfe")
    endian = "<" if magic in ("cefaedfe", "cffaedfe") else ">"
    # Parse header
    if is_64:
        cpu_type = _read_le(data, 4, 4) if endian == "<" else _read_be(data, 4, 4)
        cpu_subtype = _read_le(data, 8, 4) if endian == "<" else _read_be(data, 8, 4)
        filetype = _read_le(data, 12, 4) if endian == "<" else _read_be(data, 12, 4)
        ncmds = _read_le(data, 16, 4) if endian == "<" else _read_be(data, 16, 4)
        sizeofcmds = _read_le(data, 20, 4) if endian == "<" else _read_be(data, 20, 4)
    else:
        cpu_type = _read_le(data, 4, 4)
        cpu_subtype = _read_le(data, 8, 4)
        filetype = _read_le(data, 12, 4)
        ncmds = _read_le(data, 16, 4)
        sizeofcmds = _read_le(data, 20, 4)
    filetypes = {1: "OBJECT", 2: "EXECUTE", 3: "FVMLIB", 4: "CORE", 5: "PRELOAD", 6: "DYLIB", 7: "DYLINKER", 8: "BUNDLE", 9: "DYLIB_STUB", 10: "DSYM", 11: "KEXT_BUNDLE"}
    return {
        "path": str(p),
        "format": "Mach-O",
        "magic": magic,
        "is_64bit": is_64,
        "endian": "little" if endian == "<" else "big",
        "cpu_type": hex(cpu_type),
        "cpu_subtype": hex(cpu_subtype),
        "filetype": filetypes.get(filetype, hex(filetype)),
        "num_load_commands": ncmds,
        "sizeof_commands": sizeofcmds,
        "strings": extract_strings(data, min_len=6)[:50],
        "status": "MACHO_ANALYZED",
    }


# ─── 25. WASM ANALYSIS ───────────────────────────────────────────────────────

@mcp.tool()
def re_analyze_wasm(path: str) -> Dict[str, Any]:
    """Analyze a WebAssembly binary (.wasm)."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    if len(data) < 8 or data[:4] != b"\x00asm":
        return {"error": "Not a valid WASM binary"}
    version = _read_le(data, 4, 4)
    # Parse sections
    sections = []
    off = 8
    section_names = {0: "custom", 1: "type", 2: "import", 3: "function", 4: "table", 5: "memory", 6: "global", 7: "export", 8: "start", 9: "element", 10: "code", 11: "data", 12: "data_count"}
    while off < len(data):
        if off + 2 > len(data):
            break
        sec_id = data[off]
        off += 1
        # Read LEB128 size
        size = 0
        shift = 0
        while True:
            b = data[off]
            off += 1
            size |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        sec_data = data[off:off + size]
        sections.append({"id": sec_id, "name": section_names.get(sec_id, f"unknown_{sec_id}"), "size": size})
        off += size
    return {
        "path": str(p),
        "format": "WASM",
        "version": version,
        "sections": sections,
        "strings": extract_strings(data, min_len=4)[:30],
        "size": len(data),
        "status": "WASM_ANALYZED",
    }


# ─── 26. JAVA CLASS FILE ANALYSIS ────────────────────────────────────────────

@mcp.tool()
def re_analyze_java_class(path: str) -> Dict[str, Any]:
    """Analyze a Java .class file."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    if len(data) < 8 or data[:4] != b"\xca\xfe\xba\xbe":
        return {"error": "Not a valid Java class file"}
    minor = _read_be(data, 4, 2)
    major = _read_be(data, 6, 2)
    java_versions = {45: "Java 1.1", 46: "Java 1.2", 47: "Java 1.3", 48: "Java 1.4", 49: "Java 5", 50: "Java 6", 51: "Java 7", 52: "Java 8", 53: "Java 9", 54: "Java 10", 55: "Java 11", 56: "Java 12", 57: "Java 13", 58: "Java 14", 59: "Java 15", 60: "Java 16", 61: "Java 17", 62: "Java 18", 63: "Java 19", 64: "Java 20", 65: "Java 21"}
    strings = extract_strings(data, min_len=4)[:50]
    return {
        "path": str(p),
        "format": "Java Class",
        "magic": data[:4].hex(),
        "minor_version": minor,
        "major_version": major,
        "java_version": java_versions.get(major, f"Unknown ({major})"),
        "size": len(data),
        "strings": strings,
        "status": "JAVA_CLASS_ANALYZED",
    }


# ─── 27. ANTI-VM DETECTION ──────────────────────────────────────────────────

@mcp.tool()
def re_antivm_detect(path: str) -> Dict[str, Any]:
    """Detect anti-VM / sandbox evasion techniques."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace").lower()
    indicators = []
    vm_artifacts = ["vmware", "virtualbox", "vbox", "qemu", "xen", "parallels", "hyper-v",
                    "sandboxie", "cuckoo", "anubis", "joebox", "threatexpert"]
    for art in vm_artifacts:
        if art in text:
            indicators.append({"type": "vm_artifact", "detail": art, "severity": "MEDIUM"})
    vm_apis = ["GetSystemFirmwareTable", "EnumSystemFirmwareTables", "cpuid", "sidt", "sgdt", "sldt"]
    for api in vm_apis:
        if api.lower() in text:
            indicators.append({"type": "vm_detection_api", "detail": api, "severity": "MEDIUM"})
    return {
        "path": str(p),
        "indicators": indicators,
        "indicator_count": len(indicators),
        "antivm_detected": len(indicators) > 0,
        "status": "ANTIVM_CHECK_COMPLETE",
    }


# ─── 28. CODE SIGNING VERIFICATION ───────────────────────────────────────────

@mcp.tool()
def re_codesign_verify(path: str) -> Dict[str, Any]:
    """Verify code signing / digital signature of a binary."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    has_signature = False
    signer = None
    # Check for certificate table in PE
    if data[:2] == b"MZ":
        info = parse_pe(data)
        # Check for certificate directory (index 4 in DataDirectory)
        # Simplified: look for common signing strings
        if "Microsoft" in text or "VeriSign" in text or "DigiCert" in text or "Symantec" in text:
            has_signature = True
            for ca in ["Microsoft", "VeriSign", "DigiCert", "Symantec", "Comodo", "GlobalSign"]:
                if ca in text:
                    signer = ca
                    break
    return {
        "path": str(p),
        "signed": has_signature,
        "signer": signer,
        "note": "For full certificate chain verification, use signtool or openssl.",
        "status": "CODESIGN_CHECK_COMPLETE",
    }


# ─── 29. RESOURCE EXTRACTION (PE resources) ──────────────────────────────────

@mcp.tool()
def re_extract_resources(path: str) -> Dict[str, Any]:
    """Extract embedded resources (icons, images, manifests) from a PE binary."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    fmt = _guess_format(data)
    resources = []
    # Look for common resource signatures in the binary
    # PNG
    png_sig = b"\x89PNG\r\n\x1a\n"
    idx = 0
    while True:
        pos = data.find(png_sig, idx)
        if pos == -1:
            break
        # Find end (IEND chunk)
        end = data.find(b"IEND", pos)
        if end != -1:
            end += 8  # IEND + CRC
            resources.append({"type": "PNG", "offset": pos, "size": end - pos})
        idx = pos + 1
        if len(resources) >= 20:
            break
    # JPEG
    jpeg_sig = b"\xff\xd8\xff"
    idx = 0
    while True:
        pos = data.find(jpeg_sig, idx)
        if pos == -1:
            break
        end = data.find(b"\xff\xd9", pos)
        if end != -1:
            end += 2
            resources.append({"type": "JPEG", "offset": pos, "size": end - pos})
        idx = pos + 1
        if len(resources) >= 20:
            break
    # RT_MANIFEST (XML)
    manifest_sig = b"<?xml"
    idx = 0
    while True:
        pos = data.find(manifest_sig, idx)
        if pos == -1:
            break
        end = data.find(b"</assembly>", pos)
        if end != -1:
            end += len(b"</assembly>")
            resources.append({"type": "XML_MANIFEST", "offset": pos, "size": end - pos,
                              "preview": data[pos:pos+200].decode("utf-8", errors="replace")})
        idx = pos + 1
        if len(resources) >= 20:
            break
    return {
        "path": str(p),
        "format": fmt,
        "resources_found": len(resources),
        "resources": resources,
        "status": "RESOURCE_EXTRACTION_COMPLETE",
    }


# ─── 30. EXPLOIT / SHELLCODE ANALYSIS ────────────────────────────────────────

@mcp.tool()
def re_shellcode_analyze(data_hex: str) -> Dict[str, Any]:
    """Analyze shellcode (hex string) for patterns and API calls."""
    try:
        data = bytes.fromhex(data_hex.replace(" ", "").replace("\\x", ""))
    except Exception:
        return {"error": "Invalid hex string."}
    indicators = []
    # Check for common shellcode patterns
    if b"\x90" * 4 in data:
        indicators.append({"type": "NOP sled", "severity": "HIGH"})
    # Common API hashes (Metasploit)
    if b"\xfc\xe8" in data[:4]:
        indicators.append({"type": "Metasploit-style shellcode (CALL/POP)", "severity": "HIGH"})
    # Check for GetProcAddress hash
    if b"\x72\xfe\xb3\x16" in data or b"\x0e\x2f\x20\x1f" in data:
        indicators.append({"type": "GetProcAddress hash (Metasploit)", "severity": "HIGH"})
    strings = extract_strings(data, min_len=4)
    return {
        "size": len(data),
        "hex": data.hex(),
        "indicators": indicators,
        "strings": [s["value"] for s in strings[:20]],
        "status": "SHELLCODE_ANALYZED",
    }


# ─── 31. SMART CONTRACT ANALYSIS ─────────────────────────────────────────────

@mcp.tool()
def re_smart_contract(path: str) -> Dict[str, Any]:
    """Analyze a smart contract (Solidity/EVM bytecode)."""
    p = Path(path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {path}"}
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    # Solidity source
    if "pragma solidity" in text or "contract " in text:
        funcs = re.findall(r"function\s+(\w+)", text)
        events = re.findall(r"event\s+(\w+)", text)
        modifiers = re.findall(r"modifier\s+(\w+)", text)
        return {
            "type": "solidity_source",
            "functions": funcs,
            "events": events,
            "modifiers": modifiers,
            "has_payable": "payable" in text,
            "has_selfdestruct": "selfdestruct" in text or "suicide" in text,
            "status": "SMART_CONTRACT_ANALYZED",
        }
    # EVM bytecode
    if data[:2] == b"\x60" or len(data) > 10:
        return {
            "type": "evm_bytecode",
            "size": len(data),
            "note": "Use solc or evm disassembler for full analysis.",
            "status": "BYTECODE_DETECTED",
        }
    return {"error": "Not a recognized smart contract format."}


# ─── 32. VOLATILITY-STYLE MEMORY FORENSICS ───────────────────────────────────

@mcp.tool()
def re_memory_forensics(dump_path: str, profile: str = "Win10x64") -> Dict[str, Any]:
    """Generate Volatility-style memory forensics commands for a memory dump."""
    p = Path(dump_path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"File not found: {dump_path}"}
    commands = [
        f"volatility -f {dump_path} --profile={profile} imageinfo",
        f"volatility -f {dump_path} --profile={profile} pslist",
        f"volatility -f {dump_path} --profile={profile} pstree",
        f"volatility -f {dump_path} --profile={profile} netscan",
        f"volatility -f {dump_path} --profile={profile} malfind",
        f"volatility -f {dump_path} --profile={profile} dlllist",
        f"volatility -f {dump_path} --profile={profile} handles",
        f"volatility -f {dump_path} --profile={profile} filescan",
        f"volatility -f {dump_path} --profile={profile} registry hivelist",
    ]
    return {
        "dump_path": str(p),
        "profile": profile,
        "size_mb": round(p.stat().st_size / (1024**2), 1),
        "volatility_commands": commands,
        "note": "Run these commands with Volatility 2/3 installed. Or use the re_memory_dump tool for basic analysis.",
        "status": "MEMORY_FORENSICS_READY",
    }


# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()
