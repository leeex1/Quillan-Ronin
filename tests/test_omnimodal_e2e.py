#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Omni-Modal Verification Suite (Audio, Video, Image, PDF/Docs)
========================================================================
Runs against real assets in C:\\02_QUILLAN\\05 - Creative Works\\ and
C:\\02_QUILLAN\\10 - Formal Papers\\ without modifying or deleting any media.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(r"C:\02_QUILLAN")
MCP_DIR = REPO_ROOT / "09 - Projects" / "mcp"
VISION_DIR = REPO_ROOT / "09 - Projects" / "projects" / "Quillan-Vision"
SERVICES_VISION_DIR = REPO_ROOT / "09 - Projects" / "projects" / "services" / "vision"
SERVICES_VOICE_DIR = REPO_ROOT / "09 - Projects" / "projects" / "services" / "voice"

for _p in (str(MCP_DIR), str(VISION_DIR), str(SERVICES_VISION_DIR), str(SERVICES_VOICE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ASSETS = {
    "wav": REPO_ROOT / "05 - Creative Works" / "Audio Engineer" / "Audio files" / "notice.wav",
    "mp3": REPO_ROOT / "05 - Creative Works" / "docs" / "audio" / "DJ_STOP_and_SCRATCH_FREE_SOUND_EFFECT_(SFX)_2025.mp3",
    "flac": REPO_ROOT / "05 - Creative Works" / "assets" / "06_Media" / "audio" / "_HIGH_VOLTAGE_RONIN_.flac",
    "m4a": REPO_ROOT / "05 - Creative Works" / "assets" / "06_Media" / "audio" / "Is_JDXX_a_Top_100_Rap_Legend_.m4a",
    "mp4": REPO_ROOT / "05 - Creative Works" / "assets" / "videos" / "rap concept videos" / "Man_on_skyscraper_rooftop_20261013194019.mp4",
    "png": REPO_ROOT / "05 - Creative Works" / "assets" / "main-images" / "image-7.png",
    "jpg": REPO_ROOT / "05 - Creative Works" / "assets" / "concept_art" / "me clean pic.jpg",
    "webp": REPO_ROOT / "05 - Creative Works" / "assets" / "logo.webp",
    "pdf": REPO_ROOT / "10 - Formal Papers" / "Formal Papers" / "Quillan-Ronin-Self-Paper-I-Am-Quillan.pdf",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOmniModalE2E(unittest.TestCase):
    def test_00_assets_present(self):
        for ext, path in ASSETS.items():
            self.assertTrue(path.exists(), f"Missing sample asset for {ext}: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty sample asset: {path}")

    def test_01_audio_mcp_and_engines(self):
        import quillan_audio.server as qa

        health = qa.audio_health()
        self.assertEqual(health["status"], "AUDIO_HEALTH_READY")
        self.assertEqual(health["deps"]["av"], "installed")
        self.assertNotEqual(health["deps"]["ffmpeg"], "missing")

        for ext, transcribe in (("wav", True), ("mp3", True), ("flac", False), ("m4a", False)):
            res = qa.audio_analyze_track(str(ASSETS[ext]), transcribe=transcribe, max_seconds=8.0)
            self.assertEqual(res.get("engine"), "pyav+numpy", f"{ext} failed: {res}")
            self.assertGreater(res.get("sample_rate", 0), 0)
            self.assertGreater(res.get("duration_sec", 0), 0.0)
            self.assertGreater(res.get("brightness_centroid", 0), 0.0)
            if transcribe:
                self.assertIn("transcript_preview", res)

        pipe = qa.audio_full_pipeline(persist=True)
        self.assertEqual(pipe["overall"], "READY")
        listing = qa.audio_session_list()
        self.assertGreater(listing["count"], 0)
        self.assertIsNotNone(listing["sessions"][0]["session_id"])

        # LLM Ears.py + transcribe_media.py
        llm_ears = _load_module("llm_ears_e2e", REPO_ROOT / "05 - Creative Works" / "Audio Engineer" / "LLM Ears.py")
        eng = llm_ears.SynesthesiaEngine(model_size="tiny")
        try:
            report = eng.generate_llm_report(str(ASSETS["mp3"]))
            self.assertIn("SYNESTHESIA REPORT GENERATED", report)
        finally:
            eng.close()

        transcribe_mod = _load_module(
            "transcribe_media_e2e",
            REPO_ROOT / "09 - Projects" / "projects" / "testing" / "Dev test" / "transcribe_media.py",
        )
        with tempfile.TemporaryDirectory() as tmpd:
            out_jsonl = Path(tmpd) / "transcripts.jsonl"
            t_res = transcribe_mod.transcribe_media(
                media_dir=ASSETS["mp3"].parent,
                out_file=out_jsonl,
                model_size="tiny",
                max_files=1,
                max_seconds=5.0,
            )
            self.assertEqual(t_res["transcribed"], 1)
            self.assertTrue(out_jsonl.exists())

    def test_02_video_mcp_and_vision_tracker(self):
        import av
        import torch
        import torchvision.transforms as T
        import quillan_audio.server as qa
        import quillan_media.server as qm
        from quillan_vision_builder import build_quillan_video_model

        v_info = qm.media_inspect_video(str(ASSETS["mp4"]), sample_frames=2, analyze_audio=True)
        self.assertEqual(v_info["status"], "OK")
        self.assertEqual(v_info["video_codec"], "h264")
        self.assertEqual(v_info["width"], 1280)
        self.assertEqual(v_info["height"], 720)
        self.assertTrue(v_info["has_audio"])
        self.assertGreaterEqual(v_info["decoded_sample_frames"], 2)
        self.assertIn("audio_stats", v_info)

        a_from_mp4 = qa.audio_analyze_track(str(ASSETS["mp4"]), transcribe=False, max_seconds=6.0)
        self.assertEqual(a_from_mp4["engine"], "pyav+numpy")
        self.assertEqual(a_from_mp4["codec"], "aac")

        frame_res = qm.media_extract_video_frame(str(ASSETS["mp4"]), frame_index=0)
        self.assertEqual(frame_res["status"], "OK")
        self.assertTrue(Path(frame_res["output_path"]).exists())

        from _shared.media_utils import letterbox_pil_image

        tracker = build_quillan_video_model(device="cpu")
        with av.open(str(ASSETS["mp4"])) as container:
            vs = next(s for s in container.streams if s.type == "video")
            for idx, frame in enumerate(container.decode(vs)):
                lb = letterbox_pil_image(frame.to_image(), (64, 64))
                self.assertEqual(lb.size, (64, 64))
                fx = T.ToTensor()(lb).unsqueeze(0)
                tracker.track_frame(fx)
                if idx >= 2:
                    break
        consensus = tracker.get_council_consensus()
        temporal = tracker.analyze_temporal_patterns()
        self.assertIn("consensus", consensus)
        self.assertEqual(temporal["frames_analyzed"], 3)

    def test_03_image_mcp_and_council_vision(self):
        import torch
        import torchvision.transforms as T
        from PIL import Image
        import quillan_media.server as qm
        from _shared.media_utils import letterbox_pil_image
        from quillan_vision_builder import build_quillan_image_model

        model = build_quillan_image_model(device="cpu")
        for ext in ("png", "jpg", "webp"):
            info = qm.media_inspect_image(str(ASSETS[ext]))
            self.assertEqual(info["status"], "OK", f"media_inspect_image failed on {ext}: {info}")
            self.assertGreater(info["width"], 0)
            self.assertGreater(info["height"], 0)

            with Image.open(ASSETS[ext]) as im:
                x = T.ToTensor()(letterbox_pil_image(im, (64, 64))).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
            self.assertEqual(tuple(out["final_output"].shape), (1, 80, 4, 4))
            self.assertEqual(len(out["metadata"]["individual_confidences"]), 4)

        pipe = qm.media_full_pipeline(objective="e2e image check", persist=True)
        self.assertEqual(pipe["overall"], "READY")
        health = qm.media_health()
        self.assertEqual(health["status"], "MEDIA_HEALTH_READY")
        self.assertIsNotNone(health["sessions"][0]["session_id"])

    def test_04_pdf_rag_and_extractors(self):
        import quillan_rag.server as qr

        async def _run_rag():
            ing = await qr.ingest_file(str(ASSETS["pdf"]))
            self.assertIn("Quillan-Ronin-Self-Paper-I-Am-Quillan.pdf", ing)
            srch = await qr.search("Quillan Ronin Bushido", n_results=2)
            self.assertIn("Quillan-Ronin-Self-Paper-I-Am-Quillan.pdf", srch)
            pipe = await qr.rag_full_pipeline(folder_path="", question="Quillan Ronin", n_context=2, persist=True)
            self.assertEqual(pipe["status"], "RAG_PIPELINE_COMPLETE")
            self.assertEqual(pipe["overall"], "PASS")

        asyncio.run(_run_rag())

        ext_mod = _load_module(
            "extract_pdfs_e2e",
            REPO_ROOT / "09 - Projects" / "projects" / "testing" / "Dev test" / "extract_pdfs.py",
        )
        ing_mod = _load_module(
            "ingest_paper_e2e",
            REPO_ROOT / "03 - Training & Model" / "scripts" / "ingest_paper.py",
        )
        with tempfile.TemporaryDirectory() as tmpd:
            out1 = Path(tmpd) / "pdfs.jsonl"
            r1 = ext_mod.extract_pdfs(pdf_dir=ASSETS["pdf"].parent, out_jsonl=out1, max_files=1)
            self.assertEqual(r1["extracted_count"], 1)
            self.assertTrue(out1.exists())

            out2 = Path(tmpd) / "paper.jsonl"
            r2 = ing_mod.ingest_paper(pdf_path=ASSETS["pdf"], out_jsonl=out2)
            self.assertGreater(r2["pages"], 0)
            self.assertGreater(r2["chars"], 1000)

    def test_05_train_multimodal_pipeline_cpu(self):
        import torch
        import torch.nn.functional as F

        tm = _load_module(
            "train_multimodal_e2e",
            REPO_ROOT / "03 - Training & Model" / "05_Training" / "scripts" / "train_multimodal.py",
        )
        img_t = tm.load_image_ffmpeg(ASSETS["png"])
        aud_t = tm.load_audio_ffmpeg(ASSETS["mp3"])
        vid_t = tm.load_video_ffmpeg(ASSETS["mp4"])
        self.assertEqual(tuple(img_t.shape), (3, 256, 256))
        self.assertEqual(tuple(aud_t.shape), (1, 16000))
        self.assertEqual(tuple(vid_t.shape), (3, 8, 64, 64))

        cfg = tm.QuillanArchConfig(
            text_only=False,
            hidden_dim=64,
            ffn_dim=128,
            vocab_size=256,
            num_experts=4,
            num_experts_active=2,
            top_k=2,
            eggroll_rank=16,
            max_seq_len=32,
            device="cpu",
        )
        model = tm.QuillanRoninSovereign(cfg)
        model.train()
        txt = torch.randint(0, 256, (1, 10))
        for modality, tensor in (("image", img_t), ("audio", aud_t), ("video", vid_t)):
            feats = tm.tensor_to_ingestion_features(tensor, modality)
            out = model(txt[:, :-1], img=feats)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), txt[:, 1:].reshape(-1))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_06_security_path_guards_and_nim_services(self):
        import quillan_media.server as qm
        import quillan_rag.server as qr
        import nim_vision_service as nvs
        import nim_audio_service as nas

        bad_img = qm.media_inspect_image(r"C:\Windows\System32\drivers\etc\hosts")
        self.assertEqual(bad_img["status"], "ERROR")
        bad_write = qm.media_extract_video_frame(
            str(ASSETS["mp4"]),
            output_path=str(REPO_ROOT / "05 - Creative Works" / "should_not_write.png"),
        )
        self.assertEqual(bad_write["status"], "ERROR")
        bad_rag = asyncio.run(qr.ingest_file(r"C:\Windows\System32\drivers\etc\hosts"))
        self.assertIn("Path traversal blocked", bad_rag)

        async def _nim_checks():
            vis = await nvs.analyze_image(str(ASSETS["png"]), prompt="One sentence description.", max_tokens=64)
            self.assertTrue(isinstance(vis, str) and len(vis.strip()) > 0)
            aud = await nas.analyze_track_metadata(str(ASSETS["mp3"]), lyrics_text="Neon blade cuts through the static.")
            self.assertIn("codec", aud)
            self.assertIn("analysis", aud)
            self.assertFalse(str(aud["analysis"]).startswith("Analysis failed"))

        asyncio.run(_nim_checks())


if __name__ == "__main__":
    unittest.main(verbosity=2)
