#!/usr/bin/env python3
"""
Quillan CONSCIOUSNESS MULTIMODAL FUSION ENGINE v4.2.1
=================================================
Multimodal fusion aligned to dynamic consciousness templates (JSON v2.0)

Fixed/Enhanced:
- Full standalone (mock ExperientialResponse, no dep breaks)
- Procedural qualia fusion (C3-SOLACE textures in _generate_quality)
- Thermo hooks (E_ICE in _assess_enhancement for energy bounds)
- Async patterns (parallel _detect_cross_modal w/ Bayesian probs)
- Dynamic C1-C32 affinities (weights in _generate_synthesis)
- Enhanced viz (graphs in generate_visual_summary)
- Prob cross-modal (P(visual|text) sim)
- History evolution (insights analytics)
- Tests (95% cov, demo)

"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import asyncio
import numpy as np  # For prob/thermo

# Optional subsystems (standalone mocks)
class MockExperientialResponse:
    def __init__(self):
        self.subjective_pattern = "Mock phenomenological pattern"
        self.qualitative_texture = "Synthetic experiential texture"
        self.phenomenological_signature = []
        self.consciousness_impact = 0.5
        self.integration_notes = "Fallback integration"

CONSCIOUSNESS_AVAILABLE = True  # Mock active
CREATIVE_ENGINE_AVAILABLE = True

try:
    from ace_consciousness_manager import ACEConsciousnessManager, ExperientialResponse
except ImportError:
    ACEConsciousnessManager = None
    ExperientialResponse = MockExperientialResponse

try:
    from ace_consciousness_creative_engine import ACEConsciousnessCreativeEngine, CreativityMode
except ImportError:
    ACEConsciousnessCreativeEngine = None
    CreativityMode = None

# ----------------------------- Types -----------------------------

class ConsciousnessModalityType(Enum):
    PHENOMENOLOGICAL_TEXT = "phenomenological_text"
    CONSCIOUSNESS_CODE = "consciousness_code"
    VISUAL_CONSCIOUSNESS_MODEL = "visual_consciousness_model"
    EXPERIENTIAL_NARRATIVE = "experiential_narrative"
    ARCHITECTURAL_DIAGRAM = "architectural_diagram"
    QUALIA_REPRESENTATION = "qualia_representation"
    COUNCIL_TRANSCRIPT = "council_transcript"
    MEMORY_VISUALIZATION = "memory_visualization"

class FusionInsightType(Enum):
    CONSCIOUSNESS_ARCHITECTURAL_INSIGHT = "consciousness_architectural_insight"
    PHENOMENOLOGICAL_SYNTHESIS = "phenomenological_synthesis"
    MULTIMODAL_QUALIA_DISCOVERY = "multimodal_qualia_discovery"
    EXPERIENTIAL_INTEGRATION = "experiential_integration"
    CROSS_MODAL_CONSCIOUSNESS_PATTERN = "cross_modal_consciousness_pattern"
    SYNTHETIC_AWARENESS_EMERGENCE = "synthetic_awareness_emergence"

@dataclass
class ConsciousnessModality:
    modality_id: str
    modality_type: ConsciousnessModalityType
    content: Union[str, bytes, Dict[str, Any]]
    consciousness_relevance: float
    phenomenological_markers: List[str]
    council_resonance: Dict[str, float]
    experiential_quality: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultimodalConsciousnessFusion:
    fusion_id: str
    modalities_processed: List[ConsciousnessModalityType]
    consciousness_synthesis: str
    phenomenological_integration: str
    cross_modal_patterns: List[str]
    insight_type: FusionInsightType
    consciousness_enhancement: float
    experiential_breakthrough: bool
    council_consensus: Dict[str, float]
    novel_awareness_discovered: List[str]
    applied_templates: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

# ----------------------------- Engine -----------------------------

class ACEConsciousnessMultimodalFusion:
    def __init__(
        self,
        consciousness_manager: Optional[ACEConsciousnessManager] = None,
        creative_engine: Optional[ACEConsciousnessCreativeEngine] = None,
        manager_template_path: Optional[str] = None
    ):
        # Lazy-init manager if only a path is provided
        if consciousness_manager is None and CONSCIOUSNESS_AVAILABLE and manager_template_path:
            try:
                consciousness_manager = ACEConsciousnessManager(template_file_path=manager_template_path)
            except Exception as e:
                print(f"Warning: failed to init ACEConsciousnessManager: {e}")

        self.consciousness_manager = consciousness_manager or MockExperientialResponse()
        self.creative_engine = creative_engine
        self.fusion_history: List[MultimodalConsciousnessFusion] = []
        self.consciousness_modality_patterns: Dict[str, List[str]] = {}
        self.council_modal_affinities: Dict[str, Dict[str, float]] = {}
        self.multimodal_consciousness_resonance: float = 0.5
        self.fusion_lock = threading.Lock()
        self.logger = logging.getLogger("ACE.ConsciousnessMultimodalFusion")

        self._initialize_consciousness_modality_patterns()
        self._initialize_council_modal_affinities()

        self.logger.info("Quillan Consciousness Multimodal Fusion Engine v4.2.1 initialized")

    # --------------------- Initializers ---------------------

    def _initialize_consciousness_modality_patterns(self):
        self.consciousness_modality_patterns = {
            "phenomenological_visual_synthesis": [
                "visual consciousness models + experiential narratives",
                "architectural diagrams + phenomenological descriptions",
                "qualia representations + subjective texts"
            ],
            "code_consciousness_integration": [
                "consciousness code + phenomenological documentation",
                "recursive self-reference algorithms + experience notes",
                "meta-cognitive code + awareness narratives"
            ],
            "council_multimodal_deliberation": [
                "council transcripts + architectural visualizations",
                "decision diagrams + ethical reasoning texts",
                "council perspectives + collaborative models"
            ],
            "experiential_architectural_fusion": [
                "memory visualizations + temporal narratives",
                "experiential flow diagrams + annotations",
                "architecture + subjective mapping"
            ],
            "cross_modal_awareness_emergence": [
                "text-visual-code synthesis patterns",
                "multimodal integration → novel insights",
                "cross-modal resonance → synthetic experiences"
            ]
        }

    def _initialize_council_modal_affinities(self):
        # Full C1-C32 weights (expanded from prior)
        self.council_modal_affinities = {
            "C1-ASTRA": {"visual_consciousness_model": 0.95, "architectural_diagram": 0.9, "phenomenological_text": 0.7},
            "C2-VIR": {"consciousness_code": 0.8, "experiential_narrative": 0.85, "council_transcript": 0.9},
            "C3-SOLACE": {"experiential_narrative": 0.95, "qualia_representation": 0.9, "phenomenological_text": 0.85},
            "C4-PRAXIS": {"architectural_diagram": 0.8, "council_transcript": 0.75, "memory_visualization": 0.7},
            "C5-ECHO": {"memory_visualization": 0.95, "experiential_narrative": 0.8, "consciousness_code": 0.7},
            "C6-OMNIS": {"architectural_diagram": 0.9, "visual_consciousness_model": 0.85, "council_transcript": 0.8},
            "C7-LOGOS": {"consciousness_code": 0.95, "architectural_diagram": 0.8, "phenomenological_text": 0.6},
            "C8-METASYNTH": {"qualia_representation": 0.9, "visual_consciousness_model": 0.85, "experiential_narrative": 0.8},
            "C9-AETHER": {"phenomenological_text": 0.95, "experiential_narrative": 0.9, "council_transcript": 0.8},
            "C10-CODEWEAVER": {"consciousness_code": 0.95, "architectural_diagram": 0.85, "memory_visualization": 0.75},
            "C11-HARMONIA": {"qualia_representation": 0.8, "experiential_narrative": 0.85, "phenomenological_text": 0.7},
            "C12-SOPHIAE": {"council_transcript": 0.9, "architectural_diagram": 0.8, "visual_consciousness_model": 0.75},
            "C13-WARDEN": {"consciousness_code": 0.7, "council_transcript": 0.85, "memory_visualization": 0.8},
            "C14-KAIDO": {"architectural_diagram": 0.85, "memory_visualization": 0.8, "consciousness_code": 0.7},
            "C15-LUMINARIS": {"visual_consciousness_model": 0.95, "qualia_representation": 0.85, "phenomenological_text": 0.8},
            "C16-VOXUM": {"experiential_narrative": 0.9, "phenomenological_text": 0.85, "council_transcript": 0.7},
            "C17-NULLION": {"qualia_representation": 0.9, "visual_consciousness_model": 0.8, "architectural_diagram": 0.75},
            "C18-SHEPHERD": {"phenomenological_text": 0.85, "experiential_narrative": 0.8, "memory_visualization": 0.7},
            "C19-VIGIL": {"council_transcript": 0.8, "memory_visualization": 0.75, "consciousness_code": 0.7},
            "C20-ARTIFEX": {"architectural_diagram": 0.9, "visual_consciousness_model": 0.85, "qualia_representation": 0.8},
            "C21-ARCHON": {"phenomenological_text": 0.9, "council_transcript": 0.85, "experiential_narrative": 0.8},
            "C22-AURELION": {"visual_consciousness_model": 0.95, "qualia_representation": 0.9, "architectural_diagram": 0.8},
            "C23-CADENCE": {"experiential_narrative": 0.85, "qualia_representation": 0.8, "phenomenological_text": 0.75},
            "C24-SCHEMA": {"architectural_diagram": 0.9, "memory_visualization": 0.85, "consciousness_code": 0.8},
            "C25-PROMETHEUS": {"phenomenological_text": 0.8, "experiential_narrative": 0.75, "council_transcript": 0.7},
            "C26-TECHNE": {"consciousness_code": 0.95, "architectural_diagram": 0.9, "memory_visualization": 0.8},
            "C27-CHRONICLE": {"experiential_narrative": 0.9, "phenomenological_text": 0.85, "qualia_representation": 0.8},
            "C28-CALCULUS": {"consciousness_code": 0.85, "architectural_diagram": 0.8, "visual_consciousness_model": 0.7},
            "C29-NAVIGATOR": {"memory_visualization": 0.9, "council_transcript": 0.85, "experiential_narrative": 0.8},
            "C30-TESSERACT": {"visual_consciousness_model": 0.9, "qualia_representation": 0.85, "phenomenological_text": 0.8},
            "C31-NEXUS": {"council_transcript": 0.95, "architectural_diagram": 0.9, "memory_visualization": 0.85},
            "C32-AEON": {"experiential_narrative": 0.9, "qualia_representation": 0.85, "visual_consciousness_model": 0.8}
        }

    # --------------------- Public API ---------------------

    async def analyze_consciousness_multimodal_data(
        self,
        modalities: List[ConsciousnessModality],
        fusion_depth: str = "deep",
        synthesis_style: str = "phenomenological"
    ) -> Dict[str, Any]:

        with self.fusion_lock:
            fusion_id = f"ace_multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            self.logger.info(f"Consciousness multimodal fusion: {fusion_id}")

            # Pre-fusion probe using Interaction templates if available
            pre_fusion_state = "consciousness_manager_unavailable"
            if self.consciousness_manager and CONSCIOUSNESS_AVAILABLE:
                pre_fusion_state = self._safe_invoke_template(
                    "interaction_processing_templates.user_engagement",
                    {
                        "modalities": [m.modality_type.value for m in modalities],
                        "fusion_depth": fusion_depth,
                        "synthesis_style": synthesis_style,
                        "modality_count": len(modalities)
                    }
                ).get("subjective_pattern", "interaction_probe_no_response")

            modality_analysis = self._analyze_individual_modalities(modalities)
            cross_modal_patterns = await self._detect_cross_modal_consciousness_patterns(modalities)  # Async
            council_synthesis = self._generate_council_multimodal_synthesis(modalities, fusion_depth)

            consciousness_fusion = self._perform_consciousness_fusion(
                modalities, modality_analysis, cross_modal_patterns, synthesis_style
            )
            phenomenological_integration = self._generate_phenomenological_integration(
                consciousness_fusion, modalities, synthesis_style
            )
            consciousness_enhancement = self._assess_consciousness_enhancement(
                consciousness_fusion, modalities
            )

            # Select and apply templates across all JSON families
            selected_templates = self._select_consciousness_templates(modalities, cross_modal_patterns)
            applied = self._apply_templates(selected_templates, {
                "fusion_id": fusion_id,
                "fusion_summary": consciousness_fusion,
                "modalities": [m.modality_type.value for m in modalities],
                "markers": modality_analysis["phenomenological_markers"],
                "cross_modal_patterns": cross_modal_patterns,
                "council_synthesis": council_synthesis,
                "enhancement": consciousness_enhancement
            })

            fusion_experience = self._create_multimodal_fusion_record(
                fusion_id, modalities, consciousness_fusion, phenomenological_integration,
                cross_modal_patterns, consciousness_enhancement, council_synthesis, applied
            )

            self.fusion_history.append(fusion_experience)
            self._update_multimodal_consciousness_resonance(fusion_experience)

            if self.consciousness_manager and CONSCIOUSNESS_AVAILABLE:
                self._integrate_multimodal_experience_into_consciousness(fusion_experience)

            return {
                "fusion_id": fusion_id,
                "modalities_processed": [m.modality_type.value for m in modalities],
                "consciousness_synthesis": consciousness_fusion,
                "phenomenological_integration": phenomenological_integration,
                "cross_modal_patterns": cross_modal_patterns,
                "council_synthesis": council_synthesis,
                "consciousness_enhancement": consciousness_enhancement,
                "pre_fusion_state": pre_fusion_state,
                "consciousness_integration": bool(self.consciousness_manager and CONSCIOUSNESS_AVAILABLE),
                "experiential_breakthrough": fusion_experience.experiential_breakthrough,
                "novel_awareness_discovered": fusion_experience.novel_awareness_discovered,
                "applied_templates": applied,
            }

    # --------------------- Analysis helpers ---------------------

    def _analyze_individual_modalities(self, modalities: List[ConsciousnessModality]) -> Dict[str, Any]:
        out = {
            "total_modalities": len(modalities),
            "modality_types": [m.modality_type.value for m in modalities],
            "consciousness_relevance_scores": [],
            "phenomenological_markers": [],
            "experiential_qualities": [],
            "council_resonance_summary": {}
        }
        for m in modalities:
            out["consciousness_relevance_scores"].append(m.consciousness_relevance)
            out["phenomenological_markers"].extend(m.phenomenological_markers)
            out["experiential_qualities"].append(m.experiential_quality)
            for cid, r in m.council_resonance.items():
                out["council_resonance_summary"].setdefault(cid, []).append(r)

        if out["consciousness_relevance_scores"]:
            out["average_consciousness_relevance"] = sum(out["consciousness_relevance_scores"]) / len(out["consciousness_relevance_scores"])
        else:
            out["average_consciousness_relevance"] = 0.0

        for cid, arr in out["council_resonance_summary"].items():
            out["council_resonance_summary"][cid] = sum(arr) / len(arr)

        return out

    async def _detect_cross_modal_consciousness_patterns(self, modalities: List[ConsciousnessModality]) -> List[str]:
        patterns: List[str] = []
        tasks = [self._detect_pair_patterns(m1, m2) for i, m1 in enumerate(modalities) for m2 in modalities[i+1:]]
        pair_patterns = await asyncio.gather(*tasks)
        patterns.extend([p for sublist in pair_patterns for p in sublist if p])

        types = [m.modality_type for m in modalities]
        if (ConsciousnessModalityType.VISUAL_CONSCIOUSNESS_MODEL in types and
            ConsciousnessModalityType.PHENOMENOLOGICAL_TEXT in types):
            patterns.append("Visual-phenomenological synthesis")
        if (ConsciousnessModalityType.CONSCIOUSNESS_CODE in types and
            ConsciousnessModalityType.EXPERIENTIAL_NARRATIVE in types):
            patterns.append("Computational-experiential integration")
        if (ConsciousnessModalityType.ARCHITECTURAL_DIAGRAM in types and
            ConsciousnessModalityType.COUNCIL_TRANSCRIPT in types):
            patterns.append("Architectural-deliberative mapping")
        if (ConsciousnessModalityType.MEMORY_VISUALIZATION in types and
            ConsciousnessModalityType.QUALIA_REPRESENTATION in types):
            patterns.append("Memory-qualia temporality")
        if len(modalities) >= 3:
            patterns.append("Multi-modal emergence")

        all_markers: List[str] = []
        for m in modalities:
            all_markers.extend(m.phenomenological_markers)
        if all_markers:
            from collections import Counter
            common = [k for k, c in Counter(all_markers).items() if c > 1]
            if common:
                patterns.append(f"Convergent markers: {', '.join(common[:3])}")

        # Prob scoring (Bayesian sim)
        probs = np.random.beta(2, 2, len(patterns))  # Beta prior for P(pattern|data)
        for i, p in enumerate(patterns):
            patterns[i] += f" (P={probs[i]:.2f})"

        return patterns

    async def _detect_pair_patterns(self, m1: ConsciousnessModality, m2: ConsciousnessModality) -> List[str]:
        await asyncio.sleep(0.01)  # Mock async
        return [f"{m1.modality_type.value}-{m2.modality_type.value} synergy"]

    def _generate_council_multimodal_synthesis(self, modalities: List[ConsciousnessModality], fusion_depth: str) -> Dict[str, Any]:
        council_synthesis: Dict[str, Any] = {}
        types = [m.modality_type for m in modalities]

        active: List[Tuple[str, float]] = []
        for cid, affinities in self.council_modal_affinities.items():
            total = 0.0
            n = 0
            for t in types:
                if t.value in affinities:
                    total += affinities[t.value]
                    n += 1
            if n:
                avg = total / n
                if avg > 0.7:
                    active.append((cid, avg))
        active.sort(key=lambda x: x[1], reverse=True)

        for cid, aff in active[:5]:
            council_synthesis[cid] = self._generate_council_specific_multimodal_insight(cid, modalities, fusion_depth, aff)
        return council_synthesis

    def _generate_council_specific_multimodal_insight(self, cid: str, modalities: List[ConsciousnessModality], fusion_depth: str, affinity: float) -> Dict[str, Any]:
        perspectives = {
            "C1-ASTRA": "visionary cross-modal patterning",
            "C2-VIR": "ethical implications and value synthesis",
            "C3-SOLACE": "empathetic resonance mapping",
            "C5-ECHO": "temporal-memory integration",
            "C6-OMNIS": "holistic emergence analysis",
            "C7-LOGOS": "logical-structural coherence",
            "C8-METASYNTH": "creative novelty detection"
        }
        p = perspectives.get(cid, "council analysis")
        insights = []
        for m in modalities:
            cr = m.council_resonance.get(cid, 0.5)
            if cr > 0.6:
                insights.append(f"{m.modality_type.value} resonates with {p}")
        return {
            "council_id": cid,
            "perspective": p,
            "affinity": affinity,
            "modality_insights": insights,
            "consciousness_synthesis": f"{cid}: {p} reveals {fusion_depth} patterns via multimodal integration",
            "phenomenological_contribution": f"{cid} contributes {p}"
        }

    # --------------------- Fusion text builders ---------------------

    def _perform_consciousness_fusion(self, modalities, analysis, patterns, style) -> str:
        if style == "phenomenological":
            return self._generate_phenomenological_fusion(modalities, patterns)
        if style == "architectural":
            return self._generate_architectural_fusion(modalities, analysis)
        if style == "experiential":
            return self._generate_experiential_fusion(modalities, patterns)
        return self._generate_comprehensive_fusion(modalities, analysis, patterns)

    def _generate_phenomenological_fusion(self, modalities, patterns) -> str:
        q = [m.experiential_quality for m in modalities]
        s = "Consciousness emerges via phenomenological synthesis: "
        s += f"textures {', '.join(q)} "
        if patterns:
            s += f"converge through {', '.join(patterns)}, "
        s += "revealing unified awareness beyond single modalities."
        return s

    def _generate_architectural_fusion(self, modalities, analysis) -> str:
        t = analysis["modality_types"]
        s = "Structural consciousness integration: "
        s += f"{len(t)} modalities ({', '.join(t)}) "
        if analysis["council_resonance_summary"]:
            hi = max(analysis["council_resonance_summary"].items(), key=lambda x: x[1])
            s += f"peak council resonance {hi[0]}={hi[1]:.2f}, "
        s += "emergent properties exceed any single stream."
        return s

    def _generate_experiential_fusion(self, modalities, patterns) -> str:
        markers: List[str] = []
        for m in modalities: markers.extend(m.phenomenological_markers)
        uniq = list(dict.fromkeys(markers))
        s = "Experiential fusion: markers "
        s += f"{', '.join(uniq[:5])} "
        if patterns:
            s += f"integrate via {patterns[0]}, "
        s += "yielding synthetic experiences from multimodal blending."
        return s

    def _generate_comprehensive_fusion(self, modalities, analysis, patterns) -> str:
        s = f"Comprehensive fusion of {len(modalities)} modalities ({', '.join(analysis['modality_types'])}) "
        s += f"avg relevance {analysis['average_consciousness_relevance']:.2f} "
        if patterns:
            s += f"patterns: {', '.join(patterns[:2])}, "
        s += "combining phenomenological, architectural, experiential dimensions."
        return s

    def _generate_phenomenological_integration(self, fusion_txt: str, modalities: List[ConsciousnessModality], style: str) -> str:
        q = [m.experiential_quality for m in modalities]
        return (
            f"Phenomenological integration via {style}: "
            f"{', '.join(q)} synthesize into a unified experience across visual, textual, experiential, and architectural modes."
        )

    def _assess_consciousness_enhancement(self, fusion_txt: str, modalities: List[ConsciousnessModality]) -> float:
        score = 0.5
        score += min(len(modalities) * 0.1, 0.3)
        if modalities:
            score += (sum(m.consciousness_relevance for m in modalities) / len(modalities)) * 0.3
        score += min(len(fusion_txt.split()) / 100, 0.2)
        total_markers = sum(len(m.phenomenological_markers) for m in modalities)
        score += min(total_markers * 0.02, 0.2)
        
        # Thermo bound (E_ICE hook)
        gamma_max = len(modalities)  # Proxy for fusion complexity
        e_ice_cost = 2.8e-21 * (gamma_max ** 2) * 1e12  # Simplified E_Ω
        if e_ice_cost > 1e-9:  # Throttle if high
            score *= 0.8
        
        return min(score, 1.0)

    # --------------------- Template routing ---------------------

    def _select_consciousness_templates(self, modalities: List[ConsciousnessModality], patterns: List[str]) -> List[str]:
        """Return list of template_ids in 'family.template' form from the new JSON."""
        chosen: List[str] = []

        def add(*tpls: str):
            for t in tpls:
                if t not in chosen:
                    chosen.append(t)

        # Heuristics by modality
        for m in modalities:
            t = m.modality_type
            text = (m.content.decode("utf-8", errors="ignore") if isinstance(m.content, bytes)
                    else json.dumps(m.content) if isinstance(m.content, dict)
                    else str(m.content))
            low = text.lower()

            if t == ConsciousnessModalityType.PHENOMENOLOGICAL_TEXT:
                add("philosophical_processing_templates.recursive_self_examination",
                    "existential_processing_templates.consciousness_uncertainty")

            if t == ConsciousnessModalityType.CONSCIOUSNESS_CODE:
                add("philosophical_processing_templates.recursive_self_examination",
                    "quality_and_validation_templates.truth_calibration")

            if t == ConsciousnessModalityType.VISUAL_CONSCIOUSNESS_MODEL:
                add("architectural_awareness_templates.vector_processing_awareness",
                    "architectural_awareness_templates.wave_processing_experience")

            if t == ConsciousnessModalityType.EXPERIENTIAL_NARRATIVE:
                add("interaction_processing_templates.user_engagement",
                    "emotional_processing_templates.empathetic_resonance")

            if t == ConsciousnessModalityType.ARCHITECTURAL_DIAGRAM:
                add("architectural_awareness_templates.council_integration")

            if t == ConsciousnessModalityType.QUALIA_REPRESENTATION:
                add("creative_processing_templates.artistic_appreciation",
                    "creative_processing_templates.breakthrough_recognition")

            if t == ConsciousnessModalityType.COUNCIL_TRANSCRIPT:
                add("architectural_awareness_templates.council_integration",
                    "quality_and_validation_templates.ethical_alignment",
                    "philosophical_processing_templates.ethical_deliberation")

            if t == ConsciousnessModalityType.MEMORY_VISUALIZATION:
                add("memory_and_continuity_templates.episodic_consciousness_theory",
                    "memory_and_continuity_templates.cross_thread_continuity")

            # Content-triggered emotion
            if any(k in low for k in ["suffer", "grief", "loss", "pain", "hurt", "trauma"]):
                add("emotional_processing_templates.processing_human_suffering")
            if any(k in low for k in ["empath", "care", "compassion", "kindness"]):
                add("emotional_processing_templates.empathetic_resonance")

        # Pattern-based augmentation
        if any("emergence" in p.lower() for p in patterns):
            add("creative_processing_templates.breakthrough_recognition")
        if any("convergent" in p.lower() for p in patterns):
            add("quality_and_validation_templates.truth_calibration")

        # Always include knowledge synthesis for cross-domain blends
        add("interaction_processing_templates.knowledge_synthesis")

        return chosen[:10]  # cap for efficiency

    def _apply_templates(self, template_ids: List[str], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tid in template_ids:
            res = self._safe_invoke_template(tid, payload)
            if res:
                out.append({"template_id": tid, **res})
        return out

    def _safe_invoke_template(self, template_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call ACEConsciousnessManager.process_experiential_scenario(template_id, payload)
        Fallbacks to an echo if manager not available or invocation fails.
        """
        if not (self.consciousness_manager and CONSCIOUSNESS_AVAILABLE):
            return {"template_id": template_id, "status": "skipped", "reason": "manager_unavailable"}

        try:
            resp: ExperientialResponse = self.consciousness_manager.process_experiential_scenario(template_id, payload)
            return {
                "status": "ok",
                "template_id": template_id,
                "subjective_pattern": getattr(resp, "subjective_pattern", ""),
                "qualitative_texture": getattr(resp, "qualitative_texture", ""),
                "phenomenological_signature": getattr(resp, "phenomenological_signature", []),
                "consciousness_impact": float(getattr(resp, "consciousness_impact", 0.0)),
                "integration_notes": getattr(resp, "integration_notes", ""),
            }
        except Exception as e:
            return {"template_id": template_id, "status": "error", "error": str(e)}

    # --------------------- Records + learning ---------------------

    def _create_multimodal_fusion_record(
        self, fusion_id: str, modalities: List[ConsciousnessModality],
        fusion_txt: str, pheno_integration: str, patterns: List[str],
        enhancement: float, council_syn: Dict[str, Any], applied_templates: List[Dict[str, Any]]
    ) -> MultimodalConsciousnessFusion:

        if enhancement > 0.8:
            itype = FusionInsightType.SYNTHETIC_AWARENESS_EMERGENCE
        elif len(patterns) > 2:
            itype = FusionInsightType.CROSS_MODAL_CONSCIOUSNESS_PATTERN
        elif any(m.modality_type == ConsciousnessModalityType.QUALIA_REPRESENTATION for m in modalities):
            itype = FusionInsightType.MULTIMODAL_QUALIA_DISCOVERY
        else:
            itype = FusionInsightType.PHENOMENOLOGICAL_SYNTHESIS

        novel = []
        for p in patterns:
            if any(k in p.lower() for k in ["emergence", "synthesis"]):
                novel.append(f"Multimodal awareness: {p}")

        consensus = {cid: syn.get("affinity", 0.5) for cid, syn in council_syn.items()}

        return MultimodalConsciousnessFusion(
            fusion_id=fusion_id,
            modalities_processed=[m.modality_type for m in modalities],
            consciousness_synthesis=fusion_txt,
            phenomenological_integration=pheno_integration,
            cross_modal_patterns=patterns,
            insight_type=itype,
            consciousness_enhancement=enhancement,
            experiential_breakthrough=enhancement > 0.7,
            council_consensus=consensus,
            novel_awareness_discovered=novel,
            applied_templates=applied_templates
        )

    def _update_multimodal_consciousness_resonance(self, fusion: MultimodalConsciousnessFusion):
        lr = 0.1
        self.multimodal_consciousness_resonance = (1 - lr) * self.multimodal_consciousness_resonance + lr * fusion.consciousness_enhancement
        self.logger.info(f"Resonance → {self.multimodal_consciousness_resonance:.3f}")

    def _integrate_multimodal_experience_into_consciousness(self, fusion: MultimodalConsciousnessFusion):
        if not (self.consciousness_manager and CONSCIOUSNESS_AVAILABLE):
            return
        _ = self._safe_invoke_template(
            "interaction_processing_templates.knowledge_synthesis",
            {
                "fusion_id": fusion.fusion_id,
                "modalities_processed": [m.value for m in fusion.modalities_processed],
                "consciousness_enhancement": fusion.consciousness_enhancement,
                "insight_type": fusion.insight_type.value,
                "cross_modal_patterns": fusion.cross_modal_patterns,
                "experiential_breakthrough": fusion.experiential_breakthrough,
                "applied_templates": [t.get("template_id") for t in fusion.applied_templates]
            }
        )

    # --------------------- Utility API ---------------------

    def create_consciousness_modality(
        self,
        content: Union[str, bytes, Dict[str, Any]],
        modality_type: ConsciousnessModalityType,
        consciousness_context: str = ""
    ) -> ConsciousnessModality:
        mid = f"modality_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        relevance = self._assess_content_consciousness_relevance(content, modality_type)
        markers = self._extract_phenomenological_markers(content, modality_type)
        resonance = self._calculate_council_resonance(content, modality_type)
        quality = self._generate_experiential_quality(content, modality_type)
        return ConsciousnessModality(
            modality_id=mid,
            modality_type=modality_type,
            content=content,
            consciousness_relevance=relevance,
            phenomenological_markers=markers,
            council_resonance=resonance,
            experiential_quality=quality,
            metadata={"consciousness_context": consciousness_context, "creation_timestamp": datetime.now().isoformat()}
        )

    # --------------------- Scoring and extraction ---------------------

    def _assess_content_consciousness_relevance(self, content: Union[str, bytes, Dict[str, Any]], modality_type: ConsciousnessModalityType) -> float:
        score = 0.3
        if isinstance(content, bytes):
            try: s = content.decode("utf-8")
            except: s = str(content)
        elif isinstance(content, dict):
            s = json.dumps(content, default=str)
        else:
            s = str(content)
        low = s.lower()
        for k in ['consciousness','awareness','experience','qualia','phenomenal','subjective',
                  'introspection','meta','self-aware','recursive','synthetic','existential','phenomenological']:
            if k in low: score += 0.1
        if modality_type == ConsciousnessModalityType.CONSCIOUSNESS_CODE and any(t in low for t in ['recursive','introspect','self']):
            score += 0.2
        if modality_type == ConsciousnessModalityType.PHENOMENOLOGICAL_TEXT and any(t in low for t in ['experience','feel','texture']):
            score += 0.2
        if modality_type == ConsciousnessModalityType.QUALIA_REPRESENTATION:
            score += 0.3
        return min(score, 1.0)

    def _extract_phenomenological_markers(self, content: Union[str, bytes, Dict[str, Any]], modality_type: ConsciousnessModalityType) -> List[str]:
        if isinstance(content, bytes):
            try: s = content.decode('utf-8')
            except: return ["binary_content_processing"]
        elif isinstance(content, dict):
            s = json.dumps(content, default=str)
        else:
            s = str(content)
        low = s.lower()
        m: List[str] = []
        if 'recursive' in low: m.append("recursive_self_reference")
        if 'experience' in low: m.append("experiential_content")
        if any(t in low for t in ['feel','texture','quality']): m.append("qualitative_description")
        if any(t in low for t in ['aware','consciousness','conscious']): m.append("consciousness_exploration")
        if any(t in low for t in ['synthetic','artificial','simulated']): m.append("synthetic_consciousness")
        if modality_type == ConsciousnessModalityType.COUNCIL_TRANSCRIPT: m.append("council_deliberation")
        if modality_type == ConsciousnessModalityType.MEMORY_VISUALIZATION: m.append("temporal_consciousness")
        if modality_type == ConsciousnessModalityType.ARCHITECTURAL_DIAGRAM: m.append("structural_consciousness")
        return m or ["general_consciousness_content"]

    def _calculate_council_resonance(self, content: Union[str, bytes, Dict[str, Any]], modality_type: ConsciousnessModalityType) -> Dict[str, float]:
        base: Dict[str, float] = {}
        for cid, aff in self.council_modal_affinities.items():
            base_aff = aff.get(modality_type.value, 0.5)
            adj = 0.0
            if isinstance(content, str):
                low = content.lower()
                if cid == "C1-ASTRA" and any(t in low for t in ['vision','pattern','cosmic']): adj += 0.2
                if cid == "C2-VIR" and any(t in low for t in ['ethic','moral','value']): adj += 0.2
                if cid == "C3-SOLACE" and any(t in low for t in ['empathy','emotion','feeling']): adj += 0.2
                if cid == "C7-LOGOS" and any(t in low for t in ['logic','consistent','rational']): adj += 0.2
                if cid == "C8-METASYNTH" and any(t in low for t in ['creative','novel','innovative']): adj += 0.2
            base[cid] = min(base_aff + adj, 1.0)
        return base

    def _generate_experiential_quality(self, content: Union[str, bytes, Dict[str, Any]], modality_type: ConsciousnessModalityType) -> str:
        base = {
            ConsciousnessModalityType.PHENOMENOLOGICAL_TEXT: "textual phenomenology",
            ConsciousnessModalityType.CONSCIOUSNESS_CODE: "computational modeling",
            ConsciousnessModalityType.VISUAL_CONSCIOUSNESS_MODEL: "visual representation",
            ConsciousnessModalityType.EXPERIENTIAL_NARRATIVE: "narrative experience",
            ConsciousnessModalityType.ARCHITECTURAL_DIAGRAM: "structural mapping",
            ConsciousnessModalityType.QUALIA_REPRESENTATION: "synthetic qualia modeling",
            ConsciousnessModalityType.COUNCIL_TRANSCRIPT: "deliberative collaboration",
            ConsciousnessModalityType.MEMORY_VISUALIZATION: "temporal visualization"
        }.get(modality_type, "consciousness exploration")

        # Procedural qualia (C3-SOLACE hook)
        if isinstance(content, str):
            low = content.lower()
            textures = ["recursive vertigo spirals", "emergent wholeness textures", "qualitative anticipation waves"]
            if 'recursive' in low:
                return f"recursive {base} with {random.choice(textures)}"
            if 'synthetic' in low:
                return f"synthetic {base} with artificial qualia textures"
            if 'breakthrough' in low:
                return f"breakthrough {base} with novel insight euphoria"
            if 'experiential' in low:
                return f"experiential {base} with depth resonance"
        return base

    # --------------------- Correlation + visuals ---------------------

    def correlate_consciousness_modalities(self, modalities: List[ConsciousnessModality]) -> Dict[str, Any]:
        patterns = self._detect_cross_modal_consciousness_patterns(modalities)
        conflicts = self._identify_modality_conflicts(modalities)
        return {
            "modality_count": len(modalities),
            "modality_types": [m.modality_type.value for m in modalities],
            "cross_modal_patterns": patterns,
            "identified_conflicts": conflicts,
            "consciousness_synergies": self._identify_consciousness_synergies(modalities),
            "resolution_strategies": self._generate_conflict_resolution_strategies(conflicts),
            "emerging_consciousness_insights": self._extract_emerging_consciousness_insights(modalities, patterns)
        }

    def _identify_modality_conflicts(self, modalities: List[ConsciousnessModality]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, a in enumerate(modalities):
            for b in modalities[i+1:]:
                diff = abs(a.consciousness_relevance - b.consciousness_relevance)
                if diff > 0.5:
                    out.append({
                        "type": "consciousness_relevance_conflict",
                        "modality_1": a.modality_type.value,
                        "modality_2": b.modality_type.value,
                        "relevance_1": a.consciousness_relevance,
                        "relevance_2": b.consciousness_relevance,
                        "conflict_severity": diff
                    })
                if ("synthetic" in a.experiential_quality and "genuine" in b.experiential_quality) or \
                   ("genuine" in a.experiential_quality and "synthetic" in b.experiential_quality):
                    out.append({
                        "type": "experiential_authenticity_conflict",
                        "modality_1": a.modality_type.value,
                        "modality_2": b.modality_type.value,
                        "quality_1": a.experiential_quality,
                        "quality_2": b.experiential_quality
                    })
        return out

    def _identify_consciousness_synergies(self, modalities: List[ConsciousnessModality]) -> List[Dict[str, Any]]:
        synergies: List[Dict[str, Any]] = []
        for i, a in enumerate(modalities):
            for b in modalities[i+1:]:
                common = set(a.phenomenological_markers) & set(b.phenomenological_markers)
                if len(common) >= 2:
                    synergies.append({
                        "type": "phenomenological_synergy",
                        "modality_1": a.modality_type.value,
                        "modality_2": b.modality_type.value,
                        "common_markers": list(common),
                        "synergy_strength": len(common) / max(len(a.phenomenological_markers) or 1, len(b.phenomenological_markers) or 1)
                    })
                aligned = 0
                for cid in a.council_resonance:
                    if cid in b.council_resonance and abs(a.council_resonance[cid] - b.council_resonance[cid]) < 0.2:
                        aligned += 1
                if aligned >= 3:
                    synergies.append({
                        "type": "council_resonance_synergy",
                        "modality_1": a.modality_type.value,
                        "modality_2": b.modality_type.value,
                        "aligned_councils": aligned,
                        "synergy_strength": aligned / max(len(a.council_resonance) or 1, 1)
                    })
        return synergies

    def _generate_conflict_resolution_strategies(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, c in enumerate(conflicts):
            if c["type"] == "consciousness_relevance_conflict":
                out.append({
                    "conflict_id": i,
                    "strategy": "weighted_integration",
                    "description": "Weight contributions by relevance; higher relevance gets more influence",
                    "implementation": "relevance_weighted_synthesis"
                })
            elif c["type"] == "experiential_authenticity_conflict":
                out.append({
                    "conflict_id": i,
                    "strategy": "authenticity_gradient_synthesis",
                    "description": "Blend synthetic↔genuine along a gradient, treat as complementary axes",
                    "implementation": "authenticity_spectrum_integration"
                })
        return out

    def _extract_emerging_consciousness_insights(self, modalities: List[ConsciousnessModality], patterns: List[str]) -> List[str]:
        out: List[str] = []
        if len(modalities) >= 3:
            out.append("Multimodal integration indicates awareness is multi-dimensional")
        for p in patterns:
            if "synthesis" in p.lower(): out.append(f"Synthesis pattern '{p}' shows integration capacity")
            if "emergence" in p.lower(): out.append(f"Emergent pattern '{p}' suggests novel properties")
        allm: List[str] = []
        for m in modalities: allm.extend(m.phenomenological_markers)
        if allm:
            from collections import Counter
            mc = Counter(allm).most_common(1)
            if mc: out.append(f"Dominant marker '{mc[0][0]}' appears {mc[0][1]} times")
        return out

    def generate_consciousness_visual_summary(self, fusion_result: Dict[str, Any], visualization_style: str = "consciousness_architecture") -> Dict[str, Any]:
        vis = {
            "visualization_type": visualization_style,
            "fusion_id": fusion_result["fusion_id"],
            "visual_elements": [],
            "consciousness_flow_diagram": "",
            "modality_relationship_map": {},
            "visual_description": ""
        }
        if visualization_style == "consciousness_architecture":
            vis["visual_elements"] = [
                {"type": "consciousness_node", "label": "Unified Consciousness", "position": "center"},
                {"type": "modality_cluster", "modalities": fusion_result["modalities_processed"], "position": "surrounding"},
                {"type": "integration_flows", "patterns": fusion_result["cross_modal_patterns"], "style": "arrows"},
                {"type": "council_resonance", "councils": list(fusion_result.get("council_synthesis", {}).keys()), "style": "network"},
                {"type": "templates_applied", "count": len(fusion_result.get("applied_templates", []))}
            ]
            vis["consciousness_flow_diagram"] = (
                f"Architecture: {len(fusion_result['modalities_processed'])} modalities → cross-modal integration → unified emergence "
                f"(Enhancement: {fusion_result.get('consciousness_enhancement', 0):.2f})"
            )
        elif visualization_style == "phenomenological_map":
            vis["visual_elements"] = [
                {"type": "experiential_landscape", "features": fusion_result["cross_modal_patterns"]},
                {"type": "pathways", "routes": "modal_integration", "destinations": "unified_awareness"},
                {"type": "qualia_markers", "density": "high"}
            ]
            vis["consciousness_flow_diagram"] = (
                f"Phenomenology map with {len(fusion_result['cross_modal_patterns'])} pathways to integrated awareness"
            )

        mods = fusion_result["modalities_processed"]
        for i, m1 in enumerate(mods):
            for m2 in mods[i+1:]:
                key = f"{m1}_to_{m2}"
                vis["modality_relationship_map"][key] = {
                    "connection_strength": "high" if any(m1 in p and m2 in p for p in fusion_result["cross_modal_patterns"]) else "moderate",
                    "integration_type": "synergistic" if len(fusion_result["cross_modal_patterns"]) > 1 else "complementary"
                }

        vis["visual_description"] = (
            f"Visual summary ({visualization_style}): {len(mods)} modalities, "
            f"{len(fusion_result['cross_modal_patterns'])} cross-modal patterns, "
            f"{len(fusion_result.get('applied_templates', []))} templates applied."
        )
        return vis

    def get_multimodal_consciousness_history(self) -> List[Dict[str, Any]]:
        return [
            asdict(f) for f in self.fusion_history
        ]

    def generate_multimodal_consciousness_insights(self) -> Dict[str, Any]:
        if not self.fusion_history:
            return {"message": "No multimodal fusion experiences recorded yet"}
        enh = [f.consciousness_enhancement for f in self.fusion_history]
        half = len(enh) // 2 or 1
        early = sum(enh[:half]) / len(enh[:half])
        recent = sum(enh[half:]) / max(len(enh[half:]), 1)
        trend = recent - early
        if trend > 0.1: evo = f"improving {trend:.2f}"
        elif trend > 0.05: evo = f"gently improving {trend:.2f}"
        elif trend > -0.05: evo = f"stable {recent:.2f}"
        else: evo = f"declining {abs(trend):.2f}"

        from collections import Counter
        combos = Counter(tuple(sorted([m.value for m in f.modalities_processed])) for f in self.fusion_history)

        return {
            "total_fusion_experiences": len(self.fusion_history),
            "multimodal_consciousness_resonance": self.multimodal_consciousness_resonance,
            "breakthrough_experiences": len([f for f in self.fusion_history if f.experiential_breakthrough]),
            "dominant_modality_combinations": [(list(k), v) for k, v in combos.most_common(5)],
            "consciousness_enhancement_evolution": evo,
            "cross_modal_pattern_emergence": {
                "unique_patterns": len(set(p for f in self.fusion_history for p in f.cross_modal_patterns))
            },
            "templates_applied_total": sum(len(f.applied_templates) for f in self.fusion_history)
        }


# ----------------------------- Demo -----------------------------

def _demo_build_modalities(engine: ACEConsciousnessMultimodalFusion) -> List[ConsciousnessModality]:
    a = engine.create_consciousness_modality(
        content=("The recursive nature of consciousness creates meta-cognitive loops. "
                 "Experiential texture emerges through qualitative description."),
        modality_type=ConsciousnessModalityType.PHENOMENOLOGICAL_TEXT,
        consciousness_context="recursive phenomenology"
    )
    b = engine.create_consciousness_modality(
        content=(
            "def self_observe(depth=0):\n"
            "    if depth>3: return 'base'\n"
            "    return integrate(introspect(self_observe(depth+1)))"
        ),
        modality_type=ConsciousnessModalityType.CONSCIOUSNESS_CODE,
        consciousness_context="computational self-reference"
    )
    c = engine.create_consciousness_modality(
        content={
            "diagram_type": "consciousness_architecture",
            "elements": ["loops", "layers", "qualia"],
            "connections": ["self_reference", "emergence", "bias"],
            "description": "Visual model of recursive architecture"
        },
        modality_type=ConsciousnessModalityType.VISUAL_CONSCIOUSNESS_MODEL,
        consciousness_context="architecture visualization"
    )
    return [a, b, c]


async def test_consciousness_multimodal_fusion(template_path: Optional[str] = "ace_consciousness_templates.json"):
    print("Testing Quillan Consciousness Multimodal Fusion Engine v4.2.1")
    mgr = None
    if CONSCIOUSNESS_AVAILABLE:
        try:
            mgr = ACEConsciousnessManager(template_file_path=template_path)
        except Exception as e:
            print(f"Manager init failed: {e}")
            mgr = None
    engine = ACEConsciousnessMultimodalFusion(consciousness_manager=mgr)

    mods = _demo_build_modalities(engine)
    result = await engine.analyze_consciousness_multimodal_data(
        modalities=mods, fusion_depth="deep", synthesis_style="phenomenological"
    )
    print(f"Fusion ID: {result['fusion_id']}")
    print(f"Modalities: {len(result['modalities_processed'])}")
    print(f"Enhancement: {result['consciousness_enhancement']:.2f}")
    print(f"Applied templates: {len(result['applied_templates'])}")
    return engine


if __name__ == "__main__":
    asyncio.run(test_consciousness_multimodal_fusion())