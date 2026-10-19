Below: a single-file JSON spec you can drop into a prompt-compiler, validator, or generator pipeline. Use it as the canonical “media generation manifest.”

Final template — Expert Hybrid Spec (drop-in JSON)

```json
{
  "Meta": {
    "version": "1.0",
    "id": "{{uuid_or_name}}",
    "mode": "{{image|video|audio|multimodal}}",
    "created_by": "{{author}}",
    "created_at": "{{YYYY-MM-DD}}",
    "determinism": {
      "seed": "{{integer|null}}",
      "consistency": "{{strict|adaptive|exploratory}}",
      "randomness_budget": "{{0.0-1.0}}"
    }
  },

  "Authority": {
    "visual_priority_order": [
      "Intent",
      "Constraints.Hard",
      "Priority",
      "Creative",
      "Style",
      "Physical",
      "Composition",
      "Capture",
      "Technical",
      "Output"
    ],
    "field_precedence_overrides": {
      "{{field_path_overriding}}": "{{field_path_overridden}}"
    }
  },

  "Intent": {
    "objective": "{{exact short sentence (what to achieve)}}",
    "description": "{{long description / creative brief}}",
    "deliverable": {
      "type": "{{image|video|audio|code|dataset|other}}",
      "count": "{{integer}}",
      "aspect_ratio": "{{16:9|4:5|1:1|9:16|3:2|custom}}",
      "size_px": "{{e.g., 2048x1152}}",
      "duration_sec": "{{for video/audio}}",
      "frame_rate_fps": "{{for video}}",
      "file_format": "{{PNG|JPG|MP4|MOV|WAV|FLAC|SVG|PDF}}",
      "color_profile": "{{sRGB|Display-P3|AdobeRGB}}"
    },
    "audience": "{{target audience}}",
    "intended_impact": "{{inspire|educate|sell|document|evoke}}"
  },

  "Creative": {
    "theme": "{{core idea or message}}",
    "tone": "{{warm|playful|authoritative|urgent|melancholic}}",
    "symbolism": ["{{motif1}}", "{{motif2}}"],
    "abstraction_level": "{{literal|stylized|abstract}}",
    "originality_bias": "{{0.0-1.0 (0 conservative,1 exploratory)}}",
    "semantic_targets": ["recognizability", "emotional_resonance", "narrative_clarity"]
  },

  "Content": {
    "subject": {
      "primary": "{{noun-based concrete subject}}",
      "secondary": ["{{optional}}"],
      "attributes": ["{{color, age, material, expression}}"]
    },
    "primary_elements": ["{{people|objects|creatures}}"],
    "background_elements": ["{{location, props, contextual items}}"],
    "narrative": "{{action or state}}",
    "story_points": ["{{beat1}}", "{{beat2}}"]
  },

  "Style": {
    "design_language": "{{minimal|editorial|cinematic|brutalist|illustrative}}",
    "genre": "{{documentary|fantasy|sci-fi|corporate|fashion}}",
    "palette": ["{{#HEX or color names}}"],
    "lighting_intent": "{{soft daylight|neon|chiaroscuro|rim-light}}",
    "composition_intent": "{{rule_of_thirds|centered|golden_triangle|dynamic}}",
    "perspective": "{{eye-level|aerial|macro|85mm}}",
    "references": [
      {
        "type": "style|scene|color|lighting",
        "source": "{{title_or_url}}",
        "weight": "{{0.0-1.0}}"
      }
    ],
    "style_consistency_target": "{{0.0-1.0}}"
  },

  "Physical": {
    "lighting": {
      "type": "{{natural|studio|neon|practical}}",
      "direction": "{{key|fill|back|rim|ambient}}",
      "quality": "{{soft|hard}}",
      "color_temperature_k": "{{e.g., 3200|5600}}",
      "intensity": "{{relative numeric}}"
    },
    "materials": {
      "finish": "{{matte|satin|glossy|metallic}}",
      "material_fidelity": "{{0.0-1.0}}"
    },
    "environment": {
      "location": "{{interior|exterior|studio|fantasy}}",
      "time_of_day": "{{dawn|noon|dusk|night}}",
      "atmospherics": ["{{fog|dust|rain|smoke}}"]
    }
  },

  "Composition": {
    "framing": "{{close-up|mid|wide|panoramic}}",
    "depth_structure": {
      "foreground": "{{elements}}",
      "midground": "{{elements}}",
      "background": "{{elements}}"
    },
    "focus_hierarchy": ["primary", "secondary", "tertiary"],
    "motion_design": "{{static|panning|tracking|cinematic_slow}}",
    "rule_overrides": {
      "safe_action_zone": "{{px or %}}",
      "title_safe_margin": "{{px or %}}"
    }
  },

  "Capture": {
    "enabled": "{{true|false}}",
    "mode": "{{photo|cinema|synthetic|render}}",
    "camera_anchor": {
      "model": "{{optional realism anchor e.g., Arri Alexa, Sony A7SIII, RED}}",
      "lens": {
        "focal_length_mm": "{{e.g., 50}}",
        "aperture": "{{e.g., f/1.8}}",
        "type": "{{prime|zoom|anamorphic}}"
      }
    },
    "exposure": {
      "aperture": "{{f/2.8}}",
      "shutter": "{{1/48 or shutter_angle}}",
      "iso_or_ei": "{{100}}"
    },
    "color_science": {
      "profile": "{{Rec.709|LOG|RAW}}",
      "white_balance": "{{auto|5600K}}"
    },
    "capture_overrides": {}
  },

  "Technical": {
    "resolution": "{{e.g., 4096x2304}}",
    "bit_depth": "{{8|10|12|16}}",
    "codec_and_bitrate": "{{ProRes_422|H.264|H.265|RAW and bitrate}}",
    "compression_target": "{{max filesize or quality threshold}}",
    "scalability_notes": "{{crop/letterbox/safe zones}}",
    "post_processing_intent": ["denoise", "color_grade", "VFX", "grain"],
    "consistency_checks": ["{{brand_color_match}}", "{{logo_clear_space}}"],
    "data_sources": ["{{citation_1}}"]
  },

  "Constraints": {
    "hard": [
      "{{must include logo at x,y or never include identifiable person without release}}"
    ],
    "soft": [
      "{{avoid complex reflective surfaces if CGI budget is low}}"
    ],
    "guards": [
      "no_anatomical_errors",
      "no-watermark",
      "no-text-on-image-unless-specified",
      "respect_copyright"
    ],
    "negative_prompts": [
      "blurry",
      "low_res",
      "extra_limbs",
      "watermark",
      "overexposed",
      "underexposed"
    ]
  },

  "Priority": {
    "weights": {
      "subject": 10,
      "lighting": 9,
      "composition": 9,
      "style": 8,
      "technical": 6,
      "accessibility": 7
    },
    "tie_breaker_strategy": "favor_higher_priority_field"
  },

  "Dependencies": {
    "composition": ["Intent.deliverable.aspect_ratio", "Content.subject.primary"],
    "lighting": ["Style.lighting_intent", "Physical.time_of_day"],
    "capture": ["Style.perspective", "Physical.materials.finish"]
  },

  "GenerationControl": {
    "creativity_level": "{{low|medium|high}}",
    "variation_count": "{{integer}}",
    "batch_strategy": "{{diverse|consistent|seeded}}",
    "max_tokens_or_compute_budget": "{{optional}}"
  },

  "Evaluation": {
    "metrics": {
      "subject_clarity": 0.0,
      "style_consistency": 0.0,
      "technical_quality": 0.0,
      "accessibility_score": 0.0,
      "aesthetic_impact": 0.0
    },
    "metric_weights": {
      "subject_clarity": 0.3,
      "style_consistency": 0.25,
      "technical_quality": 0.2,
      "accessibility_score": 0.1,
      "aesthetic_impact": 0.15
    },
    "thresholds": {
      "accept": 0.90,
      "revise": 0.70
    },
    "evaluation_mode": "{{auto|human|hybrid}}"
  },

  "Iteration": {
    "max_passes": 5,
    "refinement_order": [
      "enforce subject clarity",
      "align lighting with style",
      "fix composition and safe zones",
      "eliminate artifacts",
      "final color grade and accessibility checks"
    ],
    "auto_refinement_rules": {
      "if subject_clarity < 0.8 then apply": "increase_subject_contrast, tighten_focus",
      "if technical_quality < 0.75 then apply": "reduce_noise, increase_resolution_hint"
    }
  },

  "Validation": {
    "pre_generation_checks": [
      "Intent.deliverable is fully specified",
      "hard constraints present and valid",
      "priority weights sum > 0"
    ],
    "post_generation_checks": [
      "no hard constraint violations",
      "metrics computed",
      "artifacts below threshold"
    ],
    "auto_reject_on_violation": true
  },

  "Accessibility": {
    "alt_text": "{{precise alt text}}",
    "captions": "{{true|false}}",
    "contrast_target": "{{WCAG_AA|WCAG_AAA}}",
    "localization": {
      "language": "{{en|es|...}}",
      "numeral_format": "{{arabic|western}}",
      "text_direction": "{{LTR|RTL}}"
    }
  },

  "Communication": {
    "brief_for_reviewers": "{{short blurb for human QA}}",
    "notes_for_generation_engine": "{{instructions e.g., 'prioritize face clarity over background detail'}}",
    "versioning_and_changelog": "{{notes}}"
  },

  "References": [
    {
      "name": "{{title_or_asset_name}}",
      "type": "{{style|image|video|palette}}",
      "url": "{{optional_url}}",
      "weight": "{{0.0-1.0}}",
      "notes": "{{which aspect this influences}}"
    }
  ],

  "Audit": {
    "history": [],
    "last_run": {
      "by": "{{system_or_user}}",
      "at": "{{timestamp}}",
      "seed_used": "{{seed}}",
      "result_summary": "{{pass|revise|fail}}"
    }
  }
}
```

---

Example — quick filled snippet (image, cinematic portrait) 🎯
```json
{
  "Meta": {"version":"1.0","mode":"image","determinism":{"seed":12345,"consistency":"strict"}},
  "Intent":{"objective":"Hero portrait for album cover","description":"A cinematic portrait of an androgynous musician on stage smoke, dramatic rim light","deliverable":{"type":"image","count":3,"aspect_ratio":"1:1","size_px":"4096x4096","file_format":"PNG","color_profile":"Display-P3"}},
  "Content":{"subject":{"primary":"androgynous musician","attributes":["mid-30s","leather jacket","sweat"]},"narrative":"post-song catharsis"},
  "Style":{"design_language":"cinematic","palette":["#0b0b0f","#e6b800","#ffffff"],"lighting_intent":"rim-light","references":[{"type":"style","source":"Blade Runner 2049 stills","weight":0.7}]},
  "Physical":{"lighting":{"type":"studio","direction":"rim","quality":"hard","color_temperature_k":3200,"intensity":0.9},"environment":{"location":"stage","atmospherics":["smoke"]}},
  "Priority":{"weights":{"subject":10,"lighting":9,"composition":8,"technical":6}},
  "GenerationControl":{"creativity_level":"medium","variation_count":3},
  "Evaluation":{"metrics":{"subject_clarity":0.0,"style_consistency":0.0,"technical_quality":0.0},"thresholds":{"accept":0.90}},
  "Iteration":{"max_passes":3}
}
```

---

How to use — practical flow (keep it deterministic) ✅

1. Fill manifest — fully populate Intent, Content, Style, Technical, Constraints.


2. Validate pre-generation — run Validation.pre_generation_checks. Fix missing params.


3. Compile for target engine — convert manifest to model-specific prompt (preserve priorities, seed, constraints).


4. Generate batch — use GenerationControl.variation_count and Meta.determinism.seed (or seeded variants).


5. Auto-evaluate — compute Evaluation.metrics. If >= accept → finalize. If between revise and accept → run Iteration.auto_refinement_rules. If < revise → human review.


6. Post-process — apply Technical.post_processing_intent and accessibility adjustments (alt text, captions).


7. Audit and version — append run to Audit.history.



Quick notes & suggestions (real-world tips) 🛠️

Use Authority.field_precedence_overrides when you need a field to always trump others (e.g., brand logo placement).

Keep Capture.enabled = false for pure synthetic styles; enable when you want photographic realism anchored to a camera.

originality_bias + determinism lets you do reproducible exploration: keep seed constant while varying only originality_bias.

For dataset generation: set batch_strategy to consistent and use seeds offset by a fixed increment.



---
## Connections
- [[system prompts/System prompts for models/Gemini Gem System prompt.md]]
- [[Platforms/Grok/Image-or-Video template.md]]
- [[00 - Meta/06 - Deployment & Platforms.md]]
- [[system prompts/Quillan-Samurai.md]]
