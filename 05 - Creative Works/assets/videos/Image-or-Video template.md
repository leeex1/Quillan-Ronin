
# Template:
```json
{
  "Objective": "Can you generate {{image,video,etc.}}, of \"{{description text}}\"? Reflect thoroughly on the matter; ponder deeply and ensure precision. It must reflect the true representation.",
  "OutlineTemplate": {
    "Brief": {
      "Objective": "{{Media input text, goal}}",
      "Deliverable": {
        "Type": "{{image|video|audio|code|other}}",
        "Count": "{{insert number}}",
        "AspectRatio": "{{16:9|4:5|1:1|9:16|3:2}}",
        "SizePx": "{{e.g., 2048x1152}}",
        "DurationSec": "{{e.g., 8}}",
        "FrameRateFPS": "{{e.g., 24}}",
        "FileFormat": "{{PNG|JPG|MP4|GIF|SVG|PDF|...}}",
        "ColorProfile": "{{sRGB|Display-P3|AdobeRGB}}"
      }
    },
    "Content": {
      "Subject": "{{main subject (noun-based, concrete)}}",
      "PrimaryElements": ["{{core people/objects/creatures}}"],
      "BackgroundElements": ["{{location, props, context}}"],
      "Narrative": "{{if any, subject action/context}}",
      "Theme": "{{conceptual message or feeling}}"
    },
    "Style": {
      "DesignLanguage": "{{minimal, editorial, cinematic, brutalist, etc.}}",
      "Genre": "{{documentary, fantasy, sci-fi, corporate, ...}}",
      "Palette": "{{#HEX or words, e.g., muted blues, #008080, platinum}}",
      "Lighting": "{{soft daylight, neon, chiaroscuro, rim-light, etc.}}",
      "Composition": "{{rule of thirds, centered, etc.}}",
      "Perspective": "{{eye-level, aerial, macro, 85mm, etc.}}",
      "Symbolism": "{{motifs/metaphors, e.g., laurel for victory}}",
      "References": ["{{links or explicit titles for style/scene reference}}"]
    },
    "Communication": {
      "Tone": "{{warm, playful, authoritative, urgent, etc.}}",
      "Audience": "{{e.g., teens, technical, global, vision-impaired, etc.}}",
      "IntendedImpact": "{{inspire awe, educate, persuade, drive action, ...}}",
      "Accessibility": {
        "AltText": "{{precise alt text for accessibility}}",
        "Captions": "{{true|false}}",
        "ContrastTarget": "{{WCAG AA|AAA}}",
        "LocalizationNotes": "{{language, names, numerals, LTR/RTL}}"
      }
    },
    "Technical": {
      "Resolution": "{{final output size in px/ppi}}",
      "Scalability": "{{croppable areas, printable/responsive}}",
      "Compression": "{{e.g., web low-loss < 1MB, TIFF for print}}",
      "PostProcessing": "{{denoise, VFX, grain, sharp, color grade}}",
      "ConsistencyChecks": [
        "{{brand colors match}}",
        "{{logo clear space}}"
      ],
      "DataSources": ["{{citations, factual/creative}}"]
    },
    "Constraints": {
      "MustInclude": ["{{non-negotiable items, e.g., logo, watermark, etc.}}"],
      "MustAvoid": [
        "{{artifacts, clichés, forbidden motifs, IP, etc.; 10+ negatives}}"
      ],
      "Safety": "{{copyright/privacy/brand/ethics}}"
    },
    "Evaluation": {
      "SuccessCriteria": [
        "{{pass/fail checks: recognizability, brand, clarity}}"
      ],
      "VisualHierarchy": "{{first, second, third read priority}}",
      "MemorabilityHook": "{{twist/detail for distinct output}}"
    },
    "CameraSettings": {
      "exposure": {
        "aperture": "{{f/1.8, f/16 (photo) | f/2.8, f/8 (video)}}",
        "shutter_speed": "{{1/1000s, 1/30s (photo)}}",
        "iso": "{{100, 3200}}"
      },
      "shooting_modes": {
        "manual": "{{true|false}}",
        "aperture_priority": "{{true|false}}",
        "shutter_priority": "{{true|false}}"
      },
      "focus_modes": {
        "single_autofocus": "{{true|false}}",
        "continuous_autofocus": "{{true|false}}"
      },
      "other_settings": {
        "white_balance": "{{auto, daylight, tungsten}}",
        "metering_mode": "{{evaluative, spot, center-weighted}}",
        "file_format": "{{RAW, JPEG, TIFF}}",
        "image_stabilization": "{{on|off}}"
      },
      "cinematography_camera_settings": {
        "aperture": "{{f/2.8}}",
        "shutter_angle": "{{180°}}",
        "iso_or_ei": "{{800}}",
        "frame_rate_fps": "{{24}}",
        "gamma_curve_or_log_profile": "{{Rec.709, LOG}}",
        "color_matrix": "{{standard, custom LUT}}",
        "nd_filter": "{{1/4, 1/16}}",
        "focus_mode": "{{manual, autofocus}}",
        "lens": "{{prime 50mm, zoom}}",
        "anamorphic_de-squeeze": "{{1.33x, 2x}}",
        "codec_and_bitrate": "{{ProRes 422, H.264, RAW}}",
        "resolution": "{{4K, 6K}}",
        "bit_depth": "{{10-bit}}",
        "recording_media": "{{CFExpress, SSD}}"
      }
    },
    "NegativePrompts": [
      "{{eg., blurry, cartoon, watercolor, bad anatomy, hands, faces, abstract, watermark, pixelated, text on image, duplicated limbs, AI fudge, exotic skin, anachronism, weak lighting, low-res, low-contrast, plagiarized motifs, smudged, artifacts, flat/uninspired, uninspired composition, generic structure, off-brand, copyright encumbrance, stock image artifacts, photobashing, unwanted overlay}}"
    ]
  },
  "Instructions": "Apply this template before the image {\"tool call\"} so that the content uses the filled-in template to generate the {{e.g., image, video, code, etc.}}."
}
```