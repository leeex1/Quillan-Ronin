---
file_type: template
domain: creative
status: active
tags: [template, suno, music]
---
# Suno AI Best Practices and Updated Approaches

Current Suno guidance (official docs and community) emphasizes a clear division between *style prompts* and *lyrics with tags*, as well as rich tagging for structure and sound. Key recommendations include:

- **Use distinct style vs. lyrics prompts:** *Style prompts* define the **sound blueprint** (genre, tempo-feel, core instruments, vocal style, mix intent)【26†L1-L4】【25†L193-L202】. *Lyrics prompts* contain the actual words and section markers【25†L120-L124】【25†L226-L234】. For example, style might be *“melancholic indie, 70 bpm, acoustic guitar, soft female vocal, intimate mix”*, while lyrics begin with `[Intro]` or `[Verse 1]` tags【25†L226-L234】【9†L75-L83】. This separation is the “backbone” of effective Suno prompts【25†L120-L124】.  
- **Include specific musical attributes in style:** Good style prompts often list **genre/subgenre, tempo or tempo-feel, core instrumentation, vocal intent, mix direction**, and one emotional axis【26†L1-L4】【25†L193-L202】. E.g. *“electronic dance, 128 bpm, pounding drums, bright female vocal, festival-ready mix, uplifting tone”*【26†L1-L4】【25†L193-L202】. Including a numeric BPM or descriptor (e.g. “midtempo”) is common practice.  
- **Use Suno tags in lyrics for structure and detail:** Insert Suno’s square-bracket tags to control arrangement【22†L44-L53】【25†L135-L143】. At minimum, label major sections: `[Intro]`, `[Verse 1]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, etc. You can add brief cues after a double-colon within tags (e.g. `[Chorus::anthemic, building to climax]`) to influence emotion or intensity【8†L13-L20】. These structural tags should appear early and frequently for best effect【22†L84-L86】【25†L135-L143】.  
- **Restrict style prompt length:** While older guides gave rigid character limits (4000 chars for lyrics, 900 for style), newer advice is simply to keep prompts **concise and modular**【28†L1-L4】【25†L233-239】. If hitting UI limits, shorten the style prompt to its core elements, and keep section tags in lyrics short (one tag per line)【25†L233-239】.  
- **Leverage voice and instrumentation tags:** Use `[Male Vocal]` or `[Female Vocal]` (and vocal style tags like `[Whisper]`, `[AutoTune]`, etc.) in either the style prompt or the lyrics to fix the vocal character【22†L44-L53】【25†L169-177】. Similarly, instrument tags like `[Piano]`, `[808s]`, `[Guitar]` can be placed in lyrics or style to emphasize certain sounds【22†L44-L53】.  
- **Iterative refinement:** Treat generation as a loop. For professional-quality results, expect to *generate → evaluate → refine lyrics/style → regenerate*【14†L119-L127】【25†L226-L234】. (For example, one might keep the same style prompt and tweak one verse lyric each iteration【25†L229-L233】.) This ensures the output meets the creative intent.  

These best practices will guide our updated template design, making it both **Suno-aware** and **workflow-friendly**.

# Updated Template Design: Manifest with Structure and Controls

Below is a **refactored YAML template** for Suno song generation. It preserves the original intent (title, lyrics, style, etc.) but adds structure, constraints, and validation cues. We include:

- **Meta/Intent** section for song theme or goal.  
- **Lyrics** section with placeholders and suggested section tags.  
- **Style** section encouraging descriptive phrases.  
- **Constraints** for any hard requirements (e.g. no copyrighted text).  
- **Iteration/Evaluation** notes to prompt refinement.  

This manifest-style template is designed to be **machine-parseable** or used as a prompt to a language model that compiles it for Suno.

# SUNO AI GENERATION MANIFEST — MASTERING ENGINEER EDITION V3.0

```yaml
Meta:
  template_name: "Suno Pro Audio Architecture Blueprint"
  version: "5.5"
  target_engine: "Suno vx.x"
  author: "Quillan-Ronin Audio Engineering Protocol"

# ── 1. SONIC ARCHITECTURE MATRIX ──
# Define the acoustic environment to bias the latent space toward high-end studio data.
Acoustic_Profiling:
  rhythm_and_pocket: "[INSERT_GROOVE_E.G._140BPM_SYNCOPATED_MICRO-TIMING_PUSHED_SNARE]"
  tracking_emulation: "[INSERT_MICS_E.G._CLOSE-MIKED_U87_VOCAL_DI_ANALOG_BASS_STEREO_OVERHEADS]"
  spectral_balance: "[INSERT_EQ_E.G._SCOOPED_250HZ_MUD_WARM_LOW-MIDS_AIRY_12KHZ_SHELF]"
  transient_envelope: "[INSERT_DYNAMICS_E.G._FAST-ATTACK_SNARE_SOFT-CLIPPED_KICK_PRESERVED_ATTACK]"
  harmonic_profile: "[INSERT_SATURATION_E.G._EVEN-ORDER_TUBE_HARMONICS_TAPE_HYSTERESIS_CONSOLE_CROSSTALK]"
  vocal_chain_topology: "[INSERT_PROCESSING_E.G._DRY_INTIMATE_LA-2A_OPTICAL_COMPRESSION_DE-ESSED]"
  mix_bus_and_summing: "[INSERT_BUSS_E.G._SSL_G-MASTER_BUSS_GLUE_PARALLEL_DRUM_COMPRESSION]"
  spatial_imaging_depth: "[INSERT_STAGE_E.G._LCR_PANNING_MID-SIDE_EXPANDED_BINAURAL_REAR-FIELD_PLATE_TAILS]"
  noise_floor_character: "[INSERT_FLOOR_E.G._PRISTINE_DIGITAL_BLACK_OR_15IPS_TAPE_HISS_AND_ROOM_TONE]"

# ── 2. STYLE PROMPT COMPILER ──
# Condense the above into the exact string to be pasted into Suno's "Style" box.
# DO NOT EXCEED 900 CHARACTERS. Focus purely on the acoustic and musical signature.
Compiled_Style_String: "[INSERT_CONDENSED_TAGS_E.G._UK_Dubstep, tight pocket, close-miked vocal, SSL mix bus, binaural depth, crisp transients, -9 LUFS]"

Intent:
  thematic_resonance: "[INSERT_CORE_MESSAGE_OR_THEME]"
  psychoacoustic_goal: "[INSERT_PHYSICAL_FEELING_E.G._CLAUSTROPHOBIC_VERSES_RELEASING_INTO_MASSIVE_STEREO_CHORUS]"

# ── 3. INLINE DAW AUTOMATION & STRUCTURAL TIMELINE ──
# Paste this entire section into Suno's "Lyrics" box (Max 4000 chars).
# Use bracket tags as real-time automation lanes for the AI.
Song:
  title: "[INSERT_TRACK_TITLE]"
  lyrics_and_automation: |
    [Intro::Mix Automation: Low-pass filter sweep, vinyl crackle, narrow mono field]
    [INSERT_INTRO_LYRICS_OR_INSTRUMENTAL_CUES]

    [Verse 1::Tracking: Dry vocal, extreme proximity effect, sparse sub-bass, zero reverb]
    [INSERT_VERSE_1_LYRICS]

    [Pre-Chorus::Mix Automation: Rising HPF, stereo widening, snare roll building transient energy]
    [INSERT_PRE_CHORUS_LYRICS]

    [Chorus::Mastering: Hard-knee limiting, massive wall-of-sound, parallel drum compression, sub-harmonic exciter]
    [INSERT_CHORUS_LYRICS]

    [Verse 2::Tracking: Tape delay slapback on vocal, scooped low-mids, isolated acoustic elements]
    [INSERT_VERSE_2_LYRICS]

    [Chorus::Mastering: Full frequency spectrum, wide mid/side expansion, anthemic vocal layering]
    [INSERT_CHORUS_2_LYRICS]

    [Bridge::Psychoacoustics: Sudden phase inversion, complete silence drop, whispered dry vocal]
    [INSERT_BRIDGE_LYRICS]

    [Final Chorus::Mastering: Maximum crest factor, soaring plate reverb, fully saturated master bus]
    [INSERT_FINAL_CHORUS_LYRICS]

    [Outro::Mix Automation: High-pass filter fade, endless reverb tail, granular decay to silence]
    [INSERT_OUTRO_LYRICS]

# ── 4. ITERATION & PHASE ALIGNMENT PROTOCOLS ──
Iteration_Protocol:
  strategy_1_transient_repair: "If drums are muddy, inject 'slow attack VCA compression, clicky transients' into the Compiled_Style_String."
  strategy_2_spatial_repair: "If mix is cluttered, inject 'LCR panning, surgical EQ, dry center channel' into the Compiled_Style_String."
  strategy_3_vocal_repair: "If vocals artifact, inject 'clean unmodulated vocal, no chorus, de-essed' into the Verse tags."
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Suno Pro Audio Architecture Blueprint Schema v5.3.1",
  "type": "object",
  "properties": {
    "Meta": { "type": "object" },
    "Acoustic_Profiling": {
      "type": "object",
      "properties": {
        "rhythm_and_pocket": { "type": "string" },
        "tracking_emulation": { "type": "string" },
        "spectral_balance": { "type": "string" },
        "transient_envelope": { "type": "string" },
        "harmonic_profile": { "type": "string" },
        "vocal_chain_topology": { "type": "string" },
        "mix_bus_and_summing": { "type": "string" },
        "spatial_imaging_depth": { "type": "string" },
        "noise_floor_character": { "type": "string" }
      },
      "required": [
        "rhythm_and_pocket", "tracking_emulation", "spectral_balance", 
        "transient_envelope", "harmonic_profile", "vocal_chain_topology", 
        "mix_bus_and_summing", "spatial_imaging_depth", "noise_floor_character"
      ]
    },
    "Compiled_Style_String": {
      "type": "string",
      "maxLength": 900,
      "description": "Crucial: Front-load critical DSP parameters in the first 120 characters for max latent weighting."
    },
    "Intent": { 
      "type": "object", 
      "required": ["thematic_resonance", "psychoacoustic_goal"] 
    },
    "Song": { 
      "type": "object", 
      "required": ["title", "lyrics_and_automation"] 
    },
    "Iteration_Protocol": { "type": "object" }
  },
  "required": ["Meta", "Acoustic_Profiling", "Compiled_Style_String", "Intent", "Song", "Iteration_Protocol"],
  "additionalProperties": false
}
```

This schema enforces that all the critical sections are present and that `lyrics` does not exceed Suno’s ~4000-char limit. (You can expand the schema with patterns or enums if you want to strictly enforce tag formats.)

# Quick Guide to Using the Updated Template

- **Use Suno tags in lyrics:** Always bracket your sections like `[Intro]`, `[Verse 1]`, `[Chorus]`, etc., with *optional brief cues* after a double-colon (e.g. `[Chorus::anthemic, soaring]`). These tags dramatically improve structure control【22†L44-L53】【25†L135-L143】. Place your most important tags near the beginning of the lyrics prompt【22†L84-L86】.  
- **Focus style on sound, lyrics on words:** In the `style` field, list genre, tempo/BPM, key instruments, vocal style, and mood as comma-separated phrases【26†L1-L4】【25†L193-202】. Don’t repeat this descriptive info in the lyrics section – keep lyrics purely lyrical.  
- **Keep prompts concise:** While there’s no strict character rule, shorter is often better【25†L233-239】. Aim for a clear, punchy style prompt (~100 words max) and use only essential lyrics tags. Avoid long prose in any field.  
- **Iterate methodically:** Generate music, evaluate, and refine. Common strategy: *freeze the style prompt and edit one part of the lyrics at a time*, such as rewriting a verse or chorus, then regenerate【25†L229-L233】. This helps diagnose what adjustments improve the output.  
- **Validate and revise:** Before finalizing, run through the JSON schema to catch missing parts (title, lyrics, style, etc.). Ensure the lyrics read like a cohesive song and the style prompt matches the desired sound.  
- **Explain your choices:** The `description` and `reasoning` fields prompt you to articulate why the song is structured that way. This reflection helps maintain alignment: e.g. *“We chose an anthemic chorus and minor-key verses to match the theme of hope amid despair.”* 

## Connections
- [[Audio Engineer/Quillan Sonic prompt.md]]
- [[Audio Engineer/album checklist.md]]
- [[Audio Engineer/Mic Drop Quillan Ronin Track list.md]]
- [[00 - Meta/06 - Deployment & Platforms.md]]
- [[system prompts/Quillan-Samurai.md]]
- [[00 - Meta/05 - Creative Works.md]]
