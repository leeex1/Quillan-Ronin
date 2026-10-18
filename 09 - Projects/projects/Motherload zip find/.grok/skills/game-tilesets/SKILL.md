---
name: game-tilesets
description: >
  Deep guide for game TILE assets: seamless tileable textures, terrain
  transition tilesets, autotiles, and ground/platform tiles. Use whenever
  generating tileable textures, tilesets, terrain transitions, or seamless
  patterns. Complements game-asset-core.
metadata:
  short-description: "Seamless tiles and transition sets that actually tile"
user-invocable: false
---

# Tilesets

A tile's job is invisibility in repetition.
Users state WHAT they need, not how — apply everything here even when
the request never mentions it.
 Judge everything by "will the
player notice the grid?"

## 1. Seamless single tiles

- Prompt for UNIFORM stochastic texture: even density, even lighting, no
  shadows with direction, "the pattern continues off every edge".
- The repetition killer is any distinctive motif — one recognizable clump,
  flower, or rock repeats forever. Prompt for anonymous texture; verify by
  hunting for anything you could point at twice.
- MANDATORY check: composite a real 2×2 repeat with PIL to a throwaway file
  and view it. Look for (a) seam lines at the joins, (b) any motif you can
  spot in all four quadrants, (c) large-scale tone gradients that create
  checkerboarding. Any of the three = retry.

## 2. Transition tilesets (grass→dirt etc.)

- Prompt it as ONE continuous painted image that happens to be sliceable —
  never as "tiles", "cells with borders", or anything inviting separated
  sticker-tiles with gaps. Cells must be filled edge-to-edge, painted
  content flowing across cell boundaries so adjacent tiles genuinely match.
- Layout for a 3×3: center = pure inner material; edge cells = straight
  transitions facing outward; corner cells = outer corners. Verify
  DIRECTIONALITY per cell (top-center's grass is along its top edge, etc.).

## 3. Rotation economy — make fewer, better tiles

If lighting is neutral (pure top-down, no directional shading), one straight
edge and one outer corner can be ROTATED in-engine to produce all four of
each. So when the tile count is yours to choose:

- Produce: 1 center fill, 1 straight edge, 1 outer corner, 1 inner corner —
  then spend the remaining budget on VARIATIONS of the center fill (2–3
  anonymous variants breaks up repetition far better than 4 identical
  rotated edges).
- CAVEAT — rotation only works when nothing in the art encodes direction:
  no directional light, no gravity cues (hanging grass blades, drips), no
  text/emblems. Side-view (platformer) tiles almost always encode gravity
  and light, so they need all orientations painted individually. State in
  your delivery notes which tiles are rotation-safe.
- If the request explicitly fixes the grid (e.g. "3×3 with all 8
  transitions"), deliver exactly that — mention rotation economy in notes,
  don't unilaterally change the deliverable.

## 4. Platforms and props

Isolated on a keyable background, consistent lighting with their tileset,
no baked ground shadow (engines composite shadows separately).
