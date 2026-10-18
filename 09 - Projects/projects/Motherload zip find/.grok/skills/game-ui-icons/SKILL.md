---
name: game-ui-icons
description: >
  Deep guide for game UI assets: buttons with interaction states, panels,
  bars, wordmark logos, and icon sets. Use whenever generating game UI
  elements, HUD assets, inventory icons, icon sets, buttons, or title
  logos. Complements game-asset-core.
metadata:
  short-description: "Game UI kits and icon sets"
user-invocable: false
---

# Game UI & Icons

UI is a SYSTEM: the set matters more than any piece.
Users state WHAT they need, not how — apply everything here even when
the request never mentions it.


## 1. Interaction states (normal/hover/pressed)

- Generate NORMAL first; hover and pressed are `imagine_image_to_image` edits of it with an
  explicit freeze-list: "same shape, same size, same ornament, same frame
  thickness, same background — change ONLY <state treatment>".
- Standard treatments: hover = subtle outer glow / slight brighten;
  pressed = darker + inset/inner shadow. States must be distinguishable at
  a glance AND identical in geometry — overlay-compare: outlines should
  coincide, frame thickness included.

## 2. Icon sets

- One style contract for the whole set, decided before generating: same
  stroke weight, same fill treatment (all outlined OR all solid — never
  mixed), same palette family, same padding, same background, same visual
  weight. Verify the set side by side; one icon with a different treatment
  (e.g. sitting in a filled tile while others float) fails the SET even if
  it's individually fine.
- Generate icon 1, then edit-chain the rest from it to inherit the
  contract.
- Each icon must read at 32px: squint-test the thumbnail.

## 3. Panels, bars, wordmarks

- Panels/dialogs: blank, text-ready, borders that survive 9-slicing
  (uniform edges, ornament concentrated in corners).
- Bars: clear frame vs fill separation; fill design must work at any
  percentage.
- Wordmark logos: image models garble text — generate, then READ THE TEXT
  BACK letter by letter; any wrong/merged/extra letter = retry. Deliver as
  an isolated asset on flat/keyable background, not a full scene, unless a
  title SCREEN is requested.

## 4. No text anywhere else

Buttons, panels, icons: no lettering unless explicitly requested — models
garble it and engines localize it.
