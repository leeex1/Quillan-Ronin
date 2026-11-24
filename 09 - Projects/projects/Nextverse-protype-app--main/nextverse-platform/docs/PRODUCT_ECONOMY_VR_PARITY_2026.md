# NextVerse product vision: subscription economy and VR/non-VR parity (2026)

## Owner's stated intent

NextVerse is the owner's attempt at a real-life Ready Player One-style platform. It is intended to become a **subscription-funded product** where tiers and exclusive items drive recurring income, and the rest of the world builds itself through users. **VR and non-VR users must receive the same access** — no second-class desktop mode, no VR-only content gating.

This document records that intent and turns it into testable requirements. It is a design/spec document; the current implementation does not yet satisfy these requirements.

## Non-negotiable principles

1. **Parity of access**: every zone, quest, event, economy feature and social space works in both VR and desktop mode. Only input mapping and presentation differ.
2. **Cosmetic exclusives, not pay-to-win**: paid tiers may unlock appearance, expression and convenience, never combat power, inventory capacity advantage, or quest skips.
3. **Local-first sovereignty**: an account and its items remain exportable; offline play works where technically possible.
4. **Provenance**: every exclusive item has a recorded origin, tier and issuance time.

## Proposed tier model (needs owner pricing decision)

| Tier | Intended access | Current implementation |
| --- | --- | --- |
| Free | Core zones, base avatar, social hubs, public events | Local demo zones + hub exist; no account/billing |
| Explorer | Extra cosmetic slots, hub hosting, richer avatar options | Not implemented |
| Creator | Stage reservation, asset upload pipeline, event tools | DAW sequencer + GLB pipeline exist as prototypes |
| Sovereign | Full world access, private guild halls, earliest event access | Guild zone ID exists; no entitlement system |

Pricing, currency and billing provider are **not decided**. This document does not invent them.

## Exclusive items system — requirements

- Item definition: `item_id`, `display_name`, `category`, `tier_requirement`, `issuance_rule`, `provenance_hash`.
- Entitlement check happens server-side before granting; client display is never authoritative.
- Exclusives are cosmetic/expressive: skins, emotes, hub decorations, stage themes.
- Provenance records are append-only and exportable with the account.

Current state: MMORPG design doc defines 12 items and 4 recipes in C++ static data; no entitlement or billing exists. **Estimated completion: 10%.**

## VR / non-VR parity — requirements

| Capability | VR path | Desktop path | Shared requirement |
| --- | --- | --- | --- |
| Locomotion | OpenXR controller/thumbstick | WASD + mouse | Same collision, same speed, same zone spawning |
| Presence | Avatar rig with head/hands | Avatar rig with gaze/pointer | Same hub presence record |
| Interaction | Controller ray/grab | Mouse click / keyboard | Same interaction events |
| Quests/economy | Identical | Identical | Server-authoritative state |
| Social hubs | Voice + spatial text | Text + same voice opt-in | Same rooms and moderation |

Current state: Godot 4.7.2 zones with collision and hub REST bridge exist in both the Godot and three.js clients, but OpenXR is not configured and no headset runtime was tested. **Estimated completion: 35% for shared foundation, 0% for actual VR runtime.**

## Acceptance criteria for parity

1. A free desktop user and a VR user can enter the same zone, see each other's avatars, and complete the same quest.
2. Entitlement grants issued by the server are identical in both clients.
3. No zone, event or item is hidden from one client for technical reasons.
4. A headless test verifies shared world-state endpoints; a device test verifies OpenXR and desktop input produce equivalent movement events.

## Honest assessment

The owner's intuition — that a sovereign, user-built, subscription-funded world with equal VR/desktop access is possible — is not disproven by this prototype. But the prototype is a foundation, not the product. Subscription billing, entitlement services, OpenXR, content pipeline and moderation are all still ahead. The current code provides zones, quests, presence, a DAW signal path and measured acceleration; it does not yet provide the economy or VR parity described here.
