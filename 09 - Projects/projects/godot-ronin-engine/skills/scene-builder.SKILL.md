# Skill: scene-builder
Use when the user asks to create/add/arrange anything in a Godot scene.

Rules:
- Always list existing nodes first (MCP list_project).
- One node per tool call. Prefer native types: Node2D, Sprite2D, CharacterBody2D, Control.
- After creating, call save_scene + run_game to verify, check logs for errors.
- Keep scenes under 50 nodes. No placeholder spam.
- If a request is ambiguous, ask ONE clarifying question, then build.
