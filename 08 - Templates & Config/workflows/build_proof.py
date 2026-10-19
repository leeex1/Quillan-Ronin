#!/usr/bin/env python3
"""Proof-of-life v2 with the verified working free model."""
import json
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
OR_CRED = "e6a2a4ff-6f9f-4c66-a817-221922428a98"
MODEL = open(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\working_model.txt",
             encoding="utf-8").read().strip()
IDENT = json.load(open(
    r"C:\02_QUILLAN\08 - Templates & Config\workflows\council-identities.json",
    encoding="utf-8"))
c7 = next(p for p in IDENT["personas"] if p["id"] == "C7-LOGOS")


def nid():
    return str(uuid.uuid4())


wf = {
    "id": "b7c9d0e1-2f3a-4b5c-8d9e-0f123456789a",
    "name": "LLM proof-of-life (C7)",
    "nodes": [
        {"parameters": {}, "id": nid(), "name": "Manual Trigger",
         "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1,
         "position": [240, 300]},
        {"parameters": {"assignments": {"assignments": [
            {"id": nid(), "name": "query",
             "value": "What is 12 times 12? Answer briefly.",
             "type": "string"}]}, "options": {}},
         "id": nid(), "name": "Query", "type": "n8n-nodes-base.set",
         "typeVersion": 3.4, "position": [460, 300]},
        {"parameters": {
            "promptType": "define",
            "text": "={{ $json.query }}",
            "options": {
                "systemMessage": (
                    f"You are {c7['id']}-{c7['name']}, {c7['role']} "
                    f"on the Quillan-Ronin council. {c7['system_prompt']} "
                    "Answer briefly in character."),
                "maxIterations": 3}},
         "id": nid(), "name": "AI Agent C7-LOGOS",
         "type": "@n8n/n8n-nodes-langchain.agent", "typeVersion": 3.1,
         "position": [680, 300]},
        {"parameters": {"model": MODEL, "options": {}},
         "id": nid(), "name": "OpenRouter Chat Model",
         "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter",
         "typeVersion": 1, "position": [680, 500],
         "credentials": {"openRouterApi": {"id": OR_CRED,
                                           "name": "OpenRouter"}}},
    ],
    "connections": {
        "Manual Trigger": {"main": [[{"node": "Query", "type": "main",
                                      "index": 0}]]},
        "Query": {"main": [[{"node": "AI Agent C7-LOGOS", "type": "main",
                             "index": 0}]]},
        "AI Agent C7-LOGOS": {"ai_languageModel": [[{
            "node": "OpenRouter Chat Model", "type": "ai_languageModel",
            "index": 0}]]},
    },
    "settings": {"executionOrder": "v1"},
    "staticData": None, "pinData": {},
}
OUT = (r"C:\02_QUILLAN\08 - Templates & Config\workflows\llm-proof.json")
open(OUT, "w", encoding="utf-8").write(json.dumps(wf, indent=2))
print("wrote proof with model:", MODEL)
