#!/usr/bin/env python3
"""Proof v3: webhook-triggered (server-side execution, not CLI)."""
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
    "id": "c8d9e0f1-3a4b-5c6d-9e0f-1a234567890b",
    "name": "LLM proof webhook (C7)",
    "nodes": [
        {"parameters": {"httpMethod": "POST", "path": "llm-proof",
                        "responseMode": "responseNode", "options": {}},
         "id": nid(), "name": "Proof Webhook",
         "type": "n8n-nodes-base.webhook", "typeVersion": 2,
         "position": [240, 300], "webhookId": str(uuid.uuid4())},
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
         "position": [460, 300]},
        {"parameters": {"model": MODEL, "options": {}},
         "id": nid(), "name": "OpenRouter Chat Model",
         "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter",
         "typeVersion": 1, "position": [460, 520],
         "credentials": {"openRouterApi": {"id": OR_CRED,
                                           "name": "OpenRouter"}}},
        {"parameters": {"respondWith": "json",
                        "responseBody": "={{ $json }}", "options": {}},
         "id": nid(), "name": "Respond",
         "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1,
         "position": [680, 300]},
    ],
    "connections": {
        "Proof Webhook": {"main": [[{"node": "AI Agent C7-LOGOS",
                                     "type": "main", "index": 0}]]},
        "AI Agent C7-LOGOS": {
            "main": [[{"node": "Respond", "type": "main", "index": 0}]],
            "ai_languageModel": [[{
                "node": "OpenRouter Chat Model",
                "type": "ai_languageModel", "index": 0}]]},
    },
    "settings": {"executionOrder": "v1"},
    "staticData": None, "pinData": {},
}
OUT = (r"C:\02_QUILLAN\08 - Templates & Config\workflows\llm-proof-wh.json")
open(OUT, "w", encoding="utf-8").write(json.dumps(wf, indent=2))
print("wrote", OUT)
