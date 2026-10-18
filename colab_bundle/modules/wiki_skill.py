#!/usr/bin/env python3
# WikiSkill (2608.27454) — Compile agent experience into persistent wiki for skill evolution
import json
from pathlib import Path

class WikiSkill:
    """Separates experience, knowledge, skills; consolidates experience into wiki for reuse/transfer"""
    def __init__(self, wiki_path="C:/02_QUILLAN/01_Knowledge_Base/Wiki/wiki_skill.json"):
        self.wiki_path = Path(wiki_path)
        self.wiki = {"experience": [], "knowledge": {}, "skills": {}}
        if self.wiki_path.exists():
            self.wiki = json.loads(self.wiki_path.read_text())
    def consolidate(self, experience):
        # Experience -> knowledge -> skill
        self.wiki["experience"].append(experience)
        # Simple consolidation: extract knowledge, update skills
        self.wiki["knowledge"][experience["id"]] = experience["summary"]
        self.wiki_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_path.write_text(json.dumps(self.wiki, indent=2))
        return self.wiki
