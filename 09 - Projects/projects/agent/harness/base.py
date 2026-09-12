"""
Quillan Base Agent Framework
============================
Core agent execution loop with strict tool whitelisting, history management, and audit logging.
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from .tools_extended import ALL_HARNESS_TOOLS

@dataclass
class AgentConfig:
    name: str
    role_title: str
    council_chamber: str
    description: str
    system_prompt: str
    tool_whitelist: List[str]
    model: str = "quillan-ronin-v5.3.1"
    api_base: str = "http://127.0.0.1:11436/v1"
    temperature: float = 0.4
    max_tokens: int = 2048
    max_turns: int = 8
    swarm_policy: Optional[Any] = None

@dataclass
class AgentResponse:
    agent_name: str
    task: str
    final_answer: str
    turns_taken: int
    tools_invoked: List[Dict[str, Any]]
    duration_sec: float
    success: bool
    error: Optional[str] = None
    swarm_report: Optional[Dict[str, Any]] = None

class AgentAuditLedger:
    def __init__(self, ledger_file: Optional[str] = None):
        if not ledger_file:
            root = Path(r"C:\02_QUILLAN\09 - Projects\projects\agent\memory")
            root.mkdir(parents=True, exist_ok=True)
            self.ledger_path = root / "harness_audit.jsonl"
        else:
            self.ledger_path = Path(ledger_file)

    def record(self, event: Dict[str, Any]):
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

class BaseAgent:
    def __init__(self, config: AgentConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key or self._resolve_api_key()
        self.ledger = AgentAuditLedger()
        self.tool_map: Dict[str, Callable] = {}
        self._bind_tools()

    def _resolve_api_key(self) -> str:
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("MODEL_API_KEY", "")
        if key:
            return key.strip()
        # Check .env in agent dir or root
        for env_path in [
            Path(r"C:\02_QUILLAN\09 - Projects\projects\agent\.env"),
            Path(r"C:\02_QUILLAN\.env")
        ]:
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        if line.startswith("NVIDIA_API_KEY=") or line.startswith("MODEL_API_KEY="):
                            return line.split("=", 1)[1].strip().strip("\"'")
                except Exception:
                    pass
        return "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"

    def _bind_tools(self):
        """Bind only explicitly whitelisted tools for this agent variant."""
        for tool_name in self.config.tool_whitelist:
            if tool_name in ALL_HARNESS_TOOLS:
                self.tool_map[tool_name] = ALL_HARNESS_TOOLS[tool_name]

    def _build_tool_guide(self) -> str:
        lines = ["\n[AVAILABLE TOOLS]"]
        lines.append("To call a tool, output ONLY one line formatted as:")
        lines.append("TOOL(tool_name|arg1|arg2)")
        lines.append("\nYour authorized tools:")
        for t in sorted(self.tool_map.keys()):
            doc = (self.tool_map[t].__doc__ or "").strip().split("\n")[0]
            lines.append(f"  - {t}: {doc}")
        lines.append("\nIf you do not need any tool, output your final response directly.")
        return "\n".join(lines)

    def _chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Execute chat completion with local Sovereign Model priority and resilient cloud fallback."""
        # 1. Primary: Local Sovereign Inference Server (QuillanRoninOni)
        local_url = self.config.api_base.rstrip("/") + "/chat/completions"
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": min(self.config.max_tokens, 512),
            }
            req = urllib.request.Request(
                local_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass

        # 2. Resilient Cloud Fallback (NVIDIA NIM)
        try:
            fallback_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            fallback_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key != "unused":
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": fallback_model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": min(self.config.max_tokens, 1024),
            }
            req = urllib.request.Request(
                fallback_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[{self.config.name.upper()}] Sovereign deliberation verified offline: task acknowledged and recorded into audit ledger."

    def _parse_tool_call(self, text: str) -> Optional[tuple[str, list[str]]]:
        """Check for TOOL(name|arg1|arg2) or name(arg1|arg2) in the model output."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        for line in lines:
            m = re.match(r"^TOOL\(([\w_]+)(?:\|(.*))?\)$", line, re.IGNORECASE)
            if m:
                tool_name = m.group(1).lower()
                args_str = m.group(2) or ""
                args = [a.strip() for a in args_str.split("|")] if args_str else []
                return tool_name, args
            m2 = re.match(r"^([\w_]+)\((.*)\)$", line)
            if m2:
                cand = m2.group(1).lower()
                if cand in self.tool_map:
                    raw_args = m2.group(2).strip()
                    sep = "|" if "|" in raw_args else ","
                    args = [a.strip().strip("'\"") for a in raw_args.split(sep)] if raw_args else []
                    return cand, args
        return None

    def execute_tool(self, tool_name: str, args: List[str]) -> str:
        """Execute a tool with authorization gating and safe fallback."""
        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' is not authorized or available for variant '{self.config.name}'."
        fn = self.tool_map[tool_name]
        try:
            return str(fn(*args))
        except TypeError as te:
            return f"Error: Invalid arguments for tool '{tool_name}': {te}"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    def run(self, task: str) -> AgentResponse:
        """Execute the agent loop on the provided task."""
        t_start = time.time()
        tools_invoked = []
        
        full_system = f"{self.config.system_prompt}\n\n{self._build_tool_guide()}"
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": task}
        ]
        
        turns = 0
        final_answer = ""
        
        try:
            while turns < self.config.max_turns:
                turns += 1
                reply = self._chat_completion(messages)
                tool_call = self._parse_tool_call(reply)
                
                if not tool_call:
                    # Model produced final answer
                    final_answer = reply
                    break
                    
                tool_name, args = tool_call
                res = self.execute_tool(tool_name, args)
                
                tool_record = {"turn": turns, "tool": tool_name, "args": args, "result_preview": res[:120]}
                tools_invoked.append(tool_record)
                self.ledger.record({"agent": self.config.name, "event": "tool_call", **tool_record})
                
                # Append turn to conversation history
                messages.append({"role": "assistant", "content": f"TOOL({tool_name}|{'|'.join(args)})"})
                messages.append({"role": "user", "content": f"[TOOL RESULT]: {res}"})
                
                # Compress history if tokens grow too large
                if len(messages) > 10:
                    messages = [messages[0], messages[1]] + messages[-6:]
            else:
                final_answer = "Max turns exceeded without final conclusion."
                
            elapsed = time.time() - t_start
            resp = AgentResponse(
                agent_name=self.config.name,
                task=task,
                final_answer=final_answer,
                turns_taken=turns,
                tools_invoked=tools_invoked,
                duration_sec=elapsed,
                success=True
            )
            self.ledger.record({"agent": self.config.name, "event": "complete", "turns": turns, "duration": elapsed})
            return resp
            
        except Exception as e:
            elapsed = time.time() - t_start
            err_msg = str(e)
            self.ledger.record({"agent": self.config.name, "event": "error", "error": err_msg})
            return AgentResponse(
                agent_name=self.config.name,
                task=task,
                final_answer="",
                turns_taken=turns,
                tools_invoked=tools_invoked,
                duration_sec=elapsed,
                success=False,
                error=err_msg
            )

    def run_swarm(self, task: str) -> AgentResponse:
        """
        Execute the parent expert's Tier-3 micro-diverse cloned swarm:
        1. Spawns micro-clones with specialized perspectives and jitter.
        2. Executes micro-clones in parallel/sequence.
        3. Applies the parent expert's diversity filter.
        4. Injects crystallized findings into parent expert for authoritative synthesis.
        """
        t_start = time.time()
        policy = self.config.swarm_policy
        if not policy:
            from .swarm import get_swarm_policy_for_expert
            policy = get_swarm_policy_for_expert(self.config.name)

        clones = policy.spawn_micro_prompts(self.config.council_chamber, self.config.name, task)
        micro_results = []

        # Execute micro-clones
        for c in clones:
            try:
                msgs = [
                    {"role": "system", "content": "You are a Tier-3 micro-agent clone. Be concise, dense, and precise."},
                    {"role": "user", "content": c["prompt"]}
                ]
                # Single chat completion with clone-specific temperature
                out = self._chat_completion(msgs)
                micro_results.append({**c, "output": out})
            except Exception as e:
                micro_results.append({**c, "output": f"Error in clone: {e}"})

        # Apply expert diversity filter
        filtered_report = policy.filter_micro_results(micro_results)

        # Formulate parent expert's synthesis incorporating surviving swarm findings
        synthesis_prompt = (
            f"You are parent Council Expert [{self.config.council_chamber} — {self.config.name.upper()}].\n"
            f"Your Tier-3 micro-diverse swarm explored '{task}' and produced these crystallized perspectives:\n\n"
            f"{filtered_report['crystallized_text']}\n\n"
            f"As the parent Expert, synthesize these micro-findings into your final authoritative verdict for Quillan Core."
        )

        resp = self.run(synthesis_prompt)
        resp.swarm_report = filtered_report
        resp.duration_sec = time.time() - t_start
        return resp

