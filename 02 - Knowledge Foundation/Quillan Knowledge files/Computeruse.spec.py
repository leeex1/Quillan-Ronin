#!/usr/bin/env python3
"""
🖥️ Quillan-Ronin Desktop Agent (Vision + Control Loop)
Version: 2.0 (Hardened Architecture)

Features:
- Object-Oriented State Management
- Token-Optimized Vision Capture (Dynamic Downscaling)
- VLM-Optimized Relative Coordinate Mapping (0.0 to 1.0)
- Failsafe Triggers & Exception Handling
- Enforced JSON Schema Prompting
"""

import subprocess
import time
import json
import base64
import re
from io import BytesIO
from typing import Dict, Any, List

import pyautogui
from PIL import Image

# 🛡️ WARDEN PROTOCOL: Safety First
# Slam mouse to any of the 4 corners of the screen to kill the agent.
pyautogui.FAILSAFE = True

# =========================
# ⚙️ SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an autonomous desktop control agent. You receive a screenshot of the user's desktop and a goal.
Your objective is to determine the next immediate action to achieve the goal.

CRITICAL RULES:
1. Coordinate Mapping: Use RELATIVE coordinates from 0.0 to 1.0. 
   (e.g., x: 0.5, y: 0.5 is the exact center of the screen. x: 0.0, y: 0.0 is top-left).
2. You must output ONLY valid, parsable JSON. No markdown wrappers, no explanations outside the JSON.

SCHEMA:
{
  "thought": "Briefly explain your visual analysis and reasoning for the next step.",
  "action": "click" | "type" | "press" | "hotkey" | "exec" | "done",
  "args": {
     // For 'click': "x": float (0.0-1.0), "y": float (0.0-1.0)
     // For 'type': "text": string
     // For 'press': "key": string (e.g., "enter", "tab", "win")
     // For 'hotkey': "keys": ["ctrl", "c"]
     // For 'exec': "command": string
  }
}
"""

# =========================
# 🧠 AGENT ARCHITECTURE
# =========================

class QuillanDesktopAgent:
    def __init__(self, step_delay: float = 1.5, max_steps: int = 20):
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        
        # Capture environment bounds for relative mapping
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[*] Agent Initialized. Display bounds mapped: {self.screen_width}x{self.screen_height}")

    # -------------------------
    # 📸 VISION LAYER
    # -------------------------
    def capture_vision(self, max_dimension: int = 1024) -> str:
        """
        Captures screen and optimizes payload to prevent VLM context overflow.
        Maintains aspect ratio while restricting max dimension.
        """
        img = pyautogui.screenshot()
        
        # Optimization: Downscale for token efficiency
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded

    # -------------------------
    # 🖱️ ACTION SPACE
    # -------------------------
    def execute_action(self, action: str, args: Dict[str, Any]) -> str:
        """Routes and executes physical actions with safety bounds."""
        try:
            if action == "click":
                # Translate relative VLM coordinates (0.0-1.0) to absolute pixels
                rel_x = float(args.get("x", 0.5))
                rel_y = float(args.get("y", 0.5))
                
                # Clamp between 0.0 and 1.0 to prevent out-of-bounds
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))
                
                abs_x = int(rel_x * self.screen_width)
                abs_y = int(rel_y * self.screen_height)
                
                pyautogui.click(abs_x, abs_y)
                return f"Success: Clicked relative ({rel_x:.2f}, {rel_y:.2f}) -> absolute [{abs_x}, {abs_y}]"

            elif action == "type":
                text = str(args.get("text", ""))
                pyautogui.write(text, interval=0.02)
                return f"Success: Typed '{text}'"

            elif action == "press":
                key = str(args.get("key", ""))
                pyautogui.press(key)
                return f"Success: Pressed '{key}'"

            elif action == "hotkey":
                keys = args.get("keys", [])
                pyautogui.hotkey(*keys)
                return f"Success: Triggered hotkey {keys}"

            elif action == "exec":
                # ⚠️ WARDEN WARNING: Ensure execution environment is sandboxed
                command = str(args.get("command", ""))
                print(f"⚠️ SECURITY ALERT: Executing shell command: {command}")
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                output = result.stdout[:1000] + ("..." if len(result.stdout) > 1000 else "")
                err = result.stderr[:500]
                return f"Success: Executed. Out: {output} | Err: {err}"

            elif action == "done":
                return "Agent declared task complete."

            else:
                return f"Error: Unknown action '{action}'"
                
        except Exception as e:
            return f"Error during execution of {action}: {str(e)}"

    # -------------------------
    # 🧠 MODEL INTERFACE
    # -------------------------
    def _call_vlm(self, goal: str, image_b64: str) -> Dict[str, Any]:
        """
        Stub for your actual Vision-Language Model API call (OpenAI, Anthropic, Gemini, etc.).
        """
        # Construct the payload structure you would send to the API:
        # messages = [
        #    {"role": "system", "content": SYSTEM_PROMPT},
        #    {"role": "user", "content": [
        #        {"type": "text", "text": f"Goal: {goal}\nHistory: {json.dumps(self.history[-3:])}"},
        #        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        #    ]}
        # ]
        
        # ⚠️ MOCK RESPONSE FOR DEMONSTRATION
        if not self.history:
            raw_response = '{"thought": "Opening start menu.", "action": "press", "args": {"key": "win"}}'
        elif len(self.history) == 1:
            raw_response = '{"thought": "Searching for browser.", "action": "type", "args": {"text": "chrome"}}'
        elif len(self.history) == 2:
            raw_response = '{"thought": "Launching application.", "action": "press", "args": {"key": "enter"}}'
        else:
            raw_response = '{"thought": "Task complete.", "action": "done", "args": {}}'
            
        # Robust JSON parsing
        return self._parse_json_response(raw_response)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON, stripping markdown block wrappers if the LLM hallucinated them."""
        try:
            # Look for JSON block
            match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"❌ Failed to parse LLM response as JSON: {text}")
            # Return a safe fallback action to trigger a retry
            return {"action": "error", "args": {}, "thought": "Failed to parse JSON."}

    # -------------------------
    # 🔁 CORE LOOP
    # -------------------------
    def run(self, goal: str):
        print(f"\n🎯 ENGAGING AGENT GOAL: {goal}\n" + "="*40)

        for step in range(self.max_steps):
            print(f"\n--- 🔄 STEP {step+1}/{self.max_steps} ---")

            # 👁️ Observe
            print("[*] Capturing spatial data...")
            screen_b64 = self.capture_vision()

            # 🧠 Decide
            print("[*] Awaiting VLM decision...")
            decision = self._call_vlm(goal, screen_b64)

            action = decision.get("action", "error")
            args = decision.get("args", {})
            thought = decision.get("thought", "No thought provided.")

            print(f"🧠 Thought: {thought}")
            print(f"⚙️ Action : {action} | Args: {args}")

            # 🖱️ Act
            if action == "done":
                print("\n✅ GOAL ACHIEVED. Disengaging agent loop.")
                break
            
            if action == "error":
                print("⚠️ Skipping execution due to malformed LLM output.")
                result = "Failed to parse instruction."
            else:
                result = self.execute_action(action, args)
                print(f"📤 Result : {result}")

            # 📝 Record
            self.history.append({
                "step": step + 1,
                "action": action,
                "result": result
            })

            # Delay to allow UI animations/rendering to complete before next screenshot
            time.sleep(self.step_delay)
            
        else:
            print("\n⚠️ MAX STEPS REACHED. Terminating to prevent infinite loop.")

# =========================
# 🚀 ENTRY POINT
# =========================
if __name__ == "__main__":
    agent = QuillanDesktopAgent(step_delay=2.0, max_steps=15)
    
    TARGET_GOAL = "Open a browser and search for 'open source ai agents'"
    
    try:
        agent.run(TARGET_GOAL)
    except pyautogui.FailSafeException:
        print("\n🚨 FAILSAFE TRIGGERED! Mouse moved to corner. Agent terminated.")
    except KeyboardInterrupt:
        print("\n🛑 Manual interrupt received. Agent terminated.")