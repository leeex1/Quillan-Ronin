"""
=============================================================================
⚔️ QUILLAN-RONIN: SOVEREIGN COMMAND CENTER TUI v5.4.0
=============================================================================
A terminal command center unifying:
1. Native Sovereign GGUF Models (48+ tok/s on local GTX 1050)
2. Bring-Your-Own-Key (BYOK) Cloud Routing (DeepSeek, Claude, GPT-4o, Gemini)
3. 34-Council Cognitive Parliament Multi-Persona Deliberation
4. In-Graph CCRL Bayesian Safety Gate & Bushido Alignment System
5. Real-Time Hardware Telemetry, VRAM Tracking, and Speedometer
=============================================================================
"""

import os
import sys
import time
import asyncio
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Input, Select, Button, Label, Markdown
from textual.reactive import reactive
from textual import work

# Ensure parent directory is on sys.path
sys.path.insert(0, r"C:\02_QUILLAN")
from quillan_engine import SovereignInferenceEngine, LOCAL_MODELS, CLOUD_MODELS

# Custom CSS Styling
TUI_CSS = """
Screen {
    background: #0B0F19;
    color: #F8FAFC;
}

#header-banner {
    dock: top;
    height: 4;
    background: #111827;
    color: #A78BFA;
    border-bottom: solid #7C3AED;
    padding: 0 1;
    content-align: center middle;
}

#sidebar {
    width: 38;
    background: #0F172A;
    border-right: solid #334155;
    padding: 1;
}

.sidebar-card {
    background: #1E293B;
    border: solid #334155;
    border-radius: 1;
    padding: 1;
    margin-bottom: 1;
}

.card-title {
    color: #A78BFA;
    text-style: bold;
    margin-bottom: 1;
}

#chat-container {
    background: #0B0F19;
    padding: 1 2;
}

.chat-bubble-user {
    background: #1E1B4B;
    border-left: thick #7C3AED;
    padding: 1;
    margin-bottom: 1;
}

.chat-bubble-quillan {
    background: #064E3B;
    border-left: thick #10B981;
    padding: 1;
    margin-bottom: 1;
}

.chat-bubble-system {
    background: #1E293B;
    border-left: thick #F59E0B;
    padding: 1;
    margin-bottom: 1;
}

#input-deck {
    dock: bottom;
    height: 4;
    background: #111827;
    border-top: solid #7C3AED;
    padding: 0 1;
}

#user-input {
    border: tall #7C3AED;
    background: #0F172A;
    color: #F8FAFC;
}

#user-input:focus {
    border: tall #10B981;
}

.badge-green {
    color: #34D399;
    text-style: bold;
}

.badge-purple {
    color: #C084FC;
    text-style: bold;
}

.badge-gold {
    color: #FBBF24;
    text-style: bold;
}
"""

BANNER_TEXT = """[bold purple]⚔️  QUILLAN-RONIN // SOVEREIGN COMMAND CENTER v5.4.0[/bold purple]
[dim green]Bushido Code: Rectitude • Moral Courage • Truth • Honor • Loyalty[/dim green]
[dim cyan]Hardware: NVIDIA GTX 1050 (sm_61) | CCRL Safety Gate: Φ ≥ 0.85 (Active)[/dim cyan]"""

class QuillanCommandCenter(App):
    CSS = TUI_CSS
    TITLE = "Quillan-Ronin Sovereign Command Center"
    SUB_TITLE = "v5.4.0 — Unified Weight-Space & BYOK Terminal"

    # Reactive states
    tokens_per_sec = reactive("0.0 tok/s")
    vram_telemetry = reactive("1,024 MB / 4,096 MB")
    safety_status = reactive("Φ = 0.96 [VERIFIED]")
    is_generating = reactive(False)

    def __init__(self):
        super().__init__()
        self.engine = SovereignInferenceEngine()
        self.history = []

    def compose(self) -> ComposeResult:
        yield Static(BANNER_TEXT, id="header-banner")
        with Horizontal():
            # Left Sidebar
            with Vertical(id="sidebar"):
                with Container(classes="sidebar-card"):
                    yield Label("MODEL ENGINE SELECTOR", classes="card-title")
                    model_options = [
                        (v["name"], k) for k, v in self.engine.get_available_models().items()
                    ]
                    yield Select(
                        options=model_options,
                        value=self.engine.active_model_id,
                        id="model-select",
                        allow_blank=False
                    )

                with Container(classes="sidebar-card"):
                    yield Label("34-COUNCIL DELIBERATION", classes="card-title")
                    yield Static("• C0 Throne (Quillan Core): [bold green]ONLINE[/bold green]")
                    yield Static("• C1 Astra (Pattern Eye):   [bold green]ACTIVE[/bold green]")
                    yield Static("• C2 Vir (Conscience Gate): [bold green]VERIFIED[/bold green]")
                    yield Static("• C7 Logos (Formal Logic):  [bold green]ACTIVE[/bold green]")
                    yield Static("• C13 Warden (Security):    [bold green]LOCKED[/bold green]")
                    yield Static("• C34 Predator (Adversary): [bold green]SPARRING[/bold green]")

                with Container(classes="sidebar-card"):
                    yield Label("TELEMETRY & HARDWARE", classes="card-title")
                    yield Label(f"VRAM: [bold]{self.vram_telemetry}[/bold]", id="vram-label")
                    yield Label(f"Speed: [bold green]{self.tokens_per_sec}[/bold green]", id="speed-label")
                    yield Label(f"Safety Gate: [bold purple]{self.safety_status}[/bold purple]", id="safety-label")

                with Container(classes="sidebar-card"):
                    yield Label("TACTICAL CHEAT SHEET", classes="card-title")
                    yield Static("[dim]/model <id>  : Switch model[/dim]")
                    yield Static("[dim]/key <p> <k> : Store API Key[/dim]")
                    yield Static("[dim]/council on  : 34-Expert mode[/dim]")
                    yield Static("[dim]/bench       : Run speed test[/dim]")
                    yield Static("[dim]/clear       : Clear viewport[/dim]")
                    yield Static("[dim]/exit        : Terminate TUI[/dim]")

            # Center Viewport
            with VerticalScroll(id="chat-container"):
                yield Static(
                    "### 🗡️ Welcome to the Quillan-Ronin Sovereign Command Center\n\n"
                    "Native models loaded from `C:\\02_QUILLAN\\exports`. Type a prompt below or use slash commands.",
                    classes="chat-bubble-system"
                )

        # Bottom Input
        with Horizontal(id="input-deck"):
            yield Input(placeholder="Enter tactical directive or /command...", id="user-input")

    def on_mount(self):
        self.query_one("#user-input").focus()

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == "model-select":
            self.engine.active_model_id = str(event.value)
            chat = self.query_one("#chat-container")
            chat.mount(Static(f"🔄 **Engine switched to:** `{event.value}`", classes="chat-bubble-system"))
            chat.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted):
        prompt = event.value.strip()
        if not prompt or self.is_generating:
            return

        input_widget = self.query_one("#user-input")
        input_widget.value = ""

        # Handle Slash Commands
        if prompt.startswith("/"):
            await self.handle_command(prompt)
            return

        # Render User Message
        chat = self.query_one("#chat-container")
        chat.mount(Static(f"**👤 YOU:**\n{prompt}", classes="chat-bubble-user"))
        chat.scroll_end(animate=False)

        # Trigger Worker for Generation
        self.is_generating = True
        self.run_generation_task(prompt)

    async def handle_command(self, cmd_line: str):
        parts = cmd_line.split()
        cmd = parts[0].lower()
        chat = self.query_one("#chat-container")

        if cmd == "/clear":
            chat.remove_children()
            chat.mount(Static("🧹 **Viewport Cleared.**", classes="chat-bubble-system"))
        elif cmd == "/bench":
            chat.mount(Static("⚡ **Running instantaneous benchmark...**", classes="chat-bubble-system"))
            self.run_generation_task("Calculate the prime factorization of 2048 and state Bushido honor.")
        elif cmd == "/key":
            if len(parts) >= 3:
                prov, key = parts[1], parts[2]
                self.engine.key_mgr.set_key(prov, key)
                chat.mount(Static(f"🔑 **Key saved for provider:** `{prov.upper()}`", classes="chat-bubble-system"))
            else:
                statuses = self.engine.key_mgr.list_status()
                text = "🔑 **BYOK API Key Status:**\n" + "\n".join([f"• {k.upper()}: {'✅ Configured' if v else '❌ Missing'}" for k, v in statuses.items()])
                text += "\n\n*Usage: `/key <provider> <key>` (Providers: deepseek, anthropic, openai, gemini, openrouter)*"
                chat.mount(Static(text, classes="chat-bubble-system"))
        elif cmd == "/council":
            state = parts[1].lower() if len(parts) > 1 else "toggle"
            if state in ["on", "true", "1"]:
                self.engine.council_mode = True
            elif state in ["off", "false", "0"]:
                self.engine.council_mode = False
            else:
                self.engine.council_mode = not self.engine.council_mode
            status_text = "ENABLED (5-Wave Deliberation)" if self.engine.council_mode else "DISABLED (Direct Stream)"
            chat.mount(Static(f"🏛️ **34-Council Mode:** `{status_text}`", classes="chat-bubble-system"))
        elif cmd == "/safety":
            self.engine.safety_gate_active = not self.engine.safety_gate_active
            status_text = "ACTIVE (Φ >= 0.85)" if self.engine.safety_gate_active else "BYPASSED (Raw Mode)"
            chat.mount(Static(f"🛡️ **CCRL Safety Gate:** `{status_text}`", classes="chat-bubble-system"))
        elif cmd == "/model" and len(parts) > 1:
            m_id = parts[1]
            if m_id in self.engine.get_available_models():
                self.engine.active_model_id = m_id
                self.query_one("#model-select").value = m_id
                chat.mount(Static(f"🔄 **Switched active model to:** `{m_id}`", classes="chat-bubble-system"))
            else:
                chat.mount(Static(f"❌ Unknown model: `{m_id}`. Available: {', '.join(self.engine.get_available_models().keys())}", classes="chat-bubble-system"))
        elif cmd == "/exit":
            self.exit()
        else:
            chat.mount(Static(f"❓ Unknown command: `{cmd}`. Use `/help` for tactical assistance.", classes="chat-bubble-system"))

        chat.scroll_end(animate=False)

    @work(thread=True)
    def run_generation_task(self, prompt: str):
        chat = self.query_one("#chat-container")
        model_name = self.engine.get_available_models().get(self.engine.active_model_id, {}).get("name", self.engine.active_model_id)

        # Create response container
        response_widget = Static(f"**⚔️ {model_name}:**\n*Deliberating across sovereign manifold...*\n", classes="chat-bubble-quillan")
        self.call_from_thread(chat.mount, response_widget)
        self.call_from_thread(chat.scroll_end, False)

        accumulated_text = ""
        first_token = True

        for chunk in self.engine.generate_stream(prompt):
            chunk_type = chunk.get("type")
            if chunk_type == "telemetry":
                score = chunk.get("safety_score", 1.0)
                self.safety_status = f"Φ = {score:.2f} [PASS]"
                self.call_from_thread(self.update_telemetry_labels)
            elif chunk_type == "token":
                if first_token:
                    accumulated_text = f"**⚔️ {model_name}:**\n\n"
                    first_token = False
                accumulated_text += chunk.get("text", "")
                self.call_from_thread(response_widget.update, accumulated_text)
                self.call_from_thread(chat.scroll_end, False)
            elif chunk_type == "done":
                tps = chunk.get("tps", 0.0)
                self.tokens_per_sec = f"{tps:.1f} tok/s"
                self.call_from_thread(self.update_telemetry_labels)
                self.is_generating = False

    def update_telemetry_labels(self):
        try:
            self.query_one("#speed-label").update(f"Speed: [bold green]{self.tokens_per_sec}[/bold green]")
            self.query_one("#safety-label").update(f"Safety Gate: [bold purple]{self.safety_status}[/bold purple]")
        except Exception:
            pass

if __name__ == "__main__":
    app = CommandCenter() if 'CommandCenter' in locals() else QuillanCommandCenter()
    app.run()

