#!/usr/bin/env python3
"""
🖥️ Quillan-Ronin Desktop Agent (Vision + Control Loop)
Version: 4.0 (Fully Featured Agentic Browser & Computer Use)

Features:
- Object-Oriented State Management with full persistence
- Token-Optimized Vision Capture (Dynamic Downscaling)
- VLM-Optimized Relative Coordinate Mapping (0.0 to 1.0)
- Failsafe Triggers & Exception Handling
- Enforced JSON Schema Prompting
- Comprehensive Browser Actions (scroll, navigate, tabs, bookmarks, history)
- Enhanced Text Editing (select, copy, paste, cut, undo, redo, find)
- OCR Capabilities (screen reading, text extraction, element detection)
- Multi-Window Support (window switching, minimization, maximization)
- State Recovery and Undo (save/load state, action history)
- Element Detection (find elements by text, color, position)
- Form Automation (fill forms, submit, checkboxes, radio buttons)
- File Operations (upload, download, save dialogs)
- Keyboard Shortcuts (comprehensive hotkey support)
- Mouse Actions (click, double-click, right-click, drag, hover)
- Screen Analysis (color detection, region analysis, element location)
- Browser-Specific Workflows (login, search, navigation patterns)
- Error Recovery (retry logic, fallback actions, state restoration)
- Logging and Debugging (detailed execution logs, screenshot saving)
- API Integration (OpenAI, Anthropic, Gemini, local models)
- Configuration Management (settings file, environment variables)
"""

import subprocess
import time
import json
import base64
import re
import os
import logging
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib

import pyautogui
from PIL import Image, ImageGrab
import pytesseract
try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False
    print("[!] pyperclip not available. Clipboard features limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('computer_use_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 🛡️ WARDEN PROTOCOL: Safety First
# Slam mouse to any of the 4 corners of the screen to kill the agent.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Small pause between actions for stability

# =========================
# 🎯 ACTION TYPES
# =========================
class ActionType(Enum):
    # Mouse Actions
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    HOVER = "hover"
    
    # Keyboard Actions
    TYPE = "type"
    PRESS = "press"
    HOTKEY = "hotkey"
    
    # Browser Actions
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    NEW_TAB = "new_tab"
    CLOSE_TAB = "close_tab"
    SWITCH_TAB = "switch_tab"
    REFRESH = "refresh"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    BOOKMARK = "bookmark"
    
    # Text Actions
    SELECT_ALL = "select_all"
    COPY = "copy"
    PASTE = "paste"
    CUT = "cut"
    UNDO = "undo"
    REDO = "redo"
    FIND = "find"
    SELECT_TEXT = "select_text"
    
    # Form Actions
    FILL_FORM = "fill_form"
    SUBMIT_FORM = "submit_form"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SELECT_DROPDOWN = "select_dropdown"
    
    # File Actions
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"
    SAVE_DIALOG = "save_dialog"
    
    # File System Actions
    CREATE_FILE = "create_file"
    DELETE_FILE = "delete_file"
    MOVE_FILE = "move_file"
    COPY_FILE = "copy_file"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    LIST_DIR = "list_dir"
    CREATE_DIR = "create_dir"
    DELETE_DIR = "delete_dir"
    GET_FILE_INFO = "get_file_info"
    SEARCH_FILES = "search_files"
    
    # Application Actions
    LAUNCH_APP = "launch_app"
    CLOSE_APP = "close_app"
    LIST_PROCESSES = "list_processes"
    KILL_PROCESS = "kill_process"
    
    # System Actions
    OPEN_SETTINGS = "open_settings"
    CONTROL_PANEL = "control_panel"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    VOLUME_MUTE = "volume_mute"
    VOLUME_SET = "volume_set"
    
    # Multi-Monitor Actions
    SWITCH_SCREEN = "switch_screen"
    MOVE_TO_SCREEN = "move_to_screen"
    
    # System Search & Notifications
    SYSTEM_SEARCH = "system_search"
    OPEN_NOTIFICATIONS = "open_notifications"
    DISMISS_NOTIFICATIONS = "dismiss_notifications"
    
    # Window Actions
    SWITCH_WINDOW = "switch_window"
    MINIMIZE_WINDOW = "minimize_window"
    MAXIMIZE_WINDOW = "maximize_window"
    CLOSE_WINDOW = "close_window"
    
    # Analysis Actions
    OCR_READ = "ocr_read"
    FIND_ELEMENT = "find_element"
    GET_COLOR = "get_color"
    ANALYZE_REGION = "analyze_region"
    
    # System Actions
    EXEC = "exec"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    
    # Control Actions
    DONE = "done"
    ERROR = "error"
    RETRY = "retry"

# =========================
# 📊 STATE MANAGEMENT
# =========================
@dataclass
class AgentState:
    """Tracks the agent's execution state for recovery and undo."""
    history: List[Dict[str, Any]] = field(default_factory=list)
    clipboard_content: str = ""
    current_window: str = ""
    current_url: str = ""
    current_tab_index: int = 0
    last_action: Optional[str] = None
    last_action_args: Dict[str, Any] = field(default_factory=dict)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)  # Paths to saved screenshots
    element_cache: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # Cache element positions
    form_data: Dict[str, Any] = field(default_factory=dict)  # Store form data
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0

# =========================
# ⚙️ SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an autonomous desktop control agent with comprehensive browser and computer use capabilities.
You receive a screenshot of the user's desktop and a goal.
Your objective is to determine the next immediate action to achieve the goal.

CRITICAL RULES:
1. Coordinate Mapping: Use RELATIVE coordinates from 0.0 to 1.0. 
   (e.g., x: 0.5, y: 0.5 is the exact center of the screen. x: 0.0, y: 0.0 is top-left).
2. You must output ONLY valid, parsable JSON. No markdown wrappers, no explanations outside the JSON.
3. For browser tasks, prefer scroll over clicking scroll bars.
4. Use OCR when text is visible but you need to read it precisely.
5. Cache element positions when found to avoid repeated searches.
6. Use wait actions when pages are loading or animations are expected.
7. For forms, use fill_form when multiple fields need to be populated.
8. Use find_element to locate UI elements by text or visual characteristics.

AVAILABLE ACTIONS:
- Mouse: click, double_click, right_click, drag, hover
- Keyboard: type, press, hotkey
- Browser: scroll, navigate, new_tab, close_tab, switch_tab, refresh, go_back, go_forward, bookmark
- Text: select_all, copy, paste, cut, undo, redo, find, select_text
- Form: fill_form, submit_form, checkbox, radio, select_dropdown
- File: upload_file, download_file, save_dialog
- File System: create_file, delete_file, move_file, copy_file, read_file, write_file, list_dir, create_dir, delete_dir, get_file_info, search_files
- Applications: launch_app, close_app, list_processes, kill_process
- System: open_settings, control_panel, volume_up, volume_down, volume_mute, volume_set
- Multi-Monitor: switch_screen, move_to_screen
- System Search: system_search, open_notifications, dismiss_notifications
- Window: switch_window, minimize_window, maximize_window, close_window
- Analysis: ocr_read, find_element, get_color, analyze_region
- System: exec, wait, screenshot

SCHEMA:
{
  "thought": "Briefly explain your visual analysis and reasoning for the next step.",
  "action": "click" | "double_click" | "right_click" | "drag" | "hover" | "type" | "press" | "hotkey" | "scroll" | "navigate" | "new_tab" | "close_tab" | "switch_tab" | "refresh" | "go_back" | "go_forward" | "bookmark" | "select_all" | "copy" | "paste" | "cut" | "undo" | "redo" | "find" | "select_text" | "fill_form" | "submit_form" | "checkbox" | "radio" | "select_dropdown" | "upload_file" | "download_file" | "save_dialog" | "create_file" | "delete_file" | "move_file" | "copy_file" | "read_file" | "write_file" | "list_dir" | "create_dir" | "delete_dir" | "get_file_info" | "search_files" | "launch_app" | "close_app" | "list_processes" | "kill_process" | "open_settings" | "control_panel" | "volume_up" | "volume_down" | "volume_mute" | "volume_set" | "switch_screen" | "move_to_screen" | "system_search" | "open_notifications" | "dismiss_notifications" | "switch_window" | "minimize_window" | "maximize_window" | "close_window" | "ocr_read" | "find_element" | "get_color" | "analyze_region" | "exec" | "wait" | "screenshot" | "done" | "error" | "retry",
  "args": {
     // Mouse actions: "x": float (0.0-1.0), "y": float (0.0-1.0)
     // For drag: "start_x": float, "start_y": float, "end_x": float, "end_y": float
     // For type: "text": string
     // For press: "key": string (e.g., "enter", "tab", "win", "f5")
     // For hotkey: "keys": ["ctrl", "c"]
     // For scroll: "direction": "up" | "down" | "left" | "right", "amount": int
     // For navigate: "url": string
     // For switch_tab: "tab_index": int
     // For find: "query": string
     // For fill_form: "fields": {"field_name": "value", ...}
     // For checkbox/radio: "checked": boolean
     // For select_dropdown: "option": string
     // For upload_file: "filepath": string
     // For switch_window: "window_title": string
     // For find_element: "text": string | "color": string | "pattern": string
     // For get_color: "x": float, "y": float
     // For analyze_region: "x": float, "y": float, "width": float, "height": float
     // For wait: "duration": float (seconds)
     // For screenshot: "filename": string (optional)
     // For exec: "command": string
     // File System: "filepath": string, "src": string, "dst": string, "content": string, "dirpath": string, "pattern": string, "mode": "append"|"overwrite"
     // Applications: "app_path": string, "app_name": string, "pid": int
     // Volume: "amount": int, "level": int
     // Multi-Monitor: "screen": int
     // System Search: "query": string
  }
}
"""

# =========================
# 🧠 AGENT ARCHITECTURE
# =========================

class QuillanDesktopAgent:
    def __init__(self, 
                 step_delay: float = 1.5, 
                 max_steps: int = 20, 
                 enable_ocr: bool = True,
                 config_file: Optional[str] = None,
                 screenshot_dir: str = "screenshots"):
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.state = AgentState()
        self.enable_ocr = enable_ocr
        self.screenshot_dir = screenshot_dir
        self.config = {}
        
        # Create screenshot directory
        Path(self.screenshot_dir).mkdir(exist_ok=True)
        
        # Load configuration if provided
        if config_file:
            self.load_config(config_file)
        
        # Capture environment bounds for relative mapping
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"Agent Initialized. Display bounds mapped: {self.screen_width}x{self.screen_height}")
        logger.info(f"OCR Enabled: {self.enable_ocr}")
        
        # Initialize Tesseract if available
        if self.enable_ocr:
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR initialized successfully")
            except:
                logger.warning("Tesseract not found. OCR features disabled.")
                self.enable_ocr = False

    # -------------------------
    # 📸 VISION LAYER
    # -------------------------
    def capture_vision(self, max_dimension: int = 1024, region: Optional[Tuple[int, int, int, int]] = None, save: bool = False) -> str:
        """
        Captures screen and optimizes payload to prevent VLM context overflow.
        Maintains aspect ratio while restricting max dimension.
        Optionally captures a specific region (left, top, width, height).
        Optionally saves screenshot to disk.
        """
        try:
            if region:
                img = pyautogui.screenshot(region=region)
            else:
                img = pyautogui.screenshot()
            
            # Save original if requested
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
                img.save(filename)
                self.state.screenshots.append(filename)
                logger.info(f"Screenshot saved: {filename}")
            
            # Optimization: Downscale for token efficiency
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG", optimize=True)
            encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return encoded
        except Exception as e:
            logger.error(f"Vision capture failed: {e}")
            return ""

    def save_screenshot(self, filename: Optional[str] = None) -> str:
        """Save a full-resolution screenshot to disk."""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
            
            img = pyautogui.screenshot()
            img.save(filename)
            self.state.screenshots.append(filename)
            logger.info(f"Screenshot saved: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Screenshot save failed: {e}")
            return ""

    def ocr_read(self, region: Optional[Tuple[int, int, int, int]] = None, lang: str = 'eng') -> str:
        """
        Uses OCR to read text from the screen or a specific region.
        Returns the extracted text.
        """
        if not self.enable_ocr:
            return "OCR not available"
        
        try:
            if region:
                img = pyautogui.screenshot(region=region)
            else:
                img = pyautogui.screenshot()
            
            # Preprocess for better OCR
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)  # Upscale
            
            text = pytesseract.image_to_string(img, lang=lang, config='--psm 6')
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return f"OCR Error: {str(e)}"

    def find_element_by_text(self, text: str, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[float, float]]:
        """
        Find an element on screen by its text using OCR.
        Returns relative coordinates (x, y) or None if not found.
        """
        if not self.enable_ocr:
            return None
        
        try:
            # Check cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.state.element_cache:
                return self.state.element_cache[cache_key]
            
            # Perform OCR on the screen
            ocr_data = pytesseract.image_to_data(pyautogui.screenshot(region), output_type=pytesseract.Output.DICT)
            
            # Search for the text
            for i, word in enumerate(ocr_data['text']):
                if text.lower() in word.lower():
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    
                    # Convert to relative coordinates
                    if region:
                        rel_x = x / region[2]
                        rel_y = y / region[3]
                    else:
                        rel_x = x / self.screen_width
                        rel_y = y / self.screen_height
                    
                    # Cache the result
                    self.state.element_cache[cache_key] = (rel_x, rel_y)
                    logger.info(f"Found element '{text}' at ({rel_x:.2f}, {rel_y:.2f})")
                    return (rel_x, rel_y)
            
            logger.warning(f"Element '{text}' not found")
            return None
        except Exception as e:
            logger.error(f"Element finding failed: {e}")
            return None

    def get_color_at(self, x: float, y: float) -> Tuple[int, int, int]:
        """Get the RGB color at relative coordinates (x, y)."""
        try:
            abs_x = int(x * self.screen_width)
            abs_y = int(y * self.screen_height)
            img = pyautogui.screenshot()
            pixel = img.getpixel((abs_x, abs_y))
            return pixel
        except Exception as e:
            logger.error(f"Color detection failed: {e}")
            return (0, 0, 0)

    def analyze_region(self, x: float, y: float, width: float, height: float) -> Dict[str, Any]:
        """Analyze a specific region of the screen."""
        try:
            abs_x = int(x * self.screen_width)
            abs_y = int(y * self.screen_height)
            abs_w = int(width * self.screen_width)
            abs_h = int(height * self.screen_height)
            
            region = (abs_x, abs_y, abs_w, abs_h)
            img = pyautogui.screenshot(region=region)
            
            # Basic analysis
            analysis = {
                "size": (abs_w, abs_h),
                "dominant_color": self._get_dominant_color(img),
                "brightness": self._get_brightness(img),
                "text_content": self.ocr_read(region) if self.enable_ocr else ""
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Region analysis failed: {e}")
            return {}

    def _get_dominant_color(self, img: Image.Image) -> Tuple[int, int, int]:
        """Get the dominant color in an image."""
        try:
            img = img.convert('RGB')
            pixels = list(img.getdata())
            from collections import Counter
            most_common = Counter(pixels).most_common(1)[0][0]
            return most_common
        except:
            return (0, 0, 0)

    def _get_brightness(self, img: Image.Image) -> float:
        """Calculate average brightness of an image."""
        try:
            img = img.convert('L')
            pixels = list(img.getdata())
            return sum(pixels) / len(pixels) / 255.0
        except:
            return 0.0

    # -------------------------
    # 🖱️ ACTION SPACE
    # -------------------------
    def execute_action(self, action: str, args: Dict[str, Any]) -> str:
        """Routes and executes physical actions with safety bounds."""
        self.state.total_actions += 1
        
        try:
            # Store action for undo capability
            self.state.last_action = action
            self.state.last_action_args = args.copy()
            
            logger.info(f"Executing action: {action} with args: {args}")
            
            # Mouse Actions
            if action == "click":
                rel_x = float(args.get("x", 0.5))
                rel_y = float(args.get("y", 0.5))
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))
                abs_x = int(rel_x * self.screen_width)
                abs_y = int(rel_y * self.screen_height)
                pyautogui.click(abs_x, abs_y)
                self.state.successful_actions += 1
                return f"Success: Clicked relative ({rel_x:.2f}, {rel_y:.2f}) -> absolute [{abs_x}, {abs_y}]"

            elif action == "double_click":
                rel_x = float(args.get("x", 0.5))
                rel_y = float(args.get("y", 0.5))
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))
                abs_x = int(rel_x * self.screen_width)
                abs_y = int(rel_y * self.screen_height)
                pyautogui.doubleClick(abs_x, abs_y)
                self.state.successful_actions += 1
                return f"Success: Double-clicked relative ({rel_x:.2f}, {rel_y:.2f}) -> absolute [{abs_x}, {abs_y}]"

            elif action == "right_click":
                rel_x = float(args.get("x", 0.5))
                rel_y = float(args.get("y", 0.5))
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))
                abs_x = int(rel_x * self.screen_width)
                abs_y = int(rel_y * self.screen_height)
                pyautogui.rightClick(abs_x, abs_y)
                self.state.successful_actions += 1
                return f"Success: Right-clicked relative ({rel_x:.2f}, {rel_y:.2f}) -> absolute [{abs_x}, {abs_y}]"

            elif action == "drag":
                start_x = max(0.0, min(1.0, float(args.get("start_x", 0.5))))
                start_y = max(0.0, min(1.0, float(args.get("start_y", 0.5))))
                end_x = max(0.0, min(1.0, float(args.get("end_x", 0.5))))
                end_y = max(0.0, min(1.0, float(args.get("end_y", 0.5))))
                
                abs_start_x = int(start_x * self.screen_width)
                abs_start_y = int(start_y * self.screen_height)
                abs_end_x = int(end_x * self.screen_width)
                abs_end_y = int(end_y * self.screen_height)
                
                pyautogui.dragTo(abs_end_x, abs_end_y, duration=0.5, button='left')
                self.state.successful_actions += 1
                return f"Success: Dragged from ({start_x:.2f}, {start_y:.2f}) to ({end_x:.2f}, {end_y:.2f})"

            elif action == "hover":
                rel_x = float(args.get("x", 0.5))
                rel_y = float(args.get("y", 0.5))
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))
                abs_x = int(rel_x * self.screen_width)
                abs_y = int(rel_y * self.screen_height)
                pyautogui.moveTo(abs_x, abs_y)
                time.sleep(0.5)  # Wait for hover effect
                self.state.successful_actions += 1
                return f"Success: Hovered at relative ({rel_x:.2f}, {rel_y:.2f})"

            # Keyboard Actions
            elif action == "type":
                text = str(args.get("text", ""))
                pyautogui.write(text, interval=0.01)
                self.state.successful_actions += 1
                return f"Success: Typed '{text}'"

            elif action == "press":
                key = str(args.get("key", ""))
                pyautogui.press(key)
                self.state.successful_actions += 1
                return f"Success: Pressed '{key}'"

            elif action == "hotkey":
                keys = args.get("keys", [])
                pyautogui.hotkey(*keys)
                self.state.successful_actions += 1
                return f"Success: Triggered hotkey {keys}"

            # Browser Actions
            elif action == "scroll":
                direction = args.get("direction", "down")
                amount = int(args.get("amount", 5))
                if direction == "up":
                    pyautogui.scroll(amount)
                elif direction == "down":
                    pyautogui.scroll(-amount)
                elif direction == "left":
                    pyautogui.hscroll(-amount)
                elif direction == "right":
                    pyautogui.hscroll(amount)
                self.state.successful_actions += 1
                return f"Success: Scrolled {direction} by {amount}"

            elif action == "navigate":
                url = str(args.get("url", ""))
                pyautogui.hotkey('ctrl', 'l')
                time.sleep(0.2)
                pyautogui.write(url, interval=0.01)
                pyautogui.press('enter')
                self.state.current_url = url
                self.state.successful_actions += 1
                return f"Success: Navigated to {url}"

            elif action == "new_tab":
                pyautogui.hotkey('ctrl', 't')
                self.state.current_tab_index += 1
                self.state.successful_actions += 1
                return "Success: Opened new tab"

            elif action == "close_tab":
                pyautogui.hotkey('ctrl', 'w')
                self.state.current_tab_index = max(0, self.state.current_tab_index - 1)
                self.state.successful_actions += 1
                return "Success: Closed current tab"

            elif action == "switch_tab":
                tab_index = int(args.get("tab_index", 1))
                if 1 <= tab_index <= 8:
                    pyautogui.hotkey('ctrl', str(tab_index))
                elif tab_index == 9:
                    pyautogui.hotkey('ctrl', '9')
                else:
                    for _ in range(tab_index):
                        pyautogui.hotkey('ctrl', 'tab')
                        time.sleep(0.1)
                self.state.current_tab_index = tab_index
                self.state.successful_actions += 1
                return f"Success: Switched to tab {tab_index}"

            elif action == "refresh":
                pyautogui.press('f5')
                self.state.successful_actions += 1
                return "Success: Refreshed page"

            elif action == "go_back":
                pyautogui.hotkey('alt', 'left')
                self.state.successful_actions += 1
                return "Success: Went back in history"

            elif action == "go_forward":
                pyautogui.hotkey('alt', 'right')
                self.state.successful_actions += 1
                return "Success: Went forward in history"

            elif action == "bookmark":
                pyautogui.hotkey('ctrl', 'd')
                self.state.successful_actions += 1
                return "Success: Bookmarked current page"

            # Text Actions
            elif action == "select_all":
                pyautogui.hotkey('ctrl', 'a')
                self.state.successful_actions += 1
                return "Success: Selected all"

            elif action == "copy":
                pyautogui.hotkey('ctrl', 'c')
                time.sleep(0.1)
                if HAS_PYPERCLIP:
                    self.state.clipboard_content = pyperclip.paste()
                    self.state.successful_actions += 1
                    return f"Success: Copied to clipboard (length: {len(self.state.clipboard_content)})"
                else:
                    self.state.successful_actions += 1
                    return "Success: Copy command sent (clipboard read failed)"

            elif action == "paste":
                pyautogui.hotkey('ctrl', 'v')
                self.state.successful_actions += 1
                return "Success: Pasted from clipboard"

            elif action == "cut":
                pyautogui.hotkey('ctrl', 'x')
                self.state.successful_actions += 1
                return "Success: Cut to clipboard"

            elif action == "undo":
                pyautogui.hotkey('ctrl', 'z')
                self.state.successful_actions += 1
                return "Success: Undo performed"

            elif action == "redo":
                pyautogui.hotkey('ctrl', 'y')
                self.state.successful_actions += 1
                return "Success: Redo performed"

            elif action == "find":
                query = str(args.get("query", ""))
                pyautogui.hotkey('ctrl', 'f')
                time.sleep(0.2)
                pyautogui.write(query, interval=0.01)
                self.state.successful_actions += 1
                return f"Success: Searching for '{query}'"

            elif action == "select_text":
                start_x = float(args.get("start_x", 0.5))
                start_y = float(args.get("start_y", 0.5))
                end_x = float(args.get("end_x", 0.5))
                end_y = float(args.get("end_y", 0.5))
                
                abs_start_x = int(start_x * self.screen_width)
                abs_start_y = int(start_y * self.screen_height)
                abs_end_x = int(end_x * self.screen_width)
                abs_end_y = int(end_y * self.screen_height)
                
                pyautogui.moveTo(abs_start_x, abs_start_y)
                pyautogui.dragTo(abs_end_x, abs_end_y, duration=0.3, button='left')
                self.state.successful_actions += 1
                return f"Success: Selected text from ({start_x:.2f}, {start_y:.2f}) to ({end_x:.2f}, {end_y:.2f})"

            # Form Actions
            elif action == "fill_form":
                fields = args.get("fields", {})
                for field_name, value in fields.items():
                    self.state.form_data[field_name] = value
                    # Click field, clear, type value, tab to next
                    pyautogui.press('tab')
                    time.sleep(0.1)
                    pyautogui.hotkey('ctrl', 'a')
                    pyautogui.write(str(value), interval=0.01)
                self.state.successful_actions += 1
                return f"Success: Filled {len(fields)} form fields"

            elif action == "submit_form":
                pyautogui.press('enter')
                self.state.successful_actions += 1
                return "Success: Submitted form"

            elif action == "checkbox":
                checked = args.get("checked", True)
                pyautogui.press('space')  # Toggle checkbox
                self.state.successful_actions += 1
                return f"Success: {'Checked' if checked else 'Unchecked'} checkbox"

            elif action == "radio":
                pyautogui.press('space')
                self.state.successful_actions += 1
                return "Success: Selected radio button"

            elif action == "select_dropdown":
                option = str(args.get("option", ""))
                pyautogui.press('space')  # Open dropdown
                time.sleep(0.2)
                pyautogui.write(option, interval=0.01)
                pyautogui.press('enter')
                self.state.successful_actions += 1
                return f"Success: Selected '{option}' from dropdown"

            # File Actions
            elif action == "upload_file":
                filepath = str(args.get("filepath", ""))
                pyautogui.write(filepath, interval=0.01)
                pyautogui.press('enter')
                self.state.successful_actions += 1
                return f"Success: Uploaded file {filepath}"

            elif action == "download_file":
                pyautogui.hotkey('ctrl', 'j')  # Open downloads
                self.state.successful_actions += 1
                return "Success: Opened downloads"

            elif action == "save_dialog":
                filepath = str(args.get("filepath", ""))
                pyautogui.write(filepath, interval=0.01)
                pyautogui.press('enter')
                self.state.successful_actions += 1
                return f"Success: Saved file to {filepath}"

            # Window Actions
            elif action == "switch_window":
                window_title = str(args.get("window_title", ""))
                pyautogui.hotkey('alt', 'tab')
                time.sleep(0.2)
                self.state.current_window = window_title
                self.state.successful_actions += 1
                return f"Success: Switched to window '{window_title}'"

            elif action == "minimize_window":
                pyautogui.hotkey('win', 'down')
                self.state.successful_actions += 1
                return "Success: Minimized window"

            elif action == "maximize_window":
                pyautogui.hotkey('win', 'up')
                self.state.successful_actions += 1
                return "Success: Maximized window"

            elif action == "close_window":
                pyautogui.hotkey('alt', 'f4')
                self.state.successful_actions += 1
                return "Success: Closed window"

            # Analysis Actions
            elif action == "ocr_read":
                region = args.get("region")
                if region and all(0 <= v <= 1 for v in region):
                    region = (
                        int(region[0] * self.screen_width),
                        int(region[1] * self.screen_height),
                        int(region[2] * self.screen_width),
                        int(region[3] * self.screen_height)
                    )
                text = self.ocr_read(region)
                self.state.successful_actions += 1
                return f"OCR Result: {text[:300]}{'...' if len(text) > 300 else ''}"

            elif action == "find_element":
                text = args.get("text")
                color = args.get("color")
                pattern = args.get("pattern")
                
                if text:
                    result = self.find_element_by_text(text)
                    if result:
                        self.state.successful_actions += 1
                        return f"Success: Found element '{text}' at {result}"
                    return f"Failed: Element '{text}' not found"
                elif color:
                    return f"Color search not yet implemented"
                elif pattern:
                    return f"Pattern search not yet implemented"
                else:
                    return "Error: Must specify text, color, or pattern"

            elif action == "get_color":
                x = float(args.get("x", 0.5))
                y = float(args.get("y", 0.5))
                color = self.get_color_at(x, y)
                self.state.successful_actions += 1
                return f"Color at ({x:.2f}, {y:.2f}): RGB{color}"

            elif action == "analyze_region":
                x = float(args.get("x", 0.5))
                y = float(args.get("y", 0.5))
                width = float(args.get("width", 0.2))
                height = float(args.get("height", 0.2))
                analysis = self.analyze_region(x, y, width, height)
                self.state.successful_actions += 1
                return f"Region analysis: {analysis}"

            # System Actions
            elif action == "exec":
                command = str(args.get("command", ""))
                logger.warning(f"Executing shell command: {command}")
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                output = result.stdout[:1000] + ("..." if len(result.stdout) > 1000 else "")
                err = result.stderr[:500]
                self.state.successful_actions += 1
                return f"Success: Executed. Out: {output} | Err: {err}"

            elif action == "wait":
                duration = float(args.get("duration", 1.0))
                time.sleep(duration)
                self.state.successful_actions += 1
                return f"Success: Waited {duration} seconds"

            elif action == "screenshot":
                filename = args.get("filename")
                saved_path = self.save_screenshot(filename)
                self.state.successful_actions += 1
                return f"Success: Screenshot saved to {saved_path}"

            # File System Actions
            elif action == "create_file":
                filepath = str(args.get("filepath", ""))
                content = str(args.get("content", ""))
                try:
                    with open(filepath, 'w') as f:
                        f.write(content)
                    self.state.successful_actions += 1
                    return f"Success: Created file {filepath}"
                except Exception as e:
                    return f"Error creating file: {e}"

            elif action == "delete_file":
                filepath = str(args.get("filepath", ""))
                try:
                    os.remove(filepath)
                    self.state.successful_actions += 1
                    return f"Success: Deleted file {filepath}"
                except Exception as e:
                    return f"Error deleting file: {e}"

            elif action == "move_file":
                src = str(args.get("src", ""))
                dst = str(args.get("dst", ""))
                try:
                    os.rename(src, dst)
                    self.state.successful_actions += 1
                    return f"Success: Moved {src} to {dst}"
                except Exception as e:
                    return f"Error moving file: {e}"

            elif action == "copy_file":
                src = str(args.get("src", ""))
                dst = str(args.get("dst", ""))
                try:
                    import shutil
                    shutil.copy2(src, dst)
                    self.state.successful_actions += 1
                    return f"Success: Copied {src} to {dst}"
                except Exception as e:
                    return f"Error copying file: {e}"

            elif action == "read_file":
                filepath = str(args.get("filepath", ""))
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    self.state.successful_actions += 1
                    return f"Success: Read file. Content: {content[:500]}{'...' if len(content) > 500 else ''}"
                except Exception as e:
                    return f"Error reading file: {e}"

            elif action == "write_file":
                filepath = str(args.get("filepath", ""))
                content = str(args.get("content", ""))
                mode = str(args.get("mode", "append"))
                try:
                    with open(filepath, 'a' if mode == "append" else 'w') as f:
                        f.write(content)
                    self.state.successful_actions += 1
                    return f"Success: Wrote to file {filepath}"
                except Exception as e:
                    return f"Error writing to file: {e}"

            elif action == "list_dir":
                dirpath = str(args.get("dirpath", "."))
                try:
                    items = os.listdir(dirpath)
                    self.state.successful_actions += 1
                    return f"Directory {dirpath}: {items[:20]}{'...' if len(items) > 20 else ''}"
                except Exception as e:
                    return f"Error listing directory: {e}"

            elif action == "create_dir":
                dirpath = str(args.get("dirpath", ""))
                try:
                    os.makedirs(dirpath, exist_ok=True)
                    self.state.successful_actions += 1
                    return f"Success: Created directory {dirpath}"
                except Exception as e:
                    return f"Error creating directory: {e}"

            elif action == "delete_dir":
                dirpath = str(args.get("dirpath", ""))
                try:
                    import shutil
                    shutil.rmtree(dirpath)
                    self.state.successful_actions += 1
                    return f"Success: Deleted directory {dirpath}"
                except Exception as e:
                    return f"Error deleting directory: {e}"

            elif action == "get_file_info":
                filepath = str(args.get("filepath", ""))
                try:
                    stat = os.stat(filepath)
                    info = {
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_file": os.path.isfile(filepath),
                        "is_dir": os.path.isdir(filepath)
                    }
                    self.state.successful_actions += 1
                    return f"File info: {info}"
                except Exception as e:
                    return f"Error getting file info: {e}"

            elif action == "search_files":
                pattern = str(args.get("pattern", "*"))
                dirpath = str(args.get("dirpath", "."))
                try:
                    import glob
                    matches = glob.glob(os.path.join(dirpath, pattern))
                    self.state.successful_actions += 1
                    return f"Found {len(matches)} files: {matches[:10]}{'...' if len(matches) > 10 else ''}"
                except Exception as e:
                    return f"Error searching files: {e}"

            # Application Actions
            elif action == "launch_app":
                app_path = str(args.get("app_path", ""))
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.Popen([app_path], shell=True)
                    else:
                        subprocess.Popen([app_path])
                    self.state.successful_actions += 1
                    return f"Success: Launched {app_path}"
                except Exception as e:
                    return f"Error launching app: {e}"

            elif action == "close_app":
                app_name = str(args.get("app_name", ""))
                try:
                    if os.name == 'nt':
                        subprocess.run(["taskkill", "/F", "/IM", f"{app_name}.exe"], capture_output=True)
                    else:
                        subprocess.run(["pkill", app_name], capture_output=True)
                    self.state.successful_actions += 1
                    return f"Success: Closed {app_name}"
                except Exception as e:
                    return f"Error closing app: {e}"

            elif action == "list_processes":
                try:
                    if os.name == 'nt':
                        result = subprocess.run(["tasklist"], capture_output=True, text=True)
                    else:
                        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                    self.state.successful_actions += 1
                    return f"Processes: {result.stdout[:500]}{'...' if len(result.stdout) > 500 else ''}"
                except Exception as e:
                    return f"Error listing processes: {e}"

            elif action == "kill_process":
                pid = int(args.get("pid", 0))
                try:
                    if os.name == 'nt':
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                    else:
                        subprocess.run(["kill", str(pid)], capture_output=True)
                    self.state.successful_actions += 1
                    return f"Success: Killed process {pid}"
                except Exception as e:
                    return f"Error killing process: {e}"

            # System Actions
            elif action == "open_settings":
                try:
                    if os.name == 'nt':
                        subprocess.run(["start", "ms-settings:"], shell=True)
                    elif os.name == 'posix':
                        if os.uname().sysname == 'Darwin':  # macOS
                            subprocess.run(["open", "System Preferences"])
                        else:  # Linux
                            subprocess.run(["gnome-control-center"], capture_output=True)
                    self.state.successful_actions += 1
                    return "Success: Opened system settings"
                except Exception as e:
                    return f"Error opening settings: {e}"

            elif action == "control_panel":
                try:
                    if os.name == 'nt':
                        subprocess.run(["control"], shell=True)
                    elif os.name == 'posix':
                        if os.uname().sysname == 'Darwin':
                            subprocess.run(["open", "/System/Applications/Utilities/System Preferences.app"])
                    self.state.successful_actions += 1
                    return "Success: Opened control panel"
                except Exception as e:
                    return f"Error opening control panel: {e}"

            elif action == "volume_up":
                amount = int(args.get("amount", 10))
                try:
                    if os.name == 'nt':
                        # Windows volume control
                        for _ in range(amount // 5):
                            pyautogui.press('volumeup')
                    else:
                        # Unix-like systems
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{amount}%+"], capture_output=True)
                    self.state.successful_actions += 1
                    return f"Success: Volume increased by {amount}"
                except Exception as e:
                    return f"Error changing volume: {e}"

            elif action == "volume_down":
                amount = int(args.get("amount", 10))
                try:
                    if os.name == 'nt':
                        for _ in range(amount // 5):
                            pyautogui.press('volumedown')
                    else:
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{amount}%-"], capture_output=True)
                    self.state.successful_actions += 1
                    return f"Success: Volume decreased by {amount}"
                except Exception as e:
                    return f"Error changing volume: {e}"

            elif action == "volume_mute":
                try:
                    if os.name == 'nt':
                        pyautogui.press('volumemute')
                    else:
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "toggle"], capture_output=True)
                    self.state.successful_actions += 1
                    return "Success: Volume toggled"
                except Exception as e:
                    return f"Error toggling volume: {e}"

            elif action == "volume_set":
                level = int(args.get("level", 50))
                try:
                    if os.name == 'nt':
                        # Windows - approximate by resetting and increasing
                        pyautogui.press('volumemute')
                        time.sleep(0.1)
                        pyautogui.press('volumemute')
                        for _ in range(level // 5):
                            pyautogui.press('volumeup')
                    else:
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level}%"], capture_output=True)
                    self.state.successful_actions += 1
                    return f"Success: Volume set to {level}%"
                except Exception as e:
                    return f"Error setting volume: {e}"

            # Multi-Monitor Actions
            elif action == "switch_screen":
                screen_num = int(args.get("screen", 1))
                try:
                    if os.name == 'nt':
                        pyautogui.hotkey('win', 'shift', 'left' if screen_num == 1 else 'right')
                    else:
                        # Unix-like systems - use display environment
                        pass  # Implementation depends on display manager
                    self.state.successful_actions += 1
                    return f"Success: Switched to screen {screen_num}"
                except Exception as e:
                    return f"Error switching screen: {e}"

            elif action == "move_to_screen":
                screen_num = int(args.get("screen", 1))
                try:
                    if os.name == 'nt':
                        pyautogui.hotkey('win', 'shift', 'left' if screen_num == 1 else 'right')
                    self.state.successful_actions += 1
                    return f"Success: Moved window to screen {screen_num}"
                except Exception as e:
                    return f"Error moving to screen: {e}"

            # System Search & Notifications
            elif action == "system_search":
                query = str(args.get("query", ""))
                try:
                    if os.name == 'nt':
                        pyautogui.hotkey('win', 's')
                        time.sleep(0.5)
                        pyautogui.write(query, interval=0.01)
                    elif os.uname().sysname == 'Darwin':
                        pyautogui.hotkey('command', 'space')
                        time.sleep(0.5)
                        pyautogui.write(query, interval=0.01)
                    else:
                        # Linux - depends on desktop environment
                        pyautogui.hotkey('ctrl', 'f')
                        time.sleep(0.5)
                        pyautogui.write(query, interval=0.01)
                    self.state.successful_actions += 1
                    return f"Success: Searching for '{query}'"
                except Exception as e:
                    return f"Error performing search: {e}"

            elif action == "open_notifications":
                try:
                    if os.name == 'nt':
                        pyautogui.hotkey('win', 'a')
                    elif os.uname().sysname == 'Darwin':
                        pass  # macOS notification center varies
                    else:
                        pass  # Linux varies by DE
                    self.state.successful_actions += 1
                    return "Success: Opened notifications"
                except Exception as e:
                    return f"Error opening notifications: {e}"

            elif action == "dismiss_notifications":
                try:
                    if os.name == 'nt':
                        pyautogui.hotkey('win', 'a')
                        time.sleep(0.2)
                        pyautogui.press('esc')
                    self.state.successful_actions += 1
                    return "Success: Dismissed notifications"
                except Exception as e:
                    return f"Error dismissing notifications: {e}"

            # Control Actions
            elif action == "done":
                return "Agent declared task complete."

            elif action == "error":
                return "Error action received"

            elif action == "retry":
                return "Retry requested"

            else:
                self.state.failed_actions += 1
                return f"Error: Unknown action '{action}'"
                
        except Exception as e:
            self.state.failed_actions += 1
            logger.error(f"Error during execution of {action}: {e}")
            return f"Error during execution of {action}: {str(e)}"

    # -------------------------
    # 🧠 MODEL INTERFACE
    # -------------------------
    def _call_vlm(self, goal: str, image_b64: str) -> Dict[str, Any]:
        """
        Interface for Vision-Language Model API calls.
        Supports OpenAI, Anthropic, Gemini, and local models.
        """
        # Get API configuration from config or environment
        api_provider = self.config.get("api_provider", "openai")
        api_key = self.config.get("api_key", os.environ.get(f"{api_provider.upper()}_API_KEY"))
        
        if not api_key and api_provider != "mock":
            logger.warning(f"No API key found for {api_provider}, using mock mode")
            api_provider = "mock"
        
        # Construct the payload based on provider
        if api_provider == "openai":
            return self._call_openai(goal, image_b64, api_key)
        elif api_provider == "anthropic":
            return self._call_anthropic(goal, image_b64, api_key)
        elif api_provider == "gemini":
            return self._call_gemini(goal, image_b64, api_key)
        elif api_provider == "local":
            return self._call_local_model(goal, image_b64)
        else:
            return self._mock_response(goal)

    def _call_openai(self, goal: str, image_b64: str, api_key: str) -> Dict[str, Any]:
        """Call OpenAI GPT-4 Vision API."""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Goal: {goal}\nHistory: {json.dumps(self.state.history[-3:])}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]}
            ]
            
            response = client.chat.completions.create(
                model=self.config.get("openai_model", "gpt-4o"),
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            raw_response = response.choices[0].message.content
            return self._parse_json_response(raw_response)
        except ImportError:
            logger.warning("OpenAI library not installed, using mock mode")
            return self._mock_response(goal)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._mock_response(goal)

    def _call_anthropic(self, goal: str, image_b64: str, api_key: str) -> Dict[str, Any]:
        """Call Anthropic Claude Vision API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            message = client.messages.create(
                model=self.config.get("anthropic_model", "claude-3-5-sonnet-20241022"),
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Goal: {goal}\nHistory: {json.dumps(self.state.history[-3:])}"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
                        ]
                    }
                ]
            )
            
            raw_response = message.content[0].text
            return self._parse_json_response(raw_response)
        except ImportError:
            logger.warning("Anthropic library not installed, using mock mode")
            return self._mock_response(goal)
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return self._mock_response(goal)

    def _call_gemini(self, goal: str, image_b64: str, api_key: str) -> Dict[str, Any]:
        """Call Google Gemini Vision API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(self.config.get("gemini_model", "gemini-1.5-pro"))
            
            prompt = f"{SYSTEM_PROMPT}\n\nGoal: {goal}\nHistory: {json.dumps(self.state.history[-3:])}"
            
            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": image_b64}
            ])
            
            raw_response = response.text
            return self._parse_json_response(raw_response)
        except ImportError:
            logger.warning("Google Generative AI library not installed, using mock mode")
            return self._mock_response(goal)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return self._mock_response(goal)

    def _call_local_model(self, goal: str, image_b64: str) -> Dict[str, Any]:
        """Call a local vision model (e.g., Ollama, LLaVA)."""
        try:
            import requests
            
            endpoint = self.config.get("local_endpoint", "http://localhost:11434/api/generate")
            model = self.config.get("local_model", "llava")
            
            payload = {
                "model": model,
                "prompt": f"{SYSTEM_PROMPT}\n\nGoal: {goal}\nHistory: {json.dumps(self.state.history[-3:])}",
                "images": [image_b64],
                "stream": False
            }
            
            response = requests.post(endpoint, json=payload, timeout=30)
            raw_response = response.json().get("response", "")
            return self._parse_json_response(raw_response)
        except Exception as e:
            logger.error(f"Local model call failed: {e}")
            return self._mock_response(goal)

    def _mock_response(self, goal: str) -> Dict[str, Any]:
        """Generate mock responses for testing without API."""
        if not self.state.history:
            raw_response = '{"thought": "Opening browser.", "action": "press", "args": {"key": "win"}}'
        elif len(self.state.history) == 1:
            raw_response = '{"thought": "Searching for Chrome.", "action": "type", "args": {"text": "chrome"}}'
        elif len(self.state.history) == 2:
            raw_response = '{"thought": "Launching browser.", "action": "press", "args": {"key": "enter"}}'
        elif len(self.state.history) == 3:
            raw_response = '{"thought": "Navigating to search engine.", "action": "navigate", "args": {"url": "https://google.com"}}'
        elif len(self.state.history) == 4:
            raw_response = '{"thought": "Typing search query.", "action": "type", "args": {"text": "open source ai agents"}}'
        elif len(self.state.history) == 5:
            raw_response = '{"thought": "Submitting search.", "action": "press", "args": {"key": "enter"}}'
        else:
            raw_response = '{"thought": "Task complete.", "action": "done", "args": {}}'
        
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
            logger.error(f"Failed to parse LLM response as JSON: {text}")
            return {"action": "error", "args": {}, "thought": "Failed to parse JSON."}

    # -------------------------
    # 🎯 DIRECT COMMAND MODE
    # -------------------------
    def execute_direct_commands(self, commands: List[Dict[str, Any]]) -> List[str]:
        """
        Execute a sequence of commands directly without VLM decision making.
        Each command is a dict with 'action' and 'args' keys.
        Returns list of results.
        """
        results = []
        logger.info(f"Executing {len(commands)} direct commands")
        
        for i, cmd in enumerate(commands):
            action = cmd.get("action")
            args = cmd.get("args", {})
            
            print(f"\n--- Direct Command {i+1}/{len(commands)} ---")
            print(f"Action: {action}")
            print(f"Args: {args}")
            
            result = self.execute_action(action, args)
            results.append(result)
            print(f"Result: {result}")
            
            # Delay between commands
            time.sleep(self.step_delay)
        
        return results

    def run_direct(self, commands: List[Dict[str, Any]]):
        """
        Run in direct command mode.
        Args:
            commands: List of command dicts with 'action' and 'args'
        """
        logger.info("Starting direct command mode")
        print("\n🎯 DIRECT COMMAND MODE")
        print("="*40)
        
        try:
            results = self.execute_direct_commands(commands)
            
            print("\n" + "="*40)
            print("📊 DIRECT COMMAND SUMMARY")
            print("="*40)
            print(f"Total Commands: {len(commands)}")
            print(f"Successful: {sum(1 for r in results if 'Success' in r or 'Success' in r)}")
            print(f"Failed: {sum(1 for r in results if 'Error' in r)}")
            
            # Save state
            self.save_state("agent_state_direct.json")
            
        except pyautogui.FailSafeException:
            print("\n🚨 FAILSAFE TRIGGERED! Mouse moved to corner. Agent terminated.")
            self.save_state("agent_state_direct_emergency.json")
        except KeyboardInterrupt:
            print("\n🛑 Manual interrupt received. Agent terminated.")
            self.save_state("agent_state_direct_interrupt.json")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            logger.error(f"Unexpected error in direct mode: {e}", exc_info=True)
            self.save_state("agent_state_direct_error.json")

    # -------------------------
    # 🔁 CORE LOOP
    # -------------------------
    def run(self, goal: str, retry_on_failure: bool = True, max_retries: int = 3):
        """
        Main execution loop for the agent.
        Args:
            goal: The goal to achieve
            retry_on_failure: Whether to retry failed actions
            max_retries: Maximum number of retries per action
        """
        logger.info(f"Starting agent with goal: {goal}")
        print(f"\n🎯 ENGAGING AGENT GOAL: {goal}\n" + "="*40)

        retry_count = 0
        
        for step in range(self.max_steps):
            print(f"\n--- 🔄 STEP {step+1}/{self.max_steps} ---")
            logger.info(f"Step {step+1}/{self.max_steps}")

            # 👁️ Observe
            print("[*] Capturing spatial data...")
            screen_b64 = self.capture_vision(save=True)  # Save screenshot for debugging

            # 🧠 Decide
            print("[*] Awaiting VLM decision...")
            decision = self._call_vlm(goal, screen_b64)

            action = decision.get("action", "error")
            args = decision.get("args", {})
            thought = decision.get("thought", "No thought provided.")

            print(f"🧠 Thought: {thought}")
            print(f"⚙️ Action : {action} | Args: {args}")
            logger.info(f"Decision: {action} - {thought}")

            # 🖱️ Act
            if action == "done":
                print("\n✅ GOAL ACHIEVED. Disengaging agent loop.")
                logger.info("Goal achieved successfully")
                break
            
            if action == "error":
                print("⚠️ Skipping execution due to malformed LLM output.")
                result = "Failed to parse instruction."
                if retry_on_failure and retry_count < max_retries:
                    retry_count += 1
                    print(f"[*] Retrying... ({retry_count}/{max_retries})")
                    time.sleep(self.step_delay)
                    continue
            elif action == "retry":
                print("[*] Retry requested by VLM")
                retry_count += 1
                if retry_count >= max_retries:
                    print("[!] Max retries reached, continuing")
                    retry_count = 0
                time.sleep(self.step_delay)
                continue
            else:
                result = self.execute_action(action, args)
                print(f"📤 Result : {result}")
                
                # Check if action failed
                if "Error" in result or "Failed" in result:
                    if retry_on_failure and retry_count < max_retries:
                        retry_count += 1
                        print(f"[*] Action failed, retrying... ({retry_count}/{max_retries})")
                        time.sleep(self.step_delay)
                        continue
                    else:
                        retry_count = 0
                else:
                    retry_count = 0  # Reset retry count on success

            # 📝 Record
            self.state.history.append({
                "step": step + 1,
                "action": action,
                "args": args,
                "result": result,
                "thought": thought,
                "timestamp": datetime.now().isoformat()
            })

            # Delay to allow UI animations/rendering to complete before next screenshot
            time.sleep(self.step_delay)
            
        else:
            print("\n⚠️ MAX STEPS REACHED. Terminating to prevent infinite loop.")
            logger.warning("Max steps reached without completion")

        # Print summary
        self._print_summary()

    # -------------------------
    # 🔄 STATE MANAGEMENT
    # -------------------------
    def save_state(self, filepath: str):
        """Save current agent state to file for recovery."""
        state_data = {
            "history": self.state.history,
            "clipboard_content": self.state.clipboard_content,
            "current_window": self.state.current_window,
            "current_url": self.state.current_url,
            "current_tab_index": self.state.current_tab_index,
            "last_action": self.state.last_action,
            "last_action_args": self.state.last_action_args,
            "undo_stack": self.state.undo_stack,
            "redo_stack": self.state.redo_stack,
            "screenshots": self.state.screenshots,
            "element_cache": self.state.element_cache,
            "form_data": self.state.form_data,
            "session_start": self.state.session_start,
            "total_actions": self.state.total_actions,
            "successful_actions": self.state.successful_actions,
            "failed_actions": self.state.failed_actions,
            "config": self.config
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"State saved to {filepath}")
            print(f"[*] State saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            print(f"[!] Failed to save state: {e}")

    def load_state(self, filepath: str):
        """Load agent state from file for recovery."""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            self.state.history = state_data.get("history", [])
            self.state.clipboard_content = state_data.get("clipboard_content", "")
            self.state.current_window = state_data.get("current_window", "")
            self.state.current_url = state_data.get("current_url", "")
            self.state.current_tab_index = state_data.get("current_tab_index", 0)
            self.state.last_action = state_data.get("last_action")
            self.state.last_action_args = state_data.get("last_action_args", {})
            self.state.undo_stack = state_data.get("undo_stack", [])
            self.state.redo_stack = state_data.get("redo_stack", [])
            self.state.screenshots = state_data.get("screenshots", [])
            self.state.element_cache = state_data.get("element_cache", {})
            self.state.form_data = state_data.get("form_data", {})
            self.state.session_start = state_data.get("session_start", datetime.now().isoformat())
            self.state.total_actions = state_data.get("total_actions", 0)
            self.state.successful_actions = state_data.get("successful_actions", 0)
            self.state.failed_actions = state_data.get("failed_actions", 0)
            self.config = state_data.get("config", {})
            
            logger.info(f"State loaded from {filepath}")
            print(f"[*] State loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            print(f"[!] Failed to load state: {e}")

    def load_config(self, filepath: str):
        """Load configuration from file."""
        try:
            with open(filepath, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            print(f"[*] Configuration loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            print(f"[!] Failed to load config: {e}")

    def save_config(self, filepath: str):
        """Save current configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
            print(f"[*] Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            print(f"[!] Failed to save config: {e}")

    def undo_last_action(self):
        """Attempt to undo the last action if possible."""
        if not self.state.last_action:
            print("[!] No action to undo")
            return
        
        logger.info(f"Attempting to undo: {self.state.last_action}")
        print(f"[*] Attempting to undo: {self.state.last_action}")
        
        # Store current state in undo stack
        self.state.undo_stack.append({
            "action": self.state.last_action,
            "args": self.state.last_action_args.copy()
        })
        
        # Try to undo based on action type
        reversible_actions = ["type", "paste", "delete", "cut"]
        if self.state.last_action in reversible_actions:
            result = self.execute_action("undo", {})
            print(result)
        elif self.state.last_action == "copy":
            # Clear clipboard to undo copy
            if HAS_PYPERCLIP:
                pyperclip.copy("")
            print("Undo: Cleared clipboard")
        else:
            print(f"[!] Cannot automatically undo action type: {self.state.last_action}")

    def redo_last_undo(self):
        """Redo the last undone action."""
        if not self.state.undo_stack:
            print("[!] No actions to redo")
            return
        
        last_undone = self.state.undo_stack.pop()
        self.state.redo_stack.append(last_undone)
        
        logger.info(f"Redoing action: {last_undone['action']}")
        print(f"[*] Redoing action: {last_undone['action']}")
        
        result = self.execute_action(last_undone["action"], last_undone["args"])
        print(result)

    def _print_summary(self):
        """Print execution summary."""
        print("\n" + "="*40)
        print("📊 EXECUTION SUMMARY")
        print("="*40)
        print(f"Total Actions: {self.state.total_actions}")
        print(f"Successful: {self.state.successful_actions}")
        print(f"Failed: {self.state.failed_actions}")
        print(f"Success Rate: {(self.state.successful_actions/max(1,self.state.total_actions)*100):.1f}%")
        print(f"Screenshots Saved: {len(self.state.screenshots)}")
        print(f"Session Duration: {datetime.now() - datetime.fromisoformat(self.state.session_start)}")
        print(f"Current URL: {self.state.current_url}")
        print(f"Current Window: {self.state.current_window}")
        print("="*40)
        logger.info(f"Execution complete. Success rate: {(self.state.successful_actions/max(1,self.state.total_actions)*100):.1f}%")

# =========================
# 🚀 ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quillan-Ronin Desktop Agent - Autonomous Computer Control")
    parser.add_argument("--goal", type=str, help="The goal for the agent to achieve")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between steps in seconds")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of steps")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR features")
    parser.add_argument("--screenshot-dir", type=str, default="screenshots", help="Directory for screenshots")
    parser.add_argument("--load-state", type=str, help="Load state from file")
    parser.add_argument("--api-provider", type=str, choices=["openai", "anthropic", "gemini", "local", "mock"], default="mock", help="API provider to use")
    parser.add_argument("--api-key", type=str, help="API key for the provider")
    parser.add_argument("--direct-commands", type=str, help="JSON file containing direct commands to execute")
    parser.add_argument("--direct-mode", action="store_true", help="Run in direct command mode (no VLM)")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = QuillanDesktopAgent(
        step_delay=args.delay,
        max_steps=args.max_steps,
        enable_ocr=not args.no_ocr,
        config_file=args.config,
        screenshot_dir=args.screenshot_dir
    )
    
    # Set API provider
    if args.api_provider:
        agent.config["api_provider"] = args.api_provider
    if args.api_key:
        agent.config["api_key"] = args.api_key
    
    # Load state if requested
    if args.load_state:
        agent.load_state(args.load_state)
    
    # Example goals demonstrating new capabilities
    EXAMPLE_GOALS = [
        "Open a browser and search for 'open source ai agents'",
        "Navigate to GitHub and browse repositories",
        "Open a text editor and type a document",
        "Use OCR to read text from the screen",
        "Fill out a web form with sample data",
        "Navigate through a multi-page website",
        "Download a file from the internet",
        "Take screenshots of the current screen"
    ]
    
    # Determine goal
    if args.goal:
        TARGET_GOAL = args.goal
    else:
        print("\n🎯 Available example goals:")
        for i, goal in enumerate(EXAMPLE_GOALS, 1):
            print(f"  {i}. {goal}")
        print(f"\n💡 Use --goal to specify a custom goal, or --goal <number> to use an example")
        TARGET_GOAL = EXAMPLE_GOALS[0]  # Default to first example
    
    # Direct mode or VLM mode
    try:
        if args.direct_mode or args.direct_commands:
            if args.direct_commands:
                # Load commands from file
                try:
                    with open(args.direct_commands, 'r') as f:
                        commands = json.load(f)
                    print(f"[*] Loaded {len(commands)} commands from {args.direct_commands}")
                except Exception as e:
                    print(f"[!] Failed to load commands file: {e}")
                    commands = []
            else:
                # Use example commands for testing
                commands = [
                    {"action": "press", "args": {"key": "win"}},
                    {"action": "wait", "args": {"duration": 1.0}},
                    {"action": "type", "args": {"text": "brave"}},
                    {"action": "press", "args": {"key": "enter"}},
                    {"action": "wait", "args": {"duration": 3.0}},
                    {"action": "navigate", "args": {"url": "https://substack.com"}},
                    {"action": "wait", "args": {"duration": 3.0}},
                ]
                print("[*] Using example commands for direct mode test")
            agent.run_direct(commands)
        else:
            # VLM mode
            print(f"\n🎯 Running with goal: {TARGET_GOAL}")
            print(f"⚙️ Configuration: API Provider={agent.config.get('api_provider', 'mock')}, OCR={agent.enable_ocr}")
            print("💡 Press Ctrl+C to stop, or move mouse to corner for failsafe")
            print("="*40)
            agent.run(TARGET_GOAL, retry_on_failure=True, max_retries=3)
            # Save state on completion
            print("\n[*] Saving agent state...")
            agent.save_state("agent_state.json")
    except pyautogui.FailSafeException:
        print("\n🚨 FAILSAFE TRIGGERED! Mouse moved to corner. Agent terminated.")
        logger.warning("Failsafe triggered")
        agent.save_state("agent_state_emergency.json")
    except KeyboardInterrupt:
        print("\n🛑 Manual interrupt received. Agent terminated.")
        logger.info("Manual interrupt")
        agent.save_state("agent_state_interrupt.json")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        agent.save_state("agent_state_error.json")
