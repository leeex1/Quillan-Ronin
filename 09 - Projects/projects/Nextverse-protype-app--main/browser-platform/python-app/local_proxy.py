"""
Local Proxy for Qt WebEngine ↔ File System Access
Handles CORS workarounds and local resource loading
"""

import os
import json
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSlot


class LocalProxy(QObject):
    """
    Proxy object for local file system access from React components.
    Exposed to JavaScript via QtWebChannel as 'proxy'.
    """
    
    def __init__(self):
        super().__init__()
        self.base_path = Path("./assets")
        self.base_path.mkdir(exist_ok=True)
        
    @pyqtSlot(str, result=str)
    def handle_request(self, path):
        """
        Handle file requests from React components.
        Returns file content as string.
        
        Args:
            path: Relative path to file within assets directory
            
        Returns:
            File content as string, or error message
        """
        try:
            file_path = self.base_path / path
            
            # Security check: ensure path is within assets directory
            if not self._is_safe_path(file_path):
                return json.dumps({"error": "Access denied: Path outside assets directory"})
                
            if file_path.exists() and file_path.is_file():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            else:
                return json.dumps({"error": f"File not found: {path}"})
                
        except Exception as e:
            return json.dumps({"error": f"Failed to read file: {str(e)}"})
    
    @pyqtSlot(str, result=str)
    def read_config(self, config_name):
        """
        Read configuration files from config directory.
        
        Args:
            config_name: Name of config file (e.g., "app.json")
            
        Returns:
            Configuration data as JSON string
        """
        try:
            config_path = Path("./config") / config_name
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Return default config
                default_config = {
                    "app_name": "NextVerse Browser Platform",
                    "version": "1.0.0",
                    "react_port": 3000,
                    "quantum_port": 3001,
                    "debug_mode": True
                }
                return json.dumps(default_config, indent=2)
                
        except Exception as e:
            return json.dumps({"error": f"Failed to read config: {str(e)}"})
    
    @pyqtSlot(str, str, result=bool)
    def write_file(self, path, content):
        """
        Write content to file within assets directory.
        
        Args:
            path: Relative path within assets directory
            content: Content to write
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.base_path / path
            
            # Security check
            if not self._is_safe_path(file_path):
                return False
                
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
        except Exception as e:
            print(f"Failed to write file {path}: {e}")
            return False
    
    @pyqtSlot(result=str)
    def list_assets(self):
        """
        List all files in assets directory.
        
        Returns:
            JSON array of file paths
        """
        try:
            files = []
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.base_path)
                    files.append(str(relative_path))
            
            return json.dumps(files)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to list assets: {str(e)}"})
    
    def _is_safe_path(self, path):
        """
        Check if path is within allowed directory (security measure).
        
        Args:
            path: Path to check
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            path.resolve().relative_to(self.base_path.resolve())
            return True
        except ValueError:
            return False
