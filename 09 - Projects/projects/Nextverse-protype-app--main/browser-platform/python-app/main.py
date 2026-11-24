#!/usr/bin/env python3
"""
NextVerse Browser Platform - Python/PyQt6 Implementation
Main entry point for the QWebEngineView-based browser platform
"""

import sys
import os
import subprocess
import time
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QStatusBar
from PyQt6.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt6.QtWebEngineCore import QWebEngineProfile
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QUrl, QTimer, QObject, pyqtSlot
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGL import QOpenGLWidget

from local_proxy import LocalProxy
from quantum_bridge import QuantumBridge


class MainWindow(QMainWindow):
    """Main window containing QWebEngineView for React app display"""
    
    def __init__(self):
        super().__init__()
        self.react_process = None
        self.quantum_service = None
        self.init_ui()
        self.setup_web_engine()
        self.setup_ipc_bridge()
        self.start_services()
        
    def init_ui(self):
        """Initialize the main window UI"""
        self.setWindowTitle("NextVerse Browser Platform")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create web engine view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing NextVerse Platform...")
        
    def setup_web_engine(self):
        """Configure QWebEngineView settings for optimal performance"""
        # Get default settings
        settings = QWebEngineSettings.defaultSettings()
        
        # Enable local file access (critical for Qt 6.4+)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        
        # Enable hardware acceleration and WebGL
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
        
        # Performance optimizations
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        
        # Configure profile for memory caching (Phase 2 optimization)
        profile = QWebEngineProfile.defaultProfile()
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        profile.setPersistentStoragePath(str(Path("./data/storage")))
        
        # Set persistent storage quota (workaround for local storage limits)
        profile.setHttpCacheMaximumSize(100 * 1024 * 1024)  # 100MB cache
        
    def setup_ipc_bridge(self):
        """Setup QtWebChannel for JavaScript ↔ Python communication"""
        # Create web channel
        self.channel = QWebChannel()
        
        # Create local proxy for file system access
        self.local_proxy = LocalProxy()
        self.channel.registerObject("proxy", self.local_proxy)
        
        # Create quantum bridge for C++ core communication
        self.quantum_bridge = QuantumBridge()
        self.channel.registerObject("quantum", self.quantum_bridge)
        
        # Attach channel to web page
        self.web_view.page().setWebChannel(self.channel)
        
    def start_services(self):
        """Start React dev server and quantum core service"""
        # Start React development server
        self.start_react_server()
        
        # Start quantum core service (Phase 1: mock implementation)
        self.start_quantum_service()
        
        # Load React app after services are ready
        QTimer.singleShot(3000, self.load_react_app)  # 3s delay for startup
        
    def start_react_server(self):
        """Start React development server on localhost:3000"""
        react_dir = Path("../react-frontend")
        if react_dir.exists():
            try:
                self.status_bar.showMessage("Starting React development server...")
                self.react_process = subprocess.Popen(
                    ["npm", "run", "dev", "--", "--port", "3000", "--host", "0.0.0.0"],
                    cwd=react_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print("React dev server started on port 3000")
            except Exception as e:
                print(f"Failed to start React server: {e}")
                self.status_bar.showMessage("Failed to start React server")
        else:
            print("React frontend directory not found")
            self.status_bar.showMessage("React frontend not found")
            
    def start_quantum_service(self):
        """Start quantum core service (mock for Phase 1)"""
        quantum_dir = Path("../quantum-core")
        if quantum_dir.exists():
            try:
                self.status_bar.showMessage("Starting Quantum Core service...")
                # For Phase 1: Python mock service
                self.quantum_service = subprocess.Popen(
                    ["python", "quantum_mock_service.py"],
                    cwd=quantum_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print("Quantum core service started")
            except Exception as e:
                print(f"Failed to start quantum service: {e}")
        else:
            print("Quantum core directory not found - using embedded mock")
            
    def load_react_app(self):
        """Load React app from development server"""
        react_url = QUrl("http://localhost:3000")
        self.web_view.load(react_url)
        self.status_bar.showMessage("Loading React application...")
        
        # Set up load finished callback
        self.web_view.loadFinished.connect(self.on_load_finished)
        
    def on_load_finished(self, success):
        """Handle React app load completion"""
        if success:
            self.status_bar.showMessage("NextVerse Platform Ready - React + Qt Integration Active")
            print("React app loaded successfully in QWebEngineView")
        else:
            self.status_bar.showMessage("Failed to load React application")
            print("Failed to load React app")
            
    def closeEvent(self, event):
        """Clean shutdown of services"""
        print("Shutting down NextVerse Platform...")
        
        # Terminate React dev server
        if self.react_process:
            self.react_process.terminate()
            self.react_process.wait(timeout=5)
            
        # Terminate quantum service
        if self.quantum_service:
            self.quantum_service.terminate()
            self.quantum_service.wait(timeout=5)
            
        event.accept()


def setup_opengl_context():
    """Configure OpenGL context for WebGL compatibility"""
    format = QSurfaceFormat()
    format.setVersion(4, 3)
    format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    format.setDepthBufferSize(24)
    format.setStencilBufferSize(8)
    format.setSamples(4)
    QSurfaceFormat.setDefaultFormat(format)


def main():
    """Main application entry point"""
    # Set up OpenGL context before QApplication
    setup_opengl_context()
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("NextVerse Browser Platform")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NextVerse")
    app.setOrganizationDomain("nextverse.local")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    print("NextVerse Browser Platform starting...")
    print("React dev server: http://localhost:3000")
    print("QWebEngineView integration: Active")
    print("IPC Bridge: Python ↔ JavaScript via QtWebChannel")
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
