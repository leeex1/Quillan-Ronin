#!/usr/bin/env python3
"""
NextVerse PyQt6 WebEngine Wrapper
Displays React frontend with full quantum integration
"""

import sys
import json
import asyncio
import subprocess
import time
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QStatusBar
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile, QWebEngineScript
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QObject, QUrl, pyqtSlot, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
import requests

class QuantumBridge(QObject):
    """Bridge between React frontend and Quantum Core"""
    
    quantumStatusChanged = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.quantum_master_url = "http://localhost:7777"
        self.module_urls = {
            'quantum-core': 'http://localhost:3001',
            'ai-assistant': 'http://localhost:3002',
            'game-engine': 'http://localhost:3003',
            'daw': 'http://localhost:3004',
            'mmorpg': 'http://localhost:3005',
            'orchestrator': 'http://localhost:3006'
        }
    
    @pyqtSlot(str, result=str)
    def executeQuantumTask(self, task_json):
        """Execute task using Quantum VM 11-copy acceleration"""
        try:
            task = json.loads(task_json)
            response = requests.post(f"{self.quantum_master_url}/execute", json=task, timeout=10)
            result = response.json()
            
            self.quantumStatusChanged.emit(f"Quantum boost: {result.get('total_boost', 0)}x")
            return json.dumps(result)
        except Exception as e:
            return json.dumps({'error': str(e), 'status': 'failed'})
    
    @pyqtSlot(str, str, result=str)
    def callModule(self, module_name, endpoint):
        """Call specific NextVerse module"""
        try:
            if module_name not in self.module_urls:
                return json.dumps({'error': f'Unknown module: {module_name}'})
            
            url = f"{self.module_urls[module_name]}/{endpoint}"
            response = requests.get(url, timeout=5)
            return response.text
        except Exception as e:
            return json.dumps({'error': str(e)})
    
    @pyqtSlot(result=str)
    def getSystemStatus(self):
        """Get overall system health"""
        status = {'modules': {}, 'quantum_instances': 0, 'overall_health': 'unknown'}
        
        # Check Quantum Master
        try:
            response = requests.get(f"{self.quantum_master_url}/health", timeout=2)
            quantum_status = response.json()
            status['quantum_instances'] = quantum_status.get('instances', 0)
        except:
            pass
        
        # Check each module
        for module, url in self.module_urls.items():
            try:
                response = requests.get(f"{url}/health", timeout=1)
                status['modules'][module] = 'healthy' if response.status_code == 200 else 'unhealthy'
            except:
                status['modules'][module] = 'offline'
        
        # Determine overall health
        healthy_modules = sum(1 for s in status['modules'].values() if s == 'healthy')
        if healthy_modules >= len(self.module_urls) * 0.8:  # 80% healthy threshold
            status['overall_health'] = 'healthy'
        elif healthy_modules >= len(self.module_urls) * 0.5:
            status['overall_health'] = 'degraded'
        else:
            status['overall_health'] = 'critical'
        
        return json.dumps(status)

class NextVerseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_window()
        self.setup_webengine()
        self.setup_bridge()
        self.setup_monitoring()
        
    def setup_window(self):
        """Configure main window properties"""
        self.setWindowTitle("NextVerse Platform - Quantum-Accelerated Development")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application icon (if available)
        try:
            self.setWindowIcon(QIcon("assets/nextverse-icon.png"))
        except:
            pass
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing NextVerse Platform...")
        
    def setup_webengine(self):
        """Configure QWebEngineView with optimal settings"""
        # Create web view
        self.web_view = QWebEngineView()
        self.centralWidget().layout().addWidget(self.web_view)
        
        # Configure WebEngine settings
        settings = QWebEngineSettings.globalSettings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.SpatialNavigationEnabled, True)
        
        # Enable developer tools
        settings.setAttribute(QWebEngineSettings.WebAttribute.DeveloperExtrasEnabled, True)
        
        # Configure profile for better performance
        profile = QWebEngineProfile.defaultProfile()
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        profile.setPersistentStoragePath(str(Path.home() / ".nextverse" / "storage"))
        
        # Load React development server
        self.web_view.load(QUrl("http://localhost:3000"))
        
        # Connect page load signals
        self.web_view.loadFinished.connect(self.on_page_loaded)
        
    def setup_bridge(self):
        """Setup WebChannel bridge for React ↔ Python communication"""
        self.bridge = QuantumBridge()
        self.channel = QWebChannel()
        self.channel.registerObject("nextverse", self.bridge)
        self.web_view.page().setWebChannel(self.channel)
        
        # Connect bridge signals
        self.bridge.quantumStatusChanged.connect(self.update_quantum_status)
        
    def setup_monitoring(self):
        """Setup real-time system monitoring"""
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_system_status)
        self.monitor_timer.start(5000)  # Update every 5 seconds
        
    def on_page_loaded(self, success):
        """Handle page load completion"""
        if success:
            self.status_bar.showMessage("NextVerse Platform Ready - Quantum Acceleration Active")
            
            # Inject quantum bridge initialization script
            bridge_script = """
            new QWebChannel(qt.webChannelTransport, function (channel) {
                window.nextverse = channel.objects.nextverse;
                
                // Global quantum execution function
                window.executeQuantum = function(task) {
                    return new Promise((resolve, reject) => {
                        try {
                            const result = window.nextverse.executeQuantumTask(JSON.stringify(task));
                            resolve(JSON.parse(result));
                        } catch (error) {
                            reject(error);
                        }
                    });
                };
                
                // Module communication helper
                window.callModule = function(module, endpoint) {
                    return new Promise((resolve, reject) => {
                        try {
                            const result = window.nextverse.callModule(module, endpoint);
                            resolve(JSON.parse(result));
                        } catch (error) {
                            reject(error);
                        }
                    });
                };
                
                // System status helper
                window.getSystemStatus = function() {
                    return new Promise((resolve, reject) => {
                        try {
                            const result = window.nextverse.getSystemStatus();
                            resolve(JSON.parse(result));
                        } catch (error) {
                            reject(error);
                        }
                    });
                };
                
                console.log('🚀 NextVerse Quantum Bridge initialized');
                
                // Notify React that bridge is ready
                window.dispatchEvent(new CustomEvent('nextverse-ready'));
            });
            """
            
            script = QWebEngineScript()
            script.setSourceCode(bridge_script)
            script.setName("NextVerseBridge")
            script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
            script.setRunsOnSubFrames(False)
            
            self.web_view.page().scripts().insert(script)
        else:
            self.status_bar.showMessage("Failed to load React frontend - Check if npm run dev is running")
            
    def update_quantum_status(self, status):
        """Update quantum status in status bar"""
        self.status_bar.showMessage(f"NextVerse Platform Ready - {status}")
        
    def update_system_status(self):
        """Update system health monitoring"""
        try:
            status_json = self.bridge.getSystemStatus()
            status = json.loads(status_json)
            
            health = status.get('overall_health', 'unknown')
            instances = status.get('quantum_instances', 0)
            
            health_emoji = {
                'healthy': '🟢',
                'degraded': '🟡', 
                'critical': '🔴',
                'unknown': '⚪'
            }.get(health, '⚪')
            
            self.status_bar.showMessage(
                f"System Health: {health_emoji} {health.title()} | "
                f"Quantum Instances: {instances} | "
                f"Modules Active: {len([m for m in status.get('modules', {}).values() if m == 'healthy'])}"
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"Monitoring Error: {str(e)}")

def check_react_server():
    """Check if React development server is running"""
    try:
        response = requests.get("http://localhost:3000", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_react_server():
    """Start React development server if not running"""
    if not check_react_server():
        print("🌐 Starting React development server...")
        subprocess.Popen([
            "npm", "run", "dev"
        ], cwd="../web-ui", stdout=subprocess.DEVNULL)
        
        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            if check_react_server():
                print("✅ React server started successfully")
                return True
            time.sleep(1)
        
        print("❌ Failed to start React server")
        return False
    else:
        print("✅ React server already running")
        return True

def main():
    """Main application entry point"""
    # Initialize Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("NextVerse Platform")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NextVerse")
    
    # Check and start React server
    if not start_react_server():
        print("❌ Cannot start without React frontend")
        sys.exit(1)
    
    # Create and show main window
    window = NextVerseMainWindow()
    window.show()
    
    print("🚀 NextVerse Platform launched successfully!")
    print("🌐 Web UI: http://localhost:3000")
    print("🖥️  Qt WebEngine: Ready")
    print("⚡ Quantum Bridge: Active")
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
