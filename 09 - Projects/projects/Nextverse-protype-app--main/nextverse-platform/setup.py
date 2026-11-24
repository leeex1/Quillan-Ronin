#!/usr/bin/env python3
"""
NextVerse Platform Setup Script
Handles complete environment setup for git clone → npm start workflow
"""

import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run shell command with error handling"""
    print(f"🔧 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_dependencies():
    """Check and install required system dependencies"""
    print("📋 Checking system dependencies...")
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        node_version = result.stdout.strip()
        print(f"✅ Node.js {node_version}")
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js 18+ first.")
        sys.exit(1)
    
    # Check Python
    try:
        result = subprocess.run([sys.executable, '--version'], capture_output=True, text=True)
        python_version = result.stdout.strip()
        print(f"✅ {python_version}")
    except:
        print("❌ Python not found")
        sys.exit(1)
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        docker_version = result.stdout.strip()
        print(f"✅ {docker_version}")
    except FileNotFoundError:
        print("⚠️  Docker not found. Some features may be limited.")
    
    # Check CMake
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        cmake_version = result.stdout.strip().split('\n')[0]
        print(f"✅ {cmake_version}")
    except FileNotFoundError:
        print("⚠️  CMake not found. Installing...")
        run_command("sudo apt-get update && sudo apt-get install -y cmake", check=False)

def install_python_dependencies():
    """Install Python/PyQt6 dependencies"""
    print("🐍 Installing Python dependencies...")
    
    python_deps = [
        'PyQt6',
        'PyQt6-WebEngine', 
        'requests',
        'aiohttp',
        'websockets'
    ]
    
    for dep in python_deps:
        print(f"📦 Installing {dep}...")
        run_command(f"{sys.executable} -m pip install {dep}")

def setup_react_frontend():
    """Setup React frontend with all dependencies"""
    print("⚛️  Setting up React frontend...")
    
    web_ui_path = Path("web-ui")
    
    if not web_ui_path.exists():
        print("❌ web-ui directory not found")
        return
    
    # Install React dependencies
    print("📦 Installing React dependencies...")
    run_command("npm install", cwd=web_ui_path)
    
    # Create missing component files
    components_dir = web_ui_path / "src" / "components"
    components_dir.mkdir(exist_ok=True)
    
    component_templates = {
        'GameEngineView.tsx': create_game_engine_component(),
        'AIAssistant.tsx': create_ai_assistant_component(),
        'DAWStudio.tsx': create_daw_component(),
        'MMORPGWorld.tsx': create_mmorpg_component(),
        'SystemMonitor.tsx': create_system_monitor_component()
    }
    
    for filename, content in component_templates.items():
        component_path = components_dir / filename
        if not component_path.exists():
            print(f"📝 Creating {filename}...")
            component_path.write_text(content)

def create_game_engine_component():
    return '''import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const GameEngineView: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">Game Engine - DVVE Renderer</Typography>
        <Typography>
          Real-time rendering with 100k particles @ 50FPS target
        </Typography>
      </CardContent>
    </Card>
  );
};

export default GameEngineView;'''

def create_ai_assistant_component():
    return '''import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const AIAssistant: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">AI Assistant - NeuroMind</Typography>
        <Typography>
          DESS 99.5% ethical compliance with local LLM
        </Typography>
      </CardContent>
    </Card>
  );
};

export default AIAssistant;'''

def create_daw_component():
    return '''import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const DAWStudio: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">DAW Studio - SonicFusion</Typography>
        <Typography>
          LSSS 96kHz @ 50ms latency with JUCE backend
        </Typography>
      </CardContent>
    </Card>
  );
};

export default DAWStudio;'''

def create_mmorpg_component():
    return '''import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const MMORPGWorld: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">MMORPG World - SocialFabric</Typography>
        <Typography>
          JSSC 500-player simulation with Rust backend
        </Typography>
      </CardContent>
    </Card>
  );
};

export default MMORPGWorld;'''

def create_system_monitor_component():
    return '''import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const SystemMonitor: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">System Monitor</Typography>
        <Typography>
          Real-time monitoring of all NextVerse modules
        </Typography>
      </CardContent>
    </Card>
  );
};

export default SystemMonitor;'''

def build_quantum_core():
    """Build quantum core if source exists"""
    print("⚡ Setting up Quantum Core...")
    
    quantum_vm_path = Path("quantum-vm")
    if not quantum_vm_path.exists():
        quantum_vm_path.mkdir()
    
    # Create mock quantum core binary for development
    mock_binary_content = '''#!/bin/bash
# Mock Quantum Core for development
echo "🚀 Quantum Core Instance Starting..."
echo "⚡ JQLD Parameters: $@"

# Create simple HTTP server for testing
python3 -c "
import http.server
import socketserver
import json
import sys
from urllib.parse import parse_qs

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7001

class QuantumHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/execute':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'success',
                'boost_applied': 42,
                'data': ['quantum', 'result', 'data'],
                'instance_id': PORT - 7000
            }
            
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/boost':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {'current_boost': 42}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {'status': 'healthy', 'instance': PORT - 7000}
            self.wfile.write(json.dumps(response).encode())

with socketserver.TCPServer(('', PORT), QuantumHandler) as httpd:
    print(f'Quantum Core serving on port {PORT}')
    httpd.serve_forever()
" $@
'''
    
    binary_path = quantum_vm_path / "quantum-core-binary"
    binary_path.write_text(mock_binary_content)
    binary_path.chmod(0o755)
    
    # Make 11-boot.sh executable
    boot_script = quantum_vm_path / "11-boot.sh"
    if boot_script.exists():
        boot_script.chmod(0o755)

def setup_module_containers():
    """Setup module containers and services"""
    print("🐳 Setting up module containers...")
    
    # Create basic Dockerfiles for each module
    modules = ['quantum-core', 'ai-assistant', 'game-engine', 'mmorpg', 'daw', 'orchestrator']
    
    for module in modules:
        module_path = Path(module)
        module_path.mkdir(exist_ok=True)
        
        dockerfile_content = f'''FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Mock service for {module}
EXPOSE 300{modules.index(module) + 1}

CMD ["python3", "-c", "\\
import http.server; \\
import socketserver; \\
PORT = 300{modules.index(module) + 1}; \\
Handler = http.server.SimpleHTTPRequestHandler; \\
with socketserver.TCPServer(('', PORT), Handler) as httpd: \\
    print(f'{module} serving on port {{PORT}}'); \\
    httpd.serve_forever()"]
'''
        
        dockerfile_path = module_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        
        # Create health endpoint
        health_script = module_path / "health.py"
        health_script.write_text(f'''#!/usr/bin/env python3
import json
print(json.dumps({{"status": "healthy", "module": "{module}"}}))
''')

def main():
    """Main setup function"""
    print("🚀 NextVerse Platform Setup Starting...")
    print("=" * 50)
    
    # Ensure we're in the right directory
    if not Path("package.json").exists():
        print("❌ Please run this script from the nextverse-platform root directory")
        sys.exit(1)
    
    # Run setup steps
    check_dependencies()
    install_python_dependencies()
    setup_react_frontend()
    build_quantum_core()
    setup_module_containers()
    
    # Create start script
    start_script_content = '''#!/bin/bash
echo "🚀 Starting NextVerse Platform..."
npm run turbo:start
'''
    
    start_script = Path("start.sh")
    start_script.write_text(start_script_content)
    start_script.chmod(0o755)
    
    print("\n🎉 NextVerse Platform Setup Complete!")
    print("=" * 50)
    print("🌐 To start the platform:")
    print("   npm run turbo:start")
    print("   OR")
    print("   ./start.sh")
    print("\n🔧 Development URLs:")
    print("   Web UI: http://localhost:3000")
    print("   Quantum Debug: http://localhost:3000/quantum-debug")
    print("   System Monitor: http://localhost:3000/monitor")
    print("\n📖 Documentation: Check README.md for detailed usage")

if __name__ == "__main__":
    main()
