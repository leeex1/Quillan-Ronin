import React, { useEffect, useState, useCallback } from 'react';
import './App.css';
import QuantumDashboard from './components/QuantumDashboard';
import GameEngineView from './components/GameEngineView';
import AIAssistant from './components/AIAssistant';
import DAWStudio from './components/DAWStudio';
import MMORPGWorld from './components/MMORPGWorld';
import SystemMonitor from './components/SystemMonitor';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemIcon, ListItemText, Box, Alert } from '@mui/material';
import { Dashboard, Games, SmartToy, MusicNote, Public, Computer } from '@mui/icons-material';

// Global quantum bridge interface
declare global {
  interface Window {
    nextverse: any;
    executeQuantum: (task: any) => Promise<any>;
    callModule: (module: string, endpoint: string) => Promise<any>;
    getSystemStatus: () => Promise<any>;
  }
}

interface SystemStatus {
  modules: Record<string, string>;
  quantum_instances: number;
  overall_health: 'healthy' | 'degraded' | 'critical' | 'unknown';
}

function App() {
  const [bridgeReady, setBridgeReady] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [quantumBoost, setQuantumBoost] = useState(0);

  // Initialize NextVerse bridge
  useEffect(() => {
    const handleBridgeReady = () => {
      console.log('🚀 NextVerse Quantum Bridge Ready!');
      setBridgeReady(true);
      
      // Test quantum execution
      testQuantumExecution();
      
      // Start system monitoring
      startSystemMonitoring();
    };

    // Listen for bridge ready event
    window.addEventListener('nextverse-ready', handleBridgeReady);

    // Cleanup
    return () => {
      window.removeEventListener('nextverse-ready', handleBridgeReady);
    };
  }, []);

  const testQuantumExecution = useCallback(async () => {
    if (!window.executeQuantum) return;

    try {
      const testTask = {
        operation: 'fibonacci',
        data: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        iterations: 1000000
      };

      console.log('🧮 Testing Quantum Execution...');
      const result = await window.executeQuantum(testTask);
      
      if (result.status === 'success') {
        setQuantumBoost(result.total_boost || 0);
        console.log(`✅ Quantum Test Complete - Boost: ${result.total_boost}x`);
      }
    } catch (error) {
      console.error('❌ Quantum Test Failed:', error);
    }
  }, []);

  const startSystemMonitoring = useCallback(() => {
    const updateStatus = async () => {
      if (!window.getSystemStatus) return;

      try {
        const status = await window.getSystemStatus();
        setSystemStatus(status);
      } catch (error) {
        console.error('❌ System Status Update Failed:', error);
      }
    };

    // Update immediately and then every 5 seconds
    updateStatus();
    const interval = setInterval(updateStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'critical': return 'error';
      default: return 'info';
    }
  };

  const drawerWidth = 240;

  const menuItems = [
    { text: 'Quantum Dashboard', icon: <Dashboard />, path: '/' },
    { text: 'Game Engine', icon: <Games />, path: '/game-engine' },
    { text: 'AI Assistant', icon: <SmartToy />, path: '/ai-assistant' },
    { text: 'DAW Studio', icon: <MusicNote />, path: '/daw-studio' },
    { text: 'MMORPG World', icon: <Public />, path: '/mmorpg' },
    { text: 'System Monitor', icon: <Computer />, path: '/monitor' },
  ];

  return (
    <Router>
      <Box style={{ display: 'flex' }}>
        {/* App Bar */}
        <AppBar position="fixed" style={{ zIndex: 1300 }}>
          <Toolbar>
            <Typography variant="h6" component="div" style={{ flexGrow: 1 }}>
              NextVerse Platform
              {quantumBoost > 0 && (
                <Typography variant="caption" style={{ marginLeft: 8, color: '#90caf9' }}>
                  ⚡ Quantum Boost: {quantumBoost.toFixed(1)}x
                </Typography>
              )}
            </Typography>
            {systemStatus && (
              <Alert 
                severity={getHealthColor(systemStatus.overall_health) as any} 
                variant="outlined"
                style={{ marginRight: 8 }}
              >
                {systemStatus.overall_health.toUpperCase()} | 
                {systemStatus.quantum_instances} Quantum Instances
              </Alert>
            )}
          </Toolbar>
        </AppBar>

        {/* Navigation Drawer */}
        <Drawer
          variant="permanent"
          PaperProps={{ style: { width: drawerWidth, boxSizing: 'border-box' } }}
          style={{ width: drawerWidth, flexShrink: 0 }}
        >
          <Toolbar />
          <Box style={{ overflow: 'auto' }}>
            <List>
              {menuItems.map((item) => (
                <ListItem key={item.text} component={Link} to={item.path} button>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>

        {/* Main Content */}
        <Box component="main" style={{ flexGrow: 1, padding: 24 }}>
          <Toolbar />
          
          {!bridgeReady && (
            <Alert severity="warning" style={{ marginBottom: 8 }}>
              Initializing NextVerse Quantum Bridge...
            </Alert>
          )}

          <Routes>
            <Route path="/" element={<QuantumDashboard />} />
            <Route path="/game-engine" element={<GameEngineView />} />
            <Route path="/ai-assistant" element={<AIAssistant />} />
            <Route path="/daw-studio" element={<DAWStudio />} />
            <Route path="/mmorpg" element={<MMORPGWorld />} />
            <Route path="/monitor" element={<SystemMonitor />} />
          </Routes>
        </Box>
      </Box>
    </Router>
  );
}

export default App;
