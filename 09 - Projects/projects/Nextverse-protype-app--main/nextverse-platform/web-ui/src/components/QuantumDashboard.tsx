import React, { useEffect, useState, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stats, Text, Box } from '@react-three/drei';
import { Card, CardContent, Typography, Grid, Button, Slider, Alert, LinearProgress } from '@mui/material';
import * as THREE from 'three';

interface QuantumParticle {
  id: number;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  energy: number;
  entangled: boolean;
}

const QuantumParticleSystem: React.FC<{ count: number, boost: number }> = ({ count, boost }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [particles] = useState<QuantumParticle[]>(() => 
    Array.from({ length: count }, (_, i) => ({
      id: i,
      position: new THREE.Vector3(
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ),
      velocity: new THREE.Vector3(
        (Math.random() - 0.5) * 0.1,
        (Math.random() - 0.5) * 0.1,
        (Math.random() - 0.5) * 0.1
      ),
      energy: Math.random(),
      entangled: Math.random() > 0.7
    }))
  );

  useFrame((state, delta) => {
    if (!meshRef.current) return;

    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();

    particles.forEach((particle, i) => {
      // Apply quantum boost to movement
      const boostedDelta = delta * (1 + boost / 100);
      
      // Update particle physics
      particle.position.add(particle.velocity.clone().multiplyScalar(boostedDelta));
      
      // Quantum tunneling effect
      if (particle.position.length() > 15) {
        particle.position.multiplyScalar(-0.8);
        particle.velocity.multiplyScalar(0.5);
      }
      
      // Entanglement visualization
      if (particle.entangled) {
        particle.energy = Math.sin(state.clock.elapsedTime * 5 + particle.id) * 0.5 + 0.5;
      }
      
      // Set instance matrix
      matrix.setPosition(particle.position);
      matrix.scale(new THREE.Vector3(
        0.1 + particle.energy * 0.3,
        0.1 + particle.energy * 0.3,
        0.1 + particle.energy * 0.3
      ));
      meshRef.current!.setMatrixAt(i, matrix);
      
      // Set instance color
      color.setHSL(
        particle.entangled ? 0.6 : 0.1, 
        0.8, 
        0.3 + particle.energy * 0.7
      );
      meshRef.current!.setColorAt(i, color);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 8, 6]} />
      <meshPhongMaterial />
    </instancedMesh>
  );
};

const QuantumDashboard: React.FC = () => {
  const [quantumMetrics, setQuantumMetrics] = useState({
    instances: 11,
    totalBoost: 0,
    particleCount: 50000,
    entanglementRate: 0,
    fps: 60
  });
  
  const [jqldParams, setJqldParams] = useState({
    Njq: 5,
    psiParallel: 0.8,
    boostFactor: 420
  });

  const [isRunning, setIsRunning] = useState(false);
  const [performance, setPerformance] = useState({
    cpuUsage: 0,
    memoryUsage: 0,
    renderTime: 0
  });

  useEffect(() => {
    // Simulate real-time quantum metrics
    const interval = setInterval(() => {
      if (isRunning) {
        setQuantumMetrics(prev => ({
          ...prev,
          totalBoost: prev.instances * jqldParams.boostFactor * jqldParams.psiParallel,
          entanglementRate: 0.7 + Math.random() * 0.25,
          fps: 58 + Math.random() * 4
        }));
        
        setPerformance({
          cpuUsage: 20 + Math.random() * 60,
          memoryUsage: 380 + Math.random() * 100,
          renderTime: 16 + Math.random() * 8
        });
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning, jqldParams]);

  const executeQuantumTask = async () => {
    if (!window.executeQuantum) {
      alert('Quantum bridge not available');
      return;
    }

    setIsRunning(true);
    
    try {
      const task = {
        operation: 'quantum_simulation',
        data: {
          particles: quantumMetrics.particleCount,
          jqld_params: jqldParams,
          simulation_time: 10
        }
      };

      const result = await window.executeQuantum(task);
      
      if (result.status === 'success') {
        setQuantumMetrics(prev => ({
          ...prev,
          totalBoost: result.total_boost || prev.totalBoost
        }));
      }
    } catch (error) {
      console.error('Quantum execution failed:', error);
    }
  };

  const stopQuantumTask = () => {
    setIsRunning(false);
  };

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Quantum Core Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Real-time 3D Visualization */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quantum Particle Visualization
              </Typography>
              <div style={{ height: '500px', background: '#000' }}>
                <Canvas camera={{ position: [0, 0, 30], fov: 60 }}>
                  <ambientLight intensity={0.3} />
                  <pointLight position={[10, 10, 10]} intensity={1} />
                  <QuantumParticleSystem 
                    count={Math.min(quantumMetrics.particleCount, 10000)} 
                    boost={quantumMetrics.totalBoost}
                  />
                  <OrbitControls enableZoom enablePan enableRotate />
                  <Stats />
                  <Text
                    position={[0, 12, 0]}
                    fontSize={1}
                    color="white"
                  >
                    {`Quantum Boost: ${quantumMetrics.totalBoost.toFixed(1)}x`}
                  </Text>
                </Canvas>
              </div>
            </CardContent>
          </Card>
        </Grid>

        {/* Control Panel */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            {/* JQLD Parameters */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    JQLD Formula Parameters
                  </Typography>
                  
                  <Typography gutterBottom>Njq Factor: {jqldParams.Njq}</Typography>
                  <Slider
                    value={jqldParams.Njq}
                    onChange={(_, value) => setJqldParams(prev => ({ ...prev, Njq: value as number }))}
                    min={1}
                    max={10}
                    step={1}
                    marks
                    disabled={isRunning}
                  />
                  
                  <Typography gutterBottom>ψ Parallel: {jqldParams.psiParallel}</Typography>
                  <Slider
                    value={jqldParams.psiParallel}
                    onChange={(_, value) => setJqldParams(prev => ({ ...prev, psiParallel: value as number }))}
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    disabled={isRunning}
                  />
                  
                  <Typography gutterBottom>Boost Factor: {jqldParams.boostFactor}</Typography>
                  <Slider
                    value={jqldParams.boostFactor}
                    onChange={(_, value) => setJqldParams(prev => ({ ...prev, boostFactor: value as number }))}
                    min={100}
                    max={1000}
                    step={50}
                    disabled={isRunning}
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Quantum Metrics */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Quantum Metrics
                  </Typography>
                  
                  <Typography variant="body2">
                    Active Instances: {quantumMetrics.instances}
                  </Typography>
                  <Typography variant="body2">
                    Total Boost: {quantumMetrics.totalBoost.toFixed(1)}x
                  </Typography>
                  <Typography variant="body2">
                    Entanglement Rate: {(quantumMetrics.entanglementRate * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    Render FPS: {quantumMetrics.fps.toFixed(1)}
                  </Typography>
                  
                  <LinearProgress 
                    variant="determinate" 
                    value={Math.min(quantumMetrics.totalBoost / 10, 100)} 
                    style={{ marginTop: '10px' }}
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Performance Monitor */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Performance Monitor
                  </Typography>
                  
                  <Typography variant="body2">
                    CPU Usage: {performance.cpuUsage.toFixed(1)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={performance.cpuUsage} 
                    color={performance.cpuUsage > 80 ? 'error' : 'primary'}
                  />
                  
                  <Typography variant="body2" style={{ marginTop: '10px' }}>
                    Memory: {performance.memoryUsage.toFixed(0)}MB
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(performance.memoryUsage / 1000) * 100} 
                    color={performance.memoryUsage > 800 ? 'warning' : 'primary'}
                  />
                  
                  <Typography variant="body2" style={{ marginTop: '10px' }}>
                    Render Time: {performance.renderTime.toFixed(1)}ms
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(performance.renderTime / 33) * 100} 
                    color={performance.renderTime > 25 ? 'error' : 'success'}
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Control Buttons */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Quantum Control
                  </Typography>
                  
                  {!isRunning ? (
                    <Button 
                      variant="contained" 
                      color="primary" 
                      onClick={executeQuantumTask}
                      fullWidth
                      style={{ marginBottom: '10px' }}
                    >
                      🚀 Execute Quantum Task
                    </Button>
                  ) : (
                    <Button 
                      variant="contained" 
                      color="secondary" 
                      onClick={stopQuantumTask}
                      fullWidth
                      style={{ marginBottom: '10px' }}
                    >
                      ⏹️ Stop Quantum Task
                    </Button>
                  )}
                  
                  {isRunning && (
                    <Alert severity="success">
                      Quantum simulation running with {quantumMetrics.instances} instances
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </div>
  );
};

export default QuantumDashboard;
