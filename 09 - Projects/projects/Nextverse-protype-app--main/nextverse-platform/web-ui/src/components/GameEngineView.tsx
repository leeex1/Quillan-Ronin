import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { ZONES, DEFAULT_ZONE, findZone } from '../data/zones';
import { connectHub } from '../services/hubSocket';

interface RemotePlayer { sid: string; x: number; y: number; z: number; label: string }

// Production-style zone browser: loads real Tripo GLB assets when present,
// falls back to procedural geometry, and renders live hub players.
const GameEngineView: React.FC = () => {
  const mount = useRef<HTMLDivElement>(null);
  const remoteGroupRef = useRef<THREE.Group | null>(null);
  const [zoneId, setZoneId] = useState<string>(DEFAULT_ZONE);
  const [assetCount, setAssetCount] = useState(0);
  const [players, setPlayers] = useState<RemotePlayer[]>([]);
  const zone = findZone(zoneId);

  useEffect(() => {
    const container = mount.current;
    if (!container) return;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#05070d');
    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);
    camera.position.set(0, 2.2, 6);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Image-based lighting gives MeshStandardMaterial real PBR reflections.
    const pmrem = new THREE.PMREMGenerator(renderer);
    const roomEnvironment = new RoomEnvironment();
    const environmentMap = pmrem.fromScene(roomEnvironment, 0.04).texture;
    scene.environment = environmentMap;
    roomEnvironment.dispose();
    pmrem.dispose();

    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.3);
    key.position.set(3, 5, 2);
    scene.add(key);

    const disposables: Array<{ dispose: () => void }> = [];
    const localAvatarGeometry = new THREE.CapsuleGeometry(0.3, 0.9, 4, 8);
    const localAvatarMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff9c });
    const avatar = new THREE.Mesh(localAvatarGeometry, localAvatarMaterial);
    avatar.position.set(0, -0.5, 0);
    scene.add(avatar);
    disposables.push(localAvatarGeometry, localAvatarMaterial);

    const remoteGroup = new THREE.Group();
    remoteGroupRef.current = remoteGroup;
    scene.add(remoteGroup);

    const floorGeometry = new THREE.PlaneGeometry(40, 40);
    const floorMaterial = new THREE.MeshStandardMaterial({ color: 0x0b1220, roughness: 0.85 });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.4;
    scene.add(floor);
    disposables.push(floorGeometry, floorMaterial);

    // Real Tripo assets: manifest-driven GLB loading with procedural fallback.
    const loader = new GLTFLoader();
    fetch('/assets/tripio/manifest.json')
      .then(response => response.json())
      .then((manifest: { assets: Array<{ file: string; zone_id: string }> }) => {
        const matches = (manifest.assets || []).filter(item => item.zone_id === zone.id);
        setAssetCount(matches.length);
        matches.forEach(item => {
          loader.load(`/assets/tripio/${item.file}`,
            gltf => scene.add(gltf.scene),
            undefined,
            () => setAssetCount(0));
        });
      })
      .catch(() => setAssetCount(0));

    // Procedural zone landmark when no generated asset is available.
    const landmarkGeometry = new THREE.IcosahedronGeometry(1.2, 2);
    const landmarkMaterial = new THREE.MeshStandardMaterial({
      color: zone.color, roughness: 0.24, metalness: 0.72, wireframe: true,
    });
    const landmark = new THREE.Mesh(landmarkGeometry, landmarkMaterial);
    landmark.position.set(0, 0.7, -2);
    scene.add(landmark);
    disposables.push(landmarkGeometry, landmarkMaterial, environmentMap);

    // Circle colliders keep the avatar out of zone obstacles.
    const colliders = [
      { x: 0, z: -2, radius: 1.7 },
      { x: 6, z: 3, radius: 1.2 },
      { x: -6, z: 2, radius: 1.2 },
    ];

    const socket = connectHub({
      onPresence: () => undefined,
      onMessage: () => undefined,
    });
    socket.emit('join_hub', { hub_id: 'nexus_atrium', zone_id: zone.id, character_id: 'web-avatar' });
    socket.on('player_moved', (event: RemotePlayer) => {
      setPlayers(prev => {
        const next = prev.filter(player => player.sid !== event.sid);
        next.push({ sid: event.sid, x: event.x, y: event.y, z: event.z, label: event.label });
        return next;
      });
    });

    const keys: Record<string, boolean> = {};
    const onKeyDown = (event: KeyboardEvent) => { keys[event.key.toLowerCase()] = true; };
    const onKeyUp = (event: KeyboardEvent) => { keys[event.key.toLowerCase()] = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    let frame = 0;
    let lastBroadcast = 0;
    const animate = () => {
      const speed = 3.2 / 60;
      if (keys['w']) avatar.position.z -= speed;
      if (keys['s']) avatar.position.z += speed;
      if (keys['a']) avatar.position.x -= speed;
      if (keys['d']) avatar.position.x += speed;
      avatar.position.x = Math.max(-18, Math.min(18, avatar.position.x));
      avatar.position.z = Math.max(-18, Math.min(18, avatar.position.z));
      colliders.forEach(collider => {
        const dx = avatar.position.x - collider.x;
        const dz = avatar.position.z - collider.z;
        const distance = Math.hypot(dx, dz);
        const minimum = collider.radius + 0.35;
        if (distance > 0.0001 && distance < minimum) {
          const push = (minimum - distance) / distance;
          avatar.position.x += dx * push;
          avatar.position.z += dz * push;
        }
      });
      landmark.rotation.y += 0.006;
      const now = Date.now();
      if (now - lastBroadcast > 80) {
        socket.emit('player_move', {
          hub_id: 'nexus_atrium',
          x: avatar.position.x, y: avatar.position.y, z: avatar.position.z,
        });
        lastBroadcast = now;
      }
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      socket.disconnect();
      disposables.forEach(item => item.dispose());
      renderer.dispose();
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
    };
  }, [zone.color, zone.id]);

  useEffect(() => {
    // Remote avatars update without rebuilding the WebGL scene.
    const group = remoteGroupRef.current;
    if (!group) return;
    while (group.children.length > 0) {
      const child = group.children[0] as THREE.Mesh;
      group.remove(child);
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
    players.forEach(player => {
      const geometry = new THREE.SphereGeometry(0.35, 12, 8);
      const material = new THREE.MeshStandardMaterial({ color: 0xff6b35 });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(player.x, player.y, player.z);
      group.add(mesh);
    });
  }, [players]);

  return (
    <div style={{ position: 'relative', height: '100%', minHeight: 420 }}>
      <div ref={mount} style={{ width: '100%', height: '100%' }} />
      <div style={{ position: 'absolute', left: 12, top: 12, pointerEvents: 'auto' }}>
        <h3 style={{ margin: 0 }}>Zone Browser</h3>
        <select value={zoneId} onChange={event => setZoneId(event.target.value)}>
          {ZONES.map(item => <option key={item.id} value={item.id}>{item.name} ({item.tier})</option>)}
        </select>
        <div style={{ fontSize: 12, opacity: 0.8 }}>
          {zone.id} · Tripo GLB assets: {assetCount} · hub players: {players.length} · WASD to move
        </div>
      </div>
    </div>
  );
};

export default GameEngineView;
