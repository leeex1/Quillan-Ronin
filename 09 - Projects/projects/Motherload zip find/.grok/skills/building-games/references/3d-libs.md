# 3D libraries for this stack (three.js + React Three Fiber + drei + rapier)

The app is React/TanStack. The idiomatic way to do 3D here is **React Three
Fiber (R3F)** — write three.js as React components inside a `<Canvas>` on a
route. It deploys like any other page.

> These are **not preinstalled**. Install and confirm they land in `package.json`:
> `npm i three @react-three/fiber @react-three/drei @react-three/rapier`
> (+ `npm i -D @types/three`). If the deploy build has no three, the game is blank.

## When to use what
- **three.js** — the engine. Raw three is fine for a self-contained canvas; prefer
  R3F when the game has React UI/state around it.
- **@react-three/fiber** — renders three as React; `useFrame((state, delta) => …)`
  is your game loop (delta is seconds — scale movement by it, see SKILL §1).
- **@react-three/drei** — helpers so you don't hand-roll bug-prone code:
  `<PointerLockControls/>`, `useKeyboardControls`, `<OrbitControls/>`,
  `useGLTF`, `<Environment/>`, `<Instances/>`, `<Stats/>`.
- **@react-three/rapier** — physics + **character controller** (capsule +
  autostep + snap-to-ground). Use for FPS/platformer movement and collisions
  instead of hand-rolled raycasts. `<Physics>`, `<RigidBody>`, `useRapier`.

## Minimal R3F shape
```tsx
import { Canvas, useFrame } from "@react-three/fiber";
import { PointerLockControls, useKeyboardControls } from "@react-three/drei";

function Player() {
  useFrame((_, delta) => {
    const d = Math.min(delta, 0.1);       // cap delta (SKILL §1)
    // move with a dedicated moveForward/moveRight basis (SKILL §2/§4)
  });
  return null;
}

export default function Game() {
  return (
    <Canvas camera={{ position: [0, 1.7, 5], fov: 75 }} shadows dpr={[1, 2]}>
      {/* scene */}
      <PointerLockControls />   {/* mouse-look ONLY — implement WASD yourself */}
      <Player />
    </Canvas>
  );
}
```

## Gotchas (map to the SKILL universals)
- **Controls:** drei `PointerLockControls` is mouse-look only — WASD is yours
  (SKILL §2). Gate `.lock()` behind a "click to play" overlay and dismiss it on lock.
- **Orientation:** meshes face +Z, camera looks −Z; right-handed +Y up (SKILL §3).
- **Delta:** `useFrame` delta is seconds; cap it; fixed-step physics via rapier's
  own stepping. Don't read `THREE.Clock.getDelta()` twice a frame.
- **Perf:** `<Instances>`/`InstancedMesh` for repeats; dispose on unmount (R3F
  auto-disposes objects it created, but not manual textures/loaders — dispose those).
  Set `dpr={[1, 2]}` to cap retina cost.
- **Mobile:** R3F sets pixel ratio via `dpr`; still add touch controls + `touch-action:none`.

## Alternatives
- **Babylon.js** (`references/babylon.md`) — batteries-included 3D (Havok physics,
  inspector, SceneOptimizer). Viable if you want an all-in-one engine, but it's a
  separate paradigm from React; default to three/R3F for consistency with the app.

Sources: R3F docs (https://r3f.docs.pmnd.rs), drei (https://github.com/pmndrs/drei),
react-three-rapier (https://github.com/pmndrs/react-three-rapier), three.js docs.
