# React room binding (optional — copy if it fits your app)

For the common "everyone on this app plays together" shape, copy this hook to
`src/lib/multiplayer/use-p2p-room.ts` and adapt it freely — it is yours, not
part of the kit. It captures `room`/`name` on first render, so changing them
later requires remounting the component (key it on the room code).


```ts
/**
 * React binding for P2PRoom. Identity and room id are captured once on mount
 * (useState initializers) so re-renders never tear down the mesh: the P2PRoom
 * instance lives exactly as long as the component that mounted it, and
 * changing `room`/`name` requires a remount (key the component on them).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { P2PRoom, type PeerInfo } from "./p2p";

export interface UseP2PRoomOptions {
  /** Defaults to a per-deployment room derived from the hostname. */
  room?: string;
  name?: string;
}

export interface P2PRoomHandle {
  selfId: string;
  room: string;
  /** Remote peers only (self excluded), with live connection diagnostics. */
  peers: PeerInfo[];
  joined: boolean;
  /** Unreliable game-state fanout to every connected peer. */
  broadcast: (data: unknown) => void;
  /** Reliable ordered send to one peer (or all when peerId is omitted). */
  send: (data: unknown, peerId?: string) => void;
  /** Subscribe to incoming messages; returns an unsubscribe function. */
  onMessage: (
    fn: (from: string, data: unknown, channel: "state" | "reliable") => void,
  ) => () => void;
}

function defaultRoom(): string {
  if (typeof window === "undefined") return "room-ssr";
  // DNS labels can be 63 chars; the signaling ID regex caps room ids at 64 —
  // truncate so `room-` + label always fits.
  return `room-${window.location.hostname.split(".")[0]}`.slice(0, 64);
}

export function useP2PRoom(options: UseP2PRoomOptions = {}): P2PRoomHandle {
  const [selfId] = useState(() => `p-${Math.random().toString(36).slice(2, 10)}`);
  const [room] = useState(() => options.room ?? defaultRoom());
  const [name] = useState(() => options.name ?? selfId);
  const [peers, setPeers] = useState<PeerInfo[]>([]);
  const [joined, setJoined] = useState(false);
  const roomRef = useRef<P2PRoom | null>(null);
  const listeners = useRef(
    new Set<(from: string, data: unknown, channel: "state" | "reliable") => void>(),
  );

  useEffect(() => {
    const p2p = new P2PRoom({
      room,
      selfId,
      name,
      onPeersChanged: setPeers,
      onMessage: (from, data, channel) => {
        for (const fn of listeners.current) fn(from, data, channel);
      },
      // `joined` flips on the FIRST successful poll — join() itself resolves
      // even when the first poll fails (the loop keeps retrying).
      onConnected: () => setJoined(true),
    });
    roomRef.current = p2p;
    void p2p.join();
    return () => {
      roomRef.current = null;
      p2p.close();
    };
  }, [room, selfId, name]);

  // Stable identities (both close over refs) so consumers can safely list
  // these in effect deps without re-subscribing every render.
  const broadcast = useCallback((data: unknown) => roomRef.current?.broadcast(data), []);
  const send = useCallback(
    (data: unknown, peerId?: string) => roomRef.current?.send(data, peerId),
    [],
  );
  const onMessage = useCallback(
    (fn: (from: string, data: unknown, channel: "state" | "reliable") => void) => {
      listeners.current.add(fn);
      return () => {
        listeners.current.delete(fn);
      };
    },
    [],
  );

  return { selfId, room, peers, joined, broadcast, send, onMessage };
}
```


Used in a component:


```tsx
import { useP2PRoom } from "@/lib/multiplayer";

function Game() {
  const p2p = useP2PRoom({ name: "ani" }); // room defaults per deployment
  const [positions, setPositions] = useState<Record<string, Pos>>({});

  useEffect(
    () =>
      p2p.onMessage((from, data, channel) => {
        if (channel === "state") {
          setPositions((p) => ({ ...p, [from]: data as Pos }));
        }
      }),
    [p2p.onMessage],
  );

  // Game-rate state: broadcast on the unreliable channel (stale packets drop).
  // 20-30 sends/s is plenty; interpolate between updates for smooth motion.
  useEffect(() => {
    let raf = 0;
    let lastSent = 0;
    const loop = (now: number) => {
      if (now - lastSent >= 50) {
        // ~20 sends/s
        p2p.broadcast(myPositionRef.current);
        lastSent = now;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [p2p.broadcast]);

  // One-shot actions (chat, "start game"): reliable + ordered.
  const sendChat = (text: string) => p2p.send({ chat: text });

  return <Board peers={p2p.peers} positions={positions} />;
}
```
