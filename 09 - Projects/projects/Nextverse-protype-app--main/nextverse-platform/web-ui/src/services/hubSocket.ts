import { io, Socket } from 'socket.io-client';

export interface PresenceMember {
  sid: string;
  hub_id: string;
  zone_id: string;
  character_id: string;
  presence_state: string;
  last_seen: number;
}

export interface HubMessage {
  hub_id: string;
  character_id: string;
  body: string;
  timestamp: number;
}

export interface HubHandlers {
  onPresence: (members: PresenceMember[]) => void;
  onMessage: (message: HubMessage) => void;
  onError?: (message: string) => void;
}

// Connects to the local Flask-SocketIO hub service. Defaults to loopback only.
export function connectHub(handlers: HubHandlers, url: string = 'http://127.0.0.1:8085'): Socket {
  const socket = io(url, { autoConnect: true, transports: ['websocket', 'polling'] });
  socket.on('presence_update', (payload: { members: PresenceMember[] }) => handlers.onPresence(payload.members));
  socket.on('hub_message', (message: HubMessage) => handlers.onMessage(message));
  socket.on('hub_error', (payload: { message: string }) => {
    if (handlers.onError) handlers.onError(payload.message);
  });
  return socket;
}
