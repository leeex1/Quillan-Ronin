import React, { useEffect, useRef, useState } from 'react';
import { connectHub, HubMessage, PresenceMember } from '../services/hubSocket';

// Live social hub panel backed by the local multiplayer service.
const HubPresence: React.FC = () => {
  const [hubId, setHubId] = useState('nexus_atrium');
  const [joined, setJoined] = useState(false);
  const [members, setMembers] = useState<PresenceMember[]>([]);
  const [messages, setMessages] = useState<HubMessage[]>([]);
  const [body, setBody] = useState('');
  const socketRef = useRef<ReturnType<typeof connectHub> | null>(null);

  useEffect(() => () => {
    if (socketRef.current) socketRef.current.disconnect();
  }, []);

  const join = () => {
    if (socketRef.current) socketRef.current.disconnect();
    const socket = connectHub({
      onPresence: (list) => setMembers(list),
      onMessage: (message) => setMessages(prev => [...prev, message]),
      onError: (message) => setMessages(prev => [...prev, {
        hub_id: hubId, character_id: 'system', body: message, timestamp: Date.now() / 1000,
      }]),
    });
    socketRef.current = socket;
    socket.emit('join_hub', {
      hub_id: hubId,
      zone_id: 'nexus_atrium',
      character_id: `guest-${Math.floor(Math.random() * 1000)}`,
    });
    setJoined(true);
  };

  const send = () => {
    const text = body.trim();
    if (!text || !socketRef.current) return;
    socketRef.current.emit('hub_chat', { hub_id: hubId, body: text });
    setBody('');
  };

  return (
    <div style={{ border: '1px solid #37474f', borderRadius: 6, padding: 10, marginTop: 12 }}>
      <h4 style={{ marginTop: 0 }}>Social hub — live presence</h4>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <input value={hubId} onChange={event => setHubId(event.target.value)} />
        <button type="button" onClick={join}>{joined ? 'Rejoin' : 'Join'}</button>
        <span style={{ fontSize: 12 }}>{members.length} present</span>
      </div>
      <div style={{ maxHeight: 120, overflowY: 'auto', fontSize: 12 }}>
        {messages.map((message, index) => (
          <div key={index}><b>{message.character_id}:</b> {message.body}</div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <input value={body} placeholder="Hub message"
               onChange={event => setBody(event.target.value)}
               onKeyDown={event => { if (event.key === 'Enter') send(); }} />
        <button type="button" onClick={send} disabled={!joined}>Send</button>
      </div>
    </div>
  );
};

export default HubPresence;
