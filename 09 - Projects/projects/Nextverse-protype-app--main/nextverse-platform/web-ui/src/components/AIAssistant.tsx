import React, { useState, useRef, useEffect } from 'react';
import HubPresence from './HubPresence';
import SystemBoostPanel from './SystemBoostPanel';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

const AIAssistant: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      text: 'Hello! I am the NextVerse AI Assistant. How can I help you today?',
      sender: 'assistant',
      timestamp: new Date()
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Local placeholder responder. Replace with a POST to the ai-assistant API
  // when the service is available; the UI contract stays the same.
  const simulateAIResponse = (userMessage: string) => {
    setIsTyping(true);
    setTimeout(() => {
      const responses = [
        "I'm analyzing that using the quantum core...",
        "That's an interesting question. Let me apply the JQLD formula to process it optimally.",
        "Based on my local language model and quantum-enhanced reasoning, I believe the answer is yes.",
        "The quantum core indicates this is a valid approach with 94.7% confidence.",
        "I've processed your request through my neural pathways, optimized by the 11-copy quantum VM.",
        "According to my knowledge store, that would work well for your use case."
      ];
      const response = responses[Math.floor(Math.random() * responses.length)];
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        text: response,
        sender: 'assistant',
        timestamp: new Date()
      }]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000);
  };

  const handleSend = () => {
    const text = inputText.trim();
    if (!text) return;
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      text,
      sender: 'user',
      timestamp: new Date()
    }]);
    setInputText('');
    simulateAIResponse(text);
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 12 }}>
      <h3>NextVerse AI Assistant</h3>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {messages.map(message => (
          <div key={message.id} style={{ marginBottom: 12, textAlign: message.sender === 'user' ? 'right' : 'left' }}>
            <span style={{ fontSize: 11, opacity: 0.6 }}>{message.sender}</span>
            <div style={{ padding: 8, borderRadius: 6, background: message.sender === 'user' ? '#123' : '#222' }}>
              {message.text}
            </div>
          </div>
        ))}
        {isTyping && <span style={{ opacity: 0.7 }}>thinking…</span>}
        <div ref={messagesEndRef} />
      </div>
      <div style={{ display: 'flex', gap: 8, paddingTop: 8 }}>
        <input
          style={{ flex: 1 }}
          value={inputText}
          placeholder="Message"
          onChange={event => setInputText(event.target.value)}
          onKeyDown={event => { if (event.key === 'Enter') handleSend(); }}
        />
        <button type="button" onClick={handleSend} disabled={isTyping}>Send</button>
      </div>
      <SystemBoostPanel />
      <HubPresence />
    </div>
  );
};

export default AIAssistant;
