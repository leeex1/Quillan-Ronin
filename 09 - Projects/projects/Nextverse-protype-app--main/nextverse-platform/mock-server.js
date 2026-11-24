// Simple Express server for NextVerse platform mock
const express = require('express');
const path = require('path');
const app = express();
const port = 3001;

// Mock API for quantum core status
app.get('/api/quantum-core/status', (req, res) => {
  res.json({
    status: 'healthy',
    boost: 420,
    formula: 'JQLD',
    instances: 11,
    performance: 'optimal'
  });
});

// Mock API for AI assistant
app.post('/api/ai-assistant/chat', (req, res) => {
  setTimeout(() => {
    res.json({
      response: "Hello! I'm the NextVerse AI Assistant. How can I help you today?",
      ethically_approved: true,
      timestamp: Date.now()
    });
  }, 500); // Simulate a small delay for realism
});

// Mock API for system status
app.get('/api/modules/status', (req, res) => {
  res.json({
    modules: {
      'quantum-core': 'healthy',
      'ai-assistant': 'healthy',
      'game-engine': 'healthy',
      'mmorpg': 'degraded',
      'daw': 'healthy',
      'orchestrator': 'healthy'
    },
    quantum_instances: 11,
    overall_health: 'healthy'
  });
});

// Static file serving for our basic test UI
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>NextVerse E2E Test</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .card { border: 1px solid #ccc; border-radius: 8px; padding: 16px; }
        .chat-messages { height: 200px; border: 1px solid #eee; padding: 8px; overflow-y: auto; margin-bottom: 8px; }
        .user { text-align: right; color: blue; }
        .assistant { color: green; }
        .system { color: gray; font-style: italic; }
        input { width: 80%; padding: 8px; }
        button { padding: 8px 16px; }
        .status { display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 8px 0; }
        .healthy { color: green; }
        .degraded { color: orange; }
        .critical { color: red; }
      </style>
    </head>
    <body>
      <h1>NextVerse Platform E2E Test</h1>
      <div class="system-health">
        <h2>System Health</h2>
        <div id="system-status"></div>
      </div>

      <div class="dashboard">
        <div class="card">
          <h2>AI Assistant</h2>
          <div class="chat-messages" id="chat-messages">
            <div class="assistant">Hello! I am the NextVerse AI Assistant. How can I help you today?</div>
          </div>
          <div class="chat-input">
            <input type="text" id="message-input" placeholder="Type your message...">
            <button id="send-button">Send</button>
          </div>
        </div>
        
        <div class="card">
          <h2>Quantum Core</h2>
          <div id="quantum-stats">
            <div>Status: <span class="healthy">healthy</span></div>
            <div>Boost: <span id="quantum-boost">420</span>x</div>
            <div>Formula: JQLD</div>
            <div>Instances: <span id="quantum-instances">11</span></div>
          </div>
        </div>
      </div>

      <script>
        // Get elements
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const chatMessages = document.getElementById('chat-messages');
        const systemStatus = document.getElementById('system-status');

        // Function to add chat message
        function addMessage(role, content) {
          const messageElement = document.createElement('div');
          messageElement.classList.add(role);
          messageElement.textContent = content;
          chatMessages.appendChild(messageElement);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Send message function
        async function sendMessage() {
          const message = messageInput.value.trim();
          if (!message) return;

          // Add user message
          addMessage('user', message);
          messageInput.value = '';

          try {
            // Call mock API
            const response = await fetch('/api/ai-assistant/chat', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ message }),
            });

            const data = await response.json();
            
            // Add AI response
            addMessage('assistant', data.response);
            
            // Update quantum stats (simulate increased usage)
            document.getElementById('quantum-boost').textContent = Math.floor(420 + Math.random() * 10);
            document.getElementById('quantum-instances').textContent = 11 + Math.floor(Math.random() * 3);
            
          } catch (error) {
            addMessage('system', 'Error: Could not reach AI Assistant');
            console.error('Error:', error);
          }
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
          if (e.key === 'Enter') {
            sendMessage();
          }
        });

        // Load system status
        async function loadSystemStatus() {
          try {
            const response = await fetch('/api/modules/status');
            const data = await response.json();
            
            // Clear previous status
            systemStatus.innerHTML = '';
            
            // Add module status
            Object.entries(data.modules).forEach(([module, status]) => {
              const statusElement = document.createElement('div');
              statusElement.classList.add('status');
              statusElement.innerHTML = `
                <div>${module}</div>
                <div class="${status}">${status}</div>
              `;
              systemStatus.appendChild(statusElement);
            });
            
            // Overall status
            const overallElement = document.createElement('div');
            overallElement.classList.add('status');
            overallElement.innerHTML = `
              <div><strong>Overall:</strong></div>
              <div class="${data.overall_health}"><strong>${data.overall_health}</strong></div>
            `;
            systemStatus.appendChild(overallElement);
            
          } catch (error) {
            systemStatus.innerHTML = '<div class="status"><div>Error loading system status</div></div>';
            console.error('Error loading status:', error);
          }
        }

        // Initial load
        loadSystemStatus();
        // Refresh status every 5 seconds
        setInterval(loadSystemStatus, 5000);
      </script>
    </body>
    </html>
  `);
});

// Start the server
app.listen(port, () => {
  console.log(`NextVerse mock server running at http://localhost:${port}`);
});
