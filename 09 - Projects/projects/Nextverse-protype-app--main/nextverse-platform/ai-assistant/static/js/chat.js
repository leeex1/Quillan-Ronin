document.addEventListener('DOMContentLoaded', () => {
    // Connect to the Socket.IO server.
    // The URL should match your server address and port.
    const socket = io(); // Assumes server is on the same host/port by default

    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');

    // Basic user/session ID generation for demonstration
    // In a real app, these would be managed more robustly (e.g., server-side sessions, login)
    let userId = localStorage.getItem('ai_chat_user_id');
    if (!userId) {
        userId = 'user_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('ai_chat_user_id', userId);
    }

    let sessionId = sessionStorage.getItem('ai_chat_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        sessionStorage.setItem('ai_chat_session_id', sessionId);
    }

    // Function to display messages in the chat window
    function displayMessage(sender, messageText, metaInfo = '') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'assistant-message');

        const metaDiv = document.createElement('div');
        metaDiv.classList.add('meta');
        metaDiv.textContent = sender === 'user' ? `You (${userId})` : `AI Assistant`;
        if (metaInfo) {
            metaDiv.textContent += ` | ${metaInfo}`;
        }

        const textDiv = document.createElement('div');
        textDiv.classList.add('text');
        textDiv.textContent = messageText;

        messageDiv.appendChild(metaDiv);
        messageDiv.appendChild(textDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the bottom
    }

    function displayErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.classList.add('message', 'error-message');
        errorDiv.textContent = message;
        chatMessages.appendChild(errorDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }


    // Function to send a message
    function sendMessage() {
        const messageText = messageInput.value.trim();
        if (messageText) {
            displayMessage('user', messageText);
            socket.emit('user_message', {
                message: messageText,
                user_id: userId,
                session_id: sessionId
            });
            messageInput.value = '';
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // SocketIO event listeners
    socket.on('connect', () => {
        console.log('Connected to server with SID:', socket.id);
        displayMessage('system', 'Connected to AI Assistant.');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server.');
        displayMessage('system', 'Disconnected from AI Assistant.');
    });

    socket.on('connection_status', (data) => {
        console.log('Connection status:', data);
    });

    socket.on('ai_response', (data) => {
        console.log('AI Response received:', data);
        let meta = `Intent: ${data.nlu_details.intent}, Sentiment: ${data.nlu_details.sentiment}`;
        // Add ethical compliance info to meta if desired
        meta += ` | Input OK: ${data.input_ethical_check.is_compliant}`;
        if(data.output_ethical_check) {
             meta += ` | Output OK: ${data.output_ethical_check.is_compliant}`;
        }
        displayMessage('assistant', data.ai_response, meta);
    });

    socket.on('error_response', (data) => {
        console.error('Server Error:', data.error);
        displayErrorMessage(`Server error: ${data.error}`);
    });

    // Initial system message
    displayMessage('system', 'Welcome! Connecting to the AI Assistant...');
});
