const API_BASE_URL = window.location.origin;
// Usa el mismo origen para evitar CORS
let accessToken = localStorage.getItem('accessToken');
let currentUsername = localStorage.getItem('currentUsername');

document.addEventListener('DOMContentLoaded', () => {
    checkLoginStatus();
});

function showMessage(elementId, text, isSuccess = true) {
    const element = document.getElementById(elementId);
    element.textContent = text;
    element.className = `message ${isSuccess ? 'success' : 'error'}`;
    element.style.display = 'block';
    setTimeout(() => {
        element.style.display = 'none';
    }, 5000);
}

function clearMessages() {
    document.querySelectorAll('.message').forEach(el => el.style.display = 'none');
}

async function register() {
    clearMessages();
    const username = document.getElementById('reg-username').value;
    const password = document.getElementById('reg-password').value;

    if (!username || !password) {
        showMessage('reg-message', 'Usuario y contraseña son requeridos.', false);
        return;
    }

    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
        const response = await fetch(`${API_BASE_URL}/register`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            showMessage('reg-message', `Usuario ${data.username} registrado exitosamente. Por favor, inicia sesión.`, true);
            document.getElementById('reg-username').value = '';
            document.getElementById('reg-password').value = '';
        } else {
            showMessage('reg-message', data.detail || 'Error al registrar usuario.', false);
        }
    } catch (error) {
        console.error('Error de red:', error);
        showMessage('reg-message', 'Error de conexión. Intenta de nuevo.', false);
    }
}

async function login() {
    clearMessages();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    if (!username || !password) {
        showMessage('login-message', 'Usuario y contraseña son requeridos.', false);
        return;
    }

    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    try {
        const response = await fetch(`${API_BASE_URL}/token`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            accessToken = data.access_token;
            currentUsername = username;
            localStorage.setItem('accessToken', accessToken);
            localStorage.setItem('currentUsername', currentUsername);
            checkLoginStatus();
            showMessage('login-message', `Bienvenido, ${username}!`, true);
        } else {
            showMessage('login-message', data.detail || 'Error al iniciar sesión.', false);
        }
    } catch (error) {
        console.error('Error de red:', error);
        showMessage('login-message', 'Error de conexión. Intenta de nuevo.', false);
    }
}

function logout() {
    accessToken = null;
    currentUsername = null;
    localStorage.removeItem('accessToken');
    localStorage.removeItem('currentUsername');
    checkLoginStatus();
    document.getElementById('chat-box').innerHTML = ''; // Limpiar chat al cerrar sesión
    showMessage('login-message', 'Has cerrado sesión.', true);
}

function checkLoginStatus() {
    if (accessToken && currentUsername) {
        document.getElementById('auth-section').style.display = 'none';
        document.getElementById('app-section').style.display = 'block';
        document.getElementById('welcome-username').textContent = currentUsername;
    } else {
        document.getElementById('auth-section').style.display = 'flex';
        document.getElementById('app-section').style.display = 'none';
    }
}

async function uploadPdf() {
    clearMessages();
    const fileInput = document.getElementById('pdf-file');
    const file = fileInput.files[0];

    if (!file) {
        showMessage('upload-message', 'Por favor, selecciona un archivo PDF.', false);
        return;
    }
    if (!accessToken) {
        showMessage('upload-message', 'Debes iniciar sesión para subir PDFs.', false);
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`
            },
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            showMessage('upload-message', data.message, true);
            fileInput.value = ''; // Clear file input
        } else {
            showMessage('upload-message', data.detail || 'Error al subir PDF.', false);
        }
    } catch (error) {
        console.error('Error de red:', error);
        showMessage('upload-message', 'Error de conexión. Intenta de nuevo.', false);
    }
}

function addChatMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

async function sendMessage() {
    clearMessages();
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();

    if (!message) return;
    if (!accessToken) {
        showMessage('chat-message', 'Debes iniciar sesión para chatear.', false);
        return;
    }

    addChatMessage(message, 'user');
    chatInput.value = '';

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${accessToken}`
            },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        if (response.ok) {
            addChatMessage(data.answer, 'bot');
        } else {
            addChatMessage(data.detail || 'Error al obtener respuesta del bot.', 'bot');
            showMessage('chat-message', data.detail || 'Error en el chat.', false);
        }
    } catch (error) {
        console.error('Error de red:', error);
        showMessage('chat-message', 'Error de conexión con el chatbot. Intenta de nuevo.', false);
        addChatMessage('Lo siento, no pude conectar con el asistente.', 'bot');
    }
}

function handleChatInput(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}