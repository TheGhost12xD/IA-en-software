import os
import faiss
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv

# Dependencias de Langchain/PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Dependencias de FastAPI y Uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from ollama import Client

# --- Configuración Inicial ---
load_dotenv()

# Configuración de Modelos y Rutas
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma:2b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKING_PARAMS = {"chunk_size": 1000, "chunk_overlap": 200}

# Inicialización del cliente Ollama
ollama_client = Client(host=OLLAMA_BASE_URL)

# --- Clases y Modelos ---

class User(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    query: str

# Clase para manejar la persistencia del índice FAISS
class IndexStore:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatL2] = None
        self.texts: List[str] = []
        self.embedding_dim: Optional[int] = None

    def load_index(self):
        """Carga el índice FAISS y los textos si existen."""
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                print(f"Cargando índice FAISS desde: {FAISS_INDEX_PATH}")
                
                # Cargar el índice FAISS
                index_data = faiss.read_index(FAISS_INDEX_PATH)
                self.index = index_data
                
                # Cargar los textos asociados
                texts_path = FAISS_INDEX_PATH.replace(".bin", "_texts.npy")
                if os.path.exists(texts_path):
                    self.texts = np.load(texts_path, allow_pickle=True).tolist()
                    
                # Obtener la dimensión del embedding para usar en las consultas
                self.embedding_dim = self.index.d
                print(f"Índice FAISS cargado. Dimensión: {self.embedding_dim}, Documentos: {len(self.texts)}")

            except Exception as e:
                print(f"Error al cargar el índice FAISS: {e}. Inicializando vacío.")
                self.index = None
                self.texts = []

    def save_index(self):
        """Guarda el índice FAISS y los textos asociados."""
        if self.index:
            try:
                print(f"Guardando índice FAISS en: {FAISS_INDEX_PATH}")
                faiss.write_index(self.index, FAISS_INDEX_PATH)
                
                # Guardar los textos
                texts_path = FAISS_INDEX_PATH.replace(".bin", "_texts.npy")
                np.save(texts_path, np.array(self.texts))
                print("Índice FAISS guardado correctamente.")

            except Exception as e:
                print(f"Error al guardar el índice FAISS: {e}")

    def add_documents(self, chunks: List[str]):
        """Genera embeddings y añade los fragmentos de texto al índice."""
        print(f"Generando embeddings para {len(chunks)} fragmentos de texto...")
        
        # 1. Generar Embeddings de los Chunks (Ollama API /api/embeddings)
        embeddings = []
        for chunk in chunks:
            response = ollama_client.embeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                prompt=chunk
            )
            embeddings.append(response['embedding'])

        new_embeddings = np.array(embeddings).astype('float32')
        
        # Inicializar índice si es la primera vez
        if self.index is None:
            self.embedding_dim = new_embeddings.shape[1]
            # Usar L2 (Euclidean distance) para la similitud
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.texts = []
        
        # Añadir al índice y a la lista de textos
        self.index.add(new_embeddings)
        self.texts.extend(chunks)

        print(f"Añadidos {len(embeddings)} embeddings al índice FAISS. Total: {len(self.texts)}.")
        self.save_index()

    def search(self, query: str, k: int = 4) -> List[str]:
        """Busca los fragmentos más relevantes para la consulta."""
        if self.index is None or not self.texts:
            print("Índice FAISS vacío o no cargado.")
            return []
        
        # 1. Generar Embedding de la Consulta (Ollama API /api/embeddings)
        print(f"Buscando contexto para la consulta: {query}")
        
        try:
            # AQUI SE USA EL MODELO DE EMBEDDING ESPECIFICO
            query_response = ollama_client.embeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                prompt=query
            )
            query_vector = np.array([query_response['embedding']]).astype('float32')
        except Exception as e:
            print(f"ERROR: No se pudo generar el embedding de la consulta con {OLLAMA_EMBEDDING_MODEL}: {e}")
            return []
            
        # 2. Buscar en FAISS
        # Asegurarse de que k no sea mayor que el número total de documentos
        k = min(k, len(self.texts))
        
        distancias, indices = self.index.search(query_vector, k) 

        # 3. Recuperar y devolver los fragmentos de texto
        context = [self.texts[i] for i in indices[0] if i != -1]
        
        if not context:
            print("No se encontraron fragmentos relevantes en la base de conocimiento.")
        
        return context


# --- Inicialización de la Aplicación ---
app = FastAPI()
db = IndexStore()
db.load_index() # Cargar el índice al inicio

# Base de datos de usuarios (simulación)
USERS = {}


# --- Funciones de Utilidad y Endpoints (sin cambios) ---

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKING_PARAMS["chunk_size"],
        chunk_overlap=CHUNKING_PARAMS["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)


def create_rag_prompt(query: str, context: List[str]) -> str:
    context_str = "\n---\n".join(context)
    system_prompt = (
        "Eres un asistente de RAG (Generación Aumentada por Recuperación) de una universidad. "
        "Tu única fuente de conocimiento son los Fragmentos de Contexto proporcionados a continuación. "
        "Responde a la Consulta del Usuario de forma concisa y profesional, basándote **SOLO** en el texto de los fragmentos. "
        "Si los fragmentos no contienen la información para responder, debes responder clara y directamente: 'No tengo información al respecto en el documento de la base de conocimiento.'."
    )
    rag_prompt = f"{system_prompt}\n\nFragmentos de Contexto:\n{context_str}\n\nConsulta del Usuario: {query}"
    return rag_prompt

@app.post("/register")
async def register(user: User):
    if user.username in USERS:
        raise HTTPException(status_code=400, detail="Usuario ya existe")
    USERS[user.username] = user.dict()
    return {"message": "Usuario registrado exitosamente"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if user and user["password"] == form_data.password:
        return {"access_token": user["username"], "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Credenciales inválidas")

def get_current_user(request: Request) -> str:
    auth_header = request.headers.get('authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Token de autenticación requerido")
    
    token = auth_header.split(' ')[1]
    
    if token in USERS:
        return token
    
    raise HTTPException(status_code=401, detail="Token inválido o expirado")

@app.post("/upload-pdf")
async def upload_pdf(user: str = Depends(get_current_user), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")
    
    print(f"Usuario {user} subiendo archivo: {file.filename}")
    
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    
    db.add_documents(chunks)
    
    return {"message": f"Archivo '{file.filename}' procesado e indexado. {len(chunks)} fragmentos añadidos."}

@app.post("/chat")
async def chat(request: ChatRequest, user: str = Depends(get_current_user)):
    query = request.query
    print(f"Consulta de {user}: {query}")

    context = db.search(query, k=4)
    print("Fragmentos recuperados para RAG:")
    print("\n--- Contexto ---\n".join(context) if context else "No se encontraron fragmentos relevantes.")

    rag_prompt = create_rag_prompt(query, context)
    
    try:
        if not context:
            final_prompt = f"El documento no tiene información al respecto. Responde a la consulta '{query}' basándote en conocimiento general, si es posible, o indica que no tienes esa información."
        else:
            final_prompt = rag_prompt

        response = ollama_client.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=final_prompt,
            stream=False
        )
        return {"response": response['response']}
        
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"No se pudo conectar con Ollama o el modelo tardó demasiado. Asegúrate de que '{OLLAMA_MODEL_NAME}' está corriendo en '{OLLAMA_BASE_URL}'. Error: {e}"
        )


# --- Endpoint de Bienvenida (HTML con Navegación de Vistas) ---
@app.get("/", response_class=HTMLResponse)
async def get_html():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot RAG Académico</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; }}
            .container {{ max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            
            /* Contenedores de Vistas */
            .view-panel {{ border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 4px; }}
            .auth-form {{ margin-top: 15px; }}
            
            /* Inputs y Botones */
            input[type="text"], input[type="password"] {{ padding: 10px; margin: 5px 0; width: 95%; border: 1px solid #ccc; border-radius: 4px; }}
            .action-button {{ padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; background-color: #007bff; color: white; margin-top: 10px; }}
            .secondary-button {{ padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; background-color: #6c757d; color: white; margin-top: 10px; }}
            
            /* Chat */
            .chat-window {{ height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; background-color: #fff; border-radius: 4px; }}
            .message-user {{ text-align: right; margin: 5px 0; }}
            .message-bot {{ text-align: left; margin: 5px 0; }}
            .bubble-user {{ background-color: #007bff; color: white; padding: 8px 12px; border-radius: 15px 15px 0 15px; display: inline-block; max-width: 70%; }}
            .bubble-bot {{ background-color: #e2e6ea; padding: 8px 12px; border-radius: 15px 15px 15px 0; display: inline-block; max-width: 70%; }}
            
            #pdfForm input[type="file"] {{ margin-right: 10px; }}
            #chatForm input[type="text"] {{ width: 80%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; margin-right: 10px; }}
            button[type="submit"] {{ background-color: #28a745; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chatbot RAG Académico</h1>
            <p><strong>Modelo de Chat:</strong> {OLLAMA_MODEL_NAME} | <strong>Modelo de Embeddings:</strong> {OLLAMA_EMBEDDING_MODEL}</p>
            
            <div id="auth-login" class="view-panel">
                <h2>1. Inicio de Sesión</h2>
                <form id="loginForm" class="auth-form">
                    <input type="text" id="loginUsername" placeholder="Usuario" required>
                    <input type="password" id="loginPassword" placeholder="Contraseña" required>
                    <button type="submit" class="action-button">Ingresar</button>
                    <p id="loginStatus" class="auth-status"></p>
                </form>
                <button onclick="showPanel('auth-register')" class="secondary-button">Ir a Registro</button>
            </div>

            <div id="auth-register" class="view-panel" style="display: none;">
                <h2>1. Registro de Usuario</h2>
                <form id="registerForm" class="auth-form">
                    <input type="text" id="regUsername" placeholder="Nuevo Usuario" required>
                    <input type="password" id="regPassword" placeholder="Nueva Contraseña" required>
                    <button type="submit" class="action-button">Registrar</button>
                    <p id="registerStatus" class="auth-status"></p>
                </form>
                <button onclick="showPanel('auth-login')" class="secondary-button">Volver a Inicio de Sesión</button>
            </div>

            <div id="app-controls" class="view-panel" style="display: none;">
                <p>Bienvenido/a: <strong id="currentUsername"></strong> 
                   <button onclick="logout()" class="secondary-button" style="float: right;">Cerrar Sesión</button>
                </p>

                <h2>2. Cargar PDF (Base de Conocimiento)</h2>
                <form id="pdfForm" enctype="multipart/form-data">
                    <input type="file" id="pdfFile" name="file" accept=".pdf" required>
                    <button type="submit">Subir y Procesar PDF</button>
                    <p id="uploadStatus"></p>
                </form>

                <h2>3. Chat con el Asistente</h2>
                <div class="chat-window" id="chatWindow">
                    </div>
                <form id="chatForm">
                    <input type="text" id="queryInput" placeholder="Escribe tu consulta académica..." required>
                    <button type="submit">Enviar</button>
                </form>
            </div>
        </div>

        <script>
            let accessToken = null;
            let username = null;

            // ===================================
            // Funciones de Utilidad (Definidas primero para evitar 'is not defined')
            // ===================================
            
            // Función principal para cambiar de vista
            function showPanel(panelId) {{
                document.getElementById('auth-login').style.display = 'none';
                document.getElementById('auth-register').style.display = 'none';
                document.getElementById('app-controls').style.display = 'none';
                document.getElementById(panelId).style.display = 'block';
            }}
            
            // Función para añadir mensajes al chat
            function addMessage(text, sender) {{
                const chatWindow = document.getElementById('chatWindow');
                const msgDiv = document.createElement('div');
                msgDiv.className = sender === 'user' ? 'message-user' : 'message-bot';
                const bubble = document.createElement('div');
                bubble.className = sender === 'user' ? 'bubble-user' : 'bubble-bot';
                bubble.textContent = text;
                msgDiv.appendChild(bubble);
                chatWindow.appendChild(msgDiv);
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }}
            
            // Función de Logout
            function logout() {{
                accessToken = null;
                username = null;
                // Limpiar y resetear el chat
                document.getElementById('chatWindow').innerHTML = 
                    '<div class="message-bot"><div class="bubble-bot">Sesión cerrada. Por favor, vuelve a iniciar sesión.</div></div>';
                showPanel('auth-login');
            }}

            // ===================================
            // Inicialización y Manejadores de Eventos
            // ===================================

            // Inicializar al cargar la página en la vista de Login
            window.onload = function() {{
                showPanel('auth-login');
            }};

            // --- Manejador de Registro ---
            document.getElementById('registerForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const regUsername = document.getElementById('regUsername').value;
                const regPassword = document.getElementById('regPassword').value;
                const status = document.getElementById('registerStatus');
                
                const response = await fetch('/register', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ username: regUsername, password: regPassword }})
                }});
                const data = await response.json();
                
                if (response.ok) {{
                    status.textContent = data.message + ' Redirigiendo a Inicio de Sesión...';
                    status.style.color = 'green';
                    setTimeout(() => showPanel('auth-login'), 1500); 
                }} else {{
                    status.textContent = 'Error de registro: ' + data.detail;
                    status.style.color = 'red';
                }}
            }});

            // --- Manejador de Login ---
            document.getElementById('loginForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const loginUsername = document.getElementById('loginUsername').value;
                const loginPassword = document.getElementById('loginPassword').value;
                const status = document.getElementById('loginStatus');
                
                const formData = new URLSearchParams();
                formData.append('username', loginUsername);
                formData.append('password', loginPassword);

                const response = await fetch('/token', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                    body: formData.toString()
                }});
                const data = await response.json();
                
                if (response.ok) {{
                    accessToken = data.access_token;
                    username = loginUsername;
                    document.getElementById('currentUsername').textContent = username;
                    
                    // Limpiar el chat y mostrar mensaje inicial
                    document.getElementById('chatWindow').innerHTML = '';
                    addMessage('¡Hola ' + username + '! Ya cargué {len(db.texts)} fragmentos de la sesión anterior. Sube un documento o haz una pregunta.', 'bot');

                    showPanel('app-controls'); 
                    status.textContent = 'Ingreso exitoso.';
                    status.style.color = 'green';
                }} else {{
                    status.textContent = 'Error de ingreso: ' + data.detail;
                    status.style.color = 'red';
                }}
            }});


            // --- Manejador de Subida de PDF ---
            document.getElementById('pdfForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const status = document.getElementById('uploadStatus');
                status.textContent = 'Procesando... esto puede tardar unos segundos.';
                
                if (!accessToken) {{
                    status.textContent = 'Error: Debe iniciar sesión primero.';
                    return;
                }}

                const fileInput = document.getElementById('pdfFile');
                if (fileInput.files.length === 0) {{
                    status.textContent = 'Error: Selecciona un archivo PDF.';
                    return;
                }}
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {{
                    const response = await fetch('/upload-pdf', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + accessToken }}, 
                        body: formData
                    }});
                    
                    const data = await response.json();

                    if (response.ok) {{
                        status.textContent = data.message + ' ¡Listo para chatear!';
                        addMessage('Nuevo documento cargado y listo.', 'bot');
                    }} else {{
                        // El detalle del error viene del servidor FastAPI (Python)
                        status.textContent = 'Error al subir el PDF (' + response.status + '): ' + (data.detail || data.message || 'Error desconocido.');
                    }}
                }} catch (error) {{
                    // Este catch atrapa los errores de red (e.g., Uvicorn apagado)
                    status.textContent = 'Error de red o servidor: ' + error.message;
                }}
            }});

            // --- Manejador de Chat ---
            document.getElementById('chatForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const input = document.getElementById('queryInput');
                const query = input.value;
                
                if (!query) return; // No enviar si está vacío

                addMessage(query, 'user');
                input.value = ''; // Borrar la entrada después de enviar

                if (!accessToken) {{
                    addMessage('Error: Debes iniciar sesión para chatear.', 'bot');
                    return;
                }}
                
                // Mostrar un mensaje de carga
                addMessage('Pensando...', 'bot');
                const lastBotBubble = document.querySelector('.message-bot:last-child .bubble-bot');
                
                try {{
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{ 
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + accessToken 
                        }},
                        body: JSON.stringify({{ query: query }})
                    }});
                    
                    // Reemplazar el mensaje de 'Pensando...' con la respuesta real
                    const data = await response.json();
                    
                    if (response.ok) {{
                        lastBotBubble.textContent = data.response;
                    }} else {{
                        lastBotBubble.textContent = 'Error (' + response.status + '): ' + (data.detail || 'Error desconocido.');
                    }}
                }} catch (error) {{
                    // Error de red o servidor
                    lastBotBubble.textContent = 'Error de red o servidor: ' + error.message;
                }}
            }});

        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)