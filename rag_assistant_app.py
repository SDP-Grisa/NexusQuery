"""
=================================================================
RAG ASSISTANT - Enhanced Document Q&A Application
=================================================================
Features:
- Advanced PDF text extraction
- Semantic chunking for better context
- Vector database with ChromaDB
- Cross-encoder reranking for accuracy
- Source attribution with relevance scores
- Interactive document analysis
- Copy & share functionality
=================================================================
"""

import streamlit as st
import mysql.connector
from mysql.connector import Error
import requests
import json
import hashlib
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import os
import io
import re
import base64
import tempfile

# RAG-specific imports
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except:
    PYMUPDF_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SEMANTIC_AVAILABLE = True
except:
    SEMANTIC_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except:
    NUMPY_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except:
    DOCX_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except:
    PPTX_AVAILABLE = False

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="📚 RAG Assistant Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    /* Main styling */
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Source card styling */
    .source-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .source-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .source-header {
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    .relevance-badge {
        background: rgba(255,255,255,0.3);
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Document upload area */
    .upload-zone {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
    }
    
    /* Copy button */
    .copy-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 15px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.9em;
        margin: 5px;
        display: inline-block;
    }
    
    .copy-button:hover {
        background-color: #45a049;
    }
    
    /* Info badge */
    .info-badge {
        background-color: #2196F3;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        display: inline-block;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ================= API CONFIGURATION =================
API_PROVIDER = "huggingface"
try:
    API_KEY = st.secrets["api"]["hf_token"]
except:
    API_KEY = None

if API_PROVIDER == "huggingface" and API_KEY:
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:fastest"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

# ================= RAG SETTINGS =================
SEMANTIC_CHUNK_SIZE = 400
SLIDING_WINDOW_SIZE = 600
WINDOW_OVERLAP = 150
INITIAL_RETRIEVAL_K = 20
RERANK_TOP_K = 10
FINAL_CONTEXT_K = 5

MAX_RECENT_MESSAGES = 10
MAX_CONTEXT_TOKENS = 3000

# ================= CACHE RAG MODELS =================
@st.cache_resource
def load_rag_models():
    """Load RAG models if available"""
    if not SEMANTIC_AVAILABLE or not CHROMADB_AVAILABLE:
        return None, None, None
    
    try:
        embed = SentenceTransformer("all-MiniLM-L6-v2")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        client = chromadb.Client(
            Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            )
        )
        return embed, reranker, client
    except Exception as e:
        st.error(f"Error loading RAG models: {e}")
        return None, None, None

# ================= DATABASE HELPER FUNCTIONS =================
def get_temp_ssl_ca(ca_b64_secret: str) -> str:
    """Decode base64 SSL CA and write to temp file."""
    if not ca_b64_secret:
        return ""
    try:
        cert_bytes = base64.b64decode(ca_b64_secret)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as temp_file:
            temp_file.write(cert_bytes)
            temp_path = temp_file.name
        return temp_path
    except Exception as e:
        st.error(f"Failed to decode SSL cert: {e}")
        return ""

ssl_ca_b64 = st.secrets.get("database", {}).get("ssl_ca_b64", "")
ssl_ca_path = get_temp_ssl_ca(ssl_ca_b64) if ssl_ca_b64 else ""

@st.cache_resource
def get_auth_db_connection():
    """Connect to authentication database"""
    try:
        if "auth_database" in st.secrets:
            connection = mysql.connector.connect(
                host=st.secrets["auth_database"]["host"],
                port=int(st.secrets["auth_database"]["port"]),
                database=st.secrets["auth_database"]["database"],
                user=st.secrets["auth_database"]["user"],
                password=st.secrets["auth_database"]["password"],
                ssl_disabled=False,
                ssl_verify_cert=True,
                ssl_ca=ssl_ca_path if ssl_ca_path else None,
                ssl_verify_identity=True,
                connect_timeout=30
            )
        else:
            connection = mysql.connector.connect(
                host='localhost',
                database='auth_db',
                user='root',
                password='password',
                connect_timeout=10
            )
        
        if connection.is_connected():
            init_auth_tables(connection)
            return connection
    except Error as e:
        st.error(f"❌ Auth Database connection failed: {e}")
        return None

def init_auth_tables(connection):
    """Initialize authentication tables"""
    cursor = connection.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                email VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                history_id INT AUTO_INCREMENT PRIMARY KEY,
                chat_id INT NOT NULL,
                user_id INT NOT NULL,
                question TEXT,
                query_generated TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            )
        """)
        
        connection.commit()
    except Error as e:
        st.error(f"Error initializing auth tables: {e}")
    finally:
        cursor.close()

# ================= USER AUTHENTICATION =================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str) -> Tuple[bool, str]:
    if not st.session_state.auth_db:
        return False, "Auth database not connected"
    
    cursor = st.session_state.auth_db.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash)
        )
        st.session_state.auth_db.commit()
        return True, "User created successfully"
    except Error as e:
        if "Duplicate entry" in str(e):
            return False, "Username already exists"
        return False, f"Error creating user: {e}"
    finally:
        cursor.close()

def authenticate_user(username: str, password: str) -> Optional[int]:
    if not st.session_state.auth_db:
        return None
    
    cursor = st.session_state.auth_db.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "SELECT user_id FROM users WHERE username = %s AND password_hash = %s",
            (username, password_hash)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    except Error as e:
        st.error(f"Authentication error: {e}")
        return None
    finally:
        cursor.close()

# ================= CHAT MANAGEMENT =================
def generate_smart_chat_title(first_question: str = None) -> str:
    """Generate a smart title for new chat"""
    if first_question and len(first_question) > 10:
        words = first_question.split()[:5]
        title = ' '.join(words)
        if len(first_question) > 30:
            title += "..."
        return title[:50]
    else:
        timestamp = datetime.now().strftime("%b %d, %I:%M %p")
        return f"Document Q&A - {timestamp}"

def create_new_chat(user_id: int, title: str = None, first_question: str = None) -> Optional[int]:
    """Create new chat"""
    if not st.session_state.auth_db:
        return None
    
    if not title:
        title = generate_smart_chat_title(first_question)
    
    cursor = st.session_state.auth_db.cursor()
    try:
        cursor.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s)",
            (user_id, title)
        )
        st.session_state.auth_db.commit()
        return cursor.lastrowid
    except Error as e:
        st.error(f"Error creating chat: {e}")
        return None
    finally:
        cursor.close()

def rename_chat(chat_id: int, user_id: int, new_title: str) -> Tuple[bool, str]:
    """Rename an existing chat"""
    if not st.session_state.auth_db:
        return False, "Auth database not connected"
    
    if not new_title or len(new_title.strip()) == 0:
        return False, "Title cannot be empty"
    
    if len(new_title) > 255:
        return False, "Title too long (max 255 characters)"
    
    cursor = st.session_state.auth_db.cursor()
    try:
        cursor.execute(
            "UPDATE chats SET title = %s WHERE chat_id = %s AND user_id = %s",
            (new_title.strip(), chat_id, user_id)
        )
        st.session_state.auth_db.commit()
        
        if cursor.rowcount > 0:
            return True, "Chat renamed successfully"
        else:
            return False, "Chat not found or unauthorized"
    except Error as e:
        return False, f"Error renaming chat: {e}"
    finally:
        cursor.close()

def get_user_chats(user_id: int) -> List[Dict]:
    if not st.session_state.auth_db:
        return []
    
    cursor = st.session_state.auth_db.cursor()
    try:
        cursor.execute(
            "SELECT chat_id, title, created_at FROM chats WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        return [
            {"chat_id": row[0], "title": row[1], "created_at": row[2]}
            for row in cursor.fetchall()
        ]
    except Error as e:
        st.error(f"Error fetching chats: {e}")
        return []
    finally:
        cursor.close()

def get_chat_history(chat_id: int, user_id: int) -> List[Dict]:
    if not st.session_state.auth_db:
        return []
    
    cursor = st.session_state.auth_db.cursor()
    try:
        cursor.execute("""
            SELECT question, query_generated, response, timestamp
            FROM chat_history
            WHERE chat_id = %s AND user_id = %s
            ORDER BY timestamp ASC
        """, (chat_id, user_id))
        
        return [
            {
                "question": row[0],
                "sources": json.loads(row[1]) if row[1] else [],
                "response": row[2],
                "timestamp": row[3]
            }
            for row in cursor.fetchall()
        ]
    except Error as e:
        st.error(f"Error fetching chat history: {e}")
        return []
    finally:
        cursor.close()

def save_chat_turn(chat_id: int, user_id: int, question: str,
                   sources: List[Dict], response: str):
    if not st.session_state.auth_db:
        return
    
    cursor = st.session_state.auth_db.cursor()
    try:
        cursor.execute("""
            INSERT INTO chat_history
            (chat_id, user_id, question, query_generated, response)
            VALUES (%s, %s, %s, %s, %s)
        """, (chat_id, user_id, question, json.dumps(sources), response))
        st.session_state.auth_db.commit()
    except Error as e:
        st.error(f"Error saving chat turn: {e}")
    finally:
        cursor.close()

def delete_chat(chat_id: int, user_id: int):
    if not st.session_state.auth_db:
        return
    
    cursor = st.session_state.auth_db.cursor()
    try:
        cursor.execute(
            "DELETE FROM chats WHERE chat_id = %s AND user_id = %s",
            (chat_id, user_id)
        )
        st.session_state.auth_db.commit()
    except Error as e:
        st.error(f"Error deleting chat: {e}")
    finally:
        cursor.close()

# ================= DOCUMENT PROCESSING =================
def clean_text(text: str) -> str:
    """Clean extracted text"""
    lines = text.split('\n')
    result = '\n'.join(lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict]:
    """Extract text from PDF using PyMuPDF"""
    if not PYMUPDF_AVAILABLE:
        return []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text().strip()
            text = clean_text(text)
            
            if text:
                pages.append({"page": i + 1, "text": text})
        
        doc.close()
        return pages
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def extract_text_from_docx(docx_bytes: bytes) -> List[Dict]:
    """Extract text from DOCX"""
    if not DOCX_AVAILABLE:
        return []
    
    try:
        doc = Document(io.BytesIO(docx_bytes))
        text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        text = clean_text(text)
        
        if text:
            return [{"page": 1, "text": text}]
        return []
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return []

def extract_text_from_txt(txt_bytes: bytes) -> List[Dict]:
    """Extract text from TXT"""
    try:
        text = txt_bytes.decode('utf-8')
        text = clean_text(text)
        
        if text:
            return [{"page": 1, "text": text}]
        return []
    except Exception as e:
        st.error(f"Error processing TXT: {e}")
        return []

def semantic_chunking(text: str, chunk_size: int = SEMANTIC_CHUNK_SIZE) -> List[str]:
    """Semantic chunking: splits text at sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_size = len(words)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_documents_for_rag(files_data: List[Dict]) -> Tuple[bool, str]:
    """Process documents and store in vector database"""
    if not CHROMADB_AVAILABLE or not SEMANTIC_AVAILABLE:
        return False, "RAG dependencies not available. Please install: pip install chromadb sentence-transformers"
    
    embedding_model, _, chroma_client = load_rag_models()
    if not embedding_model or not chroma_client:
        return False, "Failed to load RAG models"
    
    all_chunks = []
    
    for file_data in files_data:
        # Extract text based on file type
        ext = os.path.splitext(file_data['name'])[1].lower()
        
        if ext == '.pdf':
            pages = extract_text_from_pdf(file_data['bytes'])
        elif ext == '.docx':
            pages = extract_text_from_docx(file_data['bytes'])
        elif ext == '.txt':
            pages = extract_text_from_txt(file_data['bytes'])
        else:
            st.warning(f"Unsupported file type: {ext}")
            continue
        
        # Chunk the text
        for page in pages:
            chunks = semantic_chunking(page['text'])
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "page": page['page'],
                    "doc": file_data['name'],
                    "chunk_id": chunk_idx
                })
    
    if not all_chunks:
        return False, "No content extracted from documents"
    
    # Generate embeddings and store
    texts = [c["text"] for c in all_chunks]
    
    with st.spinner(f"Generating embeddings for {len(texts)} chunks..."):
        embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()
    
    try:
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection("documents")
        except:
            pass
        
        # Create new collection
        col = chroma_client.create_collection("documents")
        
        # Add documents
        col.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "page": c["page"], 
                    "doc": c["doc"],
                    "chunk_id": c["chunk_id"]
                }
                for c in all_chunks
            ],
            ids=[f"c{i}" for i in range(len(texts))]
        )
        
        return True, f"Successfully processed {len(all_chunks)} chunks from {len(files_data)} document(s)"
    except Exception as e:
        return False, f"Error storing documents: {str(e)}"

# ================= RAG QUERY =================
def rag_query(question: str, context: str = "") -> Tuple[str, List[Dict]]:
    """Query RAG system with enhanced answer generation"""
    if not CHROMADB_AVAILABLE or not SEMANTIC_AVAILABLE:
        return "❌ RAG system not available. Please install required packages.", []
    
    embedding_model, cross_encoder, chroma_client = load_rag_models()
    if not embedding_model or not chroma_client:
        return "❌ Failed to load RAG models", []
    
    try:
        col = chroma_client.get_collection("documents")
        
        # Retrieve documents
        q_emb = embedding_model.encode([question]).tolist()
        res = col.query(q_emb, n_results=INITIAL_RETRIEVAL_K)
        
        candidates = []
        for doc, metadata, distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            candidates.append({
                "text": doc,
                "page": metadata.get("page", 1),
                "doc": metadata.get("doc", "unknown"),
                "score": 1 - distance
            })
        
        if not candidates:
            return "ℹ️ No relevant information found in the uploaded documents. Please try rephrasing your question or upload more documents.", []
        
        # Rerank if cross-encoder available
        if cross_encoder:
            with st.spinner("Reranking results for better accuracy..."):
                pairs = [[question, c["text"]] for c in candidates]
                rerank_scores = cross_encoder.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    candidates[i]["score"] = float(score)
                candidates.sort(key=lambda x: x["score"], reverse=True)
        
        top_chunks = candidates[:FINAL_CONTEXT_K]
        
        # Enhanced prompting
        context_parts = []
        for i, c in enumerate(top_chunks, 1):
            score_info = f"[Relevance: {c.get('score', 0):.2f}]"
            source_info = f"**Source {i}** (Page {c['page']}, {c.get('doc', 'document')}) {score_info}:"
            context_parts.append(f"{source_info}\n{c['text']}\n")
        
        rag_context = "\n".join(context_parts)
        
        system_prompt = """You are an expert document assistant who provides clear, comprehensive, well-structured answers.

GUIDELINES:
- Answer ONLY using information from the provided document content
- Structure your answer with bullet points, paragraphs, and clear explanations
- Include specific details, dates, numbers, and examples from the documents
- Cite source numbers (e.g., "According to Source 1...") when referencing information
- If the answer is not in the documents, clearly state: "The answer is not found in the provided documents"
- Never make up or assume information not present in the documents"""
        
        user_prompt = f"""{context}

**DOCUMENT CONTENT:**
{rag_context}

**QUESTION:**
{question}

**YOUR ANSWER:**"""
        
        answer = call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=600)
        
        if not answer:
            answer = "I couldn't generate an answer. Please try again or rephrase your question."
        
        sources = [
            {
                "doc": c["doc"], 
                "page": c["page"], 
                "score": c["score"],
                "preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
            } 
            for c in top_chunks
        ]
        
        return answer, sources
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}", []

# ================= CONTEXT MANAGEMENT =================
def build_optimized_context(chat_history: List[Dict], current_question: str) -> str:
    """Build context from recent conversation"""
    if not chat_history:
        return ""
    
    context_parts = ["=== RECENT CONVERSATION ==="]
    recent_messages = chat_history[-MAX_RECENT_MESSAGES:]
    
    for msg in recent_messages:
        context_parts.append(f"User: {msg['question']}")
        context_parts.append(f"Assistant: {msg['response'][:200]}...")  # Truncate long responses
    
    return "\n".join(context_parts)

# ================= LLM CALL =================
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
    """Call LLM API"""
    if not API_KEY:
        return None
        
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=40)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        st.error(f"API failed: {e}")
        return None

# ================= UTILITY FUNCTIONS =================
def create_copy_button(text: str, button_text: str = "📋 Copy") -> str:
    """Create a copy to clipboard button"""
    escaped_text = text.replace('`', '\\`').replace('$', '\\$').replace('\\', '\\\\').replace('\n', '\\n')
    return f'<button class="copy-button" onclick="navigator.clipboard.writeText(`{escaped_text}`)">{button_text}</button>'

def display_source_card(source: Dict, index: int):
    """Display source information in a card format"""
    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">
            📄 Source {index}: {source['doc']}
        </div>
        <div>
            📍 Page: {source['page']}
            <span class="relevance-badge">
                Relevance: {source['score']:.0%}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander(f"View excerpt from Source {index}"):
        st.markdown(source.get('preview', 'No preview available'))

# ================= SESSION STATE =================
if "auth_db" not in st.session_state:
    st.session_state.auth_db = get_auth_db_connection()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "show_rename_dialog" not in st.session_state:
    st.session_state.show_rename_dialog = False

if "rename_chat_id" not in st.session_state:
    st.session_state.rename_chat_id = None

# ================= LOGIN UI =================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 📚 RAG Assistant Pro")
        st.markdown("### Ask Questions About Your Documents")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.subheader("Welcome Back!")
            login_username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            
            if st.button("🚀 Login", use_container_width=True, type="primary"):
                if login_username and login_password:
                    user_id = authenticate_user(login_username, login_password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = login_username
                        chat_id = create_new_chat(user_id, None, "Welcome!")
                        st.session_state.current_chat_id = chat_id
                        st.success(f"Welcome back, {login_username}! 👋")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
                else:
                    st.warning("⚠️ Please fill in all fields.")
        
        with tab2:
            st.subheader("Create New Account")
            signup_username = st.text_input("Choose Username", key="signup_user", placeholder="Create a unique username")
            signup_password = st.text_input("Choose Password", type="password", key="signup_pass", placeholder="Minimum 6 characters")
            signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Re-enter password")
            
            if st.button("✨ Create Account", use_container_width=True, type="primary"):
                if signup_username and signup_password:
                    if signup_password != signup_confirm:
                        st.error("❌ Passwords don't match!")
                    elif len(signup_password) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    else:
                        success, message = create_user(signup_username, signup_password)
                        if success:
                            st.success("✅ Account created successfully! Please login.")
                            st.balloons()
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please fill in all fields.")
        
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("- 📄 PDF, DOCX, TXT support")
        st.markdown("- 🧠 Semantic search with AI")
        st.markdown("- 🎯 Source attribution")
        st.markdown("- 💬 Context-aware conversations")
    
    st.stop()

# ================= MAIN APP =================
col1, col2, col3 = st.columns([2, 3, 1])
with col1:
    st.title("📚 RAG Assistant Pro")
with col2:
    st.markdown(f"### Welcome, **{st.session_state.username}**! 👋")
with col3:
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.current_chat_id = None
        st.session_state.rag_ready = False
        st.session_state.processed_files = []
        st.rerun()

st.divider()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Document Upload
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose documents to analyze",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        key="doc_uploader",
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_files and st.button("🔄 Process Documents", use_container_width=True, type="primary"):
        with st.spinner("Processing documents... This may take a moment."):
            files_data = [{'name': f.name, 'bytes': f.read()} for f in uploaded_files]
            success, message = process_documents_for_rag(files_data)
            
            if success:
                st.session_state.rag_ready = True
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("**📄 Processed Documents:**")
        for filename in st.session_state.processed_files:
            st.markdown(f"✓ {filename}")
    
    st.divider()
    
    # Chat History
    st.subheader("💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_chat_id = create_new_chat(st.session_state.user_id, None, None)
        if new_chat_id:
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
    
    st.markdown("---")
    
    user_chats = get_user_chats(st.session_state.user_id)
    
    if user_chats:
        for chat in user_chats:
            col1, col2, col3 = st.columns([6, 2, 2])
            
            with col1:
                display_title = chat['title']
                if len(display_title) > 25:
                    display_title = display_title[:22] + "..."
                
                is_active = chat['chat_id'] == st.session_state.current_chat_id
                if st.button(
                    f"{'📌 ' if is_active else '💬 '}{display_title}",
                    key=f"chat_{chat['chat_id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_chat_id = chat['chat_id']
                    st.rerun()
            
            with col2:
                if st.button("✏️", key=f"rename_{chat['chat_id']}", help="Rename"):
                    st.session_state.show_rename_dialog = True
                    st.session_state.rename_chat_id = chat['chat_id']
                    st.rerun()
            
            with col3:
                if st.button("🗑️", key=f"del_{chat['chat_id']}", help="Delete"):
                    if chat['chat_id'] == st.session_state.current_chat_id:
                        other_chats = [c for c in user_chats if c['chat_id'] != chat['chat_id']]
                        st.session_state.current_chat_id = other_chats[0]['chat_id'] if other_chats else None
                    delete_chat(chat['chat_id'], st.session_state.user_id)
                    st.rerun()
    else:
        st.info("No chats yet. Start one! 🚀")
    
    st.divider()
    st.caption("💡 **Tips:**")
    st.caption("• Upload documents first")
    st.caption("• Ask specific questions")
    st.caption("• Check source references")
    st.caption("• Refine follow-up questions")

# ================= RENAME DIALOG =================
if st.session_state.show_rename_dialog and st.session_state.rename_chat_id:
    @st.dialog("✏️ Rename Chat")
    def rename_dialog():
        user_chats = get_user_chats(st.session_state.user_id)
        current_chat = next((c for c in user_chats if c['chat_id'] == st.session_state.rename_chat_id), None)
        
        if current_chat:
            new_title = st.text_input(
                "New title:",
                value=current_chat['title'],
                max_chars=255,
                key="rename_input"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save", use_container_width=True, type="primary"):
                    if new_title and new_title.strip():
                        success, message = rename_chat(
                            st.session_state.rename_chat_id,
                            st.session_state.user_id,
                            new_title
                        )
                        
                        if success:
                            st.success(message)
                            st.session_state.show_rename_dialog = False
                            st.session_state.rename_chat_id = None
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Title cannot be empty")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_rename_dialog = False
                    st.session_state.rename_chat_id = None
                    st.rerun()
    
    rename_dialog()

# ================= MAIN CHAT INTERFACE =================
if not st.session_state.rag_ready:
    st.warning("⚠️ **Please upload and process documents first** using the sidebar before asking questions!")

st.info("📚 **Document Q&A Mode** - Ask questions about your uploaded documents!")

if not st.session_state.current_chat_id:
    # Welcome screen
    st.markdown("## 👋 Welcome to RAG Assistant Pro!")
    st.markdown("### Get started by:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1️⃣ Upload your documents")
        st.markdown("- 📄 PDF files")
        st.markdown("- 📝 DOCX files")
        st.markdown("- 📋 TXT files")
        st.markdown("- Click '🔄 Process Documents'")
    
    with col2:
        st.markdown("#### 2️⃣ Start asking questions")
        st.markdown("- Click '➕ New Chat' in sidebar")
        st.markdown("- Ask specific questions")
        st.markdown("- Get answers with sources")
        st.markdown("- Follow up for details")
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions:")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("**General:**")
        st.markdown("- 'Summarize the main points'")
        st.markdown("- 'What are the key findings?'")
        st.markdown("- 'Explain the methodology'")
    
    with example_col2:
        st.markdown("**Specific:**")
        st.markdown("- 'What does it say about X?'")
        st.markdown("- 'List all recommendations'")
        st.markdown("- 'Compare sections A and B'")

else:
    # Display chat history
    chat_history = get_chat_history(st.session_state.current_chat_id, st.session_state.user_id)
    
    for turn in chat_history:
        # User message
        with st.chat_message("user"):
            st.write(turn["question"])
            st.markdown(create_copy_button(turn["question"], "📋 Copy Question"), unsafe_allow_html=True)
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(turn["response"])
            
            # Display sources
            if turn.get("sources"):
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(turn["sources"], 1):
                        display_source_card(source, idx)
    
    # Chat input
    user_question = st.chat_input("💬 Ask a question about your documents...")
    
    if user_question:
        if not st.session_state.rag_ready:
            st.warning("⚠️ Please upload and process documents first!")
        else:
            # Update chat title if first message
            if len(chat_history) == 0:
                new_title = generate_smart_chat_title(user_question)
                rename_chat(st.session_state.current_chat_id, st.session_state.user_id, new_title)
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
                st.markdown(create_copy_button(user_question, "📋 Copy Question"), unsafe_allow_html=True)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents and generating answer..."):
                    context = build_optimized_context(chat_history, user_question)
                    answer, sources = rag_query(user_question, context)
                    
                    st.write(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources & References", expanded=True):
                            st.markdown("**Sources used to generate this answer:**")
                            for idx, source in enumerate(sources, 1):
                                display_source_card(source, idx)
                    
                    # Copy answer button
                    st.markdown(create_copy_button(answer, "📋 Copy Answer"), unsafe_allow_html=True)
                    
                    # Save to history
                    save_chat_turn(
                        st.session_state.current_chat_id,
                        st.session_state.user_id,
                        user_question,
                        sources,
                        answer
                    )

# ================= FOOTER =================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by LLaMA 3.1")
with col2:
    rag_status = "✅ Ready" if st.session_state.rag_ready else "⚠️ Upload Docs"
    st.caption(f"📚 RAG: {rag_status}")
with col3:
    st.caption(f"📄 Docs: {len(st.session_state.processed_files)}")
