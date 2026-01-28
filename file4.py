"""
SQL Assistant Pro - Enhanced Version with Groq (Meta Llama)
Features:
1. Context Retention: Recent 5 messages + summarized older messages + 3 semantically similar Q&As
2. LLM-Based Smart Query Analysis
3. Enhanced Product Display with Interactive Cards
4. Improved Authentication UI
5. Delete Confirmation Dialogs
6. FIXED: SQLite compatibility for custom databases
7. NEW: Persistent Custom SQLite Databases (file-based)
8. NEW: Custom MySQL Database Connection Support
9. NEW: User-specific previous SQLite DB listing on login
10. NEW: UI-based credential input for Custom MySQL
DATABASE MIGRATION REQUIRED:
If you're updating from a previous version, run this SQL command on your auth database:
ALTER TABLE chat_history ADD COLUMN result_data LONGTEXT AFTER response;
This adds persistent storage for query results so users can view historical data.
"""
import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import json
from datetime import datetime
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from groq import Groq
import os
import io
import sqlite3
import base64
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import shutil
import glob
# ================= CONFIGURATION =================
st.set_page_config(
    page_title="SQL Assistant Pro",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better UI
st.markdown("""
<style>
    /* Login/Signup Page Styling */
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
   
    .auth-header {
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
   
    .auth-form {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
   
    /* Delete Confirmation Dialog */
    .delete-warning {
        background: #fff5f5;
        border: 2px solid #fc8181;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
   
    /* Sidebar Chat Items */
    .chat-item {
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
   
    .chat-item:hover {
        background: #f7fafc;
    }

    /* Custom DB Form Styling */
    .custom-db-form {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }

    /* DB List Styling */
    .db-item {
        background: #e6f3ff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Create persistent DB directory if it doesn't exist
PERSISTENT_DB_DIR = "custom_dbs"
os.makedirs(PERSISTENT_DB_DIR, exist_ok=True)

# SSL Certificate Path
try:
    ssl_ca_path = st.secrets.get("ssl_ca_path", None)
except:
    ssl_ca_path = None

# Load embedding model for semantic search (cached)
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for semantic similarity"""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ================= SESSION STATE INITIALIZATION =================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'business_schema' not in st.session_state:
    st.session_state.business_schema = {}
if 'db_mode' not in st.session_state:
    st.session_state.db_mode = "system"  # system, custom_sqlite, custom_mysql
if 'active_custom_sqlite_path' not in st.session_state:
    st.session_state.active_custom_sqlite_path = None
if 'user_sqlite_dbs' not in st.session_state:
    st.session_state.user_sqlite_dbs = []  # List of user's DB paths
if 'custom_mysql_params' not in st.session_state:
    st.session_state.custom_mysql_params = {}  # Dict for MySQL creds
if 'custom_mysql_connection' not in st.session_state:
    st.session_state.custom_mysql_connection = None
if 'custom_schema' not in st.session_state:
    st.session_state.custom_schema = {}
if 'show_rename_dialog' not in st.session_state:
    st.session_state.show_rename_dialog = False
if 'rename_chat_id' not in st.session_state:
    st.session_state.rename_chat_id = None
if 'show_delete_dialog' not in st.session_state:
    st.session_state.show_delete_dialog = False
if 'delete_chat_id' not in st.session_state:
    st.session_state.delete_chat_id = None

# ================= UTILITY FUNCTIONS =================
def load_user_sqlite_dbs(user_id: int) -> List[str]:
    """Load list of user's persistent SQLite DB paths"""
    pattern = os.path.join(PERSISTENT_DB_DIR, f"{user_id}_*.db")
    db_files = glob.glob(pattern)
    # Sort by modification time, newest first
    db_files.sort(key=os.path.getmtime, reverse=True)
    return db_files

# ================= DATABASE CONNECTION FUNCTIONS =================
def get_auth_db_connection():
    """Connect to authentication database"""
    try:
        if "auth_database" in st.secrets:
            # Use provided SSL config, with option for ssl_disabled
            ssl_config = {
                'ssl_disabled': st.secrets["auth_database"].get("ssl_disabled", False),
                'ssl_verify_cert': not st.secrets["auth_database"].get("ssl_disabled", False),
                'ssl_ca': st.secrets["auth_database"].get("ssl_ca", ""),
                'ssl_verify_identity': not st.secrets["auth_database"].get("ssl_disabled", False),
            }
            connection = mysql.connector.connect(
                host=st.secrets["auth_database"]["host"],
                port=int(st.secrets["auth_database"]["port"]),
                database=st.secrets["auth_database"]["database"],
                user=st.secrets["auth_database"]["user"],
                password=st.secrets["auth_database"]["password"],
                connect_timeout=30,
                **ssl_config
            )
        else:
            connection = mysql.connector.connect(
                host='localhost',
                database='auth_db',
                user='root',
                password='password',
                connect_timeout=10,
                ssl_disabled=True  # Default to disabled for local
            )
       
        if connection.is_connected():
            init_auth_tables(connection)
        return connection
    except Error as e:
        st.error(f"❌ Auth Database connection failed: {e}")
        return None

def get_business_db_connection():
    """Connect to business database"""
    try:
        if "database" in st.secrets:
            # Use provided SSL config, with option for ssl_disabled
            ssl_config = {
                'ssl_disabled': st.secrets["database"].get("ssl_disabled", False),
                'ssl_verify_cert': not st.secrets["database"].get("ssl_disabled", False),
                'ssl_ca': st.secrets["database"].get("ssl_ca", ""),
                'ssl_verify_identity': not st.secrets["database"].get("ssl_disabled", False),
            }
            connection = mysql.connector.connect(
                host=st.secrets["database"]["host"],
                port=int(st.secrets["database"]["port"]),
                database=st.secrets["database"]["database"],
                user=st.secrets["database"]["user"],
                password=st.secrets["database"]["password"],
                connect_timeout=30,
                **ssl_config
            )
        else:
            connection = mysql.connector.connect(
                host='localhost',
                database='myntra_db',
                user='root',
                password='password',
                connect_timeout=10,
                ssl_disabled=True  # Default to disabled for local
            )
        return connection
    except Error as e:
        st.error(f"❌ Business Database connection failed: {e}")
        return None

def get_custom_mysql_connection_from_params(params: Dict) -> Optional[mysql.connector.connection.MySQLConnection]:
    """Connect to custom MySQL using provided params"""
    try:
        ssl_config = {
            'ssl_disabled': params.get("ssl_disabled", True),
            'ssl_verify_cert': not params.get("ssl_disabled", True),
            'ssl_ca': params.get("ssl_ca", ""),
            'ssl_verify_identity': not params.get("ssl_disabled", True),
        }
        connection = mysql.connector.connect(
            host=params["host"],
            port=int(params["port"]),
            database=params["database"],
            user=params["user"],
            password=params["password"],
            connect_timeout=30,
            **ssl_config
        )
        return connection
    except Error as e:
        st.error(f"❌ Custom MySQL connection failed: {e}")
        return None

def init_auth_tables(connection):
    """Initialize authentication tables if they don't exist"""
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
                mode VARCHAR(20) NOT NULL DEFAULT 'database',
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
                result_data LONGTEXT,
                mode VARCHAR(20),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            )
        """)
       
        connection.commit()
    except Error as e:
        st.error(f"Error initializing auth tables: {e}")
    finally:
        cursor.close()

# ================= AUTHENTICATION FUNCTIONS =================
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str) -> Tuple[bool, str]:
    """Create new user"""
    connection = get_auth_db_connection()
    if not connection:
        return False, "Database connection failed"
   
    try:
        cursor = connection.cursor()
        hashed_pw = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, hashed_pw)
        )
        connection.commit()
        return True, "User created successfully"
    except Error as e:
        if "Duplicate entry" in str(e):
            return False, "Username already exists"
        return False, f"Error: {e}"
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def verify_user(username: str, password: str) -> Tuple[bool, Optional[int]]:
    """Verify user credentials"""
    connection = get_auth_db_connection()
    if not connection:
        return False, None
   
    try:
        cursor = connection.cursor()
        hashed_pw = hash_password(password)
        cursor.execute(
            "SELECT user_id FROM users WHERE username = %s AND password_hash = %s",
            (username, hashed_pw)
        )
        result = cursor.fetchone()
       
        if result:
            return True, result[0]
        return False, None
    except Error as e:
        st.error(f"Login error: {e}")
        return False, None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# ================= DATABASE SCHEMA FUNCTIONS =================
def is_sqlite_connection(connection) -> bool:
    """Check if connection is SQLite"""
    return isinstance(connection, sqlite3.Connection)

def get_database_schema(connection, table_name: Optional[str] = None) -> Dict:
    """Get comprehensive database schema with relationships - works with both MySQL and SQLite"""
    schema = {}
    cursor = None
    is_sqlite = is_sqlite_connection(connection)
   
    try:
        cursor = connection.cursor()
       
        # Get all tables or specific table
        if table_name:
            tables = [table_name]
        else:
            if is_sqlite:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [table[0] for table in cursor.fetchall()]
            else:
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
       
        for table in tables:
            columns = []
           
            # Get columns - different syntax for SQLite vs MySQL
            if is_sqlite:
                # SQLite uses PRAGMA table_info
                cursor.execute(f"PRAGMA table_info({table})")
                for col in cursor.fetchall():
                    # SQLite PRAGMA returns: cid, name, type, notnull, dflt_value, pk
                    columns.append({
                        'name': col[1],
                        'type': col[2],
                        'null': 'NO' if col[3] else 'YES',
                        'key': 'PRI' if col[5] else '',
                        'default': col[4],
                        'extra': ''
                    })
            else:
                # MySQL uses DESCRIBE
                cursor.execute(f"DESCRIBE {table}")
                for col in cursor.fetchall():
                    columns.append({
                        'name': col[0],
                        'type': col[1],
                        'null': col[2],
                        'key': col[3],
                        'default': col[4],
                        'extra': col[5]
                    })
           
            # Get foreign key relationships
            relationships = []
            if is_sqlite:
                # SQLite uses PRAGMA foreign_key_list
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                for rel in cursor.fetchall():
                    # SQLite PRAGMA returns: id, seq, table, from, to, on_update, on_delete, match
                    relationships.append({
                        'column': rel[3],
                        'references_table': rel[2],
                        'references_column': rel[4]
                    })
            else:
                # MySQL uses INFORMATION_SCHEMA
                cursor.execute(f"""
                    SELECT
                        COLUMN_NAME,
                        REFERENCED_TABLE_NAME,
                        REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = '{table}'
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """)
                for rel in cursor.fetchall():
                    relationships.append({
                        'column': rel[0],
                        'references_table': rel[1],
                        'references_column': rel[2]
                    })
           
            # Get sample data (first 3 rows)
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_data = cursor.fetchall()
           
            schema[table] = {
                'columns': columns,
                'relationships': relationships,
                'sample_data': sample_data
            }
       
        return schema
    except Exception as e:
        st.error(f"Schema fetch error: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()

def format_schema_for_llm(schema: Dict, tables_to_include: Optional[List[str]] = None) -> str:
    """Format schema for LLM"""
    schema_text = "DATABASE SCHEMA:\n\n"
   
    # Filter tables if specified
    if tables_to_include:
        filtered_schema = {k: v for k, v in schema.items() if k in tables_to_include}
    else:
        filtered_schema = schema
   
    for table_name, table_info in filtered_schema.items():
        schema_text += f"TABLE: {table_name}\n"
        schema_text += "Columns:\n"
        for col in table_info['columns']:
            key_info = f" [{col['key']}]" if col['key'] else ""
            null_info = " (nullable)" if col['null'] == 'YES' else " (required)"
            schema_text += f" - {col['name']}: {col['type']}{key_info}{null_info}\n"
       
        if table_info.get('relationships'):
            schema_text += "\nRelationships:\n"
            for rel in table_info['relationships']:
                schema_text += f" - {rel['column']} → {rel['references_table']}.{rel['references_column']}\n"
       
        if table_info.get('sample_data'):
            schema_text += f"\nSample Data ({len(table_info['sample_data'])} rows):\n"
            col_names = [col['name'] for col in table_info['columns']]
            for row in table_info['sample_data'][:3]:
                row_dict = dict(zip(col_names, row))
                schema_text += f" {row_dict}\n"
       
        schema_text += "\n" + "="*80 + "\n\n"
   
    # Add relationship summary for multi-table queries
    if len(filtered_schema) > 1:
        schema_text += "RELATIONSHIP SUMMARY:\n"
        for table_name, table_info in filtered_schema.items():
            if table_info.get('relationships'):
                for rel in table_info['relationships']:
                    schema_text += f" {table_name}.{rel['column']} → {rel['references_table']}.{rel['references_column']}\n"
        schema_text += "\n"
   
    return schema_text

# ================= CONTEXT MANAGEMENT FUNCTIONS =================
def compute_embedding(text: str) -> np.ndarray:
    """Compute embedding for text using sentence transformer"""
    return embedding_model.encode(text)

def find_semantically_similar_messages(
    current_question: str,
    chat_history: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """Find top-k semantically similar Q&A pairs from chat history"""
    if not chat_history:
        return []
   
    # Compute embedding for current question
    current_embedding = compute_embedding(current_question)
   
    # Compute embeddings for all historical questions
    similarities = []
    for turn in chat_history:
        question_embedding = compute_embedding(turn['question'])
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1),
            question_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((similarity, turn))
   
    # Sort by similarity and get top-k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [turn for _, turn in similarities[:top_k]]

def summarize_old_messages(messages: List[Dict]) -> str:
    """Summarize older messages using Groq Llama"""
    if not messages:
        return ""
   
    # Prepare summary request
    summary_text = "Previous conversation summary:\n"
    for msg in messages:
        response_preview = msg.get('response', '')[:200] if msg.get('response') else ''
        summary_text += f"Q: {msg['question']}\nA: {response_preview}...\n\n"
   
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation history concisely, focusing on key context and user preferences:\n\n{summary_text}"
            }],
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Summarization failed: {e}")
        return "Previous conversation context available but not summarized."

def build_optimized_context(
    chat_history: List[Dict],
    current_question: str,
    recent_count: int = 5,
    semantic_count: int = 3
) -> Tuple[str, Dict]:
    """
    Build optimized context with:
    1. Recent 5 messages (as is)
    2. Summary of older messages
    3. 3 semantically similar Q&As
    """
    context_parts = []
    stats = {
        'total_messages': len(chat_history),
        'recent_count': 0,
        'summarized_count': 0,
        'semantic_count': 0
    }
   
    if not chat_history:
        return "", stats
   
    # 1. Recent messages (last 5)
    recent_messages = chat_history[-recent_count:] if len(chat_history) >= recent_count else chat_history
    stats['recent_count'] = len(recent_messages)
   
    if recent_messages:
        context_parts.append("RECENT CONVERSATION (Last 5 messages):")
        for turn in recent_messages:
            context_parts.append(f"User: {turn['question']}")
            if turn.get('response'):
                context_parts.append(f"Assistant: {turn['response']}")
            if turn.get('query_generated'):
                context_parts.append(f"SQL: {turn['query_generated']}")
        context_parts.append("")
   
    # 2. Summary of older messages
    older_messages = chat_history[:-recent_count] if len(chat_history) > recent_count else []
    stats['summarized_count'] = len(older_messages)
   
    if older_messages:
        summary = summarize_old_messages(older_messages)
        if summary:
            context_parts.append("EARLIER CONVERSATION SUMMARY:")
            context_parts.append(summary)
            context_parts.append("")
   
    # 3. Semantically similar messages (excluding recent ones)
    older_for_semantic = chat_history[:-recent_count] if len(chat_history) > recent_count else []
    similar_messages = find_semantically_similar_messages(
        current_question,
        older_for_semantic,
        top_k=semantic_count
    )
    stats['semantic_count'] = len(similar_messages)
   
    if similar_messages:
        context_parts.append("RELEVANT SIMILAR CONVERSATIONS:")
        for i, turn in enumerate(similar_messages, 1):
            context_parts.append(f"{i}. User: {turn['question']}")
            response_preview = turn.get('response', '')[:150] if turn.get('response') else ''
            context_parts.append(f" Assistant: {response_preview}...")
            if turn.get('query_generated'):
                context_parts.append(f" SQL: {turn['query_generated']}")
        context_parts.append("")
   
    context = "\n".join(context_parts)
    return context, stats

# ================= LLM-BASED QUERY INTENT ANALYSIS =================
def analyze_query_intent_with_llm(question: str, schema: Dict) -> Dict:
    """Use LLM to analyze query intent and determine table requirements"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
       
        # Prepare schema summary for LLM
        schema_summary = "Available Tables:\n"
        for table_name, table_info in schema.items():
            columns = [col['name'] for col in table_info['columns']]
            schema_summary += f"- {table_name}: {', '.join(columns)}\n"
            if table_info.get('relationships'):
                for rel in table_info['relationships']:
                    schema_summary += f" → {rel['column']} links to {rel['references_table']}.{rel['references_column']}\n"
       
        analysis_prompt = f"""Analyze this database query intent:
{schema_summary}
User Question: "{question}"
Determine:
1. Which tables are needed to answer this question?
2. Does it require a JOIN between tables, or can it be answered from a single table?
3. What is the query type (single_table, multi_table, aggregation, etc.)?
Return your analysis in this JSON format:
{{
    "requires_join": true/false,
    "tables_needed": ["table1", "table2"],
    "intent_type": "single_table" or "multi_table",
    "reasoning": "Brief explanation of your analysis"
}}
IMPORTANT: Prefer single-table queries when possible for better performance. Only use JOIN when data from multiple tables is absolutely necessary.
Return ONLY the JSON, no additional text."""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": analysis_prompt
            }],
            max_tokens=500,
            temperature=0.1
        )
       
        # Parse response
        result = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        result = result.replace('```json', '').replace('```', '').strip()
       
        analysis = json.loads(result)
       
        # Validate and return
        return {
            'requires_join': analysis.get('requires_join', False),
            'tables_needed': analysis.get('tables_needed', list(schema.keys())),
            'intent_type': analysis.get('intent_type', 'unknown'),
            'reasoning': analysis.get('reasoning', 'LLM analysis completed')
        }
       
    except Exception as e:
        st.warning(f"LLM intent analysis failed, using fallback: {e}")
        # Fallback: use all tables
        return {
            'requires_join': False,
            'tables_needed': list(schema.keys()),
            'intent_type': 'unknown',
            'reasoning': 'Fallback analysis - using all available tables'
        }

# ================= SMART QUERY GENERATION =================
def generate_sql_query(question: str, schema_text: str, context: str, intent_analysis: Optional[Dict] = None, is_sqlite: bool = False) -> Dict:
    """Generate SQL query using Groq Llama with smart multi-table logic"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
       
        # Determine SQL dialect
        sql_dialect = "SQLite" if is_sqlite else "MySQL"
       
        # Enhanced system prompt for smart querying
        system_prompt = f"""You are an expert SQL query generator with advanced optimization skills for {sql_dialect}.
CRITICAL MULTI-TABLE INTELLIGENCE:
1. ALWAYS analyze if the question can be answered from a SINGLE table
2. ONLY use JOINs when the question REQUIRES data from multiple tables
3. Prefer single-table queries whenever possible for performance
4. Strictly use columns from tables of database
CRITICAL CONTEXT CAPTURING INTELLIGENCE:
1. ALWAYS analyze conversation history and accumulated filters
2. For REFINEMENT queries: Combine ALL previous filters with new ones
3. For CONTEXT RESET: Ignore all previous filters and start fresh
4. For ANALYTICAL queries: Apply filters then aggregate
CONTEXT CONTINUITY EXAMPLES:
User: "I want kurti" → SELECT * FROM products WHERE category='kurti'
User: "pink" (REFINEMENT) → SELECT * FROM products WHERE category='kurti' AND color='pink'
User: "M size" (REFINEMENT) → SELECT * FROM products WHERE category='kurti' AND color='pink' AND size='M'
User: "show me shoes" (CONTEXT RESET) → SELECT * FROM products WHERE category='shoes'
DECISION FRAMEWORK:
- Question about product attributes (name, price, category, brand, color, etc.) → Use catalog table ONLY
- Question about sales metrics (quantity sold, revenue, best-sellers) → May need JOIN with sales table
- Question combining product info WITH sales data → Use JOIN
QUERY REQUIREMENTS:
- Use proper JOINs with clear ON conditions when needed
- Include all relevant columns in SELECT
- Use WHERE clauses for filtering
- Add ORDER BY for rankings, to avoid long response keep limit of 10-15
- Use DISTINCT to avoid duplicates when joining
- Always use table aliases for clarity in multi-table queries
- Return ONLY valid {sql_dialect} query without explanation, markdown, or code blocks"""
        # Build user prompt with context and schema
        user_prompt = f"""DATABASE SCHEMA:
{schema_text}
CONVERSATION CONTEXT:
{context}
CURRENT QUESTION: {question}
"""
       
        # Add intent analysis if available
        if intent_analysis:
            user_prompt += f"""
QUERY ANALYSIS (from LLM):
Intent Type: {intent_analysis['intent_type']}
Requires JOIN: {intent_analysis['requires_join']}
Tables Needed: {', '.join(intent_analysis['tables_needed'])}
Reasoning: {intent_analysis['reasoning']}
"""
       
        user_prompt += """
Generate the optimal SQL query following all the rules above.
IMPORTANT:
- If this is a refinement, include ALL accumulated filters in WHERE clause
- If analytical, use appropriate aggregate functions
- Return ONLY the SQL query, no explanations."""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
       
        query = response.choices[0].message.content.strip()
       
        # Clean up query - remove markdown code blocks if present
        query = query.replace('```sql', '').replace('```', '').strip()
       
        # Remove any explanatory text before or after the query
        lines = query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
       
        query = ' '.join(sql_lines)
       
        # Validate query
        if not query.upper().startswith('SELECT'):
            return {
                "success": False,
                "error": "Generated query is not a SELECT statement",
                "query": query
            }
       
        return {
            "success": True,
            "query": query,
            "intent": intent_analysis['intent_type'] if intent_analysis else 'unknown',
            "debug": {
                "full_schema": schema_text,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
        }
       
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": None
        }

def execute_query(connection, query: str) -> Dict:
    """Execute SQL query and return results - works with both MySQL and SQLite"""
    cursor = None
    is_sqlite = is_sqlite_connection(connection)
   
    try:
        cursor = connection.cursor()
        cursor.execute(query)
       
        # Get column names
        columns = [desc[0] for desc in cursor.description]
       
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
       
        return {
            "success": True,
            "data": df,
            "row_count": len(df)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None
        }
    finally:
        if cursor:
            cursor.close()

# ================= CHAT MANAGEMENT FUNCTIONS =================
def create_new_chat(user_id: int, title: Optional[str], first_question: Optional[str]) -> Optional[int]:
    """Create new chat session"""
    connection = get_auth_db_connection()
    if not connection:
        st.error("Failed to connect to database for chat creation")
        return None
   
    try:
        cursor = connection.cursor()
        chat_title = title or (first_question[:50] + "..." if first_question else "New Chat")
        cursor.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s)",
            (user_id, chat_title)
        )
        connection.commit()
        return cursor.lastrowid
    except Error as e:
        st.error(f"Chat creation error: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_user_chats(user_id: int) -> List[Dict]:
    """Get all chats for user"""
    connection = get_auth_db_connection()
    if not connection:
        return []
   
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT chat_id, title, created_at FROM chats WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        return cursor.fetchall()
    except Error as e:
        st.error(f"Chat fetch error: {e}")
        return []
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_chat_history(chat_id: int, user_id: int) -> List[Dict]:
    """Get chat history with verification"""
    connection = get_auth_db_connection()
    if not connection:
        return []
   
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT ch.question, ch.query_generated, ch.response, ch.result_data, ch.timestamp
            FROM chat_history ch
            JOIN chats c ON ch.chat_id = c.chat_id
            WHERE ch.chat_id = %s AND c.user_id = %s
            ORDER BY ch.timestamp ASC
        """, (chat_id, user_id))
       
        results = cursor.fetchall()
       
        # Parse result_data JSON back to DataFrame if it exists
        for result in results:
            if result.get('result_data'):
                try:
                    result['result_df'] = pd.read_json(result['result_data'])
                except:
                    result['result_df'] = None
            else:
                result['result_df'] = None
       
        return results
    except Error as e:
        st.error(f"History fetch error: {e}")
        return []
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def save_chat_turn(chat_id: int, user_id: int, question: str, query: Optional[str], response: str, result_df: Optional[pd.DataFrame] = None) -> bool:
    """Save chat turn with verification and result data"""
    connection = get_auth_db_connection()
    if not connection:
        return False
   
    try:
        cursor = connection.cursor()
       
        # Verify chat belongs to user
        cursor.execute("SELECT user_id FROM chats WHERE chat_id = %s", (chat_id,))
        result = cursor.fetchone()
       
        if not result or result[0] != user_id:
            return False
       
        # Convert DataFrame to JSON if it exists
        result_data = None
        if result_df is not None and not result_df.empty:
            result_data = result_df.to_json()
       
        cursor.execute(
            "INSERT INTO chat_history (chat_id, user_id, question, query_generated, response, result_data) VALUES (%s, %s, %s, %s, %s, %s)",
            (chat_id, user_id, question, query, response, result_data)
        )
        connection.commit()
        return True
    except Error as e:
        st.error(f"Save error: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def rename_chat(chat_id: int, user_id: int, new_title: str) -> Tuple[bool, str]:
    """Rename chat with verification"""
    connection = get_auth_db_connection()
    if not connection:
        return False, "Database connection failed"
   
    try:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE chats SET title = %s WHERE chat_id = %s AND user_id = %s",
            (new_title, chat_id, user_id)
        )
        connection.commit()
       
        if cursor.rowcount > 0:
            return True, "Chat renamed successfully"
        return False, "Chat not found or access denied"
    except Error as e:
        return False, f"Rename error: {e}"
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def delete_chat(chat_id: int, user_id: int) -> Tuple[bool, str]:
    """Delete chat with verification"""
    connection = get_auth_db_connection()
    if not connection:
        return False, "Database connection failed"
   
    try:
        cursor = connection.cursor()
       
        # Delete history first (handled by CASCADE, but being explicit)
        cursor.execute(
            "DELETE FROM chat_history WHERE chat_id = %s AND user_id = %s",
            (chat_id, user_id)
        )
       
        # Delete chat
        cursor.execute(
            "DELETE FROM chats WHERE chat_id = %s AND user_id = %s",
            (chat_id, user_id)
        )
        connection.commit()
       
        if cursor.rowcount > 0:
            return True, "Chat deleted successfully"
        return False, "Chat not found or access denied"
    except Error as e:
        return False, f"Delete error: {e}"
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def generate_smart_chat_title(question: str) -> str:
    """Generate smart chat title from first question"""
    words = question.split()
    if len(words) <= 5:
        return question
    return question[:50] + "..." if len(question) > 50 else question

# ================= RESPONSE GENERATION =================
def generate_db_response_with_presentation(
    question: str,
    query: str,
    result: Dict,
    context: str
) -> Tuple[str, Optional[pd.DataFrame], Optional[go.Figure]]:
    """Generate natural language response with visualization using Groq Llama"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
       
        df = result.get("data")
        if df is None or df.empty:
            return "No results found for your query.", None, None
       
        # Prepare data summary
        data_summary = f"Query returned {len(df)} rows with columns: {', '.join(df.columns.tolist())}\n\n"
        data_summary += f"Sample data:\n{df.head(10).to_string()}"
       
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Context: {context}
Question: {question}
SQL Query Executed: {query}
Results: {data_summary}
Provide a natural, conversational response summarizing these results. Be concise but informative. Highlight key findings."""
            }],
            max_tokens=800,
            temperature=0.7
        )
       
        summary = response.choices[0].message.content
       
        # Generate visualization if appropriate
        visualization = create_visualization_if_applicable(df, question)
       
        return summary, df, visualization
       
    except Exception as e:
        st.error(f"Response generation error: {e}")
        return f"Found {len(df)} results.", df, None

def create_visualization_if_applicable(df: pd.DataFrame, question: str) -> Optional[go.Figure]:
    """Create appropriate visualization based on data and question"""
    if df.empty or len(df) > 100:
        return None
   
    question_lower = question.lower()
   
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
   
    if not numeric_cols:
        return None
   
    # Bar chart for counts/aggregations
    if any(word in question_lower for word in ['top', 'best', 'most', 'count', 'total', 'revenue']):
        if categorical_cols and numeric_cols:
            fig = px.bar(
                df.head(15),
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=f"{categorical_cols[0].replace('_', ' ').title()} vs {numeric_cols[0].replace('_', ' ').title()}",
                color=numeric_cols[0],
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig
   
    # Pie chart for distribution
    if 'distribution' in question_lower or 'breakdown' in question_lower:
        if categorical_cols and numeric_cols:
            fig = px.pie(
                df.head(10),
                names=categorical_cols[0],
                values=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0].replace('_', ' ').title()}"
            )
            return fig
   
    return None

# ================= FILE UPLOAD FUNCTIONS =================
def create_persistent_sqlite_from_file(file_bytes: bytes, filename: str, user_id: int) -> Tuple[bool, Optional[str], str]:
    """Create persistent SQLite database from uploaded file on disk"""
    try:
        # Read file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:  # Excel
            df = pd.read_excel(io.BytesIO(file_bytes))
       
        # Generate unique file path for persistence (per user)
        safe_filename = filename.split('.')[0].replace(' ', '_').replace('-', '_').lower()
        db_filename = f"{user_id}_{safe_filename}.db"
        db_path = os.path.join(PERSISTENT_DB_DIR, db_filename)
       
        # Create SQLite connection to file (persistent)
        conn = sqlite3.connect(db_path)
       
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
       
        # Generate table name from filename
        table_name = safe_filename
       
        # Write to SQLite
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        conn.commit()
        conn.close()
       
        return True, table_name, f"Persistent database created at '{db_path}' with table '{table_name}' ({len(df)} rows)"
    except Exception as e:
        return False, None, f"File processing error: {str(e)}"

def create_temp_database_from_mysql_file(file_bytes: bytes, filename: str, mysql_conn) -> Tuple[bool, Optional[str], str]:
    """Load uploaded file into custom MySQL database as a new table"""
    try:
        # Read file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:  # Excel
            df = pd.read_excel(io.BytesIO(file_bytes))
       
        # Generate table name from filename
        safe_filename = filename.split('.')[0].replace(' ', '_').replace('-', '_').lower()
        table_name = safe_filename
       
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
       
        # Write to MySQL
        from sqlalchemy import create_engine
        engine = create_engine(f"mysql+mysqlconnector://{mysql_conn.user}:{mysql_conn.password}@{mysql_conn.host}:{mysql_conn.port}/{mysql_conn.database}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
       
        return True, table_name, f"Data loaded into custom MySQL table '{table_name}' ({len(df)} rows)"
    except Exception as e:
        return False, None, f"File processing error: {str(e)}"

# ================= UI HELPER FUNCTIONS =================
def create_copy_button(text: str, label: str = "Copy") -> str:
    """Create copy-to-clipboard button"""
    escaped_text = text.replace('`', '\\`').replace('$', '\\$').replace('"', '\\"')
    return f"""
    <button onclick="navigator.clipboard.writeText(`{escaped_text}`)" style="
        background: #667eea;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    ">{label}</button>
    """

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"""
    <a href="data:file/csv;base64,{b64}" download="{filename}" style="
        background: #48bb78;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
    ">📥 Download CSV</a>
    """

def display_product_dropdown(product: Dict, idx: int, turn_idx: int = 0):
    """Display product in expandable dropdown"""
    name = product.get('product_name') or product.get('name', 'Unknown Product')
    price = product.get('price') or product.get('selling_price', 0)
   
    # Get additional quick info for the header
    brand = product.get('brand', '')
    category = product.get('category', '')
   
    # Create header text
    header_text = f"🛍️ {name} - ₹{price:,.2f}"
    if brand:
        header_text += f" | {brand}"
    if category:
        header_text += f" | {category}"
   
    # Create unique key for expander
    expander_key = f"product_exp_{turn_idx}_{idx}"
   
    with st.expander(header_text, expanded=False):
        # Display product details in organized sections
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("### 💰 Price Information")
            st.markdown(f"**Price:** ₹{price:,.2f}")
            if 'mrp' in product and product['mrp']:
                st.markdown(f"**MRP:** ₹{product['mrp']:,.2f}")
            if 'discount' in product and product['discount']:
                st.markdown(f"**Discount:** {product['discount']}%")
           
            st.markdown("### 📦 Product Details")
            if brand:
                st.markdown(f"**Brand:** {brand}")
            if category:
                st.markdown(f"**Category:** {category}")
            if 'color' in product and product['color']:
                st.markdown(f"**Color:** {product['color']}")
            if 'size' in product and product['size']:
                st.markdown(f"**Size:** {product['size']}")
       
        with col2:
            st.markdown("### ℹ️ Additional Information")
            if 'material' in product and product['material']:
                st.markdown(f"**Material:** {product['material']}")
            if 'stock' in product and product['stock']:
                stock_status = "✅ In Stock" if product['stock'] > 0 else "❌ Out of Stock"
                st.markdown(f"**Stock:** {stock_status} ({product['stock']} units)")
            if 'rating' in product and product['rating']:
                stars = '⭐' * int(float(product['rating']))
                st.markdown(f"**Rating:** {stars} ({product['rating']})")
            if 'reviews' in product and product['reviews']:
                st.markdown(f"**Reviews:** {product['reviews']}")
       
        # Show all other attributes
        st.markdown("### 📋 All Attributes")
       
        # Collect all attributes not already displayed
        displayed_keys = ['product_name', 'name', 'price', 'selling_price', 'mrp', 'discount',
                         'brand', 'category', 'color', 'size', 'material', 'stock', 'rating', 'reviews']
       
        other_attrs = {k: v for k, v in product.items() if k not in displayed_keys and v is not None and v != ''}
       
        if other_attrs:
            for key, value in other_attrs.items():
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.caption("No additional attributes")

# ================= LOAD BUSINESS DATABASE SCHEMA =================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_business_schema():
    """Load and cache business database schema"""
    connection = get_business_db_connection()
    if connection:
        try:
            schema = get_database_schema(connection)
            return schema
        finally:
            if connection.is_connected():
                connection.close()
    return {}

# Initialize business schema on app load
if not st.session_state.business_schema:
    st.session_state.business_schema = load_business_schema()

# ================= MAIN APPLICATION =================
# ================= LOGIN/SIGNUP =================
if not st.session_state.logged_in:
    # Center the auth container
    col1, col2, col3 = st.columns([1, 2, 1])
   
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-header">', unsafe_allow_html=True)
        st.markdown("# 🗄️ SQL Assistant Pro")
        st.markdown("### Powered by Meta Llama 3.3 via Groq")
        st.markdown('</div>', unsafe_allow_html=True)
       
        tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
       
        with tab1:
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            with st.form("login_form"):
                st.markdown("### Welcome Back!")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
               
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    submit = st.form_submit_button("Login", use_container_width=True, type="primary")
               
                if submit:
                    if username and password:
                        success, user_id = verify_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user_id = user_id
                            st.session_state.username = username
                            # Load user's previous SQLite DBs
                            st.session_state.user_sqlite_dbs = load_user_sqlite_dbs(user_id)
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.warning("⚠️ Please fill all fields")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with tab2:
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            with st.form("signup_form"):
                st.markdown("### Create Account")
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
               
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    submit = st.form_submit_button("Sign Up", use_container_width=True, type="primary")
               
                if submit:
                    if new_username and new_password and confirm_password:
                        if new_password == confirm_password:
                            if len(new_password) >= 6:
                                success, message = create_user(new_username, new_password)
                                if success:
                                    st.success(f"✅ {message}")
                                    st.info("👉 Please login with your credentials")
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                st.error("❌ Password must be at least 6 characters")
                        else:
                            st.error("❌ Passwords do not match")
                    else:
                        st.warning("⚠️ Please fill all fields")
            st.markdown('</div>', unsafe_allow_html=True)
       
        st.markdown('</div>', unsafe_allow_html=True)
       
        # Feature highlights
        st.markdown("---")
        st.markdown("### ✨ Features")
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.markdown("- 🧠 Context Retention")
            st.markdown("- 🔍 Semantic Search")
            st.markdown("- ⚡ Smart Queries")
        with col_feat2:
            st.markdown("- 📊 Auto Visualization")
            st.markdown("- 🎯 LLM Intent Analysis")
            st.markdown("- 💬 Multi-Chat Support")
   
    st.stop()

# ================= MAIN APP =================
# Header
col1, col2, col3 = st.columns([2, 3, 1])
with col1:
    st.title("🗄️ SQL Assistant Pro")
with col2:
    st.markdown(f"### Welcome, **{st.session_state.username}**! 👋")
with col3:
    if st.button("🚪 Logout", type="secondary"):
        # Close MySQL connection if open
        if st.session_state.custom_mysql_connection:
            try:
                if st.session_state.custom_mysql_connection.is_connected():
                    st.session_state.custom_mysql_connection.close()
            except:
                pass
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.current_chat_id = None
        st.session_state.db_mode = "system"
        st.session_state.active_custom_sqlite_path = None
        st.session_state.user_sqlite_dbs = []
        st.session_state.custom_mysql_params = {}
        st.session_state.custom_mysql_connection = None
        st.session_state.custom_schema = {}
        st.rerun()
st.divider()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Control Panel")
   
    # Database Selection
    st.subheader("🗄️ Database Source")
    db_modes = ["System DB (MySQL)", "Custom Persistent SQLite", "Custom MySQL Host"]
    selected_mode = st.radio(
        "Select Database Mode",
        db_modes,
        index=0 if st.session_state.db_mode == "system" else 1 if st.session_state.db_mode == "custom_sqlite" else 2,
        format_func=lambda x: x
    )
   
    # Map to internal mode
    if selected_mode == "System DB (MySQL)":
        st.session_state.db_mode = "system"
    elif selected_mode == "Custom Persistent SQLite":
        st.session_state.db_mode = "custom_sqlite"
    elif selected_mode == "Custom MySQL Host":
        st.session_state.db_mode = "custom_mysql"
   
    st.markdown("---")
   
    if st.session_state.db_mode == "system":
        st.success("✅ Using pre-configured System MySQL Database")
    elif st.session_state.db_mode == "custom_sqlite":
        st.markdown('<div class="custom-db-form">', unsafe_allow_html=True)
        st.info("💡 Your Previous SQLite Databases:")
        if st.session_state.user_sqlite_dbs:
            st.markdown("### 📂 Previous Databases")
            selected_db = st.selectbox(
                "Select Existing DB",
                options=[os.path.basename(path) for path in st.session_state.user_sqlite_dbs],
                format_func=lambda x: x,
                index=0,
                key="select_existing_db"
            )
            selected_path = os.path.join(PERSISTENT_DB_DIR, selected_db)
            if st.button("🔄 Load Selected DB", use_container_width=True):
                st.session_state.active_custom_sqlite_path = selected_path
                conn = sqlite3.connect(selected_path)
                schema = get_database_schema(conn)
                st.session_state.custom_schema = schema
                conn.close()
                st.success(f"✅ Loaded {selected_db}")
                st.rerun()
        else:
            st.info("No previous databases. Upload a new one.")
        
        st.markdown("### 📤 Upload New Database")
        uploaded_db_file = st.file_uploader(
            "Upload CSV/Excel for New Persistent SQLite",
            type=['csv', 'xlsx', 'xls'],
            key="sqlite_uploader",
            help="Upload to create persistent SQLite file"
        )
       
        if uploaded_db_file and st.button("📤 Create New Persistent SQLite DB", use_container_width=True):
            with st.spinner("Processing and saving to disk..."):
                file_bytes = uploaded_db_file.read()
                success, table_name, message = create_persistent_sqlite_from_file(
                    file_bytes,
                    uploaded_db_file.name,
                    st.session_state.user_id
                )
               
                if success:
                    st.success(f"✅ {message}")
                    # Reload user's DB list
                    st.session_state.user_sqlite_dbs = load_user_sqlite_dbs(st.session_state.user_id)
                    st.session_state.active_custom_sqlite_path = os.path.join(PERSISTENT_DB_DIR, f"{st.session_state.user_id}_{uploaded_db_file.name.split('.')[0].replace(' ', '_').replace('-', '_').lower()}.db")
                    conn = sqlite3.connect(st.session_state.active_custom_sqlite_path)
                    schema = get_database_schema(conn)
                    st.session_state.custom_schema = schema
                    conn.close()
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.db_mode == "custom_mysql":
        st.markdown('<div class="custom-db-form">', unsafe_allow_html=True)
        st.info("🔌 Connect to Custom MySQL Host")
        
        # MySQL Credential Input Form
        with st.form("mysql_creds_form"):
            st.markdown("### Enter Connection Details")
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value=st.session_state.custom_mysql_params.get("host", ""), placeholder="e.g., localhost")
                port = st.number_input("Port", value=st.session_state.custom_mysql_params.get("port", 3306), min_value=1, max_value=65535)
                database = st.text_input("Database", value=st.session_state.custom_mysql_params.get("database", ""), placeholder="e.g., mydb")
            with col2:
                user = st.text_input("User", value=st.session_state.custom_mysql_params.get("user", ""), placeholder="e.g., root")
                password = st.text_input("Password", value=st.session_state.custom_mysql_params.get("password", ""), type="password", placeholder="Enter password")
                ssl_disabled = st.checkbox("Disable SSL", value=st.session_state.custom_mysql_params.get("ssl_disabled", True))
                ssl_ca = st.text_input("SSL CA Path (if enabled)", value=st.session_state.custom_mysql_params.get("ssl_ca", ""), placeholder="Path to CA cert", disabled=ssl_disabled)
            
            connect_btn = st.form_submit_button("🔄 Connect to MySQL", use_container_width=True)
            
            if connect_btn:
                if host and port and database and user and password:
                    params = {
                        "host": host,
                        "port": port,
                        "database": database,
                        "user": user,
                        "password": password,
                        "ssl_disabled": ssl_disabled,
                        "ssl_ca": ssl_ca if not ssl_disabled else ""
                    }
                    st.session_state.custom_mysql_params = params
                    with st.spinner("Connecting..."):
                        conn = get_custom_mysql_connection_from_params(params)
                        if conn:
                            st.session_state.custom_mysql_connection = conn
                            schema = get_database_schema(conn)
                            st.session_state.custom_schema = schema
                            st.success("✅ Connected to Custom MySQL!")
                            st.rerun()
                        else:
                            st.error("❌ Connection failed. Check details.")
                else:
                    st.warning("⚠️ Please fill all required fields")
        
        # Optional: Upload file to load into custom MySQL
        if st.session_state.custom_mysql_connection:
            st.info("💡 Optionally load CSV/Excel into your Custom MySQL DB")
            uploaded_file = st.file_uploader(
                "Upload File to Custom MySQL",
                type=['csv', 'xlsx', 'xls'],
                key="mysql_uploader"
            )
           
            if uploaded_file and st.button("📤 Load into Custom MySQL", use_container_width=True):
                with st.spinner("Loading into MySQL..."):
                    file_bytes = uploaded_file.read()
                    success, table_name, message = create_temp_database_from_mysql_file(
                        file_bytes,
                        uploaded_file.name,
                        st.session_state.custom_mysql_connection
                    )
                   
                    if success:
                        st.success(f"✅ {message}")
                        # Reload schema
                        schema = get_database_schema(st.session_state.custom_mysql_connection)
                        st.session_state.custom_schema = schema
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        else:
            st.warning("⚠️ Enter credentials and connect first to use uploads")
        st.markdown('</div>', unsafe_allow_html=True)
   
    st.divider()
   
    # Database Schema Viewer
    st.subheader("📊 Database Schema")
   
    # Show schema based on mode
    schema_to_show = {}
    db_name = ""
    if st.session_state.db_mode == "system":
        schema_to_show = st.session_state.business_schema
        try:
            db_name = st.secrets["database"]["database"]
        except:
            db_name = "System Database"
    elif st.session_state.db_mode == "custom_sqlite" and st.session_state.active_custom_sqlite_path:
        conn = sqlite3.connect(st.session_state.active_custom_sqlite_path)
        schema_to_show = get_database_schema(conn)
        conn.close()
        db_name = f"Active SQLite: {os.path.basename(st.session_state.active_custom_sqlite_path)}"
    elif st.session_state.db_mode == "custom_mysql" and st.session_state.custom_mysql_connection:
        schema_to_show = st.session_state.custom_schema
        db_name = f"Custom MySQL: {st.session_state.custom_mysql_params.get('database', 'Connected')}"
   
    if schema_to_show:
        st.info(f"🗄️ **Database:** {db_name}")
        st.markdown("---")
       
        # Display each table with expandable columns
        for table_name, table_info in schema_to_show.items():
            with st.expander(f"📁 **{table_name}**", expanded=False):
                st.caption(f"**Columns ({len(table_info['columns'])}):**")
                for col in table_info['columns']:
                    key_icon = ""
                    if col.get('key') == 'PRI':
                        key_icon = "🔑 "
                    elif col.get('key') == 'MUL':
                        key_icon = "🔗 "
                   
                    col_type = col['type']
                    if '(' in col_type:
                        col_type = col_type.split('(')[0]
                   
                    st.markdown(f"{key_icon}**{col['name']}** `{col_type}`")
               
                if table_info.get('relationships'):
                    st.caption("**🔗 Relationships:**")
                    for rel in table_info['relationships']:
                        st.markdown(f"→ {rel['column']} ➜ {rel['references_table']}.{rel['references_column']}")
               
                if table_info.get('sample_data'):
                    st.caption(f"📝 *Sample data available*")
    else:
        st.warning("No schema available. Configure a database first.")
   
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
                    st.session_state.show_delete_dialog = True
                    st.session_state.delete_chat_id = chat['chat_id']
                    st.rerun()
    else:
        st.info("No chats yet. Start one! 🚀")
   
    st.divider()
    st.caption("💡 **Powered by Meta Llama 3.3:**")
    st.caption("• 🧠 Context Retention (5 recent + summary)")
    st.caption("• 🔍 Semantic Similar Questions")
    st.caption("• 🎯 LLM-Based Intent Analysis")
    st.caption("• ⚡ Smart Multi-Table Queries")
    st.caption("• 🗄️ MySQL & Persistent SQLite Support")
   
    # Show available tables at the bottom
    if schema_to_show:
        st.divider()
        st.caption("📋 **Available Tables:**")
        for table_name in schema_to_show.keys():
            st.caption(f"• {table_name}")

# ================= DELETE CONFIRMATION DIALOG =================
if st.session_state.show_delete_dialog and st.session_state.delete_chat_id:
    @st.dialog("⚠️ Confirm Delete")
    def delete_dialog():
        user_chats = get_user_chats(st.session_state.user_id)
        chat_to_delete = next((c for c in user_chats if c['chat_id'] == st.session_state.delete_chat_id), None)
       
        if chat_to_delete:
            st.markdown('<div class="delete-warning">', unsafe_allow_html=True)
            st.warning("⚠️ **Warning: This action cannot be undone!**")
            st.markdown('</div>', unsafe_allow_html=True)
           
            st.markdown(f"### Are you sure you want to delete this chat?")
            st.info(f"**Chat:** {chat_to_delete['title']}")
           
            col1, col2 = st.columns(2)
           
            with col1:
                if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
                    # If deleting current chat, switch to another
                    if st.session_state.delete_chat_id == st.session_state.current_chat_id:
                        other_chats = [c for c in user_chats if c['chat_id'] != st.session_state.delete_chat_id]
                        st.session_state.current_chat_id = other_chats[0]['chat_id'] if other_chats else None
                   
                    success, message = delete_chat(st.session_state.delete_chat_id, st.session_state.user_id)
                   
                    if success:
                        st.success(message)
                        st.session_state.show_delete_dialog = False
                        st.session_state.delete_chat_id = None
                        st.rerun()
                    else:
                        st.error(message)
           
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_delete_dialog = False
                    st.session_state.delete_chat_id = None
                    st.rerun()
   
    delete_dialog()

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
st.info("🤖 **Powered by Meta Llama 3.3 70B** - Lightning-fast context-aware SQL generation with LLM-based intent analysis!")
if not st.session_state.current_chat_id:
    # Welcome screen
    st.markdown("## 👋 Welcome to SQL Assistant Pro!")
    st.markdown("### Enhanced with Meta Llama 3.3 via Groq API")
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("#### 🧠 Context Features")
        st.markdown("- ✅ **Retains Last 5 Messages** - Recent context")
        st.markdown("- 📝 **Summarizes Older Chats** - Long-term memory")
        st.markdown("- 🔍 **Semantic Search** - Finds similar Q&As")
        st.markdown("- 💡 **Smart Context Window** - Optimized tokens")
   
    with col2:
        st.markdown("#### ⚡ Query Intelligence")
        st.markdown("- 🎯 **LLM Intent Analysis** - Smart table detection")
        st.markdown("- 🔗 **JOIN When Needed** - Performance first")
        st.markdown("- 📊 **Multi-Table Analysis** - Complex queries")
        st.markdown("- 🚀 **Single-Table Preference** - Speed optimized")
   
    st.markdown("---")
    st.markdown("### 💡 Example Questions:")
   
    example_col1, example_col2 = st.columns(2)
   
    with example_col1:
        st.markdown("**Single-Table Queries:**")
        st.markdown("- 'Show me red sneakers for women'")
        st.markdown("- 'Find all Nike products'")
        st.markdown("- 'What shoes cost under ₹2000?'")
        st.markdown("- 'List athletic footwear'")
   
    with example_col2:
        st.markdown("**Multi-Table Queries:**")
        st.markdown("- 'What are our best-selling products?'")
        st.markdown("- 'Total revenue by product category'")
        st.markdown("- 'Which customers bought Nike shoes?'")
        st.markdown("- 'Sales performance analysis'")
else:
    # Display chat history
    chat_history = get_chat_history(st.session_state.current_chat_id, st.session_state.user_id)
   
    for turn_idx, turn in enumerate(chat_history):
        # User message
        with st.chat_message("user"):
            st.write(turn["question"])
            st.markdown(create_copy_button(turn["question"], "📋 Copy Question"), unsafe_allow_html=True)
       
        # Assistant message
        with st.chat_message("assistant"):
            if turn.get("response"):
                st.write(turn["response"])
           
            # Display saved results if available
            if turn.get("result_df") is not None:
                df = turn["result_df"]
               
                if not df.empty:
                    # Check if product data
                    is_product_data = any(col.lower() in ['product_name', 'name', 'brand', 'price', 'category']
                                         for col in df.columns)
                   
                    if is_product_data and len(df) <= 50:
                        # Show as product dropdowns
                        st.markdown(f"### 🛍️ Products Found ({len(df)} items)")
                        st.caption("*Click on any product to expand and view details*")
                       
                        for idx, row in df.iterrows():
                            display_product_dropdown(row.to_dict(), idx, turn_idx)
                    else:
                        # Show as table with download option
                        with st.expander(f"📊 View All Results ({len(df)} items)", expanded=False):
                            st.dataframe(df, use_container_width=True, height=400)
                            st.markdown(
                                create_download_link(
                                    df,
                                    f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                ),
                                unsafe_allow_html=True
                            )
           
            if turn.get("query_generated"):
                with st.expander("🔍 View SQL Query & Details"):
                    st.code(turn["query_generated"], language="sql")
                    st.markdown(create_copy_button(turn["query_generated"], "📋 Copy Query"), unsafe_allow_html=True)
                   
                    # Show query type
                    query_lower = turn["query_generated"].lower()
                    if "join" in query_lower:
                        st.info("🔗 Multi-table query (JOIN used)")
                    else:
                        st.success("⚡ Single-table query (Optimized)")
   
    # Chat input
    user_question = st.chat_input("💬 Ask about your data...")
   
    if user_question:
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
            with st.spinner("🤔 Analyzing with Llama 3.3..."):
                # Build optimized context
                context, context_stats = build_optimized_context(chat_history, user_question)
               
                # Determine active connection and schema based on mode
                active_conn = None
                active_schema = {}
                is_sqlite = False
               
                if st.session_state.db_mode == "system":
                    active_conn = get_business_db_connection()
                    active_schema = st.session_state.business_schema
                    is_sqlite = False
                elif st.session_state.db_mode == "custom_sqlite" and st.session_state.active_custom_sqlite_path:
                    active_conn = sqlite3.connect(st.session_state.active_custom_sqlite_path)
                    active_schema = st.session_state.custom_schema
                    is_sqlite = True
                elif st.session_state.db_mode == "custom_mysql" and st.session_state.custom_mysql_connection:
                    active_conn = st.session_state.custom_mysql_connection
                    active_schema = st.session_state.custom_schema
                    is_sqlite = False
               
                if not active_conn:
                    response = "⚠️ No database connected. Please select and configure a database mode in the sidebar."
                    st.error(response)
                    save_chat_turn(
                        st.session_state.current_chat_id,
                        st.session_state.user_id,
                        user_question,
                        None,
                        response,
                        None
                    )
                else:
                    try:
                        # Use LLM to analyze query intent
                        with st.spinner("🧠 Analyzing query intent with LLM..."):
                            intent_analysis = analyze_query_intent_with_llm(
                                user_question,
                                active_schema
                            )
                       
                        # Show intent analysis
                        st.info(f"🎯 Intent: {intent_analysis['intent_type']} | Tables: {', '.join(intent_analysis['tables_needed'])}")
                       
                        # Format schema with smart multi-table context
                        schema_text = format_schema_for_llm(
                            active_schema,
                            tables_to_include=intent_analysis['tables_needed']
                        )
                       
                        # Generate query with intent analysis and database type
                        query_result = generate_sql_query(
                            user_question,
                            schema_text,
                            context,
                            intent_analysis,
                            is_sqlite=is_sqlite
                        )
                       
                        if not query_result["success"]:
                            response = f"❌ Could not generate query: {query_result.get('error', 'Unknown')}"
                            st.error(response)
                           
                            if query_result.get("query"):
                                st.code(query_result["query"], language="sql")
                           
                            save_chat_turn(
                                st.session_state.current_chat_id,
                                st.session_state.user_id,
                                user_question,
                                query_result.get("query"),
                                response,
                                None
                            )
                        else:
                            query = query_result["query"]
                            result = execute_query(active_conn, query)
                           
                            if not result["success"]:
                                response = f"❌ Query failed: {result.get('error', 'Unknown')}"
                                st.error(response)
                               
                                # Show available tables on error
                                st.warning("**Available tables in database:**")
                                if active_schema:
                                    for table_name in active_schema.keys():
                                        st.write(f"• {table_name}")
                               
                                with st.expander("🔍 View Failed Query"):
                                    st.code(query, language="sql")
                                    st.markdown(create_copy_button(query, "📋 Copy Query"), unsafe_allow_html=True)
                            else:
                                # Generate response with visualization
                                summary, df, visualization = generate_db_response_with_presentation(
                                    user_question,
                                    query,
                                    result,
                                    context
                                )
                               
                                # Display summary
                                st.write(summary)
                               
                                # Display visualization if available
                                if visualization:
                                    st.plotly_chart(visualization, use_container_width=True)
                               
                                # Display results
                                if df is not None and not df.empty:
                                    # Check if product data
                                    is_product_data = any(col.lower() in ['product_name', 'name', 'brand', 'price', 'category']
                                                         for col in df.columns)
                                   
                                    if is_product_data and len(df) <= 50:
                                        # Show as product dropdowns
                                        st.markdown(f"### 🛍️ Products Found ({len(df)} items)")
                                        st.caption("*Click on any product to expand and view details*")
                                       
                                        # Use current turn count as turn_idx
                                        current_turn_idx = len(chat_history)
                                        for idx, row in df.iterrows():
                                            display_product_dropdown(row.to_dict(), idx, current_turn_idx)
                                    else:
                                        # Show as table with download option
                                        with st.expander(f"📊 View All Results ({len(df)} items)", expanded=True):
                                            st.dataframe(df, use_container_width=True, height=400)
                                            st.markdown(
                                                create_download_link(
                                                    df,
                                                    f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                                ),
                                                unsafe_allow_html=True
                                            )
                               
                                # Query details expander
                                with st.expander("🔍 View Query & Optimization Details"):
                                    # Query type indicator
                                    query_lower = query.lower()
                                    if "join" in query_lower:
                                        st.warning("🔗 **Multi-Table Query** - JOIN was necessary for this question")
                                    else:
                                        st.success("⚡ **Single-Table Query** - Optimized for speed!")
                                   
                                    st.subheader("📝 Generated SQL Query")
                                    db_type = "SQLite" if is_sqlite else "MySQL"
                                    st.caption(f"Database Type: {db_type}")
                                    st.code(query, language="sql")
                                    st.markdown(create_copy_button(query, "📋 Copy Query"), unsafe_allow_html=True)
                                   
                                    st.subheader("🎯 Query Intent Analysis (LLM-Based)")
                                    col1, col2 = st.columns(2)
                                   
                                    with col1:
                                        st.metric("Query Type", intent_analysis['intent_type'])
                                        st.metric("Requires JOIN", "Yes" if intent_analysis['requires_join'] else "No")
                                   
                                    with col2:
                                        st.metric("Tables Used", len(intent_analysis['tables_needed']))
                                        st.write("**Reasoning:**", intent_analysis['reasoning'])
                                   
                                    st.subheader("📊 Context Optimization")
                                    col1, col2, col3, col4 = st.columns(4)
                                   
                                    with col1:
                                        st.metric("Total Messages", context_stats['total_messages'])
                                    with col2:
                                        st.metric("Recent", context_stats['recent_count'])
                                    with col3:
                                        st.metric("Summarized", context_stats['summarized_count'])
                                    with col4:
                                        st.metric("Semantic", context_stats['semantic_count'])
                                   
                                    # Debug section
                                    if query_result.get("debug"):
                                        with st.expander("🐛 DEBUG: Schema & Prompts Sent to LLM", expanded=False):
                                            debug_info = query_result["debug"]
                                           
                                            st.subheader("📊 Full Schema Sent to LLM")
                                            st.text_area(
                                                "Schema",
                                                debug_info.get("full_schema", "Not available"),
                                                height=300,
                                                key=f"schema_{datetime.now().timestamp()}"
                                            )
                                           
                                            st.subheader("🤖 System Prompt Sent to LLM")
                                            st.text_area(
                                                "System Prompt",
                                                debug_info.get("system_prompt", "Not available"),
                                                height=400,
                                                key=f"sys_prompt_{datetime.now().timestamp()}"
                                            )
                                           
                                            st.subheader("👤 User Prompt Sent to LLM")
                                            st.text_area(
                                                "User Prompt",
                                                debug_info.get("user_prompt", "Not available"),
                                                height=200,
                                                key=f"user_prompt_{datetime.now().timestamp()}"
                                            )
                               
                                response = summary
                           
                            # Save with result DataFrame
                            save_chat_turn(
                                st.session_state.current_chat_id,
                                st.session_state.user_id,
                                user_question,
                                query,
                                response,
                                df if df is not None and not df.empty else None
                            )
                   
                    except Exception as e:
                        response = f"❌ Unexpected error: {str(e)}"
                        st.error(response)
                        st.exception(e)
                        save_chat_turn(
                            st.session_state.current_chat_id,
                            st.session_state.user_id,
                            user_question,
                            None,
                            response,
                            None
                        )
                   
                    finally:
                        # Close connection if it's not stored in session (SQLite closes automatically on context exit, but explicit for MySQL)
                        if st.session_state.db_mode != "custom_mysql" and active_conn:
                            try:
                                if not is_sqlite and active_conn.is_connected():
                                    active_conn.close()
                            except:
                                pass
# ================= FOOTER =================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Meta Llama 3.3 70B via Groq")
with col2:
    db_status = {
        "system": "System DB (MySQL)",
        "custom_sqlite": "Custom Persistent SQLite",
        "custom_mysql": "Custom MySQL Host"
    }.get(st.session_state.db_mode, "Unknown")
    st.caption(f"🗄️ {db_status}")
with col3:
    st.caption("🧠 Context-Aware + ⚡ Smart Queries")
    