"""
SQL Assistant Pro - Enhanced Version with Groq (Meta Llama) - FIXED DB CONNECTIONS
Features:
1. Context Retention: Recent 5 messages + summarized older messages + 3 semantically similar Q&As
2. Smart Multi-Table Querying: Intelligently decides when to use single table vs JOINs
3. Powered by Meta Llama via Groq API
4. Fixed: Proper database connection management with SSL support
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

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="SQL Assistant Pro",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SSL Certificate Path (if using SSL)
# Try to get from secrets, otherwise use None for local development
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
if 'use_system_db' not in st.session_state:
    st.session_state.use_system_db = True
if 'temp_db_connection' not in st.session_state:
    st.session_state.temp_db_connection = None
if 'temp_schema' not in st.session_state:
    st.session_state.temp_schema = {}
if 'show_rename_dialog' not in st.session_state:
    st.session_state.show_rename_dialog = False
if 'rename_chat_id' not in st.session_state:
    st.session_state.rename_chat_id = None

# ================= DATABASE CONNECTION FUNCTIONS =================

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
                ssl_ca=st.secrets["auth_database"].get("ssl_ca", ""),
                # ssl_ca=st.secrets["auth_database"]["ssl_ca"],
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

def get_business_db_connection():
    """Connect to business database"""
    try:
        if "database" in st.secrets:
            connection = mysql.connector.connect(
                host=st.secrets["database"]["host"],
                port=int(st.secrets["database"]["port"]),
                database=st.secrets["database"]["database"],
                user=st.secrets["database"]["user"],
                password=st.secrets["database"]["password"],
                ssl_disabled=False,
                ssl_verify_cert=True,
                ssl_ca=st.secrets["database"].get("ssl_ca", ""),
                # ssl_ca=st.secrets["database"]["ssl_ca"],
                ssl_verify_identity=True,
                connect_timeout=30
            )
        else:
            connection = mysql.connector.connect(
                host='localhost',
                database='myntra_db',
                user='root',
                password='password',
                connect_timeout=10
            )
        
        return connection
    except Error as e:
        st.error(f"❌ Business Database connection failed: {e}")
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

def get_database_schema(connection, table_name: Optional[str] = None) -> Dict:
    """Get comprehensive database schema with relationships"""
    schema = {}
    cursor = None
    try:
        cursor = connection.cursor()
        
        # Get all tables or specific table
        if table_name:
            tables = [table_name]
        else:
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
        
        for table in tables:
            # Get columns with details
            cursor.execute(f"DESCRIBE {table}")
            columns = []
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
            relationships = []
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
    except Error as e:
        st.error(f"Schema fetch error: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()

def format_schema_for_llm(schema: Dict, tables_to_include: Optional[List[str]] = None) -> str:
    """Format schema for LLM with smart multi-table context"""
    schema_text = "DATABASE SCHEMA:\n\n"
    
    # Filter tables if specified
    if tables_to_include:
        filtered_schema = {k: v for k, v in schema.items() if k in tables_to_include}
    else:
        filtered_schema = schema
    
    # Add multi-table intelligence note
    if len(filtered_schema) > 1:
        schema_text += "NOTE: Multiple tables available. Analyze the question to determine:\n"
        schema_text += "1. Can this be answered from a SINGLE table? → Use that table only\n"
        schema_text += "2. Does it require data from MULTIPLE tables? → Use appropriate JOINs\n"
        schema_text += "3. Avoid unnecessary JOINs when data exists in one table\n\n"
    
    for table_name, table_info in filtered_schema.items():
        schema_text += f"TABLE: {table_name}\n"
        schema_text += "Columns:\n"
        
        for col in table_info['columns']:
            key_info = f" [{col['key']}]" if col['key'] else ""
            null_info = " (nullable)" if col['null'] == 'YES' else " (required)"
            schema_text += f"  - {col['name']}: {col['type']}{key_info}{null_info}\n"
        
        if table_info.get('relationships'):
            schema_text += "\nRelationships:\n"
            for rel in table_info['relationships']:
                schema_text += f"  - {rel['column']} → {rel['references_table']}.{rel['references_column']}\n"
        
        if table_info.get('sample_data'):
            schema_text += f"\nSample Data ({len(table_info['sample_data'])} rows):\n"
            col_names = [col['name'] for col in table_info['columns']]
            for row in table_info['sample_data'][:3]:
                row_dict = dict(zip(col_names, row))
                schema_text += f"  {row_dict}\n"
        
        schema_text += "\n" + "="*80 + "\n\n"
    
    # Add relationship summary for multi-table queries
    if len(filtered_schema) > 1:
        schema_text += "RELATIONSHIP SUMMARY:\n"
        for table_name, table_info in filtered_schema.items():
            if table_info.get('relationships'):
                for rel in table_info['relationships']:
                    schema_text += f"  {table_name}.{rel['column']} → {rel['references_table']}.{rel['references_column']}\n"
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
            context_parts.append(f"   Assistant: {response_preview}...")
            if turn.get('query_generated'):
                context_parts.append(f"   SQL: {turn['query_generated']}")
            context_parts.append("")
    
    context = "\n".join(context_parts)
    return context, stats

# ================= SMART QUERY GENERATION =================

def analyze_query_intent(question: str, schema: Dict) -> Dict:
    """Analyze question to determine which tables are needed"""
    question_lower = question.lower()
    
    # Extract table information
    table_info = {}
    for table_name, table_data in schema.items():
        table_info[table_name] = {
            'columns': [col['name'].lower() for col in table_data['columns']],
            'keywords': table_name.lower().split('_')
        }
    
    analysis = {
        'requires_join': False,
        'tables_needed': [],
        'intent_type': 'unknown',
        'reasoning': ''
    }
    
    # Keywords that suggest multi-table queries
    join_keywords = ['best selling', 'top selling', 'revenue', 'total sales', 'customer purchased',
                     'order details', 'purchase history', 'sales analysis', 'customer bought',
                     'most popular', 'top products', 'highest revenue', 'sales performance']
    
    # Check for join indicators
    needs_join = any(keyword in question_lower for keyword in join_keywords)
    
    # Check which tables have relevant columns based on question
    for table_name, info in table_info.items():
        # Check if question mentions table-specific terms
        table_relevant = any(keyword in question_lower for keyword in info['keywords'])
        
        # Check for column mentions
        columns_mentioned = [col for col in info['columns'] if col in question_lower]
        
        if table_relevant or columns_mentioned:
            analysis['tables_needed'].append(table_name)
    
    # Determine if JOIN is needed
    if needs_join or len(analysis['tables_needed']) > 1:
        analysis['requires_join'] = True
        analysis['intent_type'] = 'multi_table'
        analysis['reasoning'] = 'Question requires data from multiple tables'
    else:
        analysis['requires_join'] = False
        analysis['intent_type'] = 'single_table'
        analysis['reasoning'] = 'Question can be answered from a single table'
    
    # If no tables identified, use all tables
    if not analysis['tables_needed']:
        analysis['tables_needed'] = list(schema.keys())
    
    return analysis

def generate_sql_query(question: str, schema_text: str, context: str, intent_analysis: Optional[Dict] = None) -> Dict:
    """Generate SQL query using Groq Llama with smart multi-table logic"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
        
        # Enhanced system prompt for smart querying
        system_prompt = """You are an expert SQL query generator with advanced optimization skills.

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
User: "I want kurti" 
→ SELECT * FROM products WHERE category='kurti'

User: "pink" (REFINEMENT)
→ SELECT * FROM products WHERE category='kurti' AND color='pink'

User: "M size" (REFINEMENT)
→ SELECT * FROM products WHERE category='kurti' AND color='pink' AND size='M'

User: "show me shoes" (CONTEXT RESET)
→ SELECT * FROM products WHERE category='shoes'

DECISION FRAMEWORK:
- Question about product attributes (name, price, category, brand, color, etc.) → Use catalog table ONLY
- Question about sales metrics (quantity sold, revenue, best-sellers) → May need JOIN with sales table
- Question combining product info WITH sales data → Use JOIN

EXAMPLES:
✓ "Show me red sneakers" → SELECT * FROM footwear_catalog WHERE color='red' (SINGLE TABLE)
✓ "Find Nike shoes under ₹2000" → SELECT * FROM footwear_catalog WHERE brand='Nike' AND price<2000 (SINGLE TABLE)
✗ "Best selling products" → SELECT c.product_name, c.brand, SUM(s.quantity_sold) FROM footwear_catalog c JOIN footwear_sales s ON c.product_id=s.product_id GROUP BY c.product_id (NEEDS JOIN)
✗ "Total revenue by product" → Requires JOIN to combine product names with sales data

QUERY REQUIREMENTS:
- Use proper JOINs with clear ON conditions when needed
- Include all relevant columns in SELECT
- Use WHERE clauses for filtering
- Add ORDER BY for rankings, to avoid long response keep limit of 10-15
- Use DISTINCT to avoid duplicates when joining
- Always use table aliases for clarity in multi-table queries
- Return ONLY valid MySQL query without explanation, markdown, or code blocks"""

        # Build user prompt with context and schema
        user_prompt = f"""DATABASE SCHEMA:
{schema_text}

CONVERSATION CONTEXT:
{context}

CURRENT QUESTION: {question}

"""
        
        # Add intent analysis if available
        if intent_analysis:
            user_prompt += f"""QUERY ANALYSIS:
Intent Type: {intent_analysis['intent_type']}
Requires JOIN: {intent_analysis['requires_join']}
Tables Needed: {', '.join(intent_analysis['tables_needed'])}
Reasoning: {intent_analysis['reasoning']}

"""
        
        # user_prompt += "Generate the optimal SQL query following the multi-table intelligence rules above. Return ONLY the SQL query, no explanations."
        user_prompt += """Generate the optimal SQL query following all the rules above.
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
    """Execute SQL query and return results"""
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=columns)
        
        return {
            "success": True,
            "data": df,
            "row_count": len(df)
        }
    except Error as e:
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
            SELECT ch.question, ch.query_generated, ch.response, ch.timestamp
            FROM chat_history ch
            JOIN chats c ON ch.chat_id = c.chat_id
            WHERE ch.chat_id = %s AND c.user_id = %s
            ORDER BY ch.timestamp ASC
        """, (chat_id, user_id))
        return cursor.fetchall()
    except Error as e:
        st.error(f"History fetch error: {e}")
        return []
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def save_chat_turn(chat_id: int, user_id: int, question: str, query: Optional[str], response: str) -> bool:
    """Save chat turn with verification"""
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
        
        cursor.execute(
            "INSERT INTO chat_history (chat_id, user_id, question, query_generated, response) VALUES (%s, %s, %s, %s, %s)",
            (chat_id, user_id, question, query, response)
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

Results:
{data_summary}

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

def create_temp_database_from_file(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str], str]:
    """Create temporary SQLite database from uploaded file"""
    try:
        # Read file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:  # Excel
            df = pd.read_excel(io.BytesIO(file_bytes))
        
        # Create SQLite connection
        conn = sqlite3.connect(':memory:')
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Generate table name from filename
        table_name = filename.split('.')[0].replace(' ', '_').replace('-', '_').lower()
        
        # Write to SQLite
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
        # Store in session state
        st.session_state.temp_db_connection = conn
        
        return True, table_name, f"Database created with table '{table_name}' ({len(df)} rows)"
    
    except Exception as e:
        return False, None, f"File processing error: {str(e)}"

# ================= UI HELPER FUNCTIONS =================

def create_copy_button(text: str, label: str = "Copy") -> str:
    """Create copy-to-clipboard button"""
    escaped_text = text.replace('`', '\\`').replace('$', '\\$').replace('"', '\\"')
    return f"""
    <button onclick="navigator.clipboard.writeText(`{escaped_text}`)" 
            style="background-color: #4CAF50; color: white; padding: 5px 10px; 
                   border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
        {label}
    </button>
    """

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link for DataFrame"""
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"""
    <a href="data:file/csv;base64,{b64}" download="{filename}" 
       style="background-color: #008CBA; color: white; padding: 8px 16px; 
              text-decoration: none; border-radius: 4px; font-size: 14px;">
        📥 Download CSV
    </a>
    """

def display_product_card(product: Dict, idx: int):
    """Display product as card"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            name = product.get('product_name') or product.get('name', 'Unknown Product')
            st.markdown(f"**{name}**")
            
            brand = product.get('brand', '')
            category = product.get('category', '')
            if brand or category:
                st.caption(f"{brand} • {category}")
        
        with col2:
            price = product.get('price') or product.get('selling_price', 0)
            st.markdown(f"**₹{price:,.2f}**")
        
        st.divider()

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
    st.title("🗄️ SQL Assistant Pro")
    st.subheader("Powered by Meta Llama 3.3 via Groq")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    success, user_id = verify_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                else:
                    st.warning("Please fill all fields")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        success, message = create_user(new_username, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill all fields")
    
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
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.current_chat_id = None
        st.rerun()

st.divider()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Database Selection
    st.subheader("🗄️ Database Source")
    use_system = st.checkbox(
        "Use System Database",
        value=st.session_state.use_system_db,
        help="Use pre-configured database"
    )
    
    if use_system != st.session_state.use_system_db:
        st.session_state.use_system_db = use_system
        st.rerun()
    
    if not use_system:
        st.info("💡 Upload CSV/Excel to create custom database")
        uploaded_db_file = st.file_uploader(
            "Upload File",
            type=['csv', 'xlsx', 'xls'],
            key="db_uploader",
            help="Upload CSV or Excel file to query"
        )
        
        if uploaded_db_file and st.button("📤 Create Database", use_container_width=True):
            with st.spinner("Processing..."):
                file_bytes = uploaded_db_file.read()
                success, table_name, message = create_temp_database_from_file(
                    file_bytes, uploaded_db_file.name
                )
                
                if success:
                    st.success(f"✅ {message}")
                    if st.session_state.temp_db_connection:
                        schema = get_database_schema(
                            st.session_state.temp_db_connection,
                            table_name
                        )
                        st.session_state.temp_schema = schema
                else:
                    st.error(f"❌ {message}")
    
    st.divider()
    
    # Database Schema Viewer
    st.subheader("📊 Database Schema")
    
    # Show schema based on which database is active
    if st.session_state.use_system_db:
        schema_to_show = st.session_state.business_schema
    elif not st.session_state.use_system_db and st.session_state.temp_db_connection:
        schema_to_show = st.session_state.get('temp_schema', {})
    else:
        schema_to_show = {}
    
    if schema_to_show:
        # Show database name
        if st.session_state.use_system_db:
            try:
                db_name = st.secrets["database"]["database"]
                st.info(f"🗄️ **Database:** {db_name}")
            except:
                st.info("🗄️ **System Database**")
        else:
            st.info(f"🗄️ **Custom Database**")
        
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
        st.warning("No schema available. Connect to a database first.")
    
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
    st.caption("💡 **Powered by Meta Llama 3.3:**")
    st.caption("• 🧠 Context Retention (5 recent + summary)")
    st.caption("• 🔍 Semantic Similar Questions")
    st.caption("• ⚡ Smart Multi-Table Queries")
    st.caption("• 🎯 Auto JOIN Detection")
    
    # Show available tables at the bottom
    if schema_to_show:
        st.divider()
        st.caption("📋 **Available Tables:**")
        for table_name in schema_to_show.keys():
            st.caption(f"• {table_name}")

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
st.info("🤖 **Powered by Meta Llama 3.3 70B** - Lightning-fast context-aware SQL generation!")

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
        st.markdown("- 🎯 **Auto Table Detection** - Smart routing")
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
    
    for turn in chat_history:
        # User message
        with st.chat_message("user"):
            st.write(turn["question"])
            st.markdown(create_copy_button(turn["question"], "📋 Copy Question"), unsafe_allow_html=True)
        
        # Assistant message
        with st.chat_message("assistant"):
            if turn.get("response"):
                st.write(turn["response"])
            
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
                
                # Get fresh business database connection
                business_conn = get_business_db_connection()
                
                if not business_conn:
                    response = "⚠️ System database not connected."
                    st.error(response)
                    save_chat_turn(
                        st.session_state.current_chat_id,
                        st.session_state.user_id,
                        user_question,
                        None,
                        response
                    )
                else:
                    try:
                        # Analyze query intent for smart table selection
                        intent_analysis = analyze_query_intent(user_question, st.session_state.business_schema)
                        
                        # Show intent analysis
                        st.info(f"🎯 Intent: {intent_analysis['intent_type']} | Tables: {', '.join(intent_analysis['tables_needed'])}")
                        
                        # Format schema with smart multi-table context
                        schema_text = format_schema_for_llm(
                            st.session_state.business_schema,
                            tables_to_include=intent_analysis['tables_needed']
                        )
                        
                        # Generate query with intent analysis
                        query_result = generate_sql_query(
                            user_question,
                            schema_text,
                            context,
                            intent_analysis
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
                                response
                            )
                        else:
                            query = query_result["query"]
                            result = execute_query(business_conn, query)
                            
                            if not result["success"]:
                                response = f"❌ Query failed: {result.get('error', 'Unknown')}"
                                st.error(response)
                                
                                # Show available tables on error
                                st.warning("**Available tables in database:**")
                                if st.session_state.business_schema:
                                    for table_name in st.session_state.business_schema.keys():
                                        st.write(f"• {table_name}")
                                
                                with st.expander("🔍 View Failed Query"):
                                    st.code(query, language="sql")
                                    st.markdown(create_copy_button(query, "📋 Copy Query"), unsafe_allow_html=True)
                            else:
                                # Generate response with visualization
                                summary, df, visualization = generate_db_response_with_presentation(
                                    user_question, query, result, context
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
                                    
                                    if is_product_data and len(df) <= 20:
                                        # Show as product cards
                                        st.markdown(f"### 🛍️ Products Found ({len(df)} items)")
                                        for idx, row in df.iterrows():
                                            display_product_card(row.to_dict(), idx)
                                    else:
                                        # Show as table with download option
                                        with st.expander(f"📊 View All Results ({len(df)} items)", expanded=True):
                                            st.dataframe(df, use_container_width=True, height=400)
                                            st.markdown(create_download_link(df, f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), unsafe_allow_html=True)
                                
                                # Query details expander
                                with st.expander("🔍 View Query & Optimization Details"):
                                    # Query type indicator
                                    query_lower = query.lower()
                                    if "join" in query_lower:
                                        st.warning("🔗 **Multi-Table Query** - JOIN was necessary for this question")
                                    else:
                                        st.success("⚡ **Single-Table Query** - Optimized for speed!")
                                    
                                    st.subheader("📝 Generated SQL Query")
                                    st.code(query, language="sql")
                                    st.markdown(create_copy_button(query, "📋 Copy Query"), unsafe_allow_html=True)
                                    
                                    st.subheader("🎯 Query Intent Analysis")
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
                            
                            save_chat_turn(
                                st.session_state.current_chat_id,
                                st.session_state.user_id,
                                user_question,
                                query,
                                response
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
                            response
                        )
                    finally:
                        # Close business database connection
                        if business_conn and business_conn.is_connected():
                            business_conn.close()

# ================= FOOTER =================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Meta Llama 3.3 70B via Groq")
with col2:
    db_status = "System DB" if st.session_state.use_system_db else "Custom DB"
    st.caption(f"🗄️ {db_status}")
with col3:
    st.caption("🧠 Context-Aware + ⚡ Smart Queries")