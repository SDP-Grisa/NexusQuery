"""
SQL Assistant Pro - Enhanced Version with Groq (Meta Llama)
COMPLETE VERSION WITH ALL IMPROVEMENTS

Features:
1. Context Retention: Recent 5 messages + summarized older messages + 3 semantically similar Q&As
2. LLM-Based Smart Query Analysis
3. DYNAMIC Product/Student/Employee Display (no hardcoding)
4. Improved Authentication UI
5. Delete Confirmation Dialogs
6. SQLite compatibility for custom databases
7. Persistent Custom SQLite Databases (file-based)
8. Custom MySQL Database Connection Support
9. User-specific previous SQLite DB listing on login
10. UI-based credential input for Custom MySQL
11. IMPROVED: Dynamic data visualization based on query intent
12. IMPROVED: Universal card display for any data type
13. IMPROVED: Intelligent chart selection (6+ chart types)

DATABASE MIGRATION:
ALTER TABLE chat_history ADD COLUMN result_data LONGTEXT AFTER response;
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
import glob

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="SQL Assistant Pro",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
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
    .delete-warning {
        background: #fff5f5;
        border: 2px solid #fc8181;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-item {
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .chat-item:hover {
        background: #f7fafc;
    }
    .custom-db-form {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .db-item {
        background: #e6f3ff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

PERSISTENT_DB_DIR = "custom_dbs"
os.makedirs(PERSISTENT_DB_DIR, exist_ok=True)

try:
    ssl_ca_path = st.secrets.get("ssl_ca_path", None)
except:
    ssl_ca_path = None

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ================= SESSION STATE =================
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
    st.session_state.db_mode = "system"
if 'active_custom_sqlite_path' not in st.session_state:
    st.session_state.active_custom_sqlite_path = None
if 'user_sqlite_dbs' not in st.session_state:
    st.session_state.user_sqlite_dbs = []
if 'custom_mysql_params' not in st.session_state:
    st.session_state.custom_mysql_params = {}
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
    db_files.sort(key=os.path.getmtime, reverse=True)
    return db_files

# ================= DATABASE CONNECTION =================
def get_auth_db_connection():
    """Connect to authentication database"""
    try:
        if "auth_database" in st.secrets:
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
                ssl_disabled=True
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
                ssl_disabled=True
            )
        return connection
    except Error as e:
        st.error(f"❌ Business Database connection failed: {e}")
        return None

def get_custom_mysql_connection_from_params(params: Dict) -> Optional[mysql.connector.connection.MySQLConnection]:
    """Connect to custom MySQL"""
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

# ================= AUTHENTICATION =================
def hash_password(password: str) -> str:
    """Hash password"""
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

# ================= DATABASE SCHEMA =================
def is_sqlite_connection(connection) -> bool:
    """Check if connection is SQLite"""
    return isinstance(connection, sqlite3.Connection)

def get_database_schema(connection, table_name: Optional[str] = None) -> Dict:
    """Get database schema - works with MySQL and SQLite"""
    schema = {}
    cursor = None
    is_sqlite = is_sqlite_connection(connection)
    try:
        cursor = connection.cursor()
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
            if is_sqlite:
                cursor.execute(f"PRAGMA table_info({table})")
                for col in cursor.fetchall():
                    columns.append({
                        'name': col[1],
                        'type': col[2],
                        'null': 'NO' if col[3] else 'YES',
                        'key': 'PRI' if col[5] else '',
                        'default': col[4],
                        'extra': ''
                    })
            else:
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
            relationships = []
            if is_sqlite:
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                for rel in cursor.fetchall():
                    relationships.append({
                        'column': rel[3],
                        'references_table': rel[2],
                        'references_column': rel[4]
                    })
            else:
                cursor.execute(f"""
                    SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
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
    if len(filtered_schema) > 1:
        schema_text += "RELATIONSHIP SUMMARY:\n"
        for table_name, table_info in filtered_schema.items():
            if table_info.get('relationships'):
                for rel in table_info['relationships']:
                    schema_text += f" {table_name}.{rel['column']} → {rel['references_table']}.{rel['references_column']}\n"
        schema_text += "\n"
    return schema_text

# ================= CONTEXT MANAGEMENT =================
def compute_embedding(text: str) -> np.ndarray:
    """Compute embedding for text"""
    return embedding_model.encode(text)

def find_semantically_similar_messages(
    current_question: str,
    chat_history: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """Find top-k semantically similar Q&A pairs"""
    if not chat_history:
        return []
    current_embedding = compute_embedding(current_question)
    similarities = []
    for turn in chat_history:
        question_embedding = compute_embedding(turn['question'])
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1),
            question_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((similarity, turn))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [turn for _, turn in similarities[:top_k]]

def summarize_old_messages(messages: List[Dict]) -> str:
    """Summarize older messages using Groq"""
    if not messages:
        return ""
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
                "content": f"Summarize this conversation history concisely:\n\n{summary_text}"
            }],
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Previous conversation context available."

def build_optimized_context(
    chat_history: List[Dict],
    current_question: str,
    recent_count: int = 5,
    semantic_count: int = 3
) -> Tuple[str, Dict]:
    """Build optimized context"""
    context_parts = []
    stats = {
        'total_messages': len(chat_history),
        'recent_count': 0,
        'summarized_count': 0,
        'semantic_count': 0
    }
    if not chat_history:
        return "", stats
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
    older_messages = chat_history[:-recent_count] if len(chat_history) > recent_count else []
    stats['summarized_count'] = len(older_messages)
    if older_messages:
        summary = summarize_old_messages(older_messages)
        if summary:
            context_parts.append("EARLIER CONVERSATION SUMMARY:")
            context_parts.append(summary)
            context_parts.append("")
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

# ================= NEW: DATA ANALYSIS FUNCTIONS =================
def analyze_dataframe_structure(df: pd.DataFrame) -> Dict:
    """Analyze DataFrame to determine optimal display strategy"""
    if df.empty:
        return {'type': 'empty', 'display_method': 'none'}
    
    # Detect column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Detect common patterns
    has_name = any(col.lower() in ['name', 'product_name', 'student_name', 'employee_name', 'customer_name'] 
                   for col in df.columns)
    has_price = any(col.lower() in ['price', 'cost', 'amount', 'salary', 'revenue'] 
                    for col in df.columns)
    has_id = any(col.lower() in ['id', 'product_id', 'student_id', 'employee_id', 'customer_id'] 
                 for col in df.columns)
    
    # Determine data type
    data_type = 'unknown'
    if any(col.lower() in ['product_name', 'brand', 'category'] for col in df.columns):
        data_type = 'product'
    elif any(col.lower() in ['student_name', 'grade', 'marks', 'score'] for col in df.columns):
        data_type = 'student'
    elif any(col.lower() in ['employee_name', 'designation', 'department'] for col in df.columns):
        data_type = 'employee'
    elif any(col.lower() in ['customer_name', 'order', 'purchase'] for col in df.columns):
        data_type = 'customer'
    
    # Determine display method
    row_count = len(df)
    col_count = len(df.columns)
    
    display_method = 'table'  # default
    if row_count <= 50 and has_name:
        display_method = 'card'
    elif row_count > 50:
        display_method = 'table_paginated'
    
    return {
        'type': data_type,
        'display_method': display_method,
        'row_count': row_count,
        'col_count': col_count,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'date_cols': date_cols,
        'has_name': has_name,
        'has_price': has_price,
        'has_id': has_id
    }

def display_data_card(row: Dict, idx: int, turn_idx: int, data_analysis: Dict):
    """Universal card display - adapts to any data type"""
    # Determine primary identifier
    primary_key = None
    primary_value = "Item"
    
    # Try to find a name column
    name_cols = [col for col in row.keys() if 'name' in col.lower()]
    if name_cols:
        primary_key = name_cols[0]
        primary_value = row[primary_key]
    elif data_analysis['has_id']:
        id_cols = [col for col in row.keys() if 'id' in col.lower()]
        if id_cols:
            primary_key = id_cols[0]
            primary_value = f"ID: {row[primary_key]}"
    else:
        primary_key = list(row.keys())[0]
        primary_value = row[primary_key]
    
    # Find secondary info for header
    secondary_info = []
    
    # Look for price/amount columns
    if data_analysis['has_price']:
        price_cols = [col for col in row.keys() if any(p in col.lower() for p in ['price', 'cost', 'amount', 'salary', 'revenue'])]
        for col in price_cols:
            if row[col] is not None and row[col] != '':
                try:
                    val = float(row[col])
                    secondary_info.append(f"₹{val:,.2f}")
                except:
                    secondary_info.append(str(row[col]))
                break
    
    # Look for category/type columns
    category_cols = [col for col in row.keys() if any(c in col.lower() for c in ['category', 'type', 'department', 'grade', 'class'])]
    for col in category_cols:
        if row[col] is not None and row[col] != '':
            secondary_info.append(str(row[col]))
            break
    
    # Build header
    header_text = f"📋 {primary_value}"
    if secondary_info:
        header_text += " | " + " | ".join(secondary_info[:2])
    
    expander_key = f"data_card_{turn_idx}_{idx}"
    
    with st.expander(header_text, expanded=False):
        all_keys = list(row.keys())
        displayed_keys = [primary_key] if primary_key else []
        
        # Display primary info
        st.markdown("### 📌 Primary Information")
        col1, col2 = st.columns(2)
        
        col_idx = 0
        for key in all_keys[:6]:
            if row[key] is not None and row[key] != '':
                with (col1 if col_idx % 2 == 0 else col2):
                    value = row[key]
                    if isinstance(value, (int, float)):
                        if 'price' in key.lower() or 'cost' in key.lower() or 'salary' in key.lower():
                            st.markdown(f"**{key.replace('_', ' ').title()}:** ₹{value:,.2f}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                displayed_keys.append(key)
                col_idx += 1
        
        # Display remaining attributes
        remaining_keys = [k for k in all_keys if k not in displayed_keys]
        if remaining_keys:
            st.markdown("### 📋 Additional Details")
            for key in remaining_keys:
                if row[key] is not None and row[key] != '':
                    value = row[key]
                    if isinstance(value, (int, float)):
                        if 'price' in key.lower() or 'cost' in key.lower() or 'salary' in key.lower():
                            st.markdown(f"**{key.replace('_', ' ').title()}:** ₹{value:,.2f}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

def create_improved_visualization(df: pd.DataFrame, question: str, data_analysis: Dict) -> Optional[go.Figure]:
    """Create intelligent visualization based on data and query"""
    if df.empty or len(df) > 100:
        return None
    
    question_lower = question.lower()
    numeric_cols = data_analysis['numeric_cols']
    categorical_cols = data_analysis['categorical_cols']
    
    if not numeric_cols:
        return None
    
    fig = None
    
    # 1. Time series
    if data_analysis['date_cols'] and numeric_cols:
        date_col = data_analysis['date_cols'][0]
        value_col = numeric_cols[0]
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            title=f"{value_col.replace('_', ' ').title()} Over Time",
            markers=True
        )
        fig.update_traces(line_color='#667eea', line_width=3)
    
    # 2. Ranking/Comparison - Bar chart
    elif any(word in question_lower for word in ['top', 'best', 'most', 'highest', 'lowest', 'compare', 'rank']):
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            plot_df = df.nlargest(min(15, len(df)), num_col)
            fig = px.bar(
                plot_df,
                x=cat_col,
                y=num_col,
                title=f"Top {len(plot_df)} by {num_col.replace('_', ' ').title()}",
                color=num_col,
                color_continuous_scale='viridis',
                text=num_col
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    
    # 3. Distribution - Donut chart
    elif any(word in question_lower for word in ['distribution', 'breakdown', 'percentage', 'share', 'proportion']):
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if len(df) > 15:
                plot_df = df.groupby(cat_col)[num_col].sum().reset_index().nlargest(10, num_col)
            else:
                plot_df = df
            fig = px.pie(
                plot_df,
                names=cat_col,
                values=num_col,
                title=f"Distribution of {num_col.replace('_', ' ').title()}",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # 4. Count/Frequency
    elif any(word in question_lower for word in ['count', 'number of', 'how many', 'frequency']):
        if categorical_cols:
            cat_col = categorical_cols[0]
            count_df = df[cat_col].value_counts().reset_index()
            count_df.columns = [cat_col, 'count']
            fig = px.bar(
                count_df.head(15),
                x=cat_col,
                y='count',
                title=f"Frequency of {cat_col.replace('_', ' ').title()}",
                color='count',
                color_continuous_scale='blues',
                text='count'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    
    # 5. Scatter for correlation
    elif len(numeric_cols) >= 2 and len(df) <= 100:
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"{numeric_cols[0].replace('_', ' ').title()} vs {numeric_cols[1].replace('_', ' ').title()}",
            color=categorical_cols[0] if categorical_cols else None,
            hover_data=df.columns.tolist()
        )
    
    # 6. Default horizontal bar
    elif categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        plot_df = df.head(15)
        fig = px.bar(
            plot_df,
            y=cat_col,
            x=num_col,
            orientation='h',
            title=f"{cat_col.replace('_', ' ').title()} Analysis",
            color=num_col,
            color_continuous_scale='teal',
            text=num_col
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    # Apply consistent styling
    if fig:
        fig.update_layout(
            height=500,
            font=dict(size=12),
            title_font_size=16,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig
# CONTINUATION OF SQL ASSISTANT PRO - PART 2
# This continues from sql_assistant_pro_complete.py

# ================= LLM QUERY INTENT ANALYSIS =================
def analyze_query_intent_with_llm(question: str, schema: Dict) -> Dict:
    """Use LLM to analyze query intent"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
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
1. Which tables are needed?
2. Requires JOIN?
3. Query type?
Return JSON:
{{
    "requires_join": true/false,
    "tables_needed": ["table1"],
    "intent_type": "single_table",
    "reasoning": "explanation"
}}
ONLY JSON, no extra text."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=500,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        result = result.replace('```json', '').replace('```', '').strip()
        analysis = json.loads(result)
        return {
            'requires_join': analysis.get('requires_join', False),
            'tables_needed': analysis.get('tables_needed', list(schema.keys())),
            'intent_type': analysis.get('intent_type', 'unknown'),
            'reasoning': analysis.get('reasoning', 'LLM analysis completed')
        }
    except Exception as e:
        return {
            'requires_join': False,
            'tables_needed': list(schema.keys()),
            'intent_type': 'unknown',
            'reasoning': 'Fallback analysis'
        }

# ================= QUERY GENERATION =================
def generate_sql_query(question: str, schema_text: str, context: str, intent_analysis: Optional[Dict] = None, is_sqlite: bool = False) -> Dict:
    """Generate SQL query using Groq"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
        sql_dialect = "SQLite" if is_sqlite else "MySQL"
        
        system_prompt = f"""You are an expert SQL query generator for {sql_dialect}.
CRITICAL RULES:
1. Prefer single-table queries
2. Only JOIN when absolutely necessary
3. Use proper WHERE clauses
4. LIMIT results to 10-15
5. Return ONLY valid SQL query, no markdown
6. For refinements, accumulate ALL filters"""

        user_prompt = f"""DATABASE SCHEMA:
{schema_text}
CONVERSATION CONTEXT:
{context}
CURRENT QUESTION: {question}
"""
        
        if intent_analysis:
            user_prompt += f"""
ANALYSIS:
Intent: {intent_analysis['intent_type']}
JOIN needed: {intent_analysis['requires_join']}
Tables: {', '.join(intent_analysis['tables_needed'])}
"""
        
        user_prompt += "\nGenerate optimal SQL query. ONLY the query, no explanation."
        
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
        query = query.replace('```sql', '').replace('```', '').strip()
        lines = query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
        query = ' '.join(sql_lines)
        
        if not query.upper().startswith('SELECT'):
            return {"success": False, "error": "Not a SELECT statement", "query": query}
        
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
        return {"success": False, "error": str(e), "query": None}

def execute_query(connection, query: str) -> Dict:
    """Execute SQL query"""
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
        return {"success": True, "data": df, "row_count": len(df)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": None}
    finally:
        if cursor:
            cursor.close()

# ================= RESPONSE GENERATION =================
def generate_db_response_with_presentation(
    question: str,
    query: str,
    result: Dict,
    context: str
) -> Tuple[str, Optional[pd.DataFrame], Optional[go.Figure], Dict]:
    """Generate response with visualization"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
        df = result.get("data")
        if df is None or df.empty:
            return "No results found.", None, None, {}
        
        # Analyze data structure
        data_analysis = analyze_dataframe_structure(df)
        
        data_summary = f"Query returned {len(df)} rows with columns: {', '.join(df.columns.tolist())}\n\n"
        data_summary += f"Sample data:\n{df.head(10).to_string()}"
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Context: {context}
Question: {question}
SQL: {query}
Results: {data_summary}
Provide a natural, conversational summary. Be concise."""
            }],
            max_tokens=800,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content
        visualization = create_improved_visualization(df, question, data_analysis)
        
        return summary, df, visualization, data_analysis
    except Exception as e:
        st.error(f"Response generation error: {e}")
        data_analysis = analyze_dataframe_structure(df) if df is not None else {}
        return f"Found {len(df)} results.", df, None, data_analysis

# ================= CHAT MANAGEMENT =================
def create_new_chat(user_id: int, title: Optional[str], first_question: Optional[str]) -> Optional[int]:
    """Create new chat"""
    connection = get_auth_db_connection()
    if not connection:
        return None
    try:
        cursor = connection.cursor()
        chat_title = title or (first_question[:50] + "..." if first_question else "New Chat")
        cursor.execute("INSERT INTO chats (user_id, title) VALUES (%s, %s)", (user_id, chat_title))
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
    """Get all user chats"""
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
    """Get chat history"""
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
    """Save chat turn"""
    connection = get_auth_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT user_id FROM chats WHERE chat_id = %s", (chat_id,))
        result = cursor.fetchone()
        if not result or result[0] != user_id:
            return False
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
    """Rename chat"""
    connection = get_auth_db_connection()
    if not connection:
        return False, "Connection failed"
    try:
        cursor = connection.cursor()
        cursor.execute("UPDATE chats SET title = %s WHERE chat_id = %s AND user_id = %s", (new_title, chat_id, user_id))
        connection.commit()
        if cursor.rowcount > 0:
            return True, "Renamed successfully"
        return False, "Not found"
    except Error as e:
        return False, f"Error: {e}"
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def delete_chat(chat_id: int, user_id: int) -> Tuple[bool, str]:
    """Delete chat"""
    connection = get_auth_db_connection()
    if not connection:
        return False, "Connection failed"
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM chat_history WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
        cursor.execute("DELETE FROM chats WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
        connection.commit()
        if cursor.rowcount > 0:
            return True, "Deleted successfully"
        return False, "Not found"
    except Error as e:
        return False, f"Error: {e}"
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def generate_smart_chat_title(question: str) -> str:
    """Generate title from question"""
    if len(question) <= 50:
        return question
    return question[:50] + "..."

# ================= FILE UPLOAD =================
def create_persistent_sqlite_from_file(file_bytes: bytes, filename: str, user_id: int) -> Tuple[bool, Optional[str], str]:
    """Create persistent SQLite from file"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        safe_filename = filename.split('.')[0].replace(' ', '_').replace('-', '_').lower()
        db_filename = f"{user_id}_{safe_filename}.db"
        db_path = os.path.join(PERSISTENT_DB_DIR, db_filename)
        conn = sqlite3.connect(db_path)
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        table_name = safe_filename
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        conn.commit()
        conn.close()
        return True, table_name, f"Database created at '{db_path}' with table '{table_name}' ({len(df)} rows)"
    except Exception as e:
        return False, None, f"Error: {str(e)}"

def create_temp_database_from_mysql_file(file_bytes: bytes, filename: str, mysql_conn) -> Tuple[bool, Optional[str], str]:
    """Load file into MySQL"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        safe_filename = filename.split('.')[0].replace(' ', '_').replace('-', '_').lower()
        table_name = safe_filename
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        from sqlalchemy import create_engine
        engine = create_engine(f"mysql+mysqlconnector://{mysql_conn.user}:{mysql_conn.password}@{mysql_conn.host}:{mysql_conn.port}/{mysql_conn.database}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        return True, table_name, f"Loaded into MySQL table '{table_name}' ({len(df)} rows)"
    except Exception as e:
        return False, None, f"Error: {str(e)}"

# ================= UI HELPERS =================
def create_copy_button(text: str, label: str = "Copy") -> str:
    """Create copy button"""
    escaped_text = text.replace('`', '\\`').replace('$', '\\$').replace('"', '\\"')
    return f"""
    <button onclick="navigator.clipboard.writeText(`{escaped_text}`)" style="
        background: #667eea;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    ">{label}</button>
    """

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link"""
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
    ">📥 Download CSV</a>
    """

# ================= LOAD SCHEMA =================
@st.cache_data(ttl=3600)
def load_business_schema():
    """Load business schema"""
    connection = get_business_db_connection()
    if connection:
        try:
            schema = get_database_schema(connection)
            return schema
        finally:
            if connection.is_connected():
                connection.close()
    return {}

if not st.session_state.business_schema:
    st.session_state.business_schema = load_business_schema()

# ================= MAIN APPLICATION UI =================
# ================= LOGIN/SIGNUP =================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-header">', unsafe_allow_html=True)
        st.markdown("# 🗄️ SQL Assistant Pro")
        st.markdown("### Powered by Meta Llama 3.3")
        st.markdown('</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
        
        with tab1:
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            with st.form("login_form"):
                st.markdown("### Welcome Back!")
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
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
                new_username = st.text_input("Username", placeholder="Choose username")
                new_password = st.text_input("Password", type="password", placeholder="Choose password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
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
                                    st.info("👉 Please login")
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                st.error("❌ Password must be 6+ characters")
                        else:
                            st.error("❌ Passwords don't match")
                    else:
                        st.warning("⚠️ Fill all fields")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### ✨ Features")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("- 🧠 Context Retention\n- 🔍 Semantic Search\n- ⚡ Smart Queries\n- 📊 Dynamic Visualization")
        with col_f2:
            st.markdown("- 🎯 Intent Analysis\n- 💬 Multi-Chat\n- 🗄️ Multi-Database\n- 🎨 Adaptive Display")
    st.stop()

# ================= MAIN APP =================
col1, col2, col3 = st.columns([2, 3, 1])
with col1:
    st.title("🗄️ SQL Assistant Pro")
with col2:
    st.markdown(f"### Welcome, **{st.session_state.username}**! 👋")
with col3:
    if st.button("🚪 Logout", type="secondary"):
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
    st.subheader("🗄️ Database Source")
    db_modes = ["System DB (MySQL)", "Custom Persistent SQLite", "Custom MySQL Host"]
    selected_mode = st.radio(
        "Select Database Mode",
        db_modes,
        index=0 if st.session_state.db_mode == "system" else 1 if st.session_state.db_mode == "custom_sqlite" else 2
    )
    
    if selected_mode == "System DB (MySQL)":
        st.session_state.db_mode = "system"
    elif selected_mode == "Custom Persistent SQLite":
        st.session_state.db_mode = "custom_sqlite"
    else:
        st.session_state.db_mode = "custom_mysql"
    
    st.markdown("---")
    
    if st.session_state.db_mode == "system":
        st.success("✅ Using System MySQL Database")
    elif st.session_state.db_mode == "custom_sqlite":
        st.markdown('<div class="custom-db-form">', unsafe_allow_html=True)
        st.info("💡 Your Previous SQLite Databases:")
        if st.session_state.user_sqlite_dbs:
            st.markdown("### 📂 Previous Databases")
            selected_db = st.selectbox(
                "Select Existing DB",
                options=[os.path.basename(path) for path in st.session_state.user_sqlite_dbs],
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
            st.info("No previous databases.")
        
        st.markdown("### 📤 Upload New Database")
        uploaded_db_file = st.file_uploader(
            "Upload CSV/Excel",
            type=['csv', 'xlsx', 'xls'],
            key="sqlite_uploader"
        )
        
        if uploaded_db_file and st.button("📤 Create Persistent SQLite DB", use_container_width=True):
            with st.spinner("Processing..."):
                file_bytes = uploaded_db_file.read()
                success, table_name, message = create_persistent_sqlite_from_file(
                    file_bytes,
                    uploaded_db_file.name,
                    st.session_state.user_id
                )
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.user_sqlite_dbs = load_user_sqlite_dbs(st.session_state.user_id)
                    safe_filename = uploaded_db_file.name.split('.')[0].replace(' ', '_').replace('-', '_').lower()
                    st.session_state.active_custom_sqlite_path = os.path.join(PERSISTENT_DB_DIR, f"{st.session_state.user_id}_{safe_filename}.db")
                    conn = sqlite3.connect(st.session_state.active_custom_sqlite_path)
                    schema = get_database_schema(conn)
                    st.session_state.custom_schema = schema
                    conn.close()
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:  # custom_mysql
        st.markdown('<div class="custom-db-form">', unsafe_allow_html=True)
        st.info("🔌 Connect to Custom MySQL Host")
        with st.form("mysql_creds_form"):
            st.markdown("### Enter Connection Details")
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value=st.session_state.custom_mysql_params.get("host", ""), placeholder="localhost")
                port = st.number_input("Port", value=st.session_state.custom_mysql_params.get("port", 3306), min_value=1, max_value=65535)
                database = st.text_input("Database", value=st.session_state.custom_mysql_params.get("database", ""))
            with col2:
                user = st.text_input("User", value=st.session_state.custom_mysql_params.get("user", ""), placeholder="root")
                password = st.text_input("Password", value=st.session_state.custom_mysql_params.get("password", ""), type="password")
                ssl_disabled = st.checkbox("Disable SSL", value=st.session_state.custom_mysql_params.get("ssl_disabled", True))
                ssl_ca = st.text_input("SSL CA Path", value=st.session_state.custom_mysql_params.get("ssl_ca", ""), disabled=ssl_disabled)
            
            connect_btn = st.form_submit_button("🔄 Connect", use_container_width=True)
            if connect_btn:
                if host and port and database and user and password:
                    params = {
                        "host": host, "port": port, "database": database,
                        "user": user, "password": password,
                        "ssl_disabled": ssl_disabled, "ssl_ca": ssl_ca if not ssl_disabled else ""
                    }
                    st.session_state.custom_mysql_params = params
                    with st.spinner("Connecting..."):
                        conn = get_custom_mysql_connection_from_params(params)
                        if conn:
                            st.session_state.custom_mysql_connection = conn
                            schema = get_database_schema(conn)
                            st.session_state.custom_schema = schema
                            st.success("✅ Connected!")
                            st.rerun()
                        else:
                            st.error("❌ Connection failed")
                else:
                    st.warning("⚠️ Fill all fields")
        
        if st.session_state.custom_mysql_connection:
            st.info("💡 Optionally load CSV/Excel")
            uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx', 'xls'], key="mysql_uploader")
            if uploaded_file and st.button("📤 Load into MySQL", use_container_width=True):
                with st.spinner("Loading..."):
                    file_bytes = uploaded_file.read()
                    success, table_name, message = create_temp_database_from_mysql_file(
                        file_bytes, uploaded_file.name, st.session_state.custom_mysql_connection
                    )
                    if success:
                        st.success(f"✅ {message}")
                        schema = get_database_schema(st.session_state.custom_mysql_connection)
                        st.session_state.custom_schema = schema
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📊 Database Schema")
    
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
        db_name = f"SQLite: {os.path.basename(st.session_state.active_custom_sqlite_path)}"
    elif st.session_state.db_mode == "custom_mysql" and st.session_state.custom_mysql_connection:
        schema_to_show = st.session_state.custom_schema
        db_name = f"MySQL: {st.session_state.custom_mysql_params.get('database', 'Connected')}"
    
    if schema_to_show:
        st.info(f"🗄️ **Database:** {db_name}")
        st.markdown("---")
        for table_name, table_info in schema_to_show.items():
            with st.expander(f"📁 **{table_name}**", expanded=False):
                st.caption(f"**Columns ({len(table_info['columns'])}):**")
                for col in table_info['columns']:
                    key_icon = ""
                    if col.get('key') == 'PRI':
                        key_icon = "🔑 "
                    elif col.get('key') == 'MUL':
                        key_icon = "🔗 "
                    col_type = col['type'].split('(')[0] if '(' in col['type'] else col['type']
                    st.markdown(f"{key_icon}**{col['name']}** `{col_type}`")
                if table_info.get('relationships'):
                    st.caption("**🔗 Relationships:**")
                    for rel in table_info['relationships']:
                        st.markdown(f"→ {rel['column']} ➜ {rel['references_table']}.{rel['references_column']}")
    else:
        st.warning("No schema available")
    
    st.divider()
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
        st.info("No chats yet 🚀")
    
    st.divider()
    st.caption("💡 **Powered by Llama 3.3:**")
    st.caption("• 🧠 Context Retention")
    st.caption("• 🔍 Semantic Search")
    st.caption("• 🎯 Intent Analysis")
    st.caption("• 📊 Dynamic Visualization")
    
    if schema_to_show:
        st.divider()
        st.caption("📋 **Available Tables:**")
        for table_name in schema_to_show.keys():
            st.caption(f"• {table_name}")

# ================= DELETE DIALOG =================
if st.session_state.show_delete_dialog and st.session_state.delete_chat_id:
    @st.dialog("⚠️ Confirm Delete")
    def delete_dialog():
        user_chats = get_user_chats(st.session_state.user_id)
        chat_to_delete = next((c for c in user_chats if c['chat_id'] == st.session_state.delete_chat_id), None)
        if chat_to_delete:
            st.markdown('<div class="delete-warning">', unsafe_allow_html=True)
            st.warning("⚠️ **This action cannot be undone!**")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"### Delete this chat?")
            st.info(f"**Chat:** {chat_to_delete['title']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
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
            new_title = st.text_input("New title:", value=current_chat['title'], max_chars=255, key="rename_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True, type="primary"):
                    if new_title and new_title.strip():
                        success, message = rename_chat(st.session_state.rename_chat_id, st.session_state.user_id, new_title)
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
st.info("🤖 **Powered by Meta Llama 3.3 70B** - Context-aware SQL generation with dynamic visualization!")

if not st.session_state.current_chat_id:
    st.markdown("## 👋 Welcome to SQL Assistant Pro!")
    st.markdown("### Enhanced with Meta Llama 3.3")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧠 Context Features")
        st.markdown("- ✅ Last 5 Messages\n- 📝 Summarizes Older Chats\n- 🔍 Semantic Search\n- 💡 Smart Context")
    with col2:
        st.markdown("#### ⚡ Query Intelligence")
        st.markdown("- 🎯 Intent Analysis\n- 🔗 Smart JOINs\n- 📊 Dynamic Charts\n- 🚀 Speed Optimized")
    st.markdown("---")
    st.markdown("### 💡 Example Questions:")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**Single-Table:**\n- 'Show red sneakers'\n- 'Find Nike products'\n- 'Shoes under ₹2000'")
    with ex_col2:
        st.markdown("**Multi-Table:**\n- 'Best-selling products'\n- 'Revenue by category'\n- 'Sales performance'")
else:
    chat_history = get_chat_history(st.session_state.current_chat_id, st.session_state.user_id)
    
    for turn_idx, turn in enumerate(chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
            st.markdown(create_copy_button(turn["question"], "📋 Copy"), unsafe_allow_html=True)
        
        with st.chat_message("assistant"):
            if turn.get("response"):
                st.write(turn["response"])
            
            # UPDATED: Dynamic data display
            if turn.get("result_df") is not None:
                df = turn["result_df"]
                if not df.empty:
                    data_analysis = analyze_dataframe_structure(df)
                    
                    if data_analysis['display_method'] == 'card' and len(df) <= 50:
                        st.markdown(f"### 📋 Results Found ({len(df)} items)")
                        st.caption("*Click to expand*")
                        for idx, row in df.iterrows():
                            display_data_card(row.to_dict(), idx, turn_idx, data_analysis)
                    else:
                        with st.expander(f"📊 View Results ({len(df)} items)", expanded=False):
                            st.dataframe(df, use_container_width=True, height=400)
                            st.markdown(create_download_link(df, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), unsafe_allow_html=True)
            
            if turn.get("query_generated"):
                with st.expander("🔍 View SQL Query"):
                    st.code(turn["query_generated"], language="sql")
                    st.markdown(create_copy_button(turn["query_generated"], "📋 Copy"), unsafe_allow_html=True)
                    query_lower = turn["query_generated"].lower()
                    if "join" in query_lower:
                        st.info("🔗 Multi-table query")
                    else:
                        st.success("⚡ Single-table query")
    
    user_question = st.chat_input("💬 Ask about your data...")
    
    if user_question:
        if len(chat_history) == 0:
            new_title = generate_smart_chat_title(user_question)
            rename_chat(st.session_state.current_chat_id, st.session_state.user_id, new_title)
        
        with st.chat_message("user"):
            st.write(user_question)
            st.markdown(create_copy_button(user_question, "📋 Copy"), unsafe_allow_html=True)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                context, context_stats = build_optimized_context(chat_history, user_question)
                
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
                    response = "⚠️ No database connected"
                    st.error(response)
                    save_chat_turn(st.session_state.current_chat_id, st.session_state.user_id, user_question, None, response, None)
                else:
                    try:
                        with st.spinner("🧠 Analyzing intent..."):
                            intent_analysis = analyze_query_intent_with_llm(user_question, active_schema)
                        st.info(f"🎯 {intent_analysis['intent_type']} | Tables: {', '.join(intent_analysis['tables_needed'])}")
                        
                        schema_text = format_schema_for_llm(active_schema, tables_to_include=intent_analysis['tables_needed'])
                        query_result = generate_sql_query(user_question, schema_text, context, intent_analysis, is_sqlite=is_sqlite)
                        
                        if not query_result["success"]:
                            response = f"❌ Query generation failed: {query_result.get('error', 'Unknown')}"
                            st.error(response)
                            if query_result.get("query"):
                                st.code(query_result["query"], language="sql")
                            save_chat_turn(st.session_state.current_chat_id, st.session_state.user_id, user_question, query_result.get("query"), response, None)
                        else:
                            query = query_result["query"]
                            result = execute_query(active_conn, query)
                            
                            if not result["success"]:
                                response = f"❌ Query failed: {result.get('error', 'Unknown')}"
                                st.error(response)
                                st.warning("**Available tables:**")
                                if active_schema:
                                    for table_name in active_schema.keys():
                                        st.write(f"• {table_name}")
                                with st.expander("🔍 View Failed Query"):
                                    st.code(query, language="sql")
                                    st.markdown(create_copy_button(query, "📋 Copy"), unsafe_allow_html=True)
                                save_chat_turn(st.session_state.current_chat_id, st.session_state.user_id, user_question, query, response, None)
                            else:
                                summary, df, visualization, data_analysis = generate_db_response_with_presentation(user_question, query, result, context)
                                st.write(summary)
                                
                                if visualization:
                                    st.plotly_chart(visualization, use_container_width=True)
                                
                                # UPDATED: Dynamic display
                                if df is not None and not df.empty:
                                    if data_analysis['display_method'] == 'card' and len(df) <= 50:
                                        st.markdown(f"### 📋 Results Found ({len(df)} items)")
                                        st.caption("*Click to expand*")
                                        current_turn_idx = len(chat_history)
                                        for idx, row in df.iterrows():
                                            display_data_card(row.to_dict(), idx, current_turn_idx, data_analysis)
                                    else:
                                        with st.expander(f"📊 View Results ({len(df)} items)", expanded=True):
                                            st.dataframe(df, use_container_width=True, height=400)
                                            st.markdown(create_download_link(df, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), unsafe_allow_html=True)
                                
                                with st.expander("🔍 Query Details"):
                                    query_lower = query.lower()
                                    if "join" in query_lower:
                                        st.warning("🔗 Multi-table query")
                                    else:
                                        st.success("⚡ Single-table query")
                                    st.subheader("📝 SQL Query")
                                    st.caption(f"Database: {'SQLite' if is_sqlite else 'MySQL'}")
                                    st.code(query, language="sql")
                                    st.markdown(create_copy_button(query, "📋 Copy"), unsafe_allow_html=True)
                                    st.subheader("🎯 Intent Analysis")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Type", intent_analysis['intent_type'])
                                        st.metric("JOIN", "Yes" if intent_analysis['requires_join'] else "No")
                                    with col2:
                                        st.metric("Tables", len(intent_analysis['tables_needed']))
                                        st.write("**Reasoning:**", intent_analysis['reasoning'])
                                    st.subheader("📊 Context Stats")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total", context_stats['total_messages'])
                                    with col2:
                                        st.metric("Recent", context_stats['recent_count'])
                                    with col3:
                                        st.metric("Summarized", context_stats['summarized_count'])
                                    with col4:
                                        st.metric("Semantic", context_stats['semantic_count'])
                                
                                response = summary
                                save_chat_turn(st.session_state.current_chat_id, st.session_state.user_id, user_question, query, response, df if df is not None and not df.empty else None)
                    
                    except Exception as e:
                        response = f"❌ Error: {str(e)}"
                        st.error(response)
                        st.exception(e)
                        save_chat_turn(st.session_state.current_chat_id, st.session_state.user_id, user_question, None, response, None)
                    
                    finally:
                        if st.session_state.db_mode != "custom_mysql" and active_conn:
                            try:
                                if not is_sqlite and active_conn.is_connected():
                                    active_conn.close()
                            except:
                                pass
        st.rerun()

# ================= FOOTER =================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Meta Llama 3.3 70B")
with col2:
    db_status = {
        "system": "System DB (MySQL)",
        "custom_sqlite": "Custom SQLite",
        "custom_mysql": "Custom MySQL"
    }.get(st.session_state.db_mode, "Unknown")
    st.caption(f"🗄️ {db_status}")
with col3:
    st.caption("🧠 Context-Aware + ⚡ Smart Queries")