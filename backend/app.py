from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
import jwt
import datetime
import os
from dotenv import load_dotenv
from functools import wraps
import tempfile
from groq import Groq

# 🔹 LangChain imports (only for document processing)
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 🔹 Load environment variables
load_dotenv()

app = Flask(__name__)

# 🔹 Allow cross-origin (React frontend)
# Allow cross-origin (React frontend). Explicitly allow Authorization header so preflight succeeds.
CORS(app,
    supports_credentials=True,
    resources={r"/api/*": {"origins": "http://localhost:3000"}},
    allow_headers=["Content-Type", "Authorization"]
)

# 🔹 Secrets and config
SECRET_KEY = os.getenv("SECRET_KEY", "mysecretkey")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# ===================== MONGODB CONNECTION =====================
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    exit()

db = client['auth_app']
users_collection = db['users']
documents_collection = db['documents']

# ===================== JWT TOKEN FUNCTIONS =====================
def generate_token(user_id, role):
    payload = {
        'id': str(user_id),
        'role': role,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Allow preflight OPTIONS requests to pass without authentication
        if request.method == 'OPTIONS':
            return ('', 200)
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            current_user = users_collection.find_one({"_id": ObjectId(decoded['id'])})
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except Exception as e:
            return jsonify({'error': f'Invalid token: {str(e)}'}), 401
        return f(current_user, *args, **kwargs)
    return decorated


# ===================== AUTH ROUTES =====================
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    if users_collection.find_one({'username': username}):
        return jsonify({'error': 'Username already exists'}), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_id = users_collection.insert_one({
        'username': username,
        'password': hashed_pw,
        'role': role
    }).inserted_id

    token = generate_token(user_id, role)
    return jsonify({'token': token, 'role': role}), 200


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = users_collection.find_one({'username': username})
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'error': 'Invalid username or password'}), 401

    token = generate_token(user['_id'], user['role'])
    return jsonify({'token': token, 'role': user['role']}), 200


@app.route('/api/profile', methods=['GET'])
@token_required
def profile(current_user):
    return jsonify({'username': current_user['username'], 'role': current_user['role']}), 200


# ===================== RAG PDF / QUERY ROUTES =====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Groq API key missing"

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Global variables for vector store
vector_store = None
embeddings_model = None


def process_pdf(pdf_path):
    """Process PDF and create vector store"""
    global vector_store, embeddings_model
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, 
        chunk_overlap=500
    )
    texts = text_splitter.split_text("\n\n".join(p.page_content for p in pages))

    # Create embeddings model (runs locally, no API needed)
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create vector store
    vector_store = Chroma.from_texts(texts, embeddings_model)
    
    return len(texts)


def query_groq_with_context(query_text, context_docs):
    """Query Groq API with retrieved context"""
    
    # Combine context documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create prompt
    prompt = f"""Context: {context}

Question: {query_text}

Answer the question using the context above. Format your answer clearly and concisely.
Always end with: "Thanks for asking!"

Answer:"""

    # Call Groq API directly
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768", "llama-3.1-70b-versatile"
        temperature=0.3,
        max_tokens=1024,
    )
    
    return chat_completion.choices[0].message.content


@app.route('/api/upload-pdf', methods=['POST', 'OPTIONS'])
@token_required
def upload_pdf(current_user):
    if request.method == 'OPTIONS':
        return '', 200

    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file found"}), 400

    file = request.files['pdf']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        chunks_count = process_pdf(tmp_path)

        # Save document metadata and empty history
        try:
            doc = {
                'user_id': current_user['_id'],
                'username': current_user['username'],
                'filename': file.filename,
                'uploaded_at': datetime.datetime.utcnow(),
                'chunks_processed': chunks_count,
                'history': []
            }
            inserted = documents_collection.insert_one(doc)
            doc_id = str(inserted.inserted_id)
        except Exception as e:
            # If DB write fails, still return success for processing but warn
            doc_id = None

        return jsonify({
            "message": "PDF processed successfully", 
            "chunks_processed": chunks_count,
            "doc_id": doc_id
        }), 200

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/api/query', methods=['POST'])
@token_required
def query(current_user):
    data = request.get_json()
    if not data or 'query' not in data or not data['query'].strip():
        return jsonify({"error": "Empty or invalid query"}), 400

    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({"error": "doc_id is required"}), 400

    if not vector_store:
        return jsonify({"error": "Upload PDF first"}), 400

    try:
        # Retrieve relevant documents
        query_text = data['query']
        relevant_docs = vector_store.similarity_search(query_text, k=3)
        
        # Query Groq with context
        answer = query_groq_with_context(query_text, relevant_docs)

        # Append to document history (best-effort)
        try:
            documents_collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$push": {"history": {
                    "query": query_text,
                    "answer": answer,
                    "timestamp": datetime.datetime.utcnow(),
                    "user": current_user['username']
                }}}
            )
        except Exception:
            # ignore errors writing history
            pass

        return jsonify({
            "answer": answer,
            "sources": [{"page": doc.metadata.get("page", "unknown")} for doc in relevant_docs]
        }), 200

    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@app.route('/api/documents/<doc_id>/history', methods=['GET'])
@token_required
def get_document_history(current_user, doc_id):
    try:
        doc = documents_collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Only allow owner to read history
        owner_id = doc.get('user_id')
        if owner_id != current_user['_id']:
            return jsonify({"error": "Forbidden"}), 403

        history = doc.get('history', [])
        # Serialize timestamps
        for item in history:
            if isinstance(item.get('timestamp'), datetime.datetime):
                item['timestamp'] = item['timestamp'].isoformat()

        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)