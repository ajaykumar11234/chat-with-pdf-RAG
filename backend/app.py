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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Configuration for Auth
SECRET_KEY = os.getenv("SECRET_KEY", "mysecretkey")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Force connection on server
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    exit()

db = client['auth_app']
users_collection = db['users']

# JWT Token Generator
def generate_token(user_id, role):
    payload = {
        'id': str(user_id),
        'role': role,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

# Auth Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].replace('Bearer ', '')

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

# =================== AUTH ROUTES ===================

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
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
    except Exception as e:
        print(f"[ERROR] /signup: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        user = users_collection.find_one({'username': username})
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'error': 'Invalid username or password'}), 401

        token = generate_token(user['_id'], user['role'])
        return jsonify({'token': token, 'role': user['role']}), 200
    except Exception as e:
        print(f"[ERROR] /login: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/api/profile', methods=['GET'])
@token_required
def profile(current_user):
    return jsonify({
        'username': current_user['username'],
        'role': current_user['role']
    }), 200

# =================== RAG PDF / QUERY ROUTES ===================

# Google API Key & model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB6Vr4ItMiOgiCPR4zcKyAJKpZZpTqAfk0")
assert GOOGLE_API_KEY, "Google API key missing"
MODEL_NAME = "gemini-1.5-flash-latest"

qa_chain = None  # Global variable for RAG chain

def generate_chain_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    texts = text_splitter.split_text("\n\n".join(p.page_content for p in pages))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(
        search_kwargs={"k": 3}
    )

    prompt_template = PromptTemplate.from_template("""
    Context: {context}
    Question: {question}

    Answer the question using the context above. 
    Format your answer clearly and concisely.
    Always end with: "Thanks for asking!"
    Answer:
    """)

    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    return RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

@app.route('/api/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    if request.method == 'OPTIONS':
        # Allow preflight CORS request
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

        global qa_chain
        qa_chain = generate_chain_from_pdf(tmp_path)

        pages_count = len(qa_chain.retriever.get_relevant_documents("test"))
        return jsonify({
            "message": "PDF processed successfully",
            "pages_processed": pages_count
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.route('/api/query', methods=['POST'])
def query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data or 'query' not in data or not data['query'].strip():
        return jsonify({"error": "Empty or invalid query"}), 400

    if not qa_chain:
        return jsonify({"error": "Upload PDF first"}), 400

    try:
        result = qa_chain.invoke({"query": data['query']})
        return jsonify({
            "answer": result['result'],
            "sources": [doc.metadata for doc in result['source_documents']]
        })
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
