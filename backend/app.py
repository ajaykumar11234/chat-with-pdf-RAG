from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId
import bcrypt
import jwt
import datetime
import os
from dotenv import load_dotenv
from functools import wraps
import tempfile
import pickle
import shutil
from groq import Groq

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# -------------------- Load env --------------------
load_dotenv()

app = Flask(__name__)

# Configure CORS - reads from environment variable for production
ALLOWED_ORIGINS = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS(app,
     supports_credentials=True,
     resources={r"/api/*": {"origins": ALLOWED_ORIGINS.split(",")}},
     allow_headers=["Content-Type", "Authorization"]
)

SECRET_KEY = os.getenv("SECRET_KEY", "mysecretkey")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PORT = int(os.getenv("PORT", 5001))

# Max upload size (50MB)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_CONTENT_LENGTH", 52428800))

if not GROQ_API_KEY:
    print("⚠️  Warning: Groq API key missing")

# -------------------- MongoDB --------------------
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    exit(1)

db = client['auth_app']
fs = GridFS(db)
users_collection = db['users']
documents_collection = db['documents']
# New collection to store vector embeddings and chunks
vectors_collection = db['vectors']

# -------------------- Groq client --------------------
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# -------------------- JWT helpers --------------------
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

# -------------------- Health check --------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "mongodb": "connected",
        "groq": "configured" if groq_client else "not configured"
    }), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "PDF Query API - MongoDB Vector Storage",
        "storage": "vectors stored in MongoDB",
        "endpoints": {
            "health": "/health",
            "signup": "/api/signup",
            "login": "/api/login",
            "profile": "/api/profile",
            "upload": "/api/upload-pdf",
            "documents": "/api/documents",
            "query": "/api/query"
        }
    }), 200

# -------------------- Auth routes --------------------
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

# -------------------- Embeddings helper --------------------
def get_embeddings():
    """Return an instance of the HuggingFaceEmbeddings used by the app."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# -------------------- Vector storage in MongoDB --------------------
def store_vectors_in_mongodb(doc_id, texts, embeddings_list, metadatas):
    """
    Store text chunks and their embeddings in MongoDB.
    
    Args:
        doc_id: Document ID from documents_collection
        texts: List of text chunks
        embeddings_list: List of embedding vectors
        metadatas: List of metadata dicts for each chunk
    """
    vectors_to_insert = []
    
    for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings_list, metadatas)):
        vector_doc = {
            'doc_id': str(doc_id),
            'chunk_index': i,
            'text': text,
            'embedding': embedding,  # Store as list
            'metadata': metadata,
            'created_at': datetime.datetime.utcnow()
        }
        vectors_to_insert.append(vector_doc)
    
    if vectors_to_insert:
        vectors_collection.insert_many(vectors_to_insert)
        print(f"✅ Stored {len(vectors_to_insert)} vectors in MongoDB")


def load_vectors_from_mongodb(doc_id):
    """
    Load vectors from MongoDB for a given document.
    
    Returns:
        tuple: (texts, embeddings, metadatas)
    """
    cursor = vectors_collection.find({'doc_id': str(doc_id)}).sort('chunk_index', 1)
    
    texts = []
    embeddings = []
    metadatas = []
    
    for vec_doc in cursor:
        texts.append(vec_doc['text'])
        embeddings.append(vec_doc['embedding'])
        metadatas.append(vec_doc.get('metadata', {}))
    
    return texts, embeddings, metadatas


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def similarity_search_mongodb(doc_id, query_text, k=3):
    """
    Perform similarity search using MongoDB stored vectors.
    
    Args:
        doc_id: Document ID
        query_text: Query string
        k: Number of results to return
        
    Returns:
        List of Document objects with page_content and metadata
    """
    # Get query embedding
    embeddings_model = get_embeddings()
    query_embedding = embeddings_model.embed_query(query_text)
    
    # Load all vectors for this document
    texts, embeddings, metadatas = load_vectors_from_mongodb(doc_id)
    
    if not texts:
        return []
    
    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((i, sim))
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    
    # Create Document objects
    results = []
    for idx, sim_score in top_k:
        doc = Document(
            page_content=texts[idx],
            metadata=metadatas[idx]
        )
        results.append(doc)
    
    return results


def process_pdf_and_store_in_mongodb(pdf_path, doc_id):
    """
    Load PDF, split into chunks, generate embeddings, and store in MongoDB.
    
    Args:
        pdf_path: Path to PDF file
        doc_id: Document ID from documents_collection
        
    Returns:
        Number of chunks processed
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    
    # Create chunks with metadata
    chunks = []
    metadatas = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.page_content)
        for chunk in page_chunks:
            chunks.append(chunk)
            metadatas.append({
                'page': page.metadata.get('page', 0),
                'source': page.metadata.get('source', 'unknown')
            })
    
    # Generate embeddings
    embeddings_model = get_embeddings()
    embeddings_list = embeddings_model.embed_documents(chunks)
    
    # Store in MongoDB
    store_vectors_in_mongodb(doc_id, chunks, embeddings_list, metadatas)
    
    return len(chunks)

# -------------------- Groq query helper --------------------
def query_groq_with_context(query_text, context_docs):
    if not groq_client:
        return "Error: Groq API is not configured. Please contact administrator."
    
    context = "\n\n".join(getattr(doc, 'page_content', str(doc)) for doc in context_docs)

    prompt = f"""Context: {context}\n\nQuestion: {query_text}\n\nAnswer the question using the context above. Format your answer clearly and concisely. Always end with: \"Thanks for asking!\"\n\nAnswer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content

# -------------------- Upload endpoint --------------------
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

    tmp_path = None
    try:
        # Save raw PDF into GridFS
        file_id = fs.put(file, filename=file.filename)
        print(f"✅ PDF saved to GridFS with ID: {file_id}")

        # Create temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.stream.seek(0)
            tmp.write(file.stream.read())
            tmp.flush()
            tmp_path = tmp.name

        # Create document record first (so we have doc_id)
        doc = {
            'user_id': current_user['_id'],
            'username': current_user['username'],
            'filename': file.filename,
            'file_id': file_id,
            'uploaded_at': datetime.datetime.utcnow(),
            'chunks_processed': 0,
            'history': [],
            'storage_type': 'mongodb'  # Indicator that vectors are in MongoDB
        }
        inserted = documents_collection.insert_one(doc)
        doc_id = inserted.inserted_id
        print(f"✅ Document record created with ID: {doc_id}")

        # Process PDF and store vectors in MongoDB
        print(f"🔄 Processing PDF and generating embeddings...")
        chunks_count = process_pdf_and_store_in_mongodb(tmp_path, doc_id)
        
        # Update document with chunk count
        documents_collection.update_one(
            {"_id": doc_id},
            {"$set": {"chunks_processed": chunks_count}}
        )

        print(f"✅ Upload complete: {chunks_count} chunks processed")
        
        return jsonify({
            "message": "PDF uploaded & processed successfully",
            "doc_id": str(doc_id),
            "chunks_processed": chunks_count,
            "storage": "mongodb"
        }), 200

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# -------------------- List documents for user --------------------
@app.route('/api/documents', methods=['GET'])
@token_required
def list_documents(current_user):
    docs = documents_collection.find({'user_id': current_user['_id']})
    out = []
    for d in docs:
        out.append({
            'doc_id': str(d['_id']),
            'filename': d.get('filename'),
            'uploaded_at': d.get('uploaded_at').isoformat() if isinstance(d.get('uploaded_at'), datetime.datetime) else d.get('uploaded_at'),
            'chunks': d.get('chunks_processed', 0),
            'storage': d.get('storage_type', 'mongodb')
        })
    return jsonify({'documents': out}), 200

# -------------------- Download PDF from GridFS --------------------
@app.route('/api/download-pdf/<doc_id>', methods=['GET'])
@token_required
def download_pdf(current_user, doc_id):
    try:
        doc = documents_collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Check ownership
        if doc.get('user_id') != current_user['_id']:
            return jsonify({"error": "Forbidden"}), 403

        file_id = doc.get('file_id')
        if not file_id:
            return jsonify({"error": "File not found for this document"}), 404

        grid_out = fs.get(file_id)
        return Response(
            grid_out.read(),
            mimetype='application/pdf',
            headers={"Content-Disposition": f"attachment;filename={doc.get('filename', 'file.pdf')}"}
        )

    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 500

@app.route('/download/<doc_id>', methods=['GET'])
def download_file(doc_id):
    """Public download endpoint (no auth required)"""
    try:
        doc = documents_collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        grid_out = fs.get(doc['file_id'])

        return Response(
            grid_out.read(),
            mimetype='application/pdf',
            headers={
                "Content-Disposition": f"attachment; filename={doc.get('filename', 'file.pdf')}"
            }
        )

    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

# -------------------- Query endpoint --------------------
@app.route('/api/query', methods=['POST'])
@token_required
def query(current_user):
    data = request.get_json()
    if not data or 'query' not in data or not data['query'].strip():
        return jsonify({"error": "Empty or invalid query"}), 400

    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({"error": "doc_id is required"}), 400

    try:
        doc = documents_collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Only owner may query their document
        if doc.get('user_id') != current_user['_id']:
            return jsonify({"error": "Forbidden"}), 403

        query_text = data['query']
        print(f"🔍 Searching for: {query_text}")
        
        # Perform similarity search using MongoDB
        relevant_docs = similarity_search_mongodb(str(doc['_id']), query_text, k=3)
        
        if not relevant_docs:
            return jsonify({
                "error": "No relevant content found in document"
            }), 404

        print(f"✅ Found {len(relevant_docs)} relevant chunks")

        # Generate answer using Groq
        answer = query_groq_with_context(query_text, relevant_docs)

        # Save to history
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)}, 
            {"$push": {"history": {
                "query": query_text,
                "answer": answer,
                "timestamp": datetime.datetime.utcnow(),
                "user": current_user['username']
            }}}
        )

        # Extract sources
        sources = []
        for doc_hit in relevant_docs:
            try:
                page = doc_hit.metadata.get('page', 'unknown')
            except Exception:
                page = 'unknown'
            sources.append({'page': page})

        return jsonify({
            "answer": answer, 
            "sources": sources,
            "chunks_found": len(relevant_docs)
        }), 200

    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

# -------------------- Document history --------------------
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
        print(f"❌ History error: {str(e)}")
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

# -------------------- Delete document endpoint --------------------
@app.route('/api/documents/<doc_id>', methods=['DELETE'])
@token_required
def delete_document(current_user, doc_id):
    try:
        doc = documents_collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Check ownership
        if doc.get('user_id') != current_user['_id']:
            return jsonify({"error": "Forbidden"}), 403

        # Delete PDF from GridFS
        if doc.get('file_id'):
            fs.delete(doc['file_id'])
            print(f"✅ Deleted PDF from GridFS")

        # Delete vectors from MongoDB
        result = vectors_collection.delete_many({'doc_id': str(doc_id)})
        print(f"✅ Deleted {result.deleted_count} vectors from MongoDB")

        # Delete document record
        documents_collection.delete_one({"_id": ObjectId(doc_id)})
        print(f"✅ Deleted document record")

        return jsonify({"message": "Document deleted successfully"}), 200

    except Exception as e:
        print(f"❌ Delete error: {str(e)}")
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500

# -------------------- Error handlers --------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 50MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# -------------------- Run --------------------
if __name__ == '__main__':
    print(f"🚀 Starting server on port {PORT}")
    print(f"📦 Vector storage: MongoDB")
    print(f"🔗 CORS allowed origins: {ALLOWED_ORIGINS}")
    app.run(debug=os.getenv("FLASK_ENV") == "development", host='0.0.0.0', port=PORT)