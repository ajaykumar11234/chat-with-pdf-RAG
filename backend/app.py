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
from groq import Groq

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------- Load env --------------------
load_dotenv()

app = Flask(__name__)

# Configure CORS - update origins for production
CORS(app,
     supports_credentials=True,
     resources={r"/api/*": {"origins": "http://localhost:3000"}},
     allow_headers=["Content-Type", "Authorization"]
)

SECRET_KEY = os.getenv("SECRET_KEY", "mysecretkey")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    # warning - Groq is required in your original code. If you don't want to use it,
    # replace query_groq_with_context with another LLM call.
    raise AssertionError("Groq API key missing")

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

# -------------------- Groq client --------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# Global state - NOTE: in production this should be per-document and not global
# we'll load vectorstore per-document using helper functions

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

# -------------------- Helpers for embeddings & vector store --------------------
def get_embeddings():
    """Return an instance of the HuggingFaceEmbeddings used by the app."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def process_pdf_to_vectorstore(pdf_path, persist_dir):
    """Load PDF from path, split, create embeddings and a persisted Chroma vectorstore.
    Returns number of chunks created.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    texts = text_splitter.split_text("\n\n".join(p.page_content for p in pages))

    embeddings_model = get_embeddings()

    # Create Chroma vectorstore and persist it
    vector_store = Chroma.from_texts(texts, embeddings_model, persist_directory=persist_dir)
    vector_store.persist()

    return len(texts)


def load_vector_store_for_document(doc_record):
    """Given a document record (from Mongo), ensure vectorstore exists and return a Chroma instance.
    If vectorstore directory is missing, re-generate from the stored PDF.
    """
    persist_dir = doc_record.get('vector_folder')
    if not persist_dir:
        raise FileNotFoundError('Vector folder not recorded for document')

    embeddings_model = get_embeddings()

    # If persist_dir doesn't exist on disk, attempt to recreate from GridFS stored PDF
    if not os.path.exists(persist_dir):
        # recreate directory
        os.makedirs(persist_dir, exist_ok=True)
        # retrieve PDF from GridFS and reprocess
        file_id = doc_record.get('file_id')
        if not file_id:
            raise FileNotFoundError('No file_id in document record to recreate vector store')
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        try:
            grid_out = fs.get(file_id)
            tmp.write(grid_out.read())
            tmp.flush()
            tmp.close()
            process_pdf_to_vectorstore(tmp.name, persist_dir)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # load persisted Chroma
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings_model)
    return vector_store

# -------------------- Groq query helper --------------------
def query_groq_with_context(query_text, context_docs):
    context = "\n\n".join(getattr(doc, 'page_content', str(doc)) for doc in context_docs)

    prompt = f"""Context: {context}\n\nQuestion: {query_text}\n\nAnswer the question using the context above. Format your answer clearly and concisely. Always end with: \"Thanks for asking!\"\n\nAnswer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content

# -------------------- Upload endpoint (stores PDF in GridFS + persists vectorstore) --------------------
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
        # Save raw PDF into GridFS
        file_id = fs.put(file, filename=file.filename)

        # create temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.stream.seek(0)
            tmp.write(file.stream.read())
            tmp.flush()
            tmp_path = tmp.name

        # prepare directory to persist vectorstore
        vector_folder = os.path.join('vectorstores', str(file_id))
        os.makedirs(vector_folder, exist_ok=True)

        chunks_count = process_pdf_to_vectorstore(tmp_path, vector_folder)

        # record in documents collection
        doc = {
            'user_id': current_user['_id'],
            'username': current_user['username'],
            'filename': file.filename,
            'file_id': file_id,
            'vector_folder': vector_folder,
            'uploaded_at': datetime.datetime.utcnow(),
            'chunks_processed': chunks_count,
            'history': []
        }
        inserted = documents_collection.insert_one(doc)
        doc_id = str(inserted.inserted_id)

        return jsonify({
            "message": "PDF uploaded & processed",
            "doc_id": doc_id,
            "chunks_processed": chunks_count
        }), 200

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
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
            'chunks': d.get('chunks_processed', 0)
        })
    return jsonify({'documents': out}), 200

# # -------------------- Download PDF from GridFS --------------------
# @app.route('/api/download-pdf/<doc_id>', methods=['GET'])
# @token_required
# def download_pdf(current_user, doc_id):
#     try:
#         doc = documents_collection.find_one({"_id": ObjectId(doc_id)})
#         if not doc:
#             return jsonify({"error": "Document not found"}), 404

#         file_id = doc.get('file_id')
#         if not file_id:
#             return jsonify({"error": "File not found for this document"}), 404

#         grid_out = fs.get(file_id)
#         return Response(grid_out.read(), mimetype='application/pdf', headers={"Content-Disposition": f"attachment;filename={doc.get('filename', 'file.pdf')}")

#     except Exception as e:
#         return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 500

# -------------------- Query endpoint (load vectorstore per doc + call Groq) --------------------
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

        # load or recreate vector store
        vector_store = load_vector_store_for_document(doc)

        query_text = data['query']
        relevant_docs = vector_store.similarity_search(query_text, k=3)

        answer = query_groq_with_context(query_text, relevant_docs)

        # push history
        documents_collection.update_one({"_id": ObjectId(doc_id)}, {"$push": {"history": {
            "query": query_text,
            "answer": answer,
            "timestamp": datetime.datetime.utcnow(),
            "user": current_user['username']
        }}})

        sources = []
        for doc_hit in relevant_docs:
            # try to get page metadata safely
            try:
                page = doc_hit.metadata.get('page', 'unknown')
            except Exception:
                page = 'unknown'
            sources.append({'page': page})

        return jsonify({"answer": answer, "sources": sources}), 200

    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

@app.route('/download/<doc_id>', methods=['GET'])
def download_file(doc_id):
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
        print("Download error:", e)
        return jsonify({"error": "Download failed"}), 500


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
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500

# -------------------- Run --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)
