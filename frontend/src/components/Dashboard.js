import React, { useState, useEffect } from 'react';

const Dashboard = ({ role, onLogout, token }) => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [documentHistory, setDocumentHistory] = useState({});

  const backendUrl = process.env.REACT_APP_BACKEND_URL;


  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${backendUrl}/documents`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok) {
        setDocuments(data.documents || []);
        if (data.documents && data.documents.length > 0 && !selectedDocId) {
          setSelectedDocId(data.documents[0].doc_id);
          fetchDocumentHistory(data.documents[0].doc_id);
        }
      } else {
        console.error('Failed to fetch documents:', data.error);
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  };

  const fetchDocumentHistory = async (docId) => {
    try {
      const res = await fetch(`${backendUrl}/documents/${docId}/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok) {
        setDocumentHistory(prev => ({
          ...prev,
          [docId]: data.history || []
        }));
      } else {
        console.error('Failed to fetch history:', data.error);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  };

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file');
      return;
    }

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      const res = await fetch(`${backendUrl}/upload-pdf`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setUploadStatus('File uploaded and processed successfully');
        setFile(null);
        await fetchDocuments();
        if (data.doc_id) {
          setSelectedDocId(data.doc_id);
          setDocumentHistory(prev => ({
            ...prev,
            [data.doc_id]: []
          }));
        }
      } else {
        setUploadStatus(data.error);
      }
    } catch (err) {
      setUploadStatus('Upload failed');
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      setResponse('Please enter a query');
      return;
    }

    if (!selectedDocId) {
      setResponse('Please select a document first');
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch(`${backendUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query, doc_id: selectedDocId }),
      });

      const data = await res.json();
      if (res.ok) {
        setResponse(data.answer);
        await fetchDocumentHistory(selectedDocId);
      } else {
        setResponse(data.error);
      }
    } catch (err) {
      setResponse('Query failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentSelect = (docId) => {
    setSelectedDocId(docId);
    setResponse('');
    setQuery('');
    if (!documentHistory[docId]) {
      fetchDocumentHistory(docId);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-200">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                📚 PDF Q&A Assistant
              </h1>
              <p className="text-gray-600 mt-2">Upload PDFs and ask questions about your documents</p>
            </div>
            <button
              onClick={onLogout}
              className="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-all duration-200 shadow-lg shadow-red-500/25 font-medium"
            >
              Logout
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Sidebar - Documents */}
          <div className="lg:col-span-1 space-y-8">
            {/* Document List */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <span className="w-6 h-6 mr-2">📁</span>
                Your Documents
              </h3>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {documents.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <div className="text-4xl mb-2">📄</div>
                    <p>No documents yet</p>
                    <p className="text-sm">Upload your first PDF to get started</p>
                  </div>
                ) : (
                  documents.map((doc) => (
                    <div
                      key={doc.doc_id}
                      onClick={() => handleDocumentSelect(doc.doc_id)}
                      className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                        selectedDocId === doc.doc_id
                          ? 'border-blue-500 bg-blue-50 shadow-md'
                          : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 truncate">{doc.filename}</h4>
                          <div className="text-sm text-gray-500 mt-1">
                            Uploaded: {new Date(doc.uploaded_at).toLocaleDateString()}
                          </div>
                          <div className="text-xs text-blue-600 font-medium mt-1">
                            Chunks: {doc.chunks}
                          </div>
                        </div>
                        {selectedDocId === doc.doc_id && (
                          <span className="bg-blue-500 text-white px-2 py-1 rounded-full text-xs font-medium">
                            Active
                          </span>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Upload Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <span className="w-6 h-6 mr-2">📤</span>
                Upload PDF
              </h3>

              <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer ${
                  isDragging
                    ? 'border-blue-500 bg-blue-50 scale-105'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="text-4xl mb-4">📁</div>
                <p className="font-semibold text-gray-900 mb-2">
                  Drop your PDF here or click to browse
                </p>
                <p className="text-sm text-gray-500 mb-4">
                  Supports PDF files up to 10MB
                </p>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-input"
                />
                <label
                  htmlFor="file-input"
                  className="inline-block px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 cursor-pointer shadow-lg shadow-blue-500/25 font-medium"
                >
                  Choose File
                </label>
              </div>

              {file && (
                <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center">
                  <span className="text-green-600 mr-2">📎</span>
                  <span className="text-green-800 font-medium truncate">{file.name}</span>
                </div>
              )}

              <button
                onClick={handleUpload}
                disabled={!file}
                className="w-full mt-4 px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-green-500/25 font-medium"
              >
                Upload & Process
              </button>

              {uploadStatus && (
                <div className={`mt-4 p-3 rounded-lg text-center font-medium ${
                  uploadStatus.includes('successfully')
                    ? 'bg-green-50 text-green-800 border border-green-200'
                    : 'bg-red-50 text-red-800 border border-red-200'
                }`}>
                  {uploadStatus}
                </div>
              )}
            </div>
          </div>

          {/* Main Content - Q&A */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <span className="w-6 h-6 mr-2">❓</span>
                Ask Questions
                {selectedDocId && (
                  <span className="text-sm font-normal text-gray-500 ml-2">
                    (Currently selected: {documents.find(d => d.doc_id === selectedDocId)?.filename})
                  </span>
                )}
              </h3>

              {!selectedDocId && (
                <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  {documents.length > 0 
                    ? 'Please select a document from the list to start asking questions.'
                    : 'Please upload a PDF document first to ask questions.'
                  }
                </div>
              )}

              <div className="space-y-4">
                <textarea
                  rows={4}
                  placeholder={
                    selectedDocId 
                      ? "Enter your query here... Ask anything about the selected document!"
                      : "Please select a document first to ask questions."
                  }
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={!selectedDocId}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 resize-none disabled:bg-gray-100 disabled:cursor-not-allowed"
                />

                <button
                  onClick={handleQuery}
                  disabled={isLoading || !selectedDocId}
                  className="w-full px-6 py-4 bg-purple-500 text-white rounded-lg hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/25 font-medium flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Processing...
                    </>
                  ) : (
                    'Submit Query'
                  )}
                </button>
              </div>

              {response && (
                <div className="mt-6 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                  <h4 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <span className="w-5 h-5 mr-2">💡</span>
                    Response
                  </h4>
                  <div className="prose prose-blue max-w-none">
                    {response.split('\n').map((line, i) => (
                      <p key={i} className="text-gray-700 leading-relaxed mb-2">{line}</p>
                    ))}
                  </div>
                </div>
              )}

              {/* Conversation History */}
              {selectedDocId && documentHistory[selectedDocId] && documentHistory[selectedDocId].length > 0 && (
                <div className="mt-8">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <span className="w-5 h-5 mr-2">📝</span>
                    Conversation History
                  </h4>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {documentHistory[selectedDocId]
                      .slice()
                      .reverse()
                      .map((item, i) => (
                        <div key={i} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                          <div className="font-semibold text-gray-900 mb-2">Q: {item.query}</div>
                          <div className="text-gray-700 mb-2">{item.answer}</div>
                          <div className="text-xs text-gray-500">
                            {new Date(item.timestamp).toLocaleString()}
                          </div>
                        </div>
                      ))
                    }
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;