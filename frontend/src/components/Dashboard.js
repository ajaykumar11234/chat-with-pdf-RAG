import React, { useState } from 'react';

const Dashboard = ({ role, onLogout, token }) => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [docId, setDocId] = useState(null);
  const [history, setHistory] = useState([]);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [hoveredButton, setHoveredButton] = useState('');

  const backendUrl = 'http://localhost:5001/api';

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
        if (data.doc_id) {
          setDocId(data.doc_id);
          setHistory([]); // clear local history for this new doc
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

    setIsLoading(true);
    try {
      const res = await fetch(`${backendUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query, doc_id: docId }),
      });

      const data = await res.json();
      if (res.ok) {
        setResponse(data.answer);
        // update local history
        setHistory((h) => [{ query, answer: data.answer, ts: new Date().toISOString() }, ...h]);
      } else {
        setResponse(data.error);
      }
    } catch (err) {
      setResponse('Query failed');
    } finally {
      setIsLoading(false);
    }
  };

  const styles = {
    dashboard: {
      minHeight: '100vh',
      backgroundColor: '#ffffff',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    },
    container: {
      maxWidth: '1200px',
      margin: '0 auto',
      backgroundColor: '#ffffff',
      borderRadius: '20px',
      padding: '40px',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
      border: '1px solid rgba(0, 0, 0, 0.1)'
    },
    header: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '40px',
      paddingBottom: '20px',
      borderBottom: '2px solid #e5e7eb'
    },
    title: {
      fontSize: '32px',
      fontWeight: '700',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text',
      margin: 0
    },
    logoutButton: {
      padding: '12px 24px',
      backgroundColor: '#ef4444',
      color: 'white',
      border: 'none',
      borderRadius: '12px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 15px rgba(239, 68, 68, 0.3)'
    },
    logoutButtonHover: {
      backgroundColor: '#dc2626',
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(239, 68, 68, 0.4)'
    },
    section: {
      background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
      borderRadius: '16px',
      padding: '30px',
      marginBottom: '30px',
      border: '1px solid rgba(255, 255, 255, 0.5)',
      boxShadow: '0 8px 25px rgba(0, 0, 0, 0.08)'
    },
    sectionTitle: {
      fontSize: '24px',
      fontWeight: '600',
      color: '#1f2937',
      marginBottom: '20px',
      display: 'flex',
      alignItems: 'center',
      gap: '10px'
    },
    uploadArea: {
      border: '2px dashed #cbd5e1',
      borderRadius: '12px',
      padding: '40px',
      textAlign: 'center',
      transition: 'all 0.3s ease',
      cursor: 'pointer',
      background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      marginBottom: '20px'
    },
    uploadAreaDragging: {
      borderColor: '#3b82f6',
      backgroundColor: '#eff6ff',
      transform: 'scale(1.02)'
    },
    uploadAreaHover: {
      borderColor: '#64748b',
      backgroundColor: '#f1f5f9'
    },
    fileInput: {
      display: 'none'
    },
    fileLabel: {
      display: 'inline-block',
      padding: '12px 24px',
      backgroundColor: '#3b82f6',
      color: 'white',
      borderRadius: '10px',
      cursor: 'pointer',
      fontSize: '16px',
      fontWeight: '600',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 15px rgba(59, 130, 246, 0.3)'
    },
    fileLabelHover: {
      backgroundColor: '#2563eb',
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(59, 130, 246, 0.4)'
    },
    uploadButton: {
      padding: '14px 28px',
      backgroundColor: '#10b981',
      color: 'white',
      border: 'none',
      borderRadius: '12px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 15px rgba(16, 185, 129, 0.3)',
      marginTop: '15px'
    },
    uploadButtonHover: {
      backgroundColor: '#059669',
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(16, 185, 129, 0.4)'
    },
    textarea: {
      width: '100%',
      padding: '16px',
      border: '2px solid #e5e7eb',
      borderRadius: '12px',
      fontSize: '16px',
      fontFamily: 'inherit',
      resize: 'vertical',
      transition: 'all 0.3s ease',
      background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      outline: 'none',
      marginBottom: '20px'
    },
    textareaFocus: {
      borderColor: '#3b82f6',
      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
      transform: 'scale(1.01)'
    },
    submitButton: {
      padding: '14px 28px',
      backgroundColor: '#8b5cf6',
      color: 'white',
      border: 'none',
      borderRadius: '12px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 15px rgba(139, 92, 246, 0.3)',
      position: 'relative',
      overflow: 'hidden'
    },
    submitButtonHover: {
      backgroundColor: '#7c3aed',
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(139, 92, 246, 0.4)'
    },
    submitButtonDisabled: {
      backgroundColor: '#9ca3af',
      cursor: 'not-allowed',
      transform: 'none',
      boxShadow: '0 2px 8px rgba(156, 163, 175, 0.2)'
    },
    loadingSpinner: {
      display: 'inline-block',
      width: '20px',
      height: '20px',
      border: '2px solid #ffffff',
      borderRadius: '50%',
      borderTopColor: 'transparent',
      animation: 'spin 1s linear infinite',
      marginRight: '8px'
    },
    response: {
      marginTop: '25px',
      padding: '20px',
      background: 'linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%)',
      borderRadius: '12px',
      border: '1px solid #e0e7ff',
      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.05)',
      maxHeight: '400px',
      overflowY: 'auto'
    },
    responseParagraph: {
      margin: '0 0 10px 0',
      lineHeight: '1.6',
      color: '#374151'
    },
    status: {
      padding: '12px 16px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
      marginTop: '15px',
      textAlign: 'center'
    },
    statusSuccess: {
      backgroundColor: '#d1fae5',
      color: '#065f46',
      border: '1px solid #a7f3d0'
    },
    statusError: {
      backgroundColor: '#fee2e2',
      color: '#991b1b',
      border: '1px solid #fca5a5'
    },
    fileName: {
      marginTop: '10px',
      padding: '8px 12px',
      backgroundColor: '#eff6ff',
      color: '#1e40af',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: '500',
      display: 'inline-block'
    }
  };

  const [focusedTextarea, setFocusedTextarea] = useState(false);
  const [hoveredUploadArea, setHoveredUploadArea] = useState(false);

  return (
    <div style={styles.dashboard}>
      <div style={styles.container}>
        <div style={styles.header}>
          <h2 style={styles.title}>
            {'� PDF Q&A'}
          </h2>
          <button
            onClick={onLogout}
            onMouseEnter={() => setHoveredButton('logout')}
            onMouseLeave={() => setHoveredButton('')}
            style={{
              ...styles.logoutButton,
              ...(hoveredButton === 'logout' ? styles.logoutButtonHover : {})
            }}
          >
            Logout
          </button>
        </div>

        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>📄 Upload PDF Document</h3>
          <div
            style={{
              ...styles.uploadArea,
              ...(isDragging ? styles.uploadAreaDragging : {}),
              ...(hoveredUploadArea && !isDragging ? styles.uploadAreaHover : {})
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onMouseEnter={() => setHoveredUploadArea(true)}
            onMouseLeave={() => setHoveredUploadArea(false)}
          >
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>📁</div>
            <p style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px', color: '#1f2937' }}>
              Drop your PDF here or click to browse
            </p>
            <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '20px' }}>
              Supports PDF files up to 10MB
            </p>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              style={styles.fileInput}
              id="file-input"
            />
            <label
              htmlFor="file-input"
              onMouseEnter={() => setHoveredButton('file')}
              onMouseLeave={() => setHoveredButton('')}
              style={{
                ...styles.fileLabel,
                ...(hoveredButton === 'file' ? styles.fileLabelHover : {})
              }}
            >
              Choose File
            </label>
          </div>
          
          {file && (
            <div style={styles.fileName}>
              📎 {file.name}
            </div>
          )}
          
          <button
            onClick={handleUpload}
            onMouseEnter={() => setHoveredButton('upload')}
            onMouseLeave={() => setHoveredButton('')}
            style={{
              ...styles.uploadButton,
              ...(hoveredButton === 'upload' ? styles.uploadButtonHover : {})
            }}
          >
            Upload & Process
          </button>
          
          {uploadStatus && (
            <div style={{
              ...styles.status,
              ...(uploadStatus.includes('successfully') ? styles.statusSuccess : styles.statusError)
            }}>
              {uploadStatus}
            </div>
          )}
        </div>

        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>❓ Ask a Question</h3>
          <textarea
            rows={4}
            placeholder="Enter your query here... Ask anything about the uploaded documents!"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setFocusedTextarea(true)}
            onBlur={() => setFocusedTextarea(false)}
            style={{
              ...styles.textarea,
              ...(focusedTextarea ? styles.textareaFocus : {})
            }}
          />
          <button
            onClick={handleQuery}
            disabled={isLoading}
            onMouseEnter={() => setHoveredButton('submit')}
            onMouseLeave={() => setHoveredButton('')}
            style={{
              ...styles.submitButton,
              ...(isLoading ? styles.submitButtonDisabled : {}),
              ...(hoveredButton === 'submit' && !isLoading ? styles.submitButtonHover : {})
            }}
          >
            {isLoading && <span style={styles.loadingSpinner}></span>}
            {isLoading ? 'Processing...' : 'Submit Query'}
          </button>
          
            {response && (
              <div style={styles.response}>
                <h4 style={{ margin: '0 0 15px 0', color: '#1f2937', fontSize: '18px', fontWeight: '600' }}>
                  💡 Response:
                </h4>
                {response.split('\n').map((line, i) => (
                  <p key={i} style={styles.responseParagraph}>{line}</p>
                ))}
              </div>
            )}

            {/* Local history for this PDF */}
            {history && history.length > 0 && (
              <div style={{ ...styles.section, marginTop: '20px' }}>
                <h4 style={{ margin: '0 0 12px 0', color: '#1f2937', fontSize: '16px', fontWeight: '600' }}>
                  🕘 Recent Q&A (this document)
                </h4>
                {history.map((item, i) => (
                  <div key={i} style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '14px', color: '#6b7280' }}>{new Date(item.ts).toLocaleString()}</div>
                    <div style={{ fontWeight: '600', color: '#111827' }}>Q: {item.query}</div>
                    <div style={{ color: '#374151', marginTop: '6px' }}>{item.answer}</div>
                  </div>
                ))}
              </div>
            )}
        </div>
      </div>
      
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          
          .dashboard *::-webkit-scrollbar {
            width: 8px;
          }
          
          .dashboard *::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
          }
          
          .dashboard *::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
          }
          
          .dashboard *::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
          }
        `}
      </style>
    </div>
  );
};

export default Dashboard;