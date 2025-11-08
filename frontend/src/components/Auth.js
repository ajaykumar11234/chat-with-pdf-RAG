import React, { useState } from 'react';

const Auth = ({ onLogin }) => {
  const [isSignup, setIsSignup] = useState(false);
  const [form, setForm] = useState({ username: '', password: '', role: 'user' });
  const [error, setError] = useState('');

  const backendUrl = 'http://localhost:5001/api';

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const endpoint = isSignup ? '/signup' : '/login';
    try {
      const res = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (res.ok) {
        // Note: localStorage not available in Claude.ai environment
        // In your actual app, uncomment the line below:
        // localStorage.setItem('token', data.token);
        onLogin(data.token, data.role); // Notify App to switch dashboard
      } else {
        setError(data.error || 'Something went wrong');
      }
    } catch (err) {
      console.error(err);
      setError('Network error');
    }
  };

  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: '#ffffff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      padding: '20px'
    },
    authCard: {
      backgroundColor: '#ffffff',
      padding: '40px',
      borderRadius: '12px',
      boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
      width: '100%',
      maxWidth: '400px',
      border: '1px solid #e5e7eb'
    },
    title: {
      fontSize: '28px',
      fontWeight: '700',
      textAlign: 'center',
      marginBottom: '30px',
      color: '#1f2937',
      letterSpacing: '-0.5px'
    },
    form: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    },
    input: {
      padding: '12px 16px',
      fontSize: '16px',
      border: '2px solid #e5e7eb',
      borderRadius: '8px',
      outline: 'none',
      transition: 'all 0.2s ease',
      backgroundColor: '#ffffff'
    },
    inputFocus: {
      borderColor: '#3b82f6',
      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)'
    },
    select: {
      padding: '12px 16px',
      fontSize: '16px',
      border: '2px solid #e5e7eb',
      borderRadius: '8px',
      outline: 'none',
      transition: 'all 0.2s ease',
      backgroundColor: '#ffffff',
      cursor: 'pointer'
    },
    button: {
      padding: '14px',
      fontSize: '16px',
      fontWeight: '600',
      backgroundColor: '#3b82f6',
      color: '#ffffff',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      marginTop: '10px'
    },
    buttonHover: {
      backgroundColor: '#2563eb',
      transform: 'translateY(-1px)',
      boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
    },
    error: {
      color: '#ef4444',
      fontSize: '14px',
      textAlign: 'center',
      marginTop: '10px',
      padding: '8px',
      backgroundColor: '#fef2f2',
      border: '1px solid #fecaca',
      borderRadius: '6px'
    },
    switchText: {
      textAlign: 'center',
      marginTop: '25px',
      fontSize: '14px',
      color: '#6b7280'
    },
    switchLink: {
      color: '#3b82f6',
      cursor: 'pointer',
      fontWeight: '600',
      textDecoration: 'none',
      transition: 'color 0.2s ease'
    },
    switchLinkHover: {
      color: '#2563eb',
      textDecoration: 'underline'
    }
  };

  const [focusedInput, setFocusedInput] = useState('');
  const [hoveredButton, setHoveredButton] = useState(false);
  const [hoveredLink, setHoveredLink] = useState(false);

  return (
    <div style={styles.container}>
      <div style={styles.authCard}>
        <h2 style={styles.title}>{isSignup ? 'Create Account' : 'Welcome Back'}</h2>
        <div onSubmit={handleSubmit} style={styles.form}>
          <input
            type="text"
            name="username"
            placeholder="Username"
            value={form.username}
            onChange={handleChange}
            onFocus={() => setFocusedInput('username')}
            onBlur={() => setFocusedInput('')}
            style={{
              ...styles.input,
              ...(focusedInput === 'username' ? styles.inputFocus : {})
            }}
            required
          />
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={form.password}
            onChange={handleChange}
            onFocus={() => setFocusedInput('password')}
            onBlur={() => setFocusedInput('')}
            style={{
              ...styles.input,
              ...(focusedInput === 'password' ? styles.inputFocus : {})
            }}
            required
          />
          {isSignup && (
            // Keep role fixed to 'user' for all signups (admin portal removed)
            <input type="hidden" name="role" value={form.role} />
          )}
          <button 
            type="submit"
            onClick={handleSubmit}
            onMouseEnter={() => setHoveredButton(true)}
            onMouseLeave={() => setHoveredButton(false)}
            style={{
              ...styles.button,
              ...(hoveredButton ? styles.buttonHover : {})
            }}
          >
            {isSignup ? 'Create Account' : 'Sign In'}
          </button>
        </div>
        {error && <div style={styles.error}>{error}</div>}
        <div style={styles.switchText}>
          {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
          <span 
            onClick={() => setIsSignup(!isSignup)}
            onMouseEnter={() => setHoveredLink(true)}
            onMouseLeave={() => setHoveredLink(false)}
            style={{
              ...styles.switchLink,
              ...(hoveredLink ? styles.switchLinkHover : {})
            }}
          >
            {isSignup ? 'Sign in here' : 'Create one here'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default Auth;