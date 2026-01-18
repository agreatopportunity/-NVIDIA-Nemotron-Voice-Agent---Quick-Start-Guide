import React, { useState, useEffect, useRef, useCallback } from 'react';

// Matrix Rain Background Component
const MatrixRain = () => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const charArray = chars.split('');
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops = [];
    
    for (let i = 0; i < columns; i++) {
      drops[i] = Math.random() * -100;
    }
    
    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      for (let i = 0; i < drops.length; i++) {
        const char = charArray[Math.floor(Math.random() * charArray.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;
        
        // Create gradient effect - brighter at the head
        const gradient = ctx.createLinearGradient(x, y - 20, x, y);
        gradient.addColorStop(0, 'rgba(0, 255, 200, 0)');
        gradient.addColorStop(0.5, 'rgba(0, 255, 200, 0.5)');
        gradient.addColorStop(1, 'rgba(0, 255, 255, 1)');
        
        ctx.fillStyle = Math.random() > 0.98 ? '#fff' : (Math.random() > 0.5 ? '#00ffc8' : '#00d4ff');
        ctx.font = `${fontSize}px monospace`;
        ctx.fillText(char, x, y);
        
        if (y > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };
    
    const interval = setInterval(draw, 33);
    return () => {
      clearInterval(interval);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);
  
  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        background: 'linear-gradient(180deg, #000510 0%, #001020 50%, #000a15 100%)'
      }}
    />
  );
};

// Animated Orb Component
const PulsingOrb = ({ isRecording, isProcessing }) => {
  return (
    <div style={{
      position: 'relative',
      width: '120px',
      height: '120px',
      margin: '0 auto 20px'
    }}>
      {/* Outer glow rings */}
      {[...Array(3)].map((_, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: `${100 + i * 30}%`,
            height: `${100 + i * 30}%`,
            transform: 'translate(-50%, -50%)',
            borderRadius: '50%',
            border: `2px solid ${isRecording ? 'rgba(255, 50, 50, 0.3)' : 'rgba(0, 255, 200, 0.2)'}`,
            animation: `pulse ${2 + i * 0.5}s ease-in-out infinite`,
            opacity: 0.5 - i * 0.15
          }}
        />
      ))}
      
      {/* Main orb */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '80px',
        height: '80px',
        borderRadius: '50%',
        background: isRecording 
          ? 'radial-gradient(circle at 30% 30%, #ff6b6b, #ff0000, #990000)'
          : isProcessing
          ? 'radial-gradient(circle at 30% 30%, #ffd700, #ff8c00, #ff4500)'
          : 'radial-gradient(circle at 30% 30%, #00ffc8, #00d4ff, #0066ff)',
        boxShadow: isRecording
          ? '0 0 40px rgba(255, 0, 0, 0.6), 0 0 80px rgba(255, 0, 0, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.2)'
          : isProcessing
          ? '0 0 40px rgba(255, 200, 0, 0.6), 0 0 80px rgba(255, 150, 0, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.2)'
          : '0 0 40px rgba(0, 255, 200, 0.6), 0 0 80px rgba(0, 212, 255, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.2)',
        animation: isProcessing ? 'spin 2s linear infinite' : 'float 3s ease-in-out infinite',
        transition: 'all 0.3s ease'
      }}>
        {/* Inner highlight */}
        <div style={{
          position: 'absolute',
          top: '15%',
          left: '20%',
          width: '30%',
          height: '30%',
          borderRadius: '50%',
          background: 'rgba(255, 255, 255, 0.4)',
          filter: 'blur(5px)'
        }} />
      </div>
      
      {/* Status indicator */}
      <div style={{
        position: 'absolute',
        bottom: '-25px',
        left: '50%',
        transform: 'translateX(-50%)',
        fontSize: '12px',
        color: isRecording ? '#ff6b6b' : isProcessing ? '#ffd700' : '#00ffc8',
        textTransform: 'uppercase',
        letterSpacing: '2px',
        fontWeight: 'bold',
        textShadow: '0 0 10px currentColor'
      }}>
        {isRecording ? '● REC' : isProcessing ? '◌ PROCESSING' : '◉ READY'}
      </div>
    </div>
  );
};

// Message Bubble Component
const MessageBubble = ({ message, isUser, attachments }) => {
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: '16px',
      animation: 'slideIn 0.3s ease-out'
    }}>
      <div style={{
        maxWidth: '85%',
        padding: '14px 18px',
        borderRadius: isUser ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
        background: isUser 
          ? 'linear-gradient(135deg, rgba(0, 100, 255, 0.9), rgba(0, 150, 255, 0.7))'
          : 'linear-gradient(135deg, rgba(0, 40, 60, 0.95), rgba(0, 60, 80, 0.85))',
        border: `1px solid ${isUser ? 'rgba(0, 200, 255, 0.3)' : 'rgba(0, 255, 200, 0.2)'}`,
        boxShadow: isUser 
          ? '0 4px 20px rgba(0, 100, 255, 0.3)'
          : '0 4px 20px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(10px)'
      }}>
        {attachments && attachments.length > 0 && (
          <div style={{ marginBottom: '10px' }}>
            {attachments.map((att, idx) => (
              <div key={idx} style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 12px',
                background: 'rgba(0, 255, 200, 0.1)',
                borderRadius: '8px',
                marginRight: '8px',
                marginBottom: '8px'
              }}>
                {att.type === 'image' ? (
                  <img src={att.preview} alt="attachment" style={{
                    width: '60px',
                    height: '60px',
                    objectFit: 'cover',
                    borderRadius: '6px'
                  }} />
                ) : (
                  <>
                    <span style={{ fontSize: '20px' }}>📎</span>
                    <span style={{ fontSize: '12px', color: '#00ffc8' }}>{att.name}</span>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
        <p style={{
          margin: 0,
          color: '#fff',
          fontSize: '15px',
          lineHeight: '1.5',
          wordBreak: 'break-word'
        }}>
          {message}
        </p>
      </div>
    </div>
  );
};

// File Upload Button Component
const FileUploadButton = ({ onFileSelect }) => {
  const fileInputRef = useRef(null);
  
  const handleClick = () => {
    fileInputRef.current?.click();
  };
  
  const handleChange = (e) => {
    const files = Array.from(e.target.files);
    onFileSelect(files);
    e.target.value = '';
  };
  
  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,.pdf,.doc,.docx,.txt,.json,.csv"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <button
        onClick={handleClick}
        style={{
          width: '44px',
          height: '44px',
          borderRadius: '50%',
          border: '2px solid rgba(0, 255, 200, 0.3)',
          background: 'rgba(0, 40, 60, 0.8)',
          color: '#00ffc8',
          fontSize: '20px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.3s ease'
        }}
        onMouseOver={(e) => {
          e.currentTarget.style.background = 'rgba(0, 255, 200, 0.2)';
          e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 255, 200, 0.4)';
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.background = 'rgba(0, 40, 60, 0.8)';
          e.currentTarget.style.boxShadow = 'none';
        }}
      >
        📎
      </button>
    </>
  );
};

// Main App Component
export default function NemotronVoiceUI() {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your Nemotron AI assistant. How can I help you today?", isUser: false }
  ]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [attachments, setAttachments] = useState([]);
  const [mode, setMode] = useState('text'); // 'text' or 'voice'
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const handleFileSelect = (files) => {
    const newAttachments = files.map(file => ({
      file,
      name: file.name,
      type: file.type.startsWith('image/') ? 'image' : 'file',
      preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : null
    }));
    setAttachments(prev => [...prev, ...newAttachments]);
  };
  
  const removeAttachment = (index) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };
  
  const simulateResponse = async (userMessage) => {
    setIsProcessing(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
    
    const responses = [
      "I've analyzed your request. Based on the Nemotron neural architecture, I can provide comprehensive assistance with that.",
      "Interesting question! Let me process that through my language understanding modules...",
      "I'm running on NVIDIA's Nemotron stack with streaming ASR and 4-bit quantized inference. Here's what I found:",
      "Processing complete. The neural pathways indicate several possible approaches to your query.",
      "Excellent! I've computed the optimal response using my transformer-based reasoning engine."
    ];
    
    setIsProcessing(false);
    return responses[Math.floor(Math.random() * responses.length)] + " " + userMessage.slice(0, 50) + "...";
  };
  
  const sendMessage = async () => {
    if (!inputText.trim() && attachments.length === 0) return;
    
    const userMessage = {
      id: Date.now(),
      text: inputText || (attachments.length > 0 ? `Sent ${attachments.length} file(s)` : ''),
      isUser: true,
      attachments: [...attachments]
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setAttachments([]);
    
    const response = await simulateResponse(inputText);
    
    setMessages(prev => [...prev, {
      id: Date.now() + 1,
      text: response,
      isUser: false
    }]);
  };
  
  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        chunksRef.current = [];
        
        mediaRecorder.ondataavailable = (e) => {
          chunksRef.current.push(e.data);
        };
        
        mediaRecorder.onstop = async () => {
          stream.getTracks().forEach(track => track.stop());
          
          // Simulate transcription
          setIsProcessing(true);
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          const simulatedTranscript = "This is a simulated voice transcription. Connect to your ASR backend for real transcription.";
          
          setMessages(prev => [...prev, {
            id: Date.now(),
            text: `🎤 "${simulatedTranscript}"`,
            isUser: true
          }]);
          
          const response = await simulateResponse(simulatedTranscript);
          
          setMessages(prev => [...prev, {
            id: Date.now() + 1,
            text: response,
            isUser: false
          }]);
        };
        
        mediaRecorder.start();
        setIsRecording(true);
      } catch (err) {
        console.error('Microphone access denied:', err);
        alert('Please allow microphone access to use voice input.');
      }
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      overflow: 'hidden',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    }}>
      {/* CSS Animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.5; }
          50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.3; }
        }
        @keyframes float {
          0%, 100% { transform: translate(-50%, -50%) translateY(0); }
          50% { transform: translate(-50%, -50%) translateY(-10px); }
        }
        @keyframes spin {
          0% { transform: translate(-50%, -50%) rotate(0deg); }
          100% { transform: translate(-50%, -50%) rotate(360deg); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 200, 0.4); }
          50% { box-shadow: 0 0 40px rgba(0, 255, 200, 0.6); }
        }
        * { box-sizing: border-box; }
        input, textarea, button { font-family: inherit; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(0, 20, 40, 0.5); }
        ::-webkit-scrollbar-thumb { background: rgba(0, 255, 200, 0.3); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 200, 0.5); }
      `}</style>
      
      {/* Matrix Rain Background */}
      <MatrixRain />
      
      {/* Main Container */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        maxWidth: '800px',
        margin: '0 auto',
        padding: '10px'
      }}>
        {/* Header */}
        <div style={{
          textAlign: 'center',
          padding: '20px 10px',
          background: 'linear-gradient(180deg, rgba(0, 20, 40, 0.9) 0%, transparent 100%)'
        }}>
          <h1 style={{
            margin: '0 0 5px 0',
            fontSize: 'clamp(20px, 5vw, 28px)',
            fontWeight: '700',
            background: 'linear-gradient(135deg, #00ffc8, #00d4ff, #0088ff)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 30px rgba(0, 255, 200, 0.3)'
          }}>
            NEMOTRON AI
          </h1>
          <p style={{
            margin: 0,
            fontSize: '12px',
            color: 'rgba(0, 255, 200, 0.6)',
            letterSpacing: '3px',
            textTransform: 'uppercase'
          }}>
            Neural Voice Assistant
          </p>
          
          {/* Mode Toggle */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '10px',
            marginTop: '15px'
          }}>
            {['text', 'voice'].map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  padding: '8px 20px',
                  borderRadius: '20px',
                  border: `1px solid ${mode === m ? '#00ffc8' : 'rgba(0, 255, 200, 0.3)'}`,
                  background: mode === m ? 'rgba(0, 255, 200, 0.2)' : 'transparent',
                  color: mode === m ? '#00ffc8' : 'rgba(255, 255, 255, 0.6)',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  transition: 'all 0.3s ease'
                }}
              >
                {m === 'text' ? '⌨️ Text' : '🎤 Voice'}
              </button>
            ))}
          </div>
        </div>
        
        {/* Messages Area */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px 15px',
          background: 'linear-gradient(180deg, transparent 0%, rgba(0, 10, 20, 0.5) 50%, transparent 100%)'
        }}>
          {messages.map(msg => (
            <MessageBubble
              key={msg.id}
              message={msg.text}
              isUser={msg.isUser}
              attachments={msg.attachments}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Voice Mode Orb */}
        {mode === 'voice' && (
          <div style={{
            padding: '20px',
            textAlign: 'center'
          }}>
            <PulsingOrb isRecording={isRecording} isProcessing={isProcessing} />
            
            <button
              onClick={toggleRecording}
              disabled={isProcessing}
              style={{
                marginTop: '30px',
                padding: '15px 40px',
                borderRadius: '30px',
                border: 'none',
                background: isRecording 
                  ? 'linear-gradient(135deg, #ff4444, #cc0000)'
                  : 'linear-gradient(135deg, #00ffc8, #00a0ff)',
                color: '#fff',
                fontSize: '16px',
                fontWeight: '700',
                cursor: isProcessing ? 'not-allowed' : 'pointer',
                textTransform: 'uppercase',
                letterSpacing: '2px',
                boxShadow: isRecording
                  ? '0 0 30px rgba(255, 0, 0, 0.5)'
                  : '0 0 30px rgba(0, 255, 200, 0.5)',
                opacity: isProcessing ? 0.5 : 1,
                transition: 'all 0.3s ease'
              }}
            >
              {isRecording ? '⏹ Stop' : isProcessing ? '◌ Processing...' : '● Record'}
            </button>
          </div>
        )}
        
        {/* Text Input Area */}
        {mode === 'text' && (
          <div style={{
            padding: '15px',
            background: 'linear-gradient(180deg, transparent 0%, rgba(0, 20, 40, 0.95) 30%)'
          }}>
            {/* Attachments Preview */}
            {attachments.length > 0 && (
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '10px',
                marginBottom: '10px',
                padding: '10px',
                background: 'rgba(0, 40, 60, 0.5)',
                borderRadius: '12px'
              }}>
                {attachments.map((att, idx) => (
                  <div key={idx} style={{
                    position: 'relative',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    background: 'rgba(0, 255, 200, 0.1)',
                    borderRadius: '8px',
                    border: '1px solid rgba(0, 255, 200, 0.3)'
                  }}>
                    {att.type === 'image' ? (
                      <img src={att.preview} alt="" style={{
                        width: '40px',
                        height: '40px',
                        objectFit: 'cover',
                        borderRadius: '4px'
                      }} />
                    ) : (
                      <span>📎</span>
                    )}
                    <span style={{ fontSize: '12px', color: '#00ffc8', maxWidth: '100px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {att.name}
                    </span>
                    <button
                      onClick={() => removeAttachment(idx)}
                      style={{
                        position: 'absolute',
                        top: '-8px',
                        right: '-8px',
                        width: '20px',
                        height: '20px',
                        borderRadius: '50%',
                        border: 'none',
                        background: '#ff4444',
                        color: '#fff',
                        fontSize: '12px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            {/* Input Row */}
            <div style={{
              display: 'flex',
              gap: '10px',
              alignItems: 'flex-end'
            }}>
              <FileUploadButton onFileSelect={handleFileSelect} />
              
              <div style={{
                flex: 1,
                position: 'relative'
              }}>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  disabled={isProcessing}
                  rows={1}
                  style={{
                    width: '100%',
                    padding: '14px 18px',
                    borderRadius: '24px',
                    border: '2px solid rgba(0, 255, 200, 0.3)',
                    background: 'rgba(0, 20, 40, 0.9)',
                    color: '#fff',
                    fontSize: '15px',
                    outline: 'none',
                    resize: 'none',
                    minHeight: '50px',
                    maxHeight: '120px',
                    transition: 'all 0.3s ease'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#00ffc8';
                    e.target.style.boxShadow = '0 0 20px rgba(0, 255, 200, 0.3)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = 'rgba(0, 255, 200, 0.3)';
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>
              
              <button
                onClick={sendMessage}
                disabled={isProcessing || (!inputText.trim() && attachments.length === 0)}
                style={{
                  width: '50px',
                  height: '50px',
                  borderRadius: '50%',
                  border: 'none',
                  background: isProcessing 
                    ? 'rgba(100, 100, 100, 0.5)'
                    : 'linear-gradient(135deg, #00ffc8, #00a0ff)',
                  color: '#fff',
                  fontSize: '20px',
                  cursor: isProcessing ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: isProcessing ? 'none' : '0 0 20px rgba(0, 255, 200, 0.4)',
                  transition: 'all 0.3s ease'
                }}
              >
                {isProcessing ? '◌' : '➤'}
              </button>
            </div>
            
            {/* Quick Actions */}
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '10px',
              marginTop: '12px',
              flexWrap: 'wrap'
            }}>
              {['🎤 Voice', '📷 Camera', '🗑️ Clear'].map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    if (action.includes('Voice')) setMode('voice');
                    if (action.includes('Clear')) setMessages([{ id: 1, text: "Chat cleared. How can I help you?", isUser: false }]);
                  }}
                  style={{
                    padding: '8px 16px',
                    borderRadius: '16px',
                    border: '1px solid rgba(0, 255, 200, 0.2)',
                    background: 'transparent',
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontSize: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.background = 'rgba(0, 255, 200, 0.1)';
                    e.currentTarget.style.color = '#00ffc8';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.color = 'rgba(255, 255, 255, 0.7)';
                  }}
                >
                  {action}
                </button>
              ))}
            </div>
          </div>
        )}
        
        {/* Footer */}
        <div style={{
          padding: '10px',
          textAlign: 'center',
          background: 'rgba(0, 10, 20, 0.8)'
        }}>
          <p style={{
            margin: 0,
            fontSize: '10px',
            color: 'rgba(0, 255, 200, 0.4)',
            letterSpacing: '1px'
          }}>
            POWERED BY NVIDIA NEMOTRON • ASR + LLM + TTS
          </p>
        </div>
      </div>
    </div>
  );
}
