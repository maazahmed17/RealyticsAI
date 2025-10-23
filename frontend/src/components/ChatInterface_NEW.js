// frontend/src/components/ChatInterface.js
// Modern RealyticsAI Property Intelligence Suite - Dark Mode UI
import React, { useState, useEffect, useRef } from 'react';

// High-quality property images
const houseImages = [
  'https://images.unsplash.com/photo-1502005229762-cf1b2da7c1a0?q=80&w=1600&auto=format&fit=crop',
  'https://images.unsplash.com/photo-1494526585095-c41746248156?q=80&w=1600&auto=format&fit=crop',
  'https://images.unsplash.com/photo-1560518883-ce09059eeffa?q=80&w=1600&auto=format&fit=crop',
  'https://images.unsplash.com/photo-1560184897-ae75f418493e?q=80&w=1600&auto=format&fit=crop',
];

const getRandomImage = () => houseImages[Math.floor(Math.random() * houseImages.length)];

// Intent detection
const detectIntent = (text) => {
  const lower = text.toLowerCase();
  if (/(find|show|search|recommend|suggest|property|apartment|under\s*\d+)/i.test(lower)) return 'RECOMMENDATION';
  if (/(price|cost|value|estimate|valuation|worth|how much)/i.test(lower)) return 'PREDICTION';
  if (/(negotiate|negotiation|offer|counter|deal|bargain)/i.test(lower)) return 'NEGOTIATION';
  return 'AUTO';
};

// Format price
const formatPrice = (price) => {
  if (typeof price === 'string') return price;
  if (price >= 100) return `₹${(price / 100).toFixed(1)} Cr`;
  return `₹${price.toFixed(1)} Lakhs`;
};

// Render markdown-style text
const renderText = (text) => {
  return text.split('\\n').map((line, i) => {
    const boldText = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    return <p key={i} dangerouslySetInnerHTML={{ __html: boldText }} style={{ marginBottom: '0.5rem' }} />;
  });
};

function ChatInterface() {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      sender: 'assistant', 
      text: 'Welcome to **RealyticsAI** — your intelligent property assistant. Ask me to find properties, get price valuations, or help with negotiations.',
      intent: 'AUTO'
    }
  ]);
  const [input, setInput] = useState('');
  const [activeFeature, setActiveFeature] = useState('AUTO');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = (messageText, forceIntent = null) => {
    const text = (typeof messageText === 'string') ? messageText : input;
    if (text.trim() === '') return;

    const userMessage = { id: Date.now(), sender: 'user', text: text };
    setMessages(prev => [...prev, userMessage]);

    const botMessageId = Date.now() + 1;
    setMessages(prev => [...prev, { id: botMessageId, sender: 'assistant', text: '⠹ 🤔 Processing...', intent: 'AUTO' }]);

    setTimeout(() => {
      const intent = forceIntent || detectIntent(text);
      let botResponse;

      if (intent === 'RECOMMENDATION' || text.toLowerCase().includes('find')) {
        botResponse = {
          id: botMessageId,
          sender: 'assistant',
          intent: 'RECOMMENDATION',
          text: `Great news! I found **351 properties** for you. Here are the best options:`,
          properties: [
            {
              id: 'p1',
              title: '3 BHK Apartment in Whitefield Old',
              location: 'Whitefield Old',
              price_lakhs: 31.9,
              total_sqft: 1796,
              bath: 3,
              balcony: 1,
              image: getRandomImage()
            },
            {
              id: 'p2',
              title: '3 BHK Apartment in Whitefield Phase 2',
              location: 'Whitefield Phase 2',
              price_lakhs: 32.1,
              total_sqft: 1727,
              bath: 2,
              balcony: 2,
              image: getRandomImage()
            },
            {
              id: 'p3',
              title: '3 BHK Apartment in Whitefield Phase 2',
              location: 'Whitefield Phase 2',
              price_lakhs: 33.3,
              total_sqft: 1650,
              bath: 3,
              balcony: 3,
              image: getRandomImage()
            },
          ]
        };
      } else if (intent === 'PREDICTION') {
        botResponse = {
          id: botMessageId,
          sender: 'assistant',
          intent: 'PREDICTION',
          text: `**Price Valuation Ready**\\n\\nI've analyzed the property parameters and generated a **data-driven estimate**.\\n\\n**Estimated Price:** ₹59.35 Lakhs\\n**Likely Range:** ₹53.4 - ₹65.3 Lakhs\\n\\nThis valuation is based on **location, BHK, size (sqft), and amenities**. Provide more details to refine accuracy.`
        };
      } else if (intent === 'NEGOTIATION') {
        botResponse = {
          id: botMessageId,
          sender: 'assistant',
          intent: 'NEGOTIATION',
          text: `**AI Negotiation Activated**\\n\\nI can draft a persuasive offer, set a **target price**, and recommend **counter strategies** to maximize your outcome.\\n\\n**Suggested Starting Offer:** ₹48 Lakhs\\n**Maximum Budget:** ₹55 Lakhs\\n**Strategy:** Start 15% below asking price, emphasize immediate purchase readiness.`
        };
      } else {
        botResponse = {
          id: botMessageId,
          sender: 'assistant',
          intent: 'AUTO',
          text: `I can help with **property recommendations**, **price valuations**, and **AI-powered negotiations**. Try asking me to find properties or estimate a price!`
        };
      }

      setMessages(prev => prev.map(msg => msg.id === botMessageId ? botResponse : msg));
    }, 1500);

    setInput('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFeatureClick = (feature) => {
    setActiveFeature(feature);
    const queries = {
      'RECOMMENDATION': 'Find me 3 BHK apartments in Whitefield under 50 lakhs',
      'PREDICTION': 'What is the price of a 3 BHK apartment in Koramangala with 1500 sqft?',
      'NEGOTIATION': 'Help me negotiate the best price for a property in HSR Layout'
    };
    if (feature !== 'AUTO') handleSendMessage(queries[feature], feature);
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#121212', 
      color: '#fff',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 10,
        backgroundColor: 'rgba(18, 18, 18, 0.9)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid #333',
        padding: '1rem 0'
      }}>
        <div style={{ maxWidth: '48rem', margin: '0 auto', padding: '0 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            <span style={{ color: '#fff' }}>r</span>
            <span style={{ color: '#00A99D' }}>AI</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#888' }}>Property Intelligence Suite</div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main style={{ flex: 1, overflowY: 'auto', padding: '2rem 1rem' }}>
        <div style={{ maxWidth: '48rem', margin: '0 auto' }}>
          {/* Feature Buttons */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
            {['AUTO', 'RECOMMENDATION', 'PREDICTION', 'NEGOTIATION'].map(feature => {
              const icons = { AUTO: '✨', RECOMMENDATION: '🏠', PREDICTION: '💰', NEGOTIATION: '🤝' };
              const labels = { AUTO: 'Auto', RECOMMENDATION: 'Property Discovery', PREDICTION: 'Price Valuation', NEGOTIATION: 'AI Negotiation' };
              const isActive = activeFeature === feature;
              return (
                <button
                  key={feature}
                  onClick={() => handleFeatureClick(feature)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.5rem 1rem',
                    borderRadius: '9999px',
                    border: `1px solid ${isActive ? '#00A99D' : '#333'}`,
                    backgroundColor: isActive ? 'rgba(0, 169, 157, 0.1)' : 'transparent',
                    color: isActive ? '#00A99D' : '#ccc',
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#00A99D'; e.currentTarget.style.color = '#00A99D'; }}
                  onMouseLeave={(e) => { if (!isActive) { e.currentTarget.style.borderColor = '#333'; e.currentTarget.style.color = '#ccc'; } }}
                >
                  <span>{icons[feature]}</span>
                  <span>{labels[feature]}</span>
                </button>
              );
            })}
          </div>

          {/* Messages */}
          {messages.map(msg => (
            <div key={msg.id} style={{ marginBottom: '1.5rem' }}>
              <div style={{
                backgroundColor: msg.sender === 'user' ? '#1e1e1e' : '#161616',
                border: '1px solid #333',
                borderRadius: '1rem',
                padding: '1.25rem',
                boxShadow: '0 0 0 1px rgba(255,255,255,0.02)'
              }}>
                <div style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#eaeaea' }}>
                  {renderText(msg.text)}
                </div>

                {/* Property Cards */}
                {msg.properties && (
                  <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {msg.properties.map(prop => (
                      <div key={prop.id} style={{
                        backgroundColor: '#1A1A1A',
                        border: '1px solid #333',
                        borderRadius: '0.75rem',
                        overflow: 'hidden',
                        display: 'grid',
                        gridTemplateRows: '18rem auto'
                      }}>
                        {/* Image */}
                        <div style={{
                          backgroundImage: `url(${prop.image})`,
                          backgroundSize: 'cover',
                          backgroundPosition: 'center'
                        }} />
                        
                        {/* Description */}
                        <div style={{ padding: '1rem' }}>
                          <h3 style={{ fontSize: '1.125rem', fontWeight: 'bold', color: '#fff', marginBottom: '0.5rem' }}>
                            {prop.title}
                          </h3>
                          <p style={{ fontSize: '0.9rem', color: '#ccc', marginBottom: '0.25rem' }}>
                            💰 Price: {formatPrice(prop.price_lakhs)}
                          </p>
                          <p style={{ fontSize: '0.85rem', color: '#EAEAEA' }}>
                            📐 Size: {prop.total_sqft} sqft | 🚿 {prop.bath} Bath | 🪟 {prop.balcony} Balcony
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <div style={{
        position: 'sticky',
        bottom: 0,
        backgroundColor: '#121212',
        borderTop: '1px solid #333',
        padding: '1rem'
      }}>
        <div style={{ maxWidth: '48rem', margin: '0 auto', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask about properties, prices, or negotiations..."
            rows={1}
            style={{
              flex: 1,
              backgroundColor: '#1e1e1e',
              border: '1px solid #333',
              borderRadius: '0.75rem',
              padding: '0.75rem 1rem',
              color: '#fff',
              fontSize: '0.95rem',
              resize: 'none',
              outline: 'none'
            }}
          />
          <button
            onClick={() => handleSendMessage()}
            disabled={!input.trim()}
            style={{
              backgroundColor: input.trim() ? '#00A99D' : '#333',
              color: '#fff',
              border: 'none',
              borderRadius: '0.75rem',
              padding: '0.75rem 1.5rem',
              cursor: input.trim() ? 'pointer' : 'not-allowed',
              fontSize: '0.9rem',
              fontWeight: '600'
            }}
          >
            Send
          </button>
        </div>
        <div style={{ maxWidth: '48rem', margin: '0.5rem auto 0', textAlign: 'center', fontSize: '0.75rem', color: '#666' }}>
          Press <kbd style={{ backgroundColor: '#333', padding: '0.125rem 0.375rem', borderRadius: '0.25rem' }}>Enter</kbd> to send
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
