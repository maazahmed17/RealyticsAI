#!/usr/bin/env python3
"""
RealyticsAI API Server
======================
Flask server to connect the frontend with the chatbot backend
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the chatbot
from src.chatbot import RealyticsAIChatbot
from src.conversation.state_manager import ConversationState

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for all routes

# Global chatbot instance
chatbot = None
conversation_state = None

def initialize_chatbot():
    """Initialize the chatbot instance"""
    global chatbot, conversation_state
    print("Initializing RealyticsAI chatbot...")
    chatbot = RealyticsAIChatbot()
    conversation_state = ConversationState()
    print("Chatbot initialized successfully!")

# Serve the frontend
@app.route('/')
def index():
    """Serve the main index.html"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory"""
    return send_from_directory('frontend', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global chatbot, conversation_state
    
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process the message through the chatbot
        response = chatbot.process_query(message)
        
        # Extract property details if available
        property_details = None
        if chatbot.state.features and chatbot.state.features.has_all_required():
            property_details = {
                'bhk': chatbot.state.features.bhk,
                'sqft': chatbot.state.features.sqft,
                'bath': chatbot.state.features.bath,
                'balcony': chatbot.state.features.balcony,
                'location': chatbot.state.features.location
            }
        
        # Check if this is a valuation response
        is_valuation = any(word in response.lower() for word in ['price', 'valuation', 'estimate', 'lakhs', 'crores'])
        
        return jsonify({
            'response': response,
            'is_valuation': is_valuation,
            'property_details': property_details,
            'has_prediction': chatbot.state.last_prediction is not None
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """Reset the chat conversation"""
    global chatbot
    
    try:
        chatbot.state.reset_features()
        return jsonify({'success': True, 'message': 'Chat reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/<feature_type>', methods=['POST'])
def use_feature(feature_type):
    """Handle feature selection from the menu"""
    
    feature_prompts = {
        'price': 'I need a price estimate for a property',
        'discover': 'Help me discover properties in Bengaluru',
        'negotiate': 'Assist me with property negotiation'
    }
    
    if feature_type not in feature_prompts:
        return jsonify({'error': 'Invalid feature type'}), 400
    
    # Process the feature request
    prompt = feature_prompts[feature_type]
    response = chatbot.process_query(prompt)
    
    return jsonify({
        'response': response,
        'feature': feature_type
    })

@app.route('/api/sample_queries', methods=['GET'])
def get_sample_queries():
    """Get sample queries for the welcome screen"""
    queries = [
        "What's the price of a 3 BHK in Whitefield?",
        "I have a 1500 sqft apartment in Koramangala",
        "Show me market trends for Electronic City",
        "Estimate price for 2 BHK in Indiranagar with 2 bathrooms",
        "What's the average price per sqft in HSR Layout?"
    ]
    return jsonify({'queries': queries})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'chatbot_initialized': chatbot is not None
    })

if __name__ == '__main__':
    # Initialize the chatbot
    initialize_chatbot()
    
    # Run the Flask server
    print("\n" + "="*60)
    print("🚀 RealyticsAI Server Starting...")
    print("="*60)
    print("\nAccess the application at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
