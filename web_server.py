#!/usr/bin/env python3
"""
RealyticsAI Unified API Server
==============================
Flask server that provides a unified interface for both price prediction 
and property recommendation services through Gemini-powered intelligent routing.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import logging
from pathlib import Path
import requests

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the unified chatbot
from realytics_ai import FixedUnifiedChatbot

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
unified_chatbot = None

def initialize_system():
    """Initialize the unified RealyticsAI system"""
    global unified_chatbot
    
    logger.info("Initializing RealyticsAI Unified System...")
    
    try:
        unified_chatbot = FixedUnifiedChatbot()
        logger.info("✅ Unified system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        return False

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
    """Main chat endpoint - handles all queries through unified routing"""
    global unified_chatbot
    
    if not unified_chatbot:
        return jsonify({
            'error': 'System not initialized', 
            'success': False,
            'response': 'I apologize, but the system is not ready yet. Please try again in a moment.'
        }), 503
    
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Processing query: '{message}'")
        
        # Process through unified chatbot
        result = unified_chatbot.process_query(message)
        
        # Extract information for frontend
        service_used = result.get('service', 'unknown')
        response_text = result.get('response', 'I could not process your request.')
        
        # Determine response type for frontend styling
        is_price_prediction = service_used == 'price_prediction'
        is_recommendation = service_used == 'property_recommendation'
        is_market_analysis = service_used == 'market_analysis'
        
        # Extract additional data
        has_prediction = result.get('has_prediction', False)
        property_details = result.get('property_details')
        recommendations = result.get('recommendations', [])
        
        response_data = {
            'response': response_text,
            'success': result.get('success', True),
            'service_used': service_used,
            'is_price_prediction': is_price_prediction,
            'is_recommendation': is_recommendation,
            'is_market_analysis': is_market_analysis,
            'has_prediction': has_prediction,
            'property_details': property_details,
            'recommendations': recommendations[:5] if recommendations else [],  # Limit to 5 for display
            'total_recommendations': len(recommendations),
            'intent_analysis': result.get('intent_analysis', {}),
            'confidence': result.get('intent_analysis', {}).get('confidence', 0.0)
        }
        
        logger.info(f"Response generated - Service: {service_used}, Success: {result.get('success', True)}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': str(e),
            'success': False,
            'response': 'I apologize, but I encountered an error processing your request. Please try again.',
            'service_used': 'error'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """Reset the chat conversation"""
    global unified_chatbot
    
    try:
        if unified_chatbot:
            unified_chatbot.reset_conversation()
            return jsonify({'success': True, 'message': 'Chat reset successfully'})
        else:
            return jsonify({'error': 'System not initialized'}), 503
    except Exception as e:
        logger.error(f"Error resetting chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_query():
    """Analyze query intent without processing - for debugging"""
    global unified_chatbot
    
    if not unified_chatbot:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Analyze query intent
        analysis = unified_chatbot.analyze_query_intent(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/services/price_prediction', methods=['POST'])
def price_prediction():
    """Direct price prediction endpoint"""
    global unified_chatbot
    
    if not unified_chatbot or not unified_chatbot.price_prediction_bot:
        return jsonify({'error': 'Price prediction service not available'}), 503
    
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Force route to price prediction
        result = unified_chatbot._handle_price_prediction(query, {})
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in direct price prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/services/recommendations', methods=['POST'])
def property_recommendations():
    """Direct property recommendation endpoint"""
    global unified_chatbot
    
    if not unified_chatbot or not unified_chatbot.property_recommender:
        return jsonify({'error': 'Recommendation service not available'}), 503
    
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Force route to recommendations
        result = unified_chatbot._handle_property_recommendation(query, {})
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in direct recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample_queries', methods=['GET'])
def get_sample_queries():
    """Get sample queries for different services"""
    queries = {
        'price_prediction': [
            "What's the price of a 3 BHK in Whitefield?",
            "I have a 1500 sqft apartment in Koramangala - what's the price?",
            "Estimate value for 2 BHK with 2 bathrooms in HSR Layout",
            "Price prediction for 4 BHK house in Electronic City"
        ],
        'property_recommendation': [
            "Find me apartments under 50 lakhs in Electronic City",
            "Show me 2 BHK properties near HSR Layout", 
            "Recommend properties similar to Indiranagar",
            "Find budget-friendly apartments with good connectivity"
        ],
        'market_analysis': [
            "What are the market trends in Bengaluru?",
            "Which areas are good for investment?",
            "Compare prices between Whitefield and Electronic City",
            "Best areas for first-time home buyers"
        ],
        'general': [
            "How does home buying work in Bangalore?",
            "What documents are needed for property purchase?",
            "Tips for property investment in IT corridors",
            "How to evaluate a property's value?"
        ]
    }
    
    return jsonify({
        'success': True,
        'sample_queries': queries
    })

# ------------------------------------------------------------
# Negotiation proxy endpoints -> forward to FastAPI (port 8000)
# ------------------------------------------------------------

@app.route('/api/negotiate/start', methods=['POST'])
def negotiate_start_proxy():
    """Proxy negotiation start to FastAPI backend."""
    try:
        resp = requests.post(
            'http://localhost:8000/api/negotiate/start',
            json=request.json,
            headers={'X-User-Id': request.headers.get('X-User-Id', 'default-user')},
            timeout=10,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logger.error(f"Negotiation start proxy error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/negotiate/respond', methods=['POST'])
def negotiate_respond_proxy():
    """Proxy negotiation respond to FastAPI backend."""
    try:
        resp = requests.post(
            'http://localhost:8000/api/negotiate/respond',
            json=request.json,
            headers={'X-User-Id': request.headers.get('X-User-Id', 'default-user')},
            timeout=10,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logger.error(f"Negotiation respond proxy error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/negotiate/history/<session_id>', methods=['GET'])
def negotiate_history_proxy(session_id):
    """Proxy negotiation history to FastAPI backend."""
    try:
        resp = requests.get(
            f'http://localhost:8000/api/negotiate/history/{session_id}',
            headers={'X-User-Id': request.headers.get('X-User-Id', 'default-user')},
            timeout=10,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logger.error(f"Negotiation history proxy error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/negotiate/end/<session_id>', methods=['POST'])
def negotiate_end_proxy(session_id):
    """Proxy negotiation end to FastAPI backend."""
    try:
        resp = requests.post(
            f'http://localhost:8000/api/negotiate/end/{session_id}',
            headers={'X-User-Id': request.headers.get('X-User-Id', 'default-user')},
            timeout=10,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logger.error(f"Negotiation end proxy error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get comprehensive system status"""
    global unified_chatbot
    
    status = {
        'system_initialized': unified_chatbot is not None,
        'services': {
            'gemini_api': False,
            'price_prediction': False,
            'property_recommendation': False,
            'intelligent_routing': False
        },
        'capabilities': [],
        'version': '2.0.0-unified'
    }
    
    if unified_chatbot:
        # Check Gemini API
        try:
            test_response = unified_chatbot.gemini_model.generate_content("Hello")
            status['services']['gemini_api'] = True
        except:
            pass
        
        # Check price prediction
        status['services']['price_prediction'] = unified_chatbot.price_prediction_bot is not None
        
        # Check recommendations
        status['services']['property_recommendation'] = (
            unified_chatbot.property_recommender is not None and 
            unified_chatbot.property_recommender.df is not None
        )
        
        # Check intelligent routing
        status['services']['intelligent_routing'] = True
        
        # Set capabilities based on available services
        if status['services']['price_prediction']:
            status['capabilities'].append('price_prediction')
        if status['services']['property_recommendation']:
            status['capabilities'].append('property_recommendation')
        if status['services']['gemini_api']:
            status['capabilities'].extend(['market_analysis', 'general_chat'])
    
    return jsonify(status)

@app.route('/api/system/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global unified_chatbot
    
    if not unified_chatbot:
        return jsonify({
            'status': 'unhealthy',
            'message': 'System not initialized'
        }), 503
    
    health_status = {
        'status': 'healthy',
        'services': {
            'unified_chatbot': True,
            'gemini_api': False,
            'price_prediction': unified_chatbot.price_prediction_bot is not None,
            'recommendations': (
                unified_chatbot.property_recommender is not None and 
                unified_chatbot.property_recommender.df is not None
            )
        },
        'version': '2.0.0-unified',
        'features': ['intelligent_routing', 'unified_interface']
    }
    
    # Test Gemini API
    try:
        unified_chatbot.gemini_model.generate_content("Health check")
        health_status['services']['gemini_api'] = True
    except:
        health_status['services']['gemini_api'] = False
    
    return jsonify(health_status)

@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history"""
    global unified_chatbot
    
    if not unified_chatbot:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        # Return last 10 conversations
        history = unified_chatbot.conversation_history[-10:] if unified_chatbot.conversation_history else []
        
        return jsonify({
            'success': True,
            'history': history,
            'total_conversations': len(unified_chatbot.conversation_history)
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the unified system
    if not initialize_system():
        print("❌ Failed to initialize system. Exiting.")
        sys.exit(1)
    
    # Run the Flask server
    print("\n" + "="*70)
    print("🚀 RealyticsAI v2.0 - Unified Real Estate AI Platform")
    print("="*70)
    print("\n🏠 Available Features:")
    print("   • 🤖 Intelligent Query Routing (Powered by Gemini)")
    print("   • 💰 Property Price Predictions")
    print("   • 🔍 Smart Property Recommendations")
    print("   • 📊 Market Analysis & Trends")
    print("   • 💬 Natural Language Chat Interface")
    print("\n🌐 Access the application at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   • Main Chat: POST /api/chat")
    print("   • System Status: GET /api/system/status")
    print("   • Health Check: GET /api/system/health")
    print("   • Sample Queries: GET /api/sample_queries")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)