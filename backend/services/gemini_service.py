"""
Gemini API Service for RealyticsAI
===================================
Provides natural language processing capabilities using Google's Gemini API.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import google.generativeai as genai
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.core.config import settings as config_settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google's Gemini API"""
    
    def __init__(self):
        """Initialize the Gemini service"""
        if not config_settings.validate_api_keys():
            raise ValueError("Invalid or missing Gemini API key")
            
        # Configure Gemini
        genai.configure(api_key=config_settings.GEMINI_API_KEY)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=config_settings.GEMINI_MODEL,
            generation_config={
                "temperature": config_settings.GEMINI_TEMPERATURE,
                "max_output_tokens": config_settings.GEMINI_MAX_TOKENS,
            }
        )
        
        # Initialize chat session
        self.chat_session = None
        
        logger.info(f"Gemini service initialized with model: {config_settings.GEMINI_MODEL}")
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response using Gemini
        
        Parameters:
        -----------
        prompt : str
            The user's query or prompt
        context : dict, optional
            Additional context for the response
            
        Returns:
        --------
        str : The generated response
        """
        try:
            # Build enhanced prompt with context
            enhanced_prompt = self._build_enhanced_prompt(prompt, context)
            
            # Generate response
            response = self.model.generate_content(enhanced_prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def start_chat_session(self, initial_context: Optional[str] = None):
        """Start a new chat session"""
        history = []
        
        if initial_context:
            history.append({
                "role": "user",
                "parts": ["System context: " + initial_context]
            })
            history.append({
                "role": "model",
                "parts": ["I understand the context. I'm ready to help with your real estate queries."]
            })
        
        self.chat_session = self.model.start_chat(history=history)
        logger.info("New chat session started")
    
    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message in the current chat session
        
        Parameters:
        -----------
        message : str
            The user's message
        context : dict, optional
            Additional context for the response
            
        Returns:
        --------
        str : The model's response
        """
        try:
            if not self.chat_session:
                self.start_chat_session()
            
            # Add context to message if provided
            if context:
                context_str = f"\n[Context: {json.dumps(context, indent=2)}]\n"
                message = context_str + message
            
            response = self.chat_session.send_message(message)
            return response.text
            
        except Exception as e:
            logger.error(f"Error in chat session: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _build_enhanced_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build an enhanced prompt with system context"""
        
        system_prompt = """You are an AI assistant for RealyticsAI, a comprehensive real estate analysis system 
        specializing in Bengaluru property market. You have access to:
        
        1. Price Prediction: Advanced ML models that can predict property prices based on features
        2. Market Analysis: Detailed insights about Bengaluru real estate market
        3. Property Data: Information about properties including location, size, amenities, and prices
        
        Provide helpful, accurate, and detailed responses about real estate queries.
        Use the provided context to give specific and data-driven answers.
        Format your responses in a clear, professional manner.
        """
        
        enhanced_prompt = system_prompt + "\n\n"
        
        if context:
            enhanced_prompt += f"Context Information:\n{json.dumps(context, indent=2)}\n\n"
        
        enhanced_prompt += f"User Query: {prompt}\n\nResponse:"
        
        return enhanced_prompt
    
    def analyze_property_query(self, query: str, property_data: Dict[str, Any]) -> str:
        """
        Analyze a property-related query with specific property data
        
        Parameters:
        -----------
        query : str
            The user's query about the property
        property_data : dict
            Property information including predictions and features
            
        Returns:
        --------
        str : Natural language analysis
        """
        prompt = f"""
        Analyze the following property query and provide a comprehensive response:
        
        Query: {query}
        
        Property Information:
        {json.dumps(property_data, indent=2)}
        
        Please provide:
        1. Direct answer to the query
        2. Key insights about the property
        3. Market comparison if relevant
        4. Recommendations or suggestions
        5. Any important considerations
        
        Format the response in a clear, conversational manner.
        """
        
        return self.generate_response(prompt)
    
    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of a price prediction
        
        Parameters:
        -----------
        prediction_data : dict
            Contains prediction results and model information
            
        Returns:
        --------
        str : Natural language explanation
        """
        prompt = f"""
        Explain the following property price prediction in natural language:
        
        Prediction Data:
        {json.dumps(prediction_data, indent=2)}
        
        Provide a clear explanation that includes:
        1. The predicted price in a conversational format
        2. Key factors that influenced this prediction
        3. Confidence level and accuracy of the prediction
        4. How this compares to market averages
        5. Any recommendations for buyers/sellers
        
        Make it informative yet easy to understand for non-technical users.
        """
        
        return self.generate_response(prompt)
    
    def generate_market_insights(self, market_data: Dict[str, Any]) -> str:
        """
        Generate market insights from statistical data
        
        Parameters:
        -----------
        market_data : dict
            Market statistics and trends
            
        Returns:
        --------
        str : Natural language market insights
        """
        prompt = f"""
        Generate comprehensive market insights from the following data:
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Provide insights covering:
        1. Current market trends
        2. Price movements and patterns
        3. Popular locations and their characteristics
        4. Investment opportunities
        5. Market forecast and recommendations
        
        Make it engaging and valuable for real estate investors and buyers.
        """
        
        return self.generate_response(prompt)
    
    def answer_general_query(self, query: str, data_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer general real estate queries
        
        Parameters:
        -----------
        query : str
            The user's question
        data_context : dict, optional
            Relevant data context
            
        Returns:
        --------
        str : Natural language response
        """
        context = {
            "type": "general_query",
            "timestamp": datetime.now().isoformat(),
            "market": "Bengaluru",
            **(data_context or {})
        }
        
        return self.generate_response(query, context)


class GeminiChatbot:
    """High-level chatbot interface using Gemini"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.gemini_service = GeminiService()
        self.conversation_history = []
        self.context = {
            "system": "RealyticsAI",
            "capabilities": [
                "price_prediction",
                "market_analysis",
                "property_insights"
            ]
        }
        
        # Start chat session with context
        initial_context = """You are RealyticsAI Assistant, specialized in Bengaluru real estate.
        You can help with:
        - Property price predictions
        - Market analysis and trends
        - Property recommendations
        - Investment insights
        - General real estate queries
        
        Be helpful, professional, and provide data-driven insights."""
        
        self.gemini_service.start_chat_session(initial_context)
    
    def process_message(self, message: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a user message and generate response
        
        Parameters:
        -----------
        message : str
            User's message
        additional_context : dict, optional
            Additional context like prediction results
            
        Returns:
        --------
        str : Bot's response
        """
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Merge contexts
        full_context = {**self.context, **(additional_context or {})}
        
        # Get response
        response = self.gemini_service.chat(message, full_context)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        history_text = "\n".join([
            f"{item['role']}: {item['message']}"
            for item in self.conversation_history[-10:]  # Last 10 messages
        ])
        
        prompt = f"""
        Summarize the following conversation concisely:
        
        {history_text}
        
        Provide a brief summary of what was discussed and any key decisions or insights.
        """
        
        return self.gemini_service.generate_response(prompt)
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.gemini_service.start_chat_session()
        logger.info("Conversation reset")


# Create singleton instances
_gemini_service = None
_gemini_chatbot = None


def get_gemini_service() -> GeminiService:
    """Get or create Gemini service instance"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def get_gemini_chatbot() -> GeminiChatbot:
    """Get or create Gemini chatbot instance"""
    global _gemini_chatbot
    if _gemini_chatbot is None:
        _gemini_chatbot = GeminiChatbot()
    return _gemini_chatbot


__all__ = [
    "GeminiService",
    "GeminiChatbot",
    "get_gemini_service",
    "get_gemini_chatbot"
]
