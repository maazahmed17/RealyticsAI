#!/usr/bin/env python3
"""
RealyticsAI Integrated Chatbot
===============================
Main chatbot that orchestrates all features using Gemini API for natural language processing.
"""

import sys
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.core.config import settings
from backend.services.gemini_service import get_gemini_chatbot
from backend.services.price_prediction.nlp_interface import create_nlp_interface
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import print as rprint

logger = logging.getLogger(__name__)
console = Console()


class IntentType(Enum):
    """Types of user intents"""
    PRICE_PREDICTION = "price_prediction"
    PROPERTY_SEARCH = "property_search"
    MARKET_ANALYSIS = "market_analysis"
    NEGOTIATION_HELP = "negotiation_help"
    GENERAL_QUERY = "general_query"
    GREETING = "greeting"
    HELP = "help"
    EXIT = "exit"


class RealyticsAIChatbot:
    """Main chatbot orchestrator for RealyticsAI"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.gemini_chatbot = get_gemini_chatbot()
        self.price_predictor = None
        self.session_data = {
            "start_time": datetime.now(),
            "interactions": 0,
            "last_intent": None
        }
        
        # Initialize feature modules
        self._initialize_features()
        
        logger.info("RealyticsAI Chatbot initialized")
    
    def _initialize_features(self):
        """Initialize feature modules"""
        try:
            if settings.ENABLE_PRICE_PREDICTION:
                self.price_predictor = create_nlp_interface()
                logger.info("Price prediction feature loaded")
        except Exception as e:
            logger.error(f"Error initializing features: {e}")
    
    def classify_intent(self, message: str) -> IntentType:
        """Classify user intent from message"""
        
        prompt = f"""
        Classify the following user message into one of these categories:
        - price_prediction: User wants to know property price or get price estimate
        - property_search: User is looking for properties or recommendations
        - market_analysis: User wants market insights, trends, or analysis
        - negotiation_help: User needs help with price negotiation
        - greeting: User is greeting or starting conversation
        - help: User is asking for help or capabilities
        - exit: User wants to end conversation
        - general_query: General real estate question
        
        Message: {message}
        
        Return ONLY the category name, nothing else.
        """
        
        response = self.gemini_chatbot.gemini_service.generate_response(prompt).strip().lower()
        
        # Map response to IntentType
        intent_map = {
            "price_prediction": IntentType.PRICE_PREDICTION,
            "property_search": IntentType.PROPERTY_SEARCH,
            "market_analysis": IntentType.MARKET_ANALYSIS,
            "negotiation_help": IntentType.NEGOTIATION_HELP,
            "greeting": IntentType.GREETING,
            "help": IntentType.HELP,
            "exit": IntentType.EXIT,
            "general_query": IntentType.GENERAL_QUERY
        }
        
        return intent_map.get(response, IntentType.GENERAL_QUERY)
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process user message and return response
        
        Parameters:
        -----------
        message : str
            User's message
            
        Returns:
        --------
        dict : Response with intent, data, and formatted message
        """
        try:
            # Update session
            self.session_data["interactions"] += 1
            
            # Classify intent
            intent = self.classify_intent(message)
            self.session_data["last_intent"] = intent
            
            # Route to appropriate handler
            if intent == IntentType.PRICE_PREDICTION:
                response = self._handle_price_prediction(message)
            elif intent == IntentType.MARKET_ANALYSIS:
                response = self._handle_market_analysis(message)
            elif intent == IntentType.PROPERTY_SEARCH:
                response = self._handle_property_search(message)
            elif intent == IntentType.NEGOTIATION_HELP:
                response = self._handle_negotiation(message)
            elif intent == IntentType.GREETING:
                response = self._handle_greeting(message)
            elif intent == IntentType.HELP:
                response = self._handle_help(message)
            elif intent == IntentType.EXIT:
                response = self._handle_exit(message)
            else:
                response = self._handle_general_query(message)
            
            return {
                "success": True,
                "intent": intent.value,
                "message": message,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": self._generate_error_response(str(e))
            }
    
    def _handle_price_prediction(self, message: str) -> str:
        """Handle price prediction queries"""
        
        if not self.price_predictor:
            return "I apologize, but the price prediction feature is currently unavailable. Please try again later."
        
        try:
            # Process through NLP interface
            result = self.price_predictor.process_query(message)
            
            if result.get("success"):
                return result.get("response", "I couldn't generate a proper response.")
            else:
                return result.get("response", "I encountered an error processing your price query.")
                
        except Exception as e:
            logger.error(f"Price prediction error: {e}")
            return f"I apologize, but I couldn't process your price query: {str(e)}"
    
    def _handle_market_analysis(self, message: str) -> str:
        """Handle market analysis queries"""
        
        try:
            if self.price_predictor and self.price_predictor.data is not None:
                # Get market insights from data
                market_insights = self.price_predictor.get_market_insights()
                
                # Add user's specific question context
                prompt = f"""
                The user asked: {message}
                
                Here are the market insights:
                {market_insights}
                
                Please provide a specific answer to the user's question using these insights.
                Make it conversational and helpful.
                """
                
                return self.gemini_chatbot.gemini_service.generate_response(prompt)
            else:
                # General market response without data
                return self.gemini_chatbot.process_message(
                    message,
                    {"context": "market_analysis", "market": "Bengaluru"}
                )
                
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return "I apologize, but I couldn't retrieve market analysis at this time."
    
    def _handle_property_search(self, message: str) -> str:
        # backend/services/chatbot_orchestrator/realytics_chatbot.py

# ... inside the RealyticsAIChatbot class ...

    def _handle_property_search(self, message: str) -> str:
        """Handle property search/recommendation queries by calling the recommendation API."""
        
        # Step 1: Extract user preferences from the message.
        # We reuse the feature extractor from the price predictor.
        if not self.price_predictor:
            return "I apologize, but the recommendation service is currently unavailable."

        try:
            # Extract features like BHK, bath, and budget (as price).
            features = self.price_predictor._extract_features_from_query(message)
            if not features:
                return "I can help you find properties! Please tell me what you're looking for, for example: 'Find me a 3 BHK with 2 bathrooms under 1.5 crores.'"

            # Step 2: Prepare the request for the recommendation API.
            # The API expects 'bhk', 'bath', and 'price' (as budget).
            api_payload = {
                "bhk": features.get("bhk"),
                "bath": features.get("bath"),
                "price": features.get("price") or features.get("budget") # Use price or budget if available
            }

            # Ensure we have the minimum required fields for a recommendation.
            if not all([api_payload["bhk"], api_payload["bath"], api_payload["price"]]):
                return "To give you good recommendations, I need at least the number of bedrooms (BHK), bathrooms, and your budget. For example: 'Recommend a 2 BHK with 2 baths for 80 lakhs.'"

            # Step 3: Call the recommendation API endpoint.
            api_url = "http://localhost:8000/api/v1/recommend" # Assumes the main app runs on port 8000
            response = requests.post(api_url, json=api_payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            recommendations = response.json()

            # Step 4: Format the API response into a user-friendly message.
            if not recommendations:
                return "I couldn't find any properties that match your criteria. You might want to try adjusting your budget or preferences."

            response_str = "Here are a few properties you might like:\n\n"
            for prop in recommendations:
                # Safely get property details with defaults
                size = prop.get('size', 'N/A')
                location = prop.get('location', 'N/A')
                price = prop.get('price', 0)
                bath = prop.get('bath', 'N/A')
                
                response_str += f"• **{size}** in **{location}**\n"
                response_str += f"  - Price: **₹{price:.2f} Lakhs**, Bathrooms: {bath}\n\n"
            
            response_str += "Would you like me to refine these recommendations or get a detailed price estimate for one of them?"
            return response_str

        except requests.exceptions.RequestException as e:
            # Handle cases where the API is down or there's a network issue
            print(f"API call failed: {e}") # Log the error for debugging
            return "I'm having trouble connecting to the recommendation service right now. Please try again in a moment."
        except Exception as e:
            print(f"An error occurred in property search: {e}") # Log the error
            return "I encountered an unexpected error while searching for properties. Please try rephrasing your request."
    #  return response
    
    def _handle_negotiation(self, message: str) -> str:
        """Handle negotiation assistance queries"""
        
        response = """I can help you with negotiation strategies!

The automated negotiation agent is under development, but I can provide valuable insights:

**Negotiation Tips for Bengaluru Real Estate:**

1. **Research Market Prices** - I can help you understand the fair price for any property
2. **Identify Negotiation Points** - Age of property, amenities, location advantages
3. **Timing Matters** - End of financial year often sees better deals
4. **Standard Negotiation Range** - 5-10% is typical in Bengaluru market

Would you like me to:
- Analyze a specific property's price to help with negotiation?
- Provide market comparisons for bargaining power?
- Suggest negotiation strategies based on current market conditions?

What specific help do you need with negotiation?"""
        
        return response
    
    def _handle_greeting(self, message: str) -> str:
        """Handle greetings"""
        
        response = """Hello! Welcome to RealyticsAI 🏠

I'm your intelligent real estate assistant specializing in Bengaluru property market. I can help you with:

✨ **Price Predictions** - Get instant property valuations
📊 **Market Analysis** - Understand trends and insights
🔍 **Property Search** - Find your dream property (coming soon)
💰 **Negotiation Help** - Get the best deals

How can I assist you today? You can ask questions like:
- "What's the price of a 2BHK in Koramangala?"
- "Show me market trends for Bengaluru"
- "Which areas are best for investment?"

Feel free to ask anything about Bengaluru real estate!"""
        
        return response
    
    def _handle_help(self, message: str) -> str:
        """Handle help requests"""
        
        response = """**RealyticsAI Help Guide** 📚

Here's what I can do for you:

**1. Price Predictions** 💰
   - Example: "Predict price for 3BHK in Whitefield with 1500 sqft"
   - Example: "What's the value of 2BHK apartment in Indiranagar?"

**2. Market Analysis** 📊
   - Example: "What's the average price in Bengaluru?"
   - Example: "Which are the most expensive areas?"
   - Example: "Show me price trends"

**3. Investment Insights** 📈
   - Example: "Is it good time to buy property?"
   - Example: "Which areas have best appreciation?"

**4. General Queries** ❓
   - Example: "What factors affect property prices?"
   - Example: "How is Bengaluru real estate market?"

**Tips for Best Results:**
- Be specific about location, size, and features
- Mention BHK type (1BHK, 2BHK, 3BHK, etc.)
- Include square footage if known
- Specify amenities (bathrooms, balconies)

What would you like to know?"""
        
        return response
    
    def _handle_exit(self, message: str) -> str:
        """Handle exit/goodbye"""
        
        # Get conversation summary
        summary = self.gemini_chatbot.get_conversation_summary()
        
        response = f"""Thank you for using RealyticsAI! 👋

**Session Summary:**
- Duration: {(datetime.now() - self.session_data['start_time']).seconds // 60} minutes
- Interactions: {self.session_data['interactions']}

{summary}

Feel free to come back anytime for:
- Property price predictions
- Market insights
- Investment advice

Have a great day and good luck with your real estate journey! 🏠✨"""
        
        return response
    
    def _handle_general_query(self, message: str) -> str:
        """Handle general queries"""
        
        # Use Gemini with context about Bengaluru real estate
        context = {
            "type": "general_real_estate",
            "market": "Bengaluru",
            "capabilities": ["price_prediction", "market_analysis"],
            "data_available": self.price_predictor is not None
        }
        
        return self.gemini_chatbot.process_message(message, context)
    
    def _generate_error_response(self, error: str) -> str:
        """Generate user-friendly error response"""
        
        return f"""I apologize for the inconvenience. 

It seems I encountered an issue: {error}

Please try:
1. Rephrasing your question
2. Being more specific about the property details
3. Asking about a different aspect

You can always type 'help' to see what I can do for you.

How else can I assist you?"""
    
    def start_interactive_session(self):
        """Start an interactive chat session"""
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]🏠 Welcome to RealyticsAI - Your Intelligent Real Estate Assistant[/bold cyan]")
        console.print("="*80)
        
        # Display greeting
        greeting = self._handle_greeting("")
        console.print(Panel(Markdown(greeting), border_style="green"))
        
        console.print("\n[dim]Type 'exit' or 'quit' to end the conversation[/dim]\n")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    # Handle exit
                    response = self._handle_exit(user_input)
                    console.print(f"\n[bold green]RealyticsAI:[/bold green]")
                    console.print(Panel(Markdown(response), border_style="yellow"))
                    break
                
                # Process message
                result = self.process_message(user_input)
                
                # Display response
                console.print(f"\n[bold green]RealyticsAI:[/bold green]")
                
                if result.get("success"):
                    console.print(Panel(
                        Markdown(result.get("response", "")),
                        title=f"[dim]Intent: {result.get('intent', 'unknown')}[/dim]",
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        result.get("response", "An error occurred"),
                        border_style="red"
                    ))
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Session interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                console.print("[yellow]Please try again or type 'help' for assistance[/yellow]")
    
    def get_status(self) -> Dict[str, Any]:
        """Get chatbot status"""
        
        return {
            "status": "active",
            "session_data": self.session_data,
            "features": {
                "price_prediction": settings.ENABLE_PRICE_PREDICTION and self.price_predictor is not None,
                "property_recommendation": settings.ENABLE_PROPERTY_RECOMMENDATION,
                "negotiation_agent": settings.ENABLE_NEGOTIATION_AGENT
            },
            "gemini_connected": self.gemini_chatbot is not None,
            "uptime_minutes": (datetime.now() - self.session_data["start_time"]).seconds // 60
        }


def main():
    """Main entry point for the chatbot"""
    
    try:
        # Install required package if needed
        import subprocess
        subprocess.run(["pip", "install", "google-generativeai", "--quiet"], check=False)
        
        # Create and start chatbot
        chatbot = RealyticsAIChatbot()
        chatbot.start_interactive_session()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Chatbot terminated by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
