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
from backend.services.price_prediction.fixed_price_predictor import get_price_predictor
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
                self.price_predictor = get_price_predictor()
                logger.info("✅ Fixed price prediction model loaded (no data leakage)")
        except Exception as e:
            logger.error(f"Error initializing features: {e}")
    
    def classify_intent(self, message: str) -> IntentType:
        """Classify user intent from message"""
        
        message_lower = message.lower()
        
        # Rule-based detection for high-confidence cases
        # Check for negotiation keywords FIRST (highest priority)
        negotiation_keywords = ['negotiate', 'negotiation', 'bargain', 'deal', 'offer', 'counter', 'budget']
        asking_price_indicators = ['asking', 'listed at', 'priced at', 'listed for']
        
        has_negotiation_keyword = any(keyword in message_lower for keyword in negotiation_keywords)
        has_asking_price = any(indicator in message_lower for indicator in asking_price_indicators)
        
        # If both negotiation keyword AND asking price mentioned → definitely negotiation
        if has_negotiation_keyword and has_asking_price:
            return IntentType.NEGOTIATION_HELP
        
        # If message has "budget" + "asking" → negotiation
        if 'budget' in message_lower and has_asking_price:
            return IntentType.NEGOTIATION_HELP
        
        # Check for other clear patterns
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return IntentType.GREETING
        
        if any(word in message_lower for word in ['help', 'how to', 'guide', 'tutorial']):
            return IntentType.HELP
        
        if any(word in message_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            return IntentType.EXIT
        
        # Use Gemini for ambiguous cases
        prompt = f"""
        Classify the following user message into one of these categories:
        - price_prediction: User wants to ESTIMATE/PREDICT property price (e.g., "what's the price of a 2BHK?")
        - property_search: User is LOOKING FOR or wants RECOMMENDATIONS for properties
        - market_analysis: User wants market insights, trends, or analysis
        - negotiation_help: User needs help with NEGOTIATING a deal, has target/budget vs asking price
        - general_query: General real estate question
        
        IMPORTANT: 
        - If message mentions BOTH an asking price AND a budget/target/offer → negotiation_help
        - If message says "help me negotiate" → negotiation_help
        - If message only asks for price estimate → price_prediction
        
        Message: {message}
        
        Return ONLY the category name, nothing else.
        """
        
        try:
            response = self.gemini_chatbot.gemini_service.generate_response(prompt).strip().lower()
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return IntentType.GENERAL_QUERY
        
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
            # Extract property features from message using Gemini
            features = self._extract_property_features(message)
            
            if not features:
                return "I couldn't extract property details from your query. Please specify location, BHK, bathrooms, balconies, and square footage."
            
            # Make prediction using fixed model
            result = self.price_predictor.predict(features)
            
            if result.get("success"):
                price = result['price_formatted']
                confidence = result['confidence']
                model_name = result['model']
                
                response = f"""Based on the property details you provided:
                
🏠 **Property Details:**
- Location: {features.get('location', 'N/A')}
- BHK: {features.get('bhk', 'N/A')}
- Bathrooms: {features.get('bath', 'N/A')}
- Balconies: {features.get('balcony', 'N/A')}
- Area: {features.get('total_sqft', 'N/A')} sq.ft

💰 **Estimated Price:** {price}
📊 **Confidence:** {confidence:.0%}
🤖 **Model:** {model_name}

This prediction is based on {result['features_used']} features and uses our latest XGBoost model trained on 150,000 Bangalore properties."""
                return response
            else:
                return f"I encountered an error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Price prediction error: {e}")
            return f"I apologize, but I couldn't process your price query: {str(e)}"
    
    def _extract_property_features(self, message: str) -> dict:
        """Extract property features from natural language using Gemini"""
        prompt = f"""Extract property details from this query and return ONLY a JSON object:
        
        Query: "{message}"
        
        Return JSON with these fields (use null if not mentioned):
        {{
            "location": "location name",
            "bhk": number,
            "bath": number,
            "balcony": number,
            "total_sqft": number
        }}
        
        Common Bangalore locations: Whitefield, Electronic City, Koramangala, HSR Layout, BTM Layout, etc.
        
        Return ONLY valid JSON, nothing else.
        """
        
        try:
            response = self.gemini_chatbot.gemini_service.generate_response(prompt)
            import re
            import json
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group())
                # Set defaults for missing values
                defaults = {'bhk': 2, 'bath': 2, 'balcony': 1, 'total_sqft': 1200}
                for key, default in defaults.items():
                    if key not in features or features[key] is None:
                        features[key] = default
                return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
        
        return {}
    
    def _handle_market_analysis(self, message: str) -> str:
        """Handle market analysis queries"""
        
        try:
            # Use Gemini for market analysis
            prompt = f"""
            The user asked about Bangalore real estate market: {message}
            
            Provide insights about:
            - Average property prices in different areas
            - Market trends
            - Popular locations
            - Price ranges for different property types
            
            Be specific and helpful. Use data about Bangalore real estate.
            """
            
            return self.gemini_chatbot.gemini_service.generate_response(prompt)
                
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return "I apologize, but I couldn't retrieve market analysis at this time."
    
    def _handle_property_search(self, message: str) -> str:
        """Handle property search/recommendation queries"""
        return "Property search and recommendations are coming soon! For now, I can help you with price predictions and market analysis."
    
    def _handle_negotiation(self, message: str) -> str:
        """Handle negotiation assistance queries"""
        
        try:
            # Extract negotiation details from message
            import re
            
            # Look for asking price pattern
            asking_match = re.search(r'asking\s+(?:price\s+)?(?:is\s+)?(?:₹)?([\d.]+)\s*(?:lakhs?|l)', message, re.IGNORECASE)
            # Look for target price pattern
            target_match = re.search(r'(?:target|budget|offer|pay)\s+(?:price\s+)?(?:is\s+)?(?:₹)?([\d.]+)\s*(?:lakhs?|l)', message, re.IGNORECASE)
            
            # Extract location/property details
            location_patterns = [
                r'in\s+([A-Za-z\s]+?)(?:\s+of|,|\.|$)',
                r'at\s+([A-Za-z\s]+?)(?:\s+of|,|\.|$)',
                r'property\s+in\s+([A-Za-z\s]+?)(?:\s+of|,|\.|$)'
            ]
            location = None
            for pattern in location_patterns:
                loc_match = re.search(pattern, message, re.IGNORECASE)
                if loc_match:
                    location = loc_match.group(1).strip()
                    break
            
            # Extract BHK
            bhk_match = re.search(r'(\d+)\s*bhk', message, re.IGNORECASE)
            bhk = int(bhk_match.group(1)) if bhk_match else None
            
            asking_price = float(asking_match.group(1)) if asking_match else None
            target_price = float(target_match.group(1)) if target_match else None
            
            # If we have both prices, perform negotiation analysis
            if asking_price and target_price:
                import httpx
                import asyncio
                
                property_id = f"prop-{location or 'unknown'}-{bhk or 'N'}bhk".replace(' ', '-')
                
                # Call negotiation API
                try:
                    async def call_negotiate_api():
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            resp = await client.post(
                                "http://localhost:8000/api/negotiate/start",
                                json={
                                    "property_id": property_id,
                                    "target_price": target_price,
                                    "user_role": "buyer",
                                    "asking_price": asking_price,
                                    "initial_message": ""
                                }
                            )
                            if resp.status_code == 200:
                                return resp.json()
                            return None
                    
                    # Run async call
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(call_negotiate_api())
                    loop.close()
                    
                    if result:
                        return result.get('agent_opening', 'Negotiation analysis completed.')
                except Exception as e:
                    logger.error(f"Negotiation API call failed: {e}")
            
            # If we have asking price but no target, ask for target
            if asking_price and not target_price:
                return f"""I can help you analyze this negotiation!

🏠 **Property Details:**
{f'- Location: {location}' if location else ''}
{f'- Type: {bhk} BHK' if bhk else ''}
- Asking Price: ₹{asking_price} Lakhs

💡 To provide negotiation advice, please tell me:
**What is your target price?**

For example: "My budget is 85 lakhs" or "I want to offer 80 lakhs"

I'll analyze if your target is compatible with the asking price and provide professional advice on whether to proceed with the offer."""
            
            # If we have target but no asking, ask for asking
            if target_price and not asking_price:
                return f"""I can help you with this negotiation!

💰 Your Budget: ₹{target_price} Lakhs
{f'🏠 Location: {location}' if location else ''}
{f'🏠 Type: {bhk} BHK' if bhk else ''}

💡 To provide negotiation advice, please tell me:
**What is the property's asking price?**

For example: "The property is asking 95 lakhs" or "Listed at 100 lakhs"

I'll analyze if your budget is compatible and provide advice on negotiation strategy."""
            
            # Generic negotiation help
            return """I can help you analyze property negotiations! 💼

**How to use:**
Tell me about the property and prices, for example:
- "Help me negotiate for a property in RT Nagar of 3BHK which is asking 100 lakhs"
- "I want to buy a 2BHK in Whitefield for 85 lakhs but it's listed at 95 lakhs"
- "Property is asking 120 lakhs, my budget is 110 lakhs"

**What I'll provide:**
✅ Compatibility assessment (Is your target reasonable?)
✅ Market perspective (Good negotiation starting point?)
✅ Clear recommendation (Should you proceed?)
✅ Next steps and strategies

Just describe your situation and I'll analyze it for you!"""
            
        except Exception as e:
            logger.error(f"Negotiation handling error: {e}")
            return """I can help with negotiation! Please provide:
1. Property asking price (e.g., "asking 100 lakhs")
2. Your target price (e.g., "my budget is 90 lakhs")

I'll analyze the compatibility and provide professional advice."""
        
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
