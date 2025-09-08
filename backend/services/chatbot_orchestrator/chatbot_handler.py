"""
RealyticsAI Chatbot Handler
============================
Natural language interface for property price predictions
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
import asyncio
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import price prediction service
from services.price_prediction.main import PricePredictionSystem, Config

class Intent(Enum):
    """User intent types"""
    PRICE_PREDICTION = "price_prediction"
    MARKET_ANALYSIS = "market_analysis"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"

class ConversationState:
    """Maintains conversation context"""
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}
        self.pending_clarification: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}
        
    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_last_n_messages(self, n: int = 5) -> List[Dict[str, str]]:
        return self.history[-n:] if len(self.history) >= n else self.history
    
    def clear_context(self):
        self.context = {}
        self.pending_clarification = None

class RealyticsAIChatbot:
    """Main chatbot class for natural language property price queries"""
    
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None):
        """
        Initialize chatbot
        
        Args:
            use_openai: Whether to use OpenAI GPT for NLU
            openai_api_key: OpenAI API key if using GPT
        """
        self.use_openai = use_openai
        self.conversation_state = ConversationState()
        self.price_predictor = None
        
        # Initialize price prediction system
        self._initialize_predictor()
        
        # If using OpenAI
        if use_openai and openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except ImportError:
                print("OpenAI library not installed. Using rule-based approach.")
                self.use_openai = False
        
        # Number patterns and mappings
        self.number_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "no": 0, "none": 0, "single": 1, "double": 2,
            "triple": 3, "a": 1, "an": 1
        }
        
        # Common location names in Bengaluru
        self.known_locations = [
            "whitefield", "electronic city", "koramangala", "indiranagar",
            "jayanagar", "btm layout", "hsr layout", "marathahalli",
            "hebbal", "yelahanka", "sarjapur", "bellandur", "jp nagar",
            "rajajinagar", "malleswaram", "basavanagudi", "banashankari"
        ]
    
    def _initialize_predictor(self):
        """Initialize the price prediction system"""
        try:
            self.price_predictor = PricePredictionSystem()
            # Load data and train model
            if self.price_predictor.load_data():
                self.price_predictor.train_model("linear")
        except Exception as e:
            print(f"Warning: Could not initialize price predictor: {e}")
    
    def classify_intent(self, query: str) -> Intent:
        """
        Classify user intent from natural language query
        
        Args:
            query: User's natural language input
            
        Returns:
            Intent enum value
        """
        query_lower = query.lower()
        
        # Greeting patterns
        if any(word in query_lower for word in ["hello", "hi", "hey", "good morning", "good evening"]):
            return Intent.GREETING
        
        # Help patterns
        if any(word in query_lower for word in ["help", "how to", "guide", "what can you"]):
            return Intent.HELP
        
        # Price prediction patterns
        price_keywords = ["price", "cost", "worth", "value", "how much", "estimate", "predict"]
        if any(keyword in query_lower for keyword in price_keywords):
            if "compare" in query_lower or "versus" in query_lower or "vs" in query_lower:
                return Intent.COMPARISON
            return Intent.PRICE_PREDICTION
        
        # Market analysis patterns
        if any(word in query_lower for word in ["market", "analysis", "trend", "average", "statistics"]):
            return Intent.MARKET_ANALYSIS
        
        # Recommendation patterns
        if any(word in query_lower for word in ["recommend", "suggest", "should i", "best", "under", "budget"]):
            return Intent.RECOMMENDATION
        
        return Intent.UNKNOWN
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract property features from natural language
        
        Args:
            query: User's natural language input
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {
            "bathrooms": None,
            "balconies": None,
            "location": None,
            "budget": None,
            "property_type": None
        }
        
        query_lower = query.lower()
        
        # Extract bathrooms
        bathroom_patterns = [
            r'(\d+)\s*(?:bath(?:room)?s?)',
            r'(?:bath(?:room)?s?)\s*[:=]?\s*(\d+)',
        ]
        
        for pattern in bathroom_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities["bathrooms"] = int(match.group(1))
                break
        
        # Check for word numbers in bathrooms
        if entities["bathrooms"] is None:
            for word, num in self.number_words.items():
                if f"{word} bath" in query_lower:
                    entities["bathrooms"] = num
                    break
        
        # Extract balconies
        balcony_patterns = [
            r'(\d+)\s*balcon(?:y|ies)',
            r'balcon(?:y|ies)\s*[:=]?\s*(\d+)',
        ]
        
        for pattern in balcony_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities["balconies"] = int(match.group(1))
                break
        
        # Check for word numbers in balconies
        if entities["balconies"] is None:
            for word, num in self.number_words.items():
                if f"{word} balcon" in query_lower:
                    entities["balconies"] = num
                    break
            
            # Check for "no balcony" or "without balcony"
            if "no balcon" in query_lower or "without balcon" in query_lower:
                entities["balconies"] = 0
        
        # Extract location
        for location in self.known_locations:
            if location in query_lower:
                entities["location"] = location.title()
                break
        
        # Extract budget (in lakhs)
        budget_patterns = [
            r'(\d+)\s*(?:lakh|lac|l)',
            r'under\s*(\d+)\s*(?:lakh|lac|l)?',
            r'budget\s*(?:of|is)?\s*(\d+)',
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities["budget"] = float(match.group(1))
                break
        
        # Extract property type
        if "apartment" in query_lower or "flat" in query_lower:
            entities["property_type"] = "apartment"
        elif "house" in query_lower or "villa" in query_lower:
            entities["property_type"] = "house"
        
        # Extract BHK if mentioned
        bhk_match = re.search(r'(\d+)\s*bhk', query_lower)
        if bhk_match:
            bhk = int(bhk_match.group(1))
            # Approximate bathrooms from BHK if not specified
            if entities["bathrooms"] is None:
                entities["bathrooms"] = max(1, bhk - 1)
        
        return entities
    
    def generate_response(self, intent: Intent, entities: Dict[str, Any], 
                         prediction: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate natural language response
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            prediction: Price prediction results
            
        Returns:
            Natural language response
        """
        if intent == Intent.GREETING:
            return ("Hello! I'm RealyticsAI, your property price assistant. "
                   "I can help you estimate property prices, analyze the market, "
                   "and provide recommendations. How can I assist you today?")
        
        elif intent == Intent.HELP:
            return self._generate_help_response()
        
        elif intent == Intent.PRICE_PREDICTION:
            if prediction:
                return self._generate_prediction_response(entities, prediction)
            else:
                return self._generate_clarification_request(entities)
        
        elif intent == Intent.MARKET_ANALYSIS:
            return self._generate_market_analysis_response(entities)
        
        elif intent == Intent.RECOMMENDATION:
            return self._generate_recommendation_response(entities)
        
        elif intent == Intent.COMPARISON:
            return self._generate_comparison_response(entities)
        
        else:
            return ("I'm not sure I understood your query. Could you please rephrase it? "
                   "I can help with property price predictions, market analysis, and recommendations.")
    
    def _generate_help_response(self) -> str:
        """Generate help message"""
        return """I can help you with:

📊 **Price Predictions**: Ask me about property prices
   Example: "What's the price of a 2 bathroom apartment with 1 balcony?"

📈 **Market Analysis**: Get insights about the real estate market
   Example: "Show me market analysis for Whitefield"

🏠 **Recommendations**: Find properties within your budget
   Example: "What can I get under 50 lakhs?"

🔍 **Comparisons**: Compare different areas or property types
   Example: "Compare prices between Whitefield and Electronic City"

Just ask me naturally, and I'll help you find the information you need!"""
    
    def _generate_prediction_response(self, entities: Dict[str, Any], 
                                     prediction: Dict[str, Any]) -> str:
        """Generate response for price prediction"""
        bath = entities.get("bathrooms", 2)
        balcony = entities.get("balconies", 1)
        
        response = f"Based on my analysis, a property with **{bath} bathroom(s) and {balcony} balcony(ies)**"
        
        if entities.get("location"):
            response += f" in **{entities['location']}**"
        
        # Add price prediction
        if "hybrid_prediction" in prediction:
            price = prediction["hybrid_prediction"]
            response += f" is estimated at **₹{price:.2f} lakhs**.\n\n"
            
            # Add confidence based on similar properties
            if prediction.get("similar_properties_count", 0) > 100:
                response += "✅ This is a high-confidence estimate based on "
            elif prediction.get("similar_properties_count", 0) > 50:
                response += "📊 This is a moderate-confidence estimate based on "
            else:
                response += "⚠️ This is a preliminary estimate based on "
            
            response += f"{prediction.get('similar_properties_count', 0)} similar properties in our database.\n\n"
            
            # Add price range
            if "price_range" in prediction:
                response += (f"💰 **Price Range**: ₹{prediction['price_range']['min']:.1f} - "
                           f"₹{prediction['price_range']['max']:.1f} lakhs\n")
            
            # Add market insight
            if prediction.get("similar_avg_price"):
                avg = prediction["similar_avg_price"]
                if price < avg * 0.9:
                    response += "📉 This estimate is below the market average - could be a good value!\n"
                elif price > avg * 1.1:
                    response += "📈 This is in the premium range for such properties.\n"
                else:
                    response += "⚖️ This is aligned with current market rates.\n"
        else:
            price = prediction.get("ml_prediction", 0)
            response += f" is estimated at **₹{price:.2f} lakhs**."
        
        # Add suggestion
        response += "\nWould you like me to show you a detailed market analysis or compare with other areas?"
        
        return response
    
    def _generate_clarification_request(self, entities: Dict[str, Any]) -> str:
        """Generate clarification request for missing information"""
        missing = []
        
        if entities.get("bathrooms") is None:
            missing.append("number of bathrooms")
        if entities.get("balconies") is None:
            missing.append("number of balconies")
        
        if missing:
            response = "I'd be happy to estimate the property price! "
            response += f"Could you please tell me the {' and '.join(missing)}?\n\n"
            response += "For example:\n"
            response += "• '2 bathrooms and 1 balcony'\n"
            response += "• 'A house with 3 bathrooms and 2 balconies'\n"
            response += "• '2BHK apartment with 2 bathrooms'"
            return response
        
        return "I need more information to provide an accurate estimate. Please specify the property features."
    
    def _generate_market_analysis_response(self, entities: Dict[str, Any]) -> str:
        """Generate market analysis response"""
        if not self.price_predictor or not self.price_predictor.data:
            return "Market analysis is currently unavailable. Please try again later."
        
        data = self.price_predictor.data
        
        response = "📊 **Current Market Analysis**\n\n"
        
        if entities.get("location"):
            location = entities["location"]
            loc_data = data[data['location'].str.contains(location, case=False, na=False)]
            
            if not loc_data.empty:
                response += f"**{location} Market Overview:**\n"
                response += f"• Properties Available: {len(loc_data)}\n"
                response += f"• Average Price: ₹{loc_data['price'].mean():.2f} lakhs\n"
                response += f"• Price Range: ₹{loc_data['price'].min():.1f} - ₹{loc_data['price'].max():.1f} lakhs\n"
                response += f"• Median Price: ₹{loc_data['price'].median():.2f} lakhs\n"
            else:
                response += f"No specific data available for {location}.\n\n"
        
        # Overall market stats
        response += "\n**Overall Bengaluru Market:**\n"
        response += f"• Total Properties: {len(data):,}\n"
        response += f"• Average Price: ₹{data['price'].mean():.2f} lakhs\n"
        response += f"• Most Active Location: {data['location'].value_counts().index[0]}\n"
        
        # Top locations
        top_3 = data['location'].value_counts().head(3)
        response += "\n**Top Locations by Activity:**\n"
        for loc, count in top_3.items():
            avg_price = data[data['location'] == loc]['price'].mean()
            response += f"• {loc}: {count} properties (Avg: ₹{avg_price:.2f}L)\n"
        
        return response
    
    def _generate_recommendation_response(self, entities: Dict[str, Any]) -> str:
        """Generate property recommendations"""
        if not self.price_predictor or not self.price_predictor.data:
            return "Recommendations are currently unavailable. Please try again later."
        
        data = self.price_predictor.data
        response = "🏠 **Property Recommendations**\n\n"
        
        if entities.get("budget"):
            budget = entities["budget"]
            
            # Filter properties under budget
            affordable = data[data['price'] <= budget]
            
            if not affordable.empty:
                response += f"**Properties under ₹{budget} lakhs:**\n\n"
                
                # Group by location and show average
                location_stats = affordable.groupby('location')['price'].agg(['mean', 'count'])
                location_stats = location_stats[location_stats['count'] >= 5]  # At least 5 properties
                location_stats = location_stats.sort_values('mean').head(5)
                
                response += "**Best Value Locations:**\n"
                for location, stats in location_stats.iterrows():
                    response += f"• **{location}**: Avg ₹{stats['mean']:.2f}L ({int(stats['count'])} properties)\n"
                
                # Property type recommendations
                response += f"\n**Typical Properties in your budget:**\n"
                bath_stats = affordable.groupby('bath')['price'].mean().sort_index()
                for bath, avg_price in bath_stats.head(3).items():
                    if not pd.isna(bath):
                        count = len(affordable[affordable['bath'] == bath])
                        response += f"• {int(bath)} bathroom(s): ₹{avg_price:.2f}L avg ({count} options)\n"
            else:
                response += f"Limited options under ₹{budget} lakhs. Consider:\n"
                response += "• Expanding your search area\n"
                response += "• Looking at emerging localities\n"
                response += "• Considering smaller configurations\n"
        else:
            response += "Please specify your budget for personalized recommendations.\n"
            response += "Example: 'Show me properties under 100 lakhs'"
        
        return response
    
    def _generate_comparison_response(self, entities: Dict[str, Any]) -> str:
        """Generate comparison between areas or property types"""
        response = "📊 **Property Comparison**\n\n"
        
        # This would need more sophisticated entity extraction for multiple locations
        response += "To compare properties, please specify:\n"
        response += "• Two locations (e.g., 'Compare Whitefield vs Electronic City')\n"
        response += "• Property configurations (e.g., '2 bathroom vs 3 bathroom properties')\n"
        response += "• Price ranges\n\n"
        response += "I'll provide detailed comparisons including price differences, availability, and market trends."
        
        return response
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Main method to process user queries
        
        Args:
            query: User's natural language input
            session_id: Optional session identifier for context management
            
        Returns:
            Natural language response
        """
        # Add to conversation history
        self.conversation_state.add_message("user", query)
        
        # Classify intent
        intent = self.classify_intent(query)
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Handle based on intent
        prediction = None
        if intent == Intent.PRICE_PREDICTION:
            # Check if we have enough information
            if entities.get("bathrooms") is not None and entities.get("balconies") is not None:
                # Get prediction from price service
                if self.price_predictor and self.price_predictor.is_trained:
                    features = {
                        "bath": entities["bathrooms"],
                        "balcony": entities["balconies"]
                    }
                    prediction = self.price_predictor.predict_price(features)
        
        # Generate response
        response = self.generate_response(intent, entities, prediction)
        
        # Add to conversation history
        self.conversation_state.add_message("assistant", response)
        
        return response
    
    def process_query_sync(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Synchronous version of process_query for non-async contexts
        
        Args:
            query: User's natural language input
            session_id: Optional session identifier
            
        Returns:
            Natural language response
        """
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_query(query, session_id))


# Example usage and testing
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = RealyticsAIChatbot(use_openai=False)
    
    # Test queries
    test_queries = [
        "Hello!",
        "What's the price of a house with 2 bathrooms and 1 balcony?",
        "How much would a 3 bathroom apartment with 2 balconies cost?",
        "Show me properties under 50 lakhs",
        "Give me market analysis for Whitefield",
        "I need a house with three bathrooms and no balcony",
        "Compare prices between 2 bathroom and 3 bathroom properties",
        "Help me find a good property",
        "What can you do?",
    ]
    
    print("=" * 80)
    print("🤖 RealyticsAI Chatbot Testing")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        response = chatbot.process_query_sync(query)
        print(f"🤖 Bot: {response}")
        print("-" * 40)
