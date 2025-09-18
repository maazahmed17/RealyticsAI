"""
Intelligent Query Router for RealyticsAI
========================================
Routes queries to appropriate services (price prediction vs property recommendation) and formats responses.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

# Import RealyticsAI services
from .gemini_service import get_gemini_service
from .recommendation_service import RecommendationEngine
from .recommendation_service.query_analyzer import QueryAnalyzer, QueryIntent

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Available services in RealyticsAI"""
    PRICE_PREDICTION = "price_prediction"
    PROPERTY_RECOMMENDATION = "property_recommendation"
    MARKET_ANALYSIS = "market_analysis"
    GENERAL_CHAT = "general_chat"

class IntelligentRouter:
    """Routes user queries to appropriate RealyticsAI services"""
    
    def __init__(self):
        """Initialize the intelligent router"""
        
        logger.info("Initializing RealyticsAI Intelligent Router")
        
        # Initialize services
        self.gemini_service = get_gemini_service()
        self.query_analyzer = QueryAnalyzer()
        self.recommendation_engine = None
        
        # Initialize recommendation engine
        self._initialize_recommendation_engine()
        
        # Routing rules
        self.routing_rules = {
            QueryIntent.PRICE_PREDICTION: ServiceType.PRICE_PREDICTION,
            QueryIntent.PROPERTY_RECOMMENDATION: ServiceType.PROPERTY_RECOMMENDATION,
            QueryIntent.MARKET_ANALYSIS: ServiceType.MARKET_ANALYSIS,
            QueryIntent.GENERAL_INQUIRY: ServiceType.GENERAL_CHAT
        }
        
        logger.info("Intelligent Router initialized successfully")
    
    def _initialize_recommendation_engine(self):
        """Initialize the recommendation engine"""
        try:
            logger.info("Initializing recommendation engine...")
            self.recommendation_engine = RecommendationEngine()
            logger.info("Recommendation engine ready")
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            self.recommendation_engine = None
    
    def process_query(self, 
                     query: str, 
                     user_context: Optional[Dict[str, Any]] = None,
                     chatbot_state: Optional[Any] = None) -> Dict[str, Any]:
        """Process a user query and route to appropriate service
        
        Args:
            query: User's query
            user_context: Optional user context
            chatbot_state: Optional chatbot state for price prediction
            
        Returns:
            Dictionary with processed response
        """
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Analyze the query
            analysis = self.query_analyzer.analyze_query(query)
            query_intent = analysis["intent"]
            confidence = analysis["confidence"]
            
            logger.info(f"Query intent: {query_intent.value} (confidence: {confidence:.2f})")
            
            # Route to appropriate service
            service_type = self.routing_rules.get(query_intent, ServiceType.GENERAL_CHAT)
            
            if service_type == ServiceType.PROPERTY_RECOMMENDATION:
                return self._handle_recommendation_query(query, analysis, user_context)
            elif service_type == ServiceType.PRICE_PREDICTION:
                return self._handle_price_prediction_query(query, analysis, chatbot_state)
            elif service_type == ServiceType.MARKET_ANALYSIS:
                return self._handle_market_analysis_query(query, analysis, user_context)
            else:
                return self._handle_general_query(query, analysis, user_context)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(query, str(e))
    
    def _handle_recommendation_query(self, 
                                   query: str, 
                                   analysis: Dict[str, Any], 
                                   user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle property recommendation queries"""
        
        if not self.recommendation_engine:
            return self._create_error_response(
                query, 
                "Recommendation service is not available. Please try again later."
            )
        
        try:
            # Get recommendations
            recommendations = self.recommendation_engine.get_recommendations(
                query=query,
                top_k=5,  # Limit for conversation
                user_context=user_context
            )
            
            if not recommendations["success"]:
                return self._create_error_response(query, recommendations.get("error", "Unknown error"))
            
            # Format response using Gemini
            formatted_response = self._format_recommendation_response(
                query, recommendations, analysis
            )
            
            return {
                "success": True,
                "service_used": "property_recommendation",
                "query": query,
                "response": formatted_response,
                "raw_data": recommendations,
                "query_analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in recommendation query: {e}")
            return self._create_error_response(query, str(e))
    
    def _handle_price_prediction_query(self, 
                                     query: str, 
                                     analysis: Dict[str, Any], 
                                     chatbot_state: Optional[Any]) -> Dict[str, Any]:
        """Handle price prediction queries"""
        
        # Extract property features from query
        features = self.query_analyzer.extract_property_features(query)
        
        # If we have chatbot state, use it for price prediction
        if chatbot_state:
            try:
                # Process the query through the existing price prediction system
                prediction_response = chatbot_state.process_query(query)
                
                # Check if we got a prediction
                has_prediction = chatbot_state.last_prediction is not None
                
                if has_prediction:
                    # Format the prediction response
                    formatted_response = self._format_price_prediction_response(
                        query, prediction_response, chatbot_state.last_prediction
                    )
                else:
                    # This is a conversation about property features
                    formatted_response = prediction_response
                
                return {
                    "success": True,
                    "service_used": "price_prediction",
                    "query": query,
                    "response": formatted_response,
                    "has_prediction": has_prediction,
                    "extracted_features": features,
                    "query_analysis": analysis
                }
                
            except Exception as e:
                logger.error(f"Error in price prediction: {e}")
                return self._create_error_response(query, str(e))
        else:
            # No chatbot state available, provide general response
            response = self.gemini_service.generate_response(
                f"The user is asking about property prices: '{query}'. "
                "Please provide helpful information about property valuation in Bangalore and "
                "ask for more specific details like location, size, BHK, etc."
            )
            
            return {
                "success": True,
                "service_used": "price_prediction",
                "query": query,
                "response": response,
                "has_prediction": False,
                "extracted_features": features,
                "query_analysis": analysis
            }
    
    def _handle_market_analysis_query(self, 
                                    query: str, 
                                    analysis: Dict[str, Any], 
                                    user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle market analysis queries"""
        
        # Get market data from recommendation system
        market_data = {}
        if self.recommendation_engine:
            try:
                stats = self.recommendation_engine.get_system_stats()
                market_data = stats.get("data_statistics", {})
            except Exception as e:
                logger.warning(f"Could not get market data: {e}")
        
        # Generate market insights using Gemini
        prompt = f"""
        User is asking about real estate market analysis: '{query}'
        
        Available market data: {market_data}
        
        Provide insightful market analysis for Bangalore real estate, including:
        1. Current market trends
        2. Price ranges and popular areas
        3. Investment advice
        4. Market predictions
        
        Be specific and data-driven where possible.
        """
        
        response = self.gemini_service.generate_response(prompt)
        
        return {
            "success": True,
            "service_used": "market_analysis", 
            "query": query,
            "response": response,
            "market_data": market_data,
            "query_analysis": analysis
        }
    
    def _handle_general_query(self, 
                            query: str, 
                            analysis: Dict[str, Any], 
                            user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle general real estate queries"""
        
        context = {
            "user_query": query,
            "available_services": [
                "Property price predictions for Bangalore",
                "Property recommendations and search",
                "Market analysis and trends",
                "Real estate investment advice"
            ],
            "user_context": user_context or {}
        }
        
        response = self.gemini_service.generate_response(query, context)
        
        return {
            "success": True,
            "service_used": "general_chat",
            "query": query,
            "response": response,
            "query_analysis": analysis
        }
    
    def _format_recommendation_response(self, 
                                     query: str, 
                                     recommendations: Dict[str, Any],
                                     analysis: Dict[str, Any]) -> str:
        """Format property recommendations into natural language"""
        
        properties = recommendations.get("recommendations", [])
        
        if not properties:
            return "I couldn't find any properties matching your criteria. Would you like to try with different requirements?"
        
        # Create context for Gemini
        context = {
            "user_query": query,
            "query_analysis": analysis,
            "total_found": len(properties),
            "recommendations": properties[:3],  # Show top 3 in detail
            "model_used": recommendations.get("model_used", "TF-IDF")
        }
        
        prompt = f"""
        A user searched for properties with the query: '{query}'
        
        I found {len(properties)} matching properties. Here are the top recommendations:
        
        {self._format_properties_for_gemini(properties[:3])}
        
        Please provide a natural, conversational response that:
        1. Acknowledges their search request
        2. Presents the recommendations in an engaging way
        3. Highlights key features and prices
        4. Asks if they'd like more details or different criteria
        
        Be friendly, helpful, and focus on the most relevant property details.
        """
        
        return self.gemini_service.generate_response(prompt, context)
    
    def _format_price_prediction_response(self, 
                                        query: str, 
                                        response: str,
                                        prediction_data: Dict[str, Any]) -> str:
        """Format price prediction response"""
        
        # If the response already looks well-formatted, return as is
        if "lakhs" in response.lower() or "crores" in response.lower():
            return response
        
        # Otherwise, enhance with Gemini
        prompt = f"""
        A user asked about property pricing: '{query}'
        
        The price prediction system provided: '{response}'
        
        Prediction details: {prediction_data}
        
        Please provide a natural, conversational response that:
        1. Presents the price estimate clearly
        2. Explains key factors affecting the price
        3. Provides context about the Bangalore market
        4. Offers additional insights or suggestions
        
        Be informative yet easy to understand.
        """
        
        return self.gemini_service.generate_response(prompt)
    
    def _format_properties_for_gemini(self, properties: list) -> str:
        """Format property list for Gemini context"""
        
        formatted = []
        for i, prop in enumerate(properties, 1):
            price_lakhs = prop.get("price_lakhs", 0)
            location = prop.get("location", "Unknown")
            size = prop.get("size", "N/A")
            sqft = prop.get("total_sqft", 0)
            
            formatted.append(
                f"{i}. {location} - {size} ({sqft} sqft) - ₹{price_lakhs:.1f} Lakhs"
            )
        
        return "\\n".join(formatted)
    
    def _create_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Create standardized error response"""
        
        return {
            "success": False,
            "service_used": "error_handler",
            "query": query,
            "response": "I apologize, but I encountered an issue processing your request. Please try rephrasing your question or try again later.",
            "error": error,
            "query_analysis": None
        }
    
    def get_service_capabilities(self) -> Dict[str, Any]:
        """Get information about available services"""
        
        recommendation_healthy = False
        if self.recommendation_engine:
            health = self.recommendation_engine.health_check()
            recommendation_healthy = health.get("system_ready", False)
        
        return {
            "available_services": {
                "price_prediction": {
                    "available": True,
                    "description": "Property price estimates for Bangalore",
                    "features": ["BHK-based pricing", "Location analysis", "Market comparisons"]
                },
                "property_recommendation": {
                    "available": recommendation_healthy,
                    "description": "Intelligent property search and recommendations",
                    "features": ["Text-based search", "Location filtering", "Price/size filtering", "Similar property matching"]
                },
                "market_analysis": {
                    "available": True,
                    "description": "Real estate market insights and trends",
                    "features": ["Market trends", "Investment advice", "Area comparisons"]
                },
                "general_chat": {
                    "available": True,
                    "description": "General real estate assistance",
                    "features": ["Q&A", "Advice", "Information"]
                }
            },
            "routing_confidence_threshold": 0.7,
            "supported_query_types": [
                "What is the price of a 3 BHK in Whitefield?",
                "Find me apartments under 50 lakhs",
                "Show me properties near Electronic City",
                "What are the market trends in HSR Layout?"
            ]
        }