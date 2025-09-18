"""
Query Analyzer for RealyticsAI
==============================
Analyzes user queries to determine intent and extract relevant information.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Possible query intents"""
    PRICE_PREDICTION = "price_prediction"
    PROPERTY_RECOMMENDATION = "property_recommendation"
    PROPERTY_DETAILS = "property_details"
    MARKET_ANALYSIS = "market_analysis"
    GENERAL_INQUIRY = "general_inquiry"

class QueryAnalyzer:
    """Analyzes user queries to determine intent and extract information"""
    
    def __init__(self):
        """Initialize the query analyzer"""
        
        # Keywords for different intents
        self.price_keywords = [
            "price", "cost", "value", "estimate", "valuation", "worth", 
            "expensive", "cheap", "affordable", "budget", "lakhs", "crores"
        ]
        
        self.recommendation_keywords = [
            "recommend", "suggest", "find", "show", "search", "looking for",
            "want", "need", "apartment", "house", "property", "flat"
        ]
        
        self.location_patterns = [
            r"\bin\s+([A-Za-z\s]+)",
            r"\bat\s+([A-Za-z\s]+)",
            r"\bnear\s+([A-Za-z\s]+)",
            r"([A-Za-z\s]+)\s+area"
        ]
        
        self.bhk_patterns = [
            r"(\d+)\s*bhk",
            r"(\d+)\s*bedroom",
            r"(\d+)\s*bed"
        ]
        
        self.sqft_patterns = [
            r"(\d+)\s*sq\s*ft",
            r"(\d+)\s*sqft",
            r"(\d+)\s*square\s*feet"
        ]
        
        self.price_patterns = [
            r"(\d+)\s*lakhs?",
            r"(\d+)\s*crores?",
            r"under\s+(\d+)",
            r"below\s+(\d+)",
            r"above\s+(\d+)"
        ]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a user query and extract intent and parameters
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing analysis results
        """
        query_lower = query.lower()
        
        # Determine intent
        intent = self._determine_intent(query_lower)
        
        # Extract parameters
        parameters = self._extract_parameters(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine confidence
        confidence = self._calculate_confidence(query_lower, intent, parameters)
        
        result = {
            "intent": intent,
            "confidence": confidence,
            "parameters": parameters,
            "entities": entities,
            "original_query": query,
            "processed_query": query_lower
        }
        
        logger.info(f"Query analysis: {intent.value} (confidence: {confidence:.2f})")
        return result
    
    def _determine_intent(self, query_lower: str) -> QueryIntent:
        """Determine the primary intent of the query"""
        
        # Check for price-related queries
        price_score = sum(1 for keyword in self.price_keywords if keyword in query_lower)
        
        # Check for recommendation-related queries
        rec_score = sum(1 for keyword in self.recommendation_keywords if keyword in query_lower)
        
        # Specific patterns for price prediction
        if any(pattern in query_lower for pattern in [
            "what is the price", "how much", "price of", "cost of", 
            "estimate", "valuation", "worth"
        ]):
            return QueryIntent.PRICE_PREDICTION
        
        # Specific patterns for recommendations
        if any(pattern in query_lower for pattern in [
            "recommend", "suggest", "find me", "show me", "looking for", 
            "want to buy", "search for"
        ]):
            return QueryIntent.PROPERTY_RECOMMENDATION
        
        # Default based on scores
        if price_score > rec_score:
            return QueryIntent.PRICE_PREDICTION
        elif rec_score > 0:
            return QueryIntent.PROPERTY_RECOMMENDATION
        else:
            return QueryIntent.GENERAL_INQUIRY
    
    def _extract_parameters(self, query_lower: str) -> Dict[str, Any]:
        """Extract structured parameters from the query"""
        parameters = {}
        
        # Extract BHK information
        for pattern in self.bhk_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parameters["bhk"] = int(match.group(1))
                break
        
        # Extract square feet information
        for pattern in self.sqft_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parameters["sqft"] = int(match.group(1))
                break
        
        # Extract price constraints
        if "under" in query_lower or "below" in query_lower:
            for pattern in self.price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    price_value = int(match.group(1))
                    if "lakh" in match.group(0):
                        parameters["max_price"] = price_value * 100000
                    elif "crore" in match.group(0):
                        parameters["max_price"] = price_value * 10000000
                    break
        
        if "above" in query_lower or "over" in query_lower:
            for pattern in self.price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    price_value = int(match.group(1))
                    if "lakh" in match.group(0):
                        parameters["min_price"] = price_value * 100000
                    elif "crore" in match.group(0):
                        parameters["min_price"] = price_value * 10000000
                    break
        
        return parameters
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from the query"""
        entities = {
            "locations": [],
            "property_types": [],
            "amenities": []
        }
        
        # Extract locations
        for pattern in self.location_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                location = match.strip()
                if len(location) > 2:  # Filter out very short matches
                    entities["locations"].append(location)
        
        # Extract property types
        property_types = ["apartment", "house", "flat", "villa", "duplex", "penthouse"]
        for prop_type in property_types:
            if prop_type in query.lower():
                entities["property_types"].append(prop_type)
        
        # Extract amenities
        amenities = ["parking", "balcony", "gym", "swimming pool", "garden", "security"]
        for amenity in amenities:
            if amenity in query.lower():
                entities["amenities"].append(amenity)
        
        return entities
    
    def _calculate_confidence(self, query_lower: str, intent: QueryIntent, parameters: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        
        base_confidence = 0.5
        
        # Boost confidence based on clear intent indicators
        if intent == QueryIntent.PRICE_PREDICTION:
            if any(keyword in query_lower for keyword in ["price", "cost", "worth", "estimate"]):
                base_confidence += 0.3
        elif intent == QueryIntent.PROPERTY_RECOMMENDATION:
            if any(keyword in query_lower for keyword in ["recommend", "suggest", "find", "show"]):
                base_confidence += 0.3
        
        # Boost confidence based on extracted parameters
        base_confidence += min(0.2, len(parameters) * 0.05)
        
        # Boost confidence based on specific patterns
        specific_patterns = [
            r"\d+\s*bhk", r"\d+\s*sq\s*ft", r"\d+\s*lakhs?", r"\d+\s*crores?",
            r"in\s+\w+", r"near\s+\w+"
        ]
        
        pattern_matches = sum(1 for pattern in specific_patterns if re.search(pattern, query_lower))
        base_confidence += min(0.2, pattern_matches * 0.05)
        
        return min(1.0, base_confidence)
    
    def extract_property_features(self, query: str) -> Dict[str, Any]:
        """Extract property features from query for price prediction
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted features
        """
        analysis = self.analyze_query(query)
        
        features = {}
        
        # Get parameters
        params = analysis.get("parameters", {})
        if "bhk" in params:
            features["bhk"] = params["bhk"]
        if "sqft" in params:
            features["sqft"] = params["sqft"]
        
        # Get location from entities
        entities = analysis.get("entities", {})
        if entities.get("locations"):
            features["location"] = entities["locations"][0]
        
        # Default values if not specified
        if "bhk" not in features:
            features["bhk"] = 2  # Default assumption
        if "sqft" not in features:
            features["sqft"] = 1000  # Default assumption
        
        return features
    
    def extract_search_filters(self, query: str) -> Dict[str, Any]:
        """Extract search filters for property recommendations
        
        Args:
            query: User query
            
        Returns:
            Dictionary of search filters
        """
        analysis = self.analyze_query(query)
        
        filters = {}
        
        # Get parameters
        params = analysis.get("parameters", {})
        for key in ["bhk", "sqft", "max_price", "min_price"]:
            if key in params:
                if key == "sqft":
                    filters["min_sqft"] = params[key]
                else:
                    filters[key] = params[key]
        
        # Get location from entities
        entities = analysis.get("entities", {})
        if entities.get("locations"):
            filters["location"] = entities["locations"][0]
        
        return filters