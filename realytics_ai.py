#!/usr/bin/env python3
"""
RealyticsAI Unified Chatbot Controller - Fixed Version
======================================================
Main chatbot interface with fixed price prediction and recommendation services
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

import google.generativeai as genai
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Import our services
from src.chatbot import RealyticsAIChatbot
from config.settings import GEMINI_API_KEY
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class ImprovedPropertyRecommender:
    """Improved property recommender that works with actual data"""
    
    def __init__(self, data_path: str):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load property data"""
        try:
            if os.path.exists(self.data_path):
                self.df = pd.read_csv(self.data_path)
                
                # Clean data
                if 'total_sqft' in self.df.columns:
                    def convert_sqft(x):
                        try:
                            if '-' in str(x):
                                parts = str(x).split('-')
                                return (float(parts[0]) + float(parts[1])) / 2
                            return float(x)
                        except:
                            return None
                    self.df['total_sqft'] = self.df['total_sqft'].apply(convert_sqft)
                
                # Extract BHK
                if 'size' in self.df.columns and 'bhk' not in self.df.columns:
                    self.df['bhk'] = self.df['size'].str.extract('(\\d+)').astype(float)
                
                # Clean data
                self.df = self.df.dropna(subset=['price', 'location']).reset_index(drop=True)
                
                logger.info(f"Loaded {len(self.df)} properties for recommendations")
            else:
                logger.warning(f"Data file not found: {self.data_path}")
                
        except Exception as e:
            logger.error(f"Error loading recommendation data: {e}")
    
    def get_recommendations(self, query: str, max_price: float = None, location: str = None, 
                          bhk: int = None, min_sqft: float = None) -> Dict[str, Any]:
        """Get property recommendations based on criteria"""
        
        if self.df is None:
            return {"success": False, "error": "No data available"}
        
        try:
            # Start with all properties
            filtered_df = self.df.copy()
            
            # Apply filters
            if max_price:
                filtered_df = filtered_df[filtered_df['price'] <= max_price]
            
            if location:
                mask = filtered_df['location'].str.contains(location, case=False, na=False)
                filtered_df = filtered_df[mask]
            
            if bhk:
                if 'bhk' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['bhk'] == bhk]
                elif 'size' in filtered_df.columns:
                    mask = filtered_df['size'].str.contains(f'{bhk} BHK', case=False, na=False)
                    filtered_df = filtered_df[mask]
            
            if min_sqft and 'total_sqft' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['total_sqft'] >= min_sqft]
            
            # Sort by price (ascending) and get top 5
            recommendations = filtered_df.nsmallest(5, 'price')
            
            # Convert to list of dictionaries
            rec_list = []
            for _, row in recommendations.iterrows():
                rec_dict = {
                    'location': row.get('location', 'Unknown'),
                    'size': row.get('size', 'Unknown'),
                    'price_lakhs': row.get('price', 0),
                    'total_sqft': row.get('total_sqft', 0),
                    'bhk': row.get('bhk', 0),
                    'bath': row.get('bath', 0),
                    'balcony': row.get('balcony', 0)
                }
                rec_list.append(rec_dict)
            
            return {
                "success": True,
                "recommendations": rec_list,
                "total_found": len(filtered_df),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error in recommendations: {e}")
            return {"success": False, "error": str(e)}

class FixedUnifiedChatbot:
    """Fixed unified chatbot with proper service integration"""
    
    def __init__(self):
        """Initialize the fixed chatbot system"""
        console.print("[yellow]🚀 Initializing Fixed RealyticsAI Unified Chatbot...[/yellow]")
        
        # Configure Gemini API
        self._setup_gemini()
        
        # Initialize services
        self.price_prediction_bot = None
        self.property_recommender = None
        self.conversation_history = []
        
        # Initialize components
        self._initialize_services()
        
        console.print("[green]✅ Fixed Unified Chatbot initialized successfully![/green]")
    
    def _setup_gemini(self):
        """Setup Gemini API configuration"""
        try:
            if not GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found. Please ensure:\n"
                    "1. You have a .env file in the project root\n"
                    "2. The .env file contains: GEMINI_API_KEY=your_api_key_here\n"
                    "3. Install python-dotenv: pip install python-dotenv"
                )
            
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
            
            # Test connection
            test_response = self.gemini_model.generate_content("Hello, test connection")
            logger.info("Gemini API connection successful")
            
        except Exception as e:
            logger.error(f"Failed to setup Gemini API: {e}")
            raise
    
    def _initialize_services(self):
        """Initialize price prediction and recommendation services"""
        try:
            # Initialize price prediction chatbot
            console.print("[dim]Loading price prediction service...[/dim]")
            self.price_prediction_bot = RealyticsAIChatbot()
            
            # Initialize property recommender with data
            console.print("[dim]Loading recommendation service...[/dim]")
            data_path = "/mnt/c/Users/Ahmed/Downloads/bengaluru_house_prices.csv"
            self.property_recommender = ImprovedPropertyRecommender(data_path)
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            console.print(f"[red]Warning: Some services may not be available: {e}[/red]")
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Use Gemini to analyze query intent and extract parameters"""
        
        analysis_prompt = f"""
        You are an expert real estate query classifier. Analyze this query and respond with ONLY valid JSON.
        
        Query: "{query}"
        
        Classification Rules:
        - PRICE_PREDICTION: asking about price, value, cost, estimate, valuation, worth of property
        - PROPERTY_RECOMMENDATION: asking to find, show, search, recommend properties with criteria
        - MARKET_ANALYSIS: asking about trends, market, investment, areas, analysis
        - GENERAL_QUERY: other questions about process, documents, advice
        
        Examples:
        "What's the price of 3 BHK?" → PRICE_PREDICTION
        "Find apartments under 50 lakhs" → PROPERTY_RECOMMENDATION  
        "Market trends in Bangalore" → MARKET_ANALYSIS
        "How to buy property?" → GENERAL_QUERY
        
        Extract numbers and locations mentioned in the query.
        
        Return ONLY this JSON format:
        {{
            "intent": "PRICE_PREDICTION",
            "confidence": 0.95,
            "reasoning": "User asking about property price",
            "extracted_params": {{
                "location": "Whitefield",
                "bhk": 3,
                "max_price": 50,
                "min_sqft": 1500
            }}
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(analysis_prompt)
            response_text = response.text.strip()
            
            # Try to extract and parse JSON
            import re
            json_match = re.search(r'\{[^{}]*(?:{[^{}]*}[^{}]*)*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                    
                    # Validate required fields
                    if "intent" not in analysis:
                        raise ValueError("Missing intent field")
                    
                    # Clean and validate extracted params
                    params = analysis.get("extracted_params", {})
                    
                    # Convert string numbers to actual numbers
                    if params.get("bhk"):
                        try:
                            params["bhk"] = int(float(str(params["bhk"]).strip()))
                        except:
                            params.pop("bhk", None)
                    
                    if params.get("max_price"):
                        try:
                            price_str = str(params["max_price"]).strip()
                            numbers = re.findall(r'\d+(?:\.\d+)?', price_str)
                            if numbers:
                                params["max_price"] = float(numbers[0])
                            else:
                                params.pop("max_price", None)
                        except:
                            params.pop("max_price", None)
                    
                    if params.get("min_sqft"):
                        try:
                            params["min_sqft"] = float(str(params["min_sqft"]).strip())
                        except:
                            params.pop("min_sqft", None)
                    
                    # Clean location
                    if params.get("location"):
                        location = str(params["location"]).strip()
                        if location.lower() in ['null', 'none', '', 'not mentioned']:
                            params.pop("location", None)
                        else:
                            params["location"] = location
                    
                    return analysis
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"JSON parsing failed: {e}")
                    pass
            
            # Fallback: Manual classification based on keywords
            return self._fallback_intent_analysis(query)
                
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return self._fallback_intent_analysis(query)
    
    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback intent analysis using keywords"""
        import re
        
        query_lower = query.lower()
        
        # Price prediction keywords
        price_keywords = ['price', 'cost', 'value', 'estimate', 'valuation', 'worth', 'how much']
        if any(keyword in query_lower for keyword in price_keywords):
            intent = "PRICE_PREDICTION"
            confidence = 0.8
        else:
            # Recommendation keywords
            rec_keywords = ['find', 'show', 'search', 'recommend', 'suggest', 'apartments', 'properties', 'under']
            if any(keyword in query_lower for keyword in rec_keywords):
                intent = "PROPERTY_RECOMMENDATION"
                confidence = 0.8
            else:
                # Market analysis keywords
                market_keywords = ['trends', 'market', 'investment', 'areas', 'analysis', 'best areas']
                if any(keyword in query_lower for keyword in market_keywords):
                    intent = "MARKET_ANALYSIS"
                    confidence = 0.8
                else:
                    intent = "GENERAL_QUERY"
                    confidence = 0.6
        
        # Extract basic parameters using regex
        params = {}
        
        # Extract BHK
        bhk_match = re.search(r'(\d+)\s*bhk', query_lower)
        if bhk_match:
            params["bhk"] = int(bhk_match.group(1))
        
        # Extract price/budget
        price_match = re.search(r'under\s*(\d+)\s*lakhs?', query_lower)
        if price_match:
            params["max_price"] = float(price_match.group(1))
        
        # Extract sqft
        sqft_match = re.search(r'(\d+)\s*sqft', query_lower)
        if sqft_match:
            params["min_sqft"] = float(sqft_match.group(1))
        
        # Extract location (common Bangalore areas)
        locations = ['whitefield', 'koramangala', 'hsr layout', 'electronic city', 'hebbal', 
                    'marathahalli', 'indiranagar', 'jayanagar', 'btm layout', 'jp nagar']
        for loc in locations:
            if loc in query_lower:
                params["location"] = loc.title()
                break
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": "Fallback classification based on keywords",
            "extracted_params": params
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the unified system"""
        
        console.print(f"[dim]Analyzing query: '{query}'[/dim]")
        
        # Analyze intent using Gemini
        intent_analysis = self.analyze_query_intent(query)
        intent = intent_analysis.get("intent", "GENERAL_QUERY")
        confidence = intent_analysis.get("confidence", 0.5)
        extracted_params = intent_analysis.get("extracted_params", {})
        
        console.print(f"[dim]Intent: {intent} (confidence: {confidence:.2f})[/dim]")
        console.print(f"[dim]Extracted params: {extracted_params}[/dim]")
        
        # Route to appropriate service
        if intent == "PRICE_PREDICTION" and self.price_prediction_bot:
            return self._handle_price_prediction(query, intent_analysis)
        
        elif intent == "PROPERTY_RECOMMENDATION" and self.property_recommender:
            return self._handle_property_recommendation(query, intent_analysis)
        
        elif intent == "MARKET_ANALYSIS":
            return self._handle_market_analysis(query, intent_analysis)
        
        else:
            return self._handle_general_query(query, intent_analysis)
    
    def _handle_price_prediction(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle price prediction queries with better feature extraction"""
        try:
            # Process through price prediction chatbot
            price_response = self.price_prediction_bot.process_query(query)
            
            # Check if we got a prediction
            has_prediction = "₹" in price_response and ("lakhs" in price_response.lower() or "crores" in price_response.lower())
            
            # Extract property details if prediction was made
            property_details = None
            if has_prediction and hasattr(self.price_prediction_bot, 'state'):
                state = self.price_prediction_bot.state
                if state.features:
                    property_details = {
                        'bhk': getattr(state.features, 'bhk', None),
                        'sqft': getattr(state.features, 'sqft', None),
                        'bath': getattr(state.features, 'bath', None),
                        'balcony': getattr(state.features, 'balcony', None),
                        'location': getattr(state.features, 'location', None)
                    }
            
            return {
                "success": True,
                "service": "price_prediction",
                "intent_analysis": analysis,
                "response": price_response,
                "has_prediction": has_prediction,
                "property_details": property_details
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {
                "success": False,
                "service": "price_prediction",
                "error": str(e),
                "response": "I apologize, but I couldn't process your price prediction request. Please try again with property details like BHK, location, and size."
            }
    
    def _handle_property_recommendation(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle property recommendation queries with actual data"""
        try:
            # Extract parameters from analysis
            params = analysis.get("extracted_params", {})
            
            # Get recommendations from our improved recommender
            recommendations_result = self.property_recommender.get_recommendations(
                query=query,
                max_price=params.get("max_price"),
                location=params.get("location"),
                bhk=params.get("bhk"),
                min_sqft=params.get("min_sqft")
            )
            
            if not recommendations_result["success"]:
                return {
                    "success": False,
                    "service": "property_recommendation",
                    "error": recommendations_result.get("error", "Unknown error"),
                    "response": "I apologize, but I couldn't find property recommendations at the moment. Please try with different criteria."
                }
            
            # Format response using Gemini with actual property data
            properties = recommendations_result["recommendations"]
            total_found = recommendations_result["total_found"]
            
            if not properties:
                response_text = "I couldn't find any properties matching your exact criteria. Try adjusting your budget, location, or size requirements."
            else:
                # Create detailed response with Gemini
                properties_text = ""
                for i, prop in enumerate(properties[:3], 1):  # Show top 3
                    properties_text += f"""
                    {i}. **{prop['location']}** - {prop['size']} 
                       • Price: ₹{prop['price_lakhs']:.1f} Lakhs
                       • Size: {prop['total_sqft']:.0f} sqft
                       • Bathrooms: {prop['bath']}, Balconies: {prop['balcony']}
                    """
                
                gemini_prompt = f"""
                User searched for: "{query}"
                
                I found {total_found} properties matching their criteria. Here are the top 3 recommendations:
                {properties_text}
                
                Create a natural, helpful response that:
                1. Acknowledges their search
                2. Presents these properties in an engaging way
                3. Highlights key features and value propositions
                4. Asks if they want more details or different criteria
                5. Provides practical advice about the locations if relevant
                
                Be conversational and focus on helping them make a good decision.
                """
                
                try:
                    gemini_response = self.gemini_model.generate_content(gemini_prompt)
                    response_text = gemini_response.text
                except Exception as e:
                    # Fallback response if Gemini fails
                    response_text = f"I found {len(properties)} great properties for you!\\n\\n{properties_text}\\n\\nWould you like more details about any of these properties or want to see more options?"
            
            return {
                "success": True,
                "service": "property_recommendation", 
                "intent_analysis": analysis,
                "response": response_text,
                "recommendations": properties,
                "total_found": total_found
            }
            
        except Exception as e:
            logger.error(f"Error in property recommendation: {e}")
            return {
                "success": False,
                "service": "property_recommendation",
                "error": str(e),
                "response": "I apologize, but I couldn't find property recommendations at the moment. Please try with different criteria."
            }
    
    def _handle_market_analysis(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market analysis queries using Gemini with real data context"""
        try:
            # Get some market data from our dataset
            market_context = ""
            if self.property_recommender and self.property_recommender.df is not None:
                df = self.property_recommender.df
                market_context = f"""
                Market Data Context:
                - Total properties in database: {len(df)}
                - Average price: ₹{df['price'].mean():.1f} Lakhs
                - Price range: ₹{df['price'].min():.0f} - ₹{df['price'].max():.0f} Lakhs
                - Popular areas: {', '.join(df['location'].value_counts().head(5).index.tolist())}
                """
            
            market_prompt = f"""
            Provide comprehensive market analysis for this Bangalore real estate query:
            
            Query: "{query}"
            
            {market_context}
            
            Include information about:
            1. Current market trends in Bangalore
            2. Popular investment areas and why
            3. Price ranges for different property types
            4. Market predictions and advice
            5. Best areas for different buyer profiles (first-time, investment, luxury)
            
            Use the market data context where relevant. Provide practical, data-driven insights.
            """
            
            response = self.gemini_model.generate_content(market_prompt)
            
            return {
                "success": True,
                "service": "market_analysis",
                "intent_analysis": analysis,
                "response": response.text
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {
                "success": False,
                "service": "market_analysis", 
                "error": str(e),
                "response": "I apologize, but I couldn't provide market analysis at the moment. Please try again."
            }
    
    def _handle_general_query(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general real estate queries using Gemini"""
        try:
            general_prompt = f"""
            You are an expert real estate assistant for Bangalore/Bengaluru market. 
            Answer this query with helpful, accurate information:
            
            Query: "{query}"
            
            Provide:
            1. Direct answer to the question
            2. Relevant context about Bangalore real estate
            3. Practical advice if applicable
            4. Suggestions for next steps if relevant
            
            Available services to mention:
            - Property price predictions (provide property details)  
            - Property search and recommendations
            - Market analysis and investment advice
            
            Keep response conversational and helpful.
            """
            
            response = self.gemini_model.generate_content(general_prompt)
            
            return {
                "success": True,
                "service": "general_query",
                "intent_analysis": analysis,
                "response": response.text
            }
            
        except Exception as e:
            logger.error(f"Error in general query: {e}")
            return {
                "success": False,
                "service": "general_query",
                "error": str(e), 
                "response": "I apologize, but I couldn't process your question at the moment. Please try rephrasing or ask about specific property details."
            }
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []
        if self.price_prediction_bot:
            self.price_prediction_bot.state.reset_features()
        console.print("[green]✅ Conversation reset[/green]")
    
    def start_interactive_chat(self):
        """Start interactive chat session"""
        
        console.print("\\n" + "="*80)
        console.print("[bold cyan]🏠 RealyticsAI Fixed Unified Chatbot - Complete Real Estate Assistant[/bold cyan]")
        console.print("="*80)
        
        greeting = """
## Welcome to RealyticsAI! 🏠 (Fixed Version)

I'm your comprehensive AI assistant for Bangalore real estate with improved accuracy:

### 🎯 **Enhanced Services:**
- **💰 Price Predictions** - Accurate property valuations with proper feature extraction
- **🔍 Property Search** - Real data-based property recommendations from 13,000+ properties
- **📊 Market Analysis** - AI insights with actual market data
- **❓ General Help** - Expert real estate guidance

### 💬 **Try These Examples:**
- "What's the price of a 3 BHK in Whitefield?"
- "Find me apartments under 50 lakhs in Electronic City"
- "Show me 2 BHK properties with 1500+ sqft"
- "What are the market trends in HSR Layout?"

### 🔧 **Commands:**
- Type `reset` to start fresh
- Type `help` for more information
- Type `exit` to quit

*Now with properly integrated models and real data! 🚀*
        """
        
        console.print(Panel(Markdown(greeting), border_style="green"))
        console.print("\\n[dim]Type your question or 'exit' to quit[/dim]\\n")
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("\\n[bold cyan]You[/bold cyan]")
                
                # Handle special commands
                if query.lower() in ['exit', 'quit', 'bye']:
                    console.print("\\n[yellow]Thank you for using RealyticsAI! Goodbye! 👋[/yellow]")
                    break
                
                if query.lower() in ['reset', 'clear']:
                    self.reset_conversation()
                    continue
                
                if query.lower() == 'help':
                    console.print(Panel(Markdown(greeting), border_style="blue"))
                    continue
                
                # Process query
                console.print("\\n[bold green]RealyticsAI:[/bold green]")
                
                with console.status("[yellow]🤔 Processing your request...[/yellow]"):
                    result = self.process_query(query)
                
                # Display response
                response_text = result.get("response", "I couldn't process your request.")
                
                # Add service indicator
                service_used = result.get("service", "unknown")
                service_icons = {
                    "price_prediction": "💰",
                    "property_recommendation": "🔍", 
                    "market_analysis": "📊",
                    "general_query": "💬"
                }
                
                service_icon = service_icons.get(service_used, "🤖")
                
                # Show which service was used
                console.print(f"[dim]{service_icon} Using: {service_used.replace('_', ' ').title()}[/dim]")
                
                console.print(Panel(Markdown(response_text), border_style="green"))
                
                # Show additional info if available
                if result.get("has_prediction"):
                    console.print("[dim]💡 Price prediction generated successfully![/dim]")
                
                if result.get("recommendations"):
                    rec_count = len(result["recommendations"])
                    total_found = result.get("total_found", rec_count)
                    console.print(f"[dim]🏡 Found {rec_count} recommendations (from {total_found} matching properties)[/dim]")
                
                # Add to conversation history
                self.conversation_history.append({
                    "query": query,
                    "response": result,
                    "service": service_used
                })
                
            except KeyboardInterrupt:
                console.print("\\n\\n[yellow]Chat interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.error(f"Chat error: {traceback.format_exc()}")

def main():
    """Main entry point"""
    try:
        # Create fixed unified chatbot
        chatbot = FixedUnifiedChatbot()
        
        # Start interactive session
        chatbot.start_interactive_chat()
        
    except KeyboardInterrupt:
        print("\\n✨ Thank you for using RealyticsAI!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("\\nPlease ensure:")
        print("1. GEMINI_API_KEY is set in config/settings.py")
        print("2. All dependencies are installed: pip install -r requirements.txt")
        print("3. Data file exists at the configured path")
        sys.exit(1)

if __name__ == "__main__":
    main()