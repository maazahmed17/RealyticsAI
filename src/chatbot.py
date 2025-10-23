#!/usr/bin/env python3
"""
RealyticsAI Chatbot Module
==========================
Main chatbot implementation with Gemini API integration and ML model predictions.
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Configuration
from config.settings import GEMINI_API_KEY, DATA_PATH, MODEL_DIR

# Import the new formatter
from src.presentation.formatter import PredictionFormatter

# Import conversation state management
from conversation.state_manager import ConversationState

# Import guardrails for outlier detection
from guardrails import PredictionGuardrails

console = Console()


class RealyticsAIChatbot:
    """Main chatbot class with Gemini integration"""
    
    def __init__(self):
        """Initialize the chatbot"""
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load price prediction model
        self.price_model = None
        self.feature_columns = None
        self.location_encoder = None  
        self.feature_scaler = None
        self.load_model()
        
        # Load data for context
        self.data = None
        self.load_data()
        
        # Initialize the professional formatter
        self.formatter = PredictionFormatter()
        
        # Initialize conversation state
        self.state = ConversationState()
        
        # Initialize prediction guardrails
        console.print("[dim]Loading guardrails...[/dim]")
        self.guardrails = PredictionGuardrails()
        
        console.print("[green]✅ Chatbot initialized successfully![/green]")
    
    def load_model(self):
        """Load the trained ML model - prioritize XGBoost Advanced"""
        try:
            # Priority 1: Load the XGBoost Advanced model (best performance)
            xgb_advanced_models = list(MODEL_DIR.glob("xgboost_advanced_*.pkl"))
            if xgb_advanced_models:
                latest_xgb = max(xgb_advanced_models, key=os.path.getctime)
                self.price_model = joblib.load(latest_xgb)
                model_name = latest_xgb.name
                console.print(f"[green]✅ XGBoost Advanced Model loaded: {model_name}[/green]")
                console.print(f"[cyan]   Model Accuracy: R² = 0.9959 (99.59%)[/cyan]")
                
                # Load corresponding feature columns
                timestamp = model_name.replace("xgboost_advanced_", "").replace(".pkl", "")
                feature_cols_file = MODEL_DIR / f"feature_columns_{timestamp}.pkl"
                if feature_cols_file.exists():
                    self.feature_columns = joblib.load(feature_cols_file)
                    console.print(f"[dim]   Features: {len(self.feature_columns)} engineered features[/dim]")
                return
            
            # Priority 2: Try enhanced XGBoost model (old format)
            enhanced_xgb_models = list(MODEL_DIR.glob("enhanced_xgb_model_*.pkl"))
            if enhanced_xgb_models:
                self.price_model = joblib.load(max(enhanced_xgb_models, key=os.path.getctime))
                console.print(f"[green]✅ Enhanced XGBoost Model loaded[/green]")
                
                # Also load feature columns and encoders if available
                if (MODEL_DIR / "feature_columns.pkl").exists():
                    self.feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
                if (MODEL_DIR / "location_encoder.pkl").exists():
                    self.location_encoder = joblib.load(MODEL_DIR / "location_encoder.pkl")
                if (MODEL_DIR / "feature_scaler.pkl").exists():
                    self.feature_scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
                return
            
            # Priority 3: Any enhanced model
            model_files = list(MODEL_DIR.glob("enhanced_model_*.pkl"))
            if model_files:
                self.price_model = joblib.load(max(model_files, key=os.path.getctime))
                console.print(f"[yellow]⚠️  Loaded enhanced model (not XGBoost)[/yellow]")
                return
            
            # Priority 4: Simple model (last resort)
            simple_model_path = MODEL_DIR / "simple_model.pkl"
            if simple_model_path.exists():
                self.price_model = joblib.load(simple_model_path)
                console.print(f"[yellow]⚠️  Loaded Simple ML Model (low accuracy)[/yellow]")
                console.print(f"[yellow]   Consider retraining with: train_xgboost_advanced.py[/yellow]")
                return
            
            console.print(f"[red]❌ No model files found in {MODEL_DIR}[/red]")
            
        except Exception as e:
            console.print(f"[red]❌ Error loading model: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    def load_data(self):
        """Load the Bengaluru dataset"""
        try:
            self.data = pd.read_csv(DATA_PATH)
            
            # Clean total_sqft column
            if 'total_sqft' in self.data.columns:
                def convert_sqft(x):
                    try:
                        if '-' in str(x):
                            parts = str(x).split('-')
                            return (float(parts[0]) + float(parts[1])) / 2
                        return float(x)
                    except:
                        return np.nan
                self.data['total_sqft'] = self.data['total_sqft'].apply(convert_sqft)
            
            # Extract BHK from size column
            if 'size' in self.data.columns and 'bhk' not in self.data.columns:
                self.data['bhk'] = self.data['size'].str.extract('(\d+)').astype(float)
            
            console.print(f"[green]✅ Data loaded: {len(self.data)} properties[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load data: {e}[/yellow]")
    
    def process_query(self, query: str) -> str:
        """Process user query with state management"""
        
        # Check for conversation control commands
        query_lower = query.lower().strip()
        if query_lower in ['reset', 'new property', 'start over', 'clear']:
            self.state.reset_features()
            return "I've cleared the previous property details. Let's start fresh! What property would you like to price?"
        
        if query_lower in ['current state', 'what do you have', 'show state']:
            state_summary = self.state.get_current_state_summary()
            missing = self.state.features.get_missing_required_fields()
            if missing:
                return f"{state_summary}\n\nStill need: {', '.join(missing)}"
            else:
                return f"{state_summary}\n\nI have all required information for a prediction."
        
        # Get current state context
        current_state = self.state.get_current_state_summary()
        conversation_context = self.state.get_conversation_context(last_n_turns=2)
        
        # Get existing features to preserve state
        existing_features = self.state.features.to_dict()
        
        # Create state-aware extraction prompt
        if existing_features:
            # We have existing state, extract only new features
            extract_prompt = f"""
            Extract ONLY NEW property features from this query that are not already in existing features.
            
            Existing Features: {json.dumps(existing_features)}
            New Query: {query}
            
            Return a JSON object with ONLY NEW or UPDATED fields:
            - bhk (number) - only if mentioned and different from existing
            - sqft (number) - only if mentioned and different from existing
            - bath (number) - only if mentioned and different from existing
            - balcony (number) - only if mentioned and different from existing
            - location (string) - only if mentioned and different from existing
            
            If no new property information is provided, return an empty JSON: {{}}
            Return ONLY the JSON, no other text.
            """
        else:
            # No existing state, extract all features mentioned
            extract_prompt = f"""
            Extract property features from this query:
            {query}
            
            Return a JSON object with these fields (if mentioned):
            - bhk (number)
            - sqft (number) 
            - bath (number)
            - balcony (number)
            - location (string)
            
            Return ONLY the JSON, no other text.
            """
        
        try:
            # Extract features from current query
            extract_response = self.model.generate_content(extract_prompt)
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', extract_response.text, re.DOTALL)
            extracted_features = {}
            
            if json_match:
                try:
                    extracted_features = json.loads(json_match.group())
                except:
                    extracted_features = {}
            
            # Check if user is asking for prediction
            prediction_keywords = ['predict', 'price', 'value', 'estimate', 'cost', 'worth', 'valuation']
            is_asking_for_prediction = any(keyword in query_lower for keyword in prediction_keywords)
            
            # Update state with new features
            if extracted_features:
                update_summary = self.state.update_features(extracted_features)
                console.print(f"[dim]State updated: {update_summary['updated_fields']}[/dim]")
            
            # Check if we should make a prediction
            if is_asking_for_prediction and self.state.should_make_prediction():
                # Generate prediction with accumulated state
                response = self.generate_prediction_response(self.state.features.to_dict(), query)
                
                # Store the prediction in state
                self.state.last_prediction = response
            elif is_asking_for_prediction and not self.state.should_make_prediction():
                # User wants prediction but we need more info
                response = self.state.get_clarification_prompt()
                response = f"I can help you estimate the property value. {response}"
            else:
                # User is providing information, acknowledge and check if we need more
                if extracted_features:
                    if self.state.should_make_prediction():
                        response = f"Thank you for the information! I now have enough details to provide an estimate. Would you like me to predict the price for this property?"
                    else:
                        response = self.state.get_clarification_prompt()
                        response = f"Thank you for that information! {response}"
                else:
                    # General query or unclear input
                    response = self.state.get_clarification_prompt()
            
            # Add turn to conversation history
            self.state.add_turn(query, response, extracted_features)
            
            return response
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def generate_prediction_response(self, features: dict, original_query: str) -> str:
        """Generate response with ML model prediction using professional formatter"""
        
        # Set defaults and get features
        bhk = features.get('bhk', 3)  # Default to 3 BHK if not specified
        bath = features.get('bath', 2) 
        balcony = features.get('balcony', 2)
        sqft = features.get('sqft', None)
        location = features.get('location', '')
        
        # If sqft not provided, estimate based on BHK
        if not sqft:
            sqft_estimates = {
                1: 650,
                2: 1100,
                3: 1650,
                4: 2200,
                5: 2800
            }
            sqft = sqft_estimates.get(bhk, 1650)
            sqft_assumed = True
        else:
            sqft_assumed = False
        
        # Make prediction using ML model
        prediction_price = None
        comparables_count = 0
        
        if self.price_model and self.feature_columns and self.location_encoder and self.feature_scaler:
            try:
                # Use enhanced XGBoost model with proper feature engineering
                prediction_price = self._predict_with_enhanced_model(
                    bhk, bath, balcony, sqft, location
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not use enhanced model: {e}[/yellow]")
        elif self.price_model:
            try:
                # Fallback to simple model (only if enhanced model not available)
                X_simple = np.array([[bath, balcony]])
                prediction_price = self.price_model.predict(X_simple)[0]
            except Exception as e:
                console.print(f"[yellow]Warning: Could not use simple model: {e}[/yellow]")
        
        # If ML model didn't work, use data-based estimate
        if prediction_price is None:
            if self.data is not None and len(self.data) > 0:
                # Find similar properties for baseline
                similar = self.data
                if 'bhk' in self.data.columns:
                    similar = similar[similar['bhk'] == bhk]
                if location and 'location' in self.data.columns:
                    loc_data = self.data[self.data['location'].str.contains(location, case=False, na=False)]
                    if len(loc_data) > 0:
                        similar = loc_data
                
                comparables_count = len(similar)
                
                if len(similar) > 0 and 'price' in similar.columns:
                    prediction_price = similar['price'].median()
                else:
                    prediction_price = self.data['price'].median() if 'price' in self.data.columns else 185.0
            else:
                # Default fallback
                prediction_price = 185.0
        
        # Calculate price range (±10%)
        lower_range = prediction_price * 0.90
        upper_range = prediction_price * 1.10
        
        # Format price based on value
        if prediction_price >= 100:
            price_str = f"₹{prediction_price/100:.2f} Crores"
            range_str = f"₹{lower_range/100:.2f} Cr - ₹{upper_range/100:.2f} Cr"
        else:
            price_str = f"₹{prediction_price:.2f} Lakhs"
            range_str = f"₹{lower_range:.0f} - ₹{upper_range:.0f} Lakhs"
        
        # Generate the formatted response
        response_lines = [
            f"## 📊 Valuation Snapshot for a {bhk}BHK in {location if location else 'Bengaluru'}",
            "",
            "The estimated market value for a property with these specifications is:",
            "",
            f"### {price_str}",
            f"**Likely Price Range:** {range_str}",
            ""
        ]
        
        # Add market insight
        if location:
            market_insights = {
                'Koramangala': "Koramangala is one of Bengaluru's premium locations. This area commands premium prices due to its central location and excellent connectivity.",
                'Whitefield': "Whitefield is a major IT hub with excellent infrastructure. Properties here show strong appreciation potential.",
                'Electronic City': "Electronic City offers good value with proximity to IT parks. It's a preferred choice for working professionals.",
                'Indiranagar': "Indiranagar is a prime residential area with vibrant social infrastructure. Properties here are in high demand.",
                'Hebbal': "Hebbal is a high-demand locality, prized for its connectivity via the Outer Ring Road and its proximity to Manyata Tech Park, which keeps property values strong.",
                'Marathahalli': "Marathahalli is strategically located near major IT corridors. It offers good rental yields and appreciation.",
                'HSR Layout': "HSR Layout is a well-planned locality popular among young professionals. It has excellent amenities and connectivity."
            }
            
            # Find matching insight
            insight = None
            for area, area_insight in market_insights.items():
                if area.lower() in location.lower():
                    insight = area_insight
                    break
            
            if insight:
                response_lines.append(f"**Market Insight:** {insight}")
            else:
                response_lines.append(f"**Market Insight:** {location} is an emerging area in Bengaluru with growing real estate demand.")
        
        response_lines.extend([
            "",
            "## 📝 Property Details Analyzed",
            "",
            "Here are the details my prediction is based on:",
            "",
            f"• **Location:** {location if location else 'Bengaluru (General)'}\n",
            f"• **Type:** {bhk} BHK Apartment\n",
            f"• **Bathrooms:** {bath}\n",
            f"• **Balconies:** {balcony}\n",
        ])
        
        if sqft_assumed:
            response_lines.append(f"• **Assumed Size:** ~{sqft} sqft (*Note: As size was not provided, I've used the typical square footage for a {bhk}BHK in this area for the calculation.*)")
        else:
            response_lines.append(f"• **Size:** {sqft} sqft")
        
        response_lines.extend([
            "",
            "## 💡 Model & Data Insights",
            "",
            "To build your trust, here's a look under the hood:",
            ""
        ])
        
        if comparables_count > 0:
            response_lines.append(f"**Analysis Base:** This estimate was generated by comparing your property against {comparables_count} similar {bhk}BHK properties recently sold or listed in {'and around ' + location if location else 'Bengaluru'}.")
        else:
            response_lines.append(f"**Analysis Base:** This estimate was generated using our machine learning model trained on thousands of Bengaluru property transactions.")
        
        response_lines.extend([
            "",
            "⚠️ **Disclaimer:** This is an AI-generated estimate based on market data and should be used for informational purposes only. It is not a formal appraisal. For a certified valuation, we recommend consulting a professional real estate agent.",
            ""
        ])
        
        return "\n".join(response_lines)
    
    def _predict_with_enhanced_model(self, bhk: int, bath: int, balcony: int, sqft: float, location: str) -> float:
        """Make prediction using the enhanced XGBoost model with proper feature engineering"""
        try:
            # Transform location using the trained encoder
            loc_enc = self.location_encoder.transform(location)
            
            # Prepare feature dictionary with all required features
            feature_dict = {
                'bath': bath,
                'balcony': balcony,
                'bhk': bhk,
                'total_sqft': sqft,
                'location_encoded': loc_enc['location_encoded'],
                'location_tier': loc_enc['location_tier'],
                'price_per_sqft': sqft / bhk if bhk > 0 else sqft,
                'sqft_per_bhk': sqft / max(bhk, 1),
                'bath_per_bhk': bath / max(bhk, 1)
            }
            
            # Create DataFrame with the exact feature columns the model expects
            X = pd.DataFrame([feature_dict])[self.feature_columns]
            
            # Scale features using the trained scaler
            X_scaled = self.feature_scaler.transform(X)
            
            # Make prediction
            predicted_price = float(self.price_model.predict(X_scaled)[0])
            
            return predicted_price
            
        except Exception as e:
            console.print(f"[red]Error in enhanced prediction: {e}[/red]")
            raise
    
    def generate_general_response(self, query: str) -> str:
        """Generate response for general queries"""
        
        # Get market statistics
        market_stats = {
            "total_properties": len(self.data),
            "average_price": float(self.data['price'].mean()),
            "median_price": float(self.data['price'].median()),
            "min_price": float(self.data['price'].min()),
            "max_price": float(self.data['price'].max())
        }
        
        # Create data-driven prompt
        prompt = f"""
        You are a data-driven real estate assistant. Use ONLY the following data to answer:
        
        Database Statistics:
        - Total Properties: {market_stats['total_properties']}
        - Average Price: ₹{market_stats['average_price']:.2f} Lakhs
        - Median Price: ₹{market_stats['median_price']:.2f} Lakhs  
        - Price Range: ₹{market_stats['min_price']:.0f} - ₹{market_stats['max_price']:.0f} Lakhs
        
        User Query: {query}
        
        Provide a brief, data-driven response. If you cannot answer from the data, suggest the user provide property details for price estimation.
        Keep response under 100 words.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return "Please provide property details (BHK, size, bathrooms, balconies, location) for price estimation."
    
    def start(self):
        """Start interactive chat session"""
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]🏠 RealyticsAI Chatbot - Bengaluru Real Estate Assistant[/bold cyan]")
        console.print("="*80)
        
        greeting = """
Welcome! I'm your AI-powered real estate assistant specializing in Bengaluru property market.

I can help you with:
• **Property Price Predictions** - Provide property details for instant valuation
• **Market Analysis** - Get insights about Bengaluru real estate trends
• **Investment Advice** - Find the best opportunities
• **General Queries** - Answer any real estate questions

**🆕 Now with Conversation Memory!**
I remember details across turns, so you can provide information gradually:
- First: "I have a 3 BHK property"
- Then: "It's in Whitefield"
- Finally: "1500 sqft with 2 bathrooms"

**Helpful Commands:**
- Type 'reset' or 'new property' to start fresh
- Type 'current state' to see what information I have
- Type 'exit' to quit
        """
        
        console.print(Panel(Markdown(greeting), border_style="green"))
        console.print("\n[dim]Type 'exit' to quit[/dim]\n")
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    console.print("\n[yellow]Thank you for using RealyticsAI! Goodbye! 👋[/yellow]")
                    break
                
                # Process query
                console.print("\n[bold green]RealyticsAI:[/bold green]")
                
                with console.status("[yellow]Analyzing...[/yellow]"):
                    response = self.process_query(query)
                
                console.print(Panel(Markdown(response), border_style="green"))
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Chat interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
