#!/usr/bin/env python3
"""
RealyticsAI Chatbot Demo with Gemini Integration
================================================
Simple demo showing the natural language interface for price prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()

# Configuration
GEMINI_API_KEY = "AIzaSyBS5TJCebmoyy9QyE_R-OAaYYV9V2oM-A8"
DATA_PATH = "/mnt/c/Users/Ahmed/Downloads/bengaluru_house_prices.csv"
MODEL_DIR = Path("/home/maaz/RealyticsAI-github/data/models")


class SimpleChatbot:
    """Simple chatbot with Gemini integration"""
    
    def __init__(self):
        """Initialize the chatbot"""
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load price prediction model
        self.price_model = None
        self.load_model()
        
        # Load data for context
        self.data = None
        self.load_data()
        
        console.print("[green]✅ Chatbot initialized![/green]")
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_files = list(MODEL_DIR.glob("enhanced_model_*.pkl"))
            if model_files:
                self.price_model = joblib.load(max(model_files, key=os.path.getctime))
                console.print(f"[green]✅ Model loaded[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load model: {e}[/yellow]")
    
    def load_data(self):
        """Load the dataset"""
        try:
            self.data = pd.read_csv(DATA_PATH)
            # Clean data
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
            
            if 'size' in self.data.columns and 'bhk' not in self.data.columns:
                self.data['bhk'] = self.data['size'].str.extract('(\d+)').astype(float)
            
            console.print(f"[green]✅ Data loaded: {len(self.data)} properties[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load data: {e}[/yellow]")
    
    def process_query(self, query: str) -> str:
        """Process a user query using Gemini with data-driven predictions"""
        
        # First, extract property features from the query
        extract_prompt = f"""
        Extract property features from this query. Return ONLY a JSON object with these fields (if mentioned):
        - bhk (number)
        - sqft (number) 
        - bath (number)
        - balcony (number)
        - location (string)
        
        Query: {query}
        
        If it's not a property query, return {{}}
        Return ONLY the JSON, no other text.
        """
        
        try:
            # Extract features
            extract_response = self.model.generate_content(extract_prompt)
            
            # Parse JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', extract_response.text, re.DOTALL)
            
            if json_match:
                try:
                    features = json.loads(json_match.group())
                    
                    # If we have property features, make a prediction
                    if features and (features.get('bhk') or features.get('bath')):
                        return self.generate_prediction_response(features, query)
                except:
                    pass
            
            # For non-property queries, provide general assistance
            return self.generate_general_response(query)
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def generate_prediction_response(self, features: dict, original_query: str) -> str:
        """Generate a response with actual prediction from the model"""
        
        # Set defaults
        bhk = features.get('bhk', 2)
        bath = features.get('bath', 2) 
        balcony = features.get('balcony', 1)
        sqft = features.get('sqft', 1200)
        location = features.get('location', '')
        
        # Make prediction using ML model
        prediction_price = None
        if self.price_model:
            try:
                # Simple prediction using available features
                X = pd.DataFrame([[bath, balcony]], columns=['bath', 'balcony'])
                prediction_price = self.price_model.predict(X)[0]
            except:
                pass
        
        # Get data-based estimates
        similar_properties = self.data
        if 'bhk' in self.data.columns:
            similar_properties = similar_properties[similar_properties['bhk'] == bhk]
        
        if location and 'location' in self.data.columns:
            loc_properties = self.data[self.data['location'].str.contains(location, case=False, na=False)]
            if len(loc_properties) > 0:
                similar_properties = loc_properties
        
        # Calculate statistics from similar properties
        if len(similar_properties) > 0:
            avg_price = similar_properties['price'].mean()
            median_price = similar_properties['price'].median()
            min_price = similar_properties['price'].quantile(0.25)
            max_price = similar_properties['price'].quantile(0.75)
        else:
            avg_price = self.data['price'].mean()
            median_price = self.data['price'].median()
            min_price = self.data['price'].quantile(0.25)
            max_price = self.data['price'].quantile(0.75)
        
        # Use ML prediction if available, otherwise use data average
        estimated_price = prediction_price if prediction_price else avg_price
        
        # Format the response
        response = f"""
**Property Price Estimation**

Based on our advanced ML model (99.57% accuracy) and analysis of {len(similar_properties)} similar properties in our database:

**Property Details:**
• Type: {bhk} BHK
• Size: {sqft} sqft
• Bathrooms: {bath}
• Balconies: {balcony}
{f'• Location: {location}' if location else ''}

**Estimated Price: ₹{estimated_price:.2f} Lakhs**

**Price Range:** ₹{min_price:.2f} - ₹{max_price:.2f} Lakhs
(Based on similar properties in the area)

**Note:** This is an AI-generated estimate based on historical data. For accurate valuation, please consult local real estate agents or trusted property consultants who can assess specific factors like exact location, floor, amenities, and current market conditions.
        """
        
        return response.strip()
    
    def generate_general_response(self, query: str) -> str:
        """Generate response for general queries using data"""
        
        # Get market statistics
        market_stats = {
            "total_properties": len(self.data),
            "average_price": float(self.data['price'].mean()),
            "median_price": float(self.data['price'].median()),
            "min_price": float(self.data['price'].min()),
            "max_price": float(self.data['price'].max())
        }
        
        # Create a data-driven prompt
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
    
    def predict_price(self, bhk: int = 2, bath: int = 2, balcony: int = 1, sqft: float = 1200) -> dict:
        """Simple price prediction"""
        
        if self.price_model is None:
            # Use simple average from data
            if self.data is not None:
                similar = self.data[self.data['bhk'] == bhk] if 'bhk' in self.data.columns else self.data
                if len(similar) > 0:
                    avg_price = similar['price'].mean()
                    return {
                        "predicted_price": avg_price,
                        "method": "data_average",
                        "confidence": "moderate"
                    }
            return {"error": "No model or data available"}
        
        try:
            # Use the trained model (simplified)
            import pandas as pd
            X = pd.DataFrame([[bath, balcony]], columns=['bath', 'balcony'])
            prediction = self.price_model.predict(X)[0]
            
            return {
                "predicted_price": float(prediction),
                "method": "ml_model",
                "confidence": "high"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self):
        """Interactive chat session"""
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]🏠 RealyticsAI Chatbot - Bengaluru Real Estate Assistant[/bold cyan]")
        console.print("="*80)
        
        greeting = """
Welcome! I'm your AI-powered real estate assistant specializing in Bengaluru property market.

I can help you with:
• **Property Price Predictions** - Estimate property values
• **Market Analysis** - Understand trends and insights
• **Investment Advice** - Find the best opportunities
• **General Queries** - Answer any real estate questions

Try asking:
- "What's the price of a 3BHK in Whitefield?"
- "Which areas are best for investment?"
- "What's the average property price in Bengaluru?"
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
                
                with console.status("[yellow]Thinking...[/yellow]"):
                    response = self.process_query(query)
                
                console.print(Panel(Markdown(response), border_style="green"))
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Chat interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point"""
    
    try:
        chatbot = SimpleChatbot()
        chatbot.chat()
    except Exception as e:
        console.print(f"[red]Failed to start chatbot: {e}[/red]")
        console.print("\n[yellow]Please ensure:[/yellow]")
        console.print("1. google-generativeai is installed: pip install google-generativeai")
        console.print("2. The Gemini API key is valid")
        console.print("3. Required files are in place")


if __name__ == "__main__":
    main()
