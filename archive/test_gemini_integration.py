#!/usr/bin/env python3
"""
Test Script for RealyticsAI Gemini Integration
==============================================
Tests the integrated chatbot with natural language queries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend.services.chatbot_orchestrator.realytics_chatbot import RealyticsAIChatbot
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def test_chatbot():
    """Test the chatbot with various queries"""
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]🧪 Testing RealyticsAI Gemini Integration[/bold cyan]")
    console.print("="*80)
    
    try:
        # Initialize chatbot
        console.print("\n[yellow]Initializing chatbot...[/yellow]")
        chatbot = RealyticsAIChatbot()
        console.print("[green]✅ Chatbot initialized successfully![/green]")
        
        # Test queries
        test_queries = [
            "Hello, what can you do?",
            "What's the price of a 3BHK apartment in Whitefield with 1500 sqft?",
            "Show me the average property prices in Bengaluru",
            "Which areas are best for investment?",
            "I need help negotiating a property deal",
            "What factors affect property prices in Bengaluru?"
        ]
        
        console.print("\n[bold]Running test queries:[/bold]\n")
        
        for i, query in enumerate(test_queries, 1):
            console.print(f"\n[bold cyan]Test {i}: {query}[/bold cyan]")
            
            # Process query
            result = chatbot.process_message(query)
            
            if result.get("success"):
                console.print(Panel(
                    Markdown(result.get("response", "")[:500] + "..." if len(result.get("response", "")) > 500 else result.get("response", "")),
                    title=f"Intent: {result.get('intent', 'unknown')}",
                    border_style="green"
                ))
                console.print(f"[green]✅ Test {i} passed[/green]")
            else:
                console.print(Panel(
                    result.get("response", "Error occurred"),
                    title="Error",
                    border_style="red"
                ))
                console.print(f"[red]❌ Test {i} failed: {result.get('error', 'Unknown error')}[/red]")
        
        # Get status
        console.print("\n[bold]Chatbot Status:[/bold]")
        status = chatbot.get_status()
        
        status_table = f"""
**Status:** {status['status']}
**Features Enabled:**
  - Price Prediction: {'✅' if status['features']['price_prediction'] else '❌'}
  - Property Recommendation: {'✅' if status['features']['property_recommendation'] else '❌'}
  - Negotiation Agent: {'✅' if status['features']['negotiation_agent'] else '❌'}
**Gemini Connected:** {'✅' if status['gemini_connected'] else '❌'}
**Interactions:** {status['session_data']['interactions']}
        """
        
        console.print(Panel(Markdown(status_table), title="System Status", border_style="cyan"))
        
        console.print("\n[bold green]✅ All tests completed![/bold green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Test failed with error: {e}[/red]")
        import traceback
        traceback.print_exc()


def test_price_prediction_nlp():
    """Test the price prediction NLP interface directly"""
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]🧪 Testing Price Prediction NLP Interface[/bold cyan]")
    console.print("="*80)
    
    try:
        from backend.services.price_prediction.nlp_interface import create_nlp_interface
        
        console.print("\n[yellow]Initializing NLP interface...[/yellow]")
        nlp = create_nlp_interface()
        console.print("[green]✅ NLP interface initialized![/green]")
        
        # Test queries
        queries = [
            "What's the price for a 2BHK in Koramangala?",
            "Predict price for 3 bedroom house in Whitefield with 2000 sqft",
            "How much would a luxury 4BHK cost in Indiranagar?"
        ]
        
        for query in queries:
            console.print(f"\n[bold cyan]Query: {query}[/bold cyan]")
            result = nlp.process_query(query)
            
            if result.get("success"):
                console.print(f"[green]Extracted Features:[/green] {result.get('extracted_features', {})}")
                
                if "prediction" in result and "predicted_price" in result["prediction"]:
                    price = result["prediction"]["predicted_price"]
                    console.print(f"[yellow]Predicted Price:[/yellow] ₹{price:.2f} Lakhs")
                
                # Show truncated response
                response = result.get("response", "")
                console.print(Panel(
                    Markdown(response[:300] + "..." if len(response) > 300 else response),
                    title="Natural Language Response",
                    border_style="green"
                ))
            else:
                console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
        
        # Get market insights
        console.print("\n[bold]Generating Market Insights...[/bold]")
        insights = nlp.get_market_insights()
        console.print(Panel(
            Markdown(insights[:500] + "..." if len(insights) > 500 else insights),
            title="Market Insights",
            border_style="cyan"
        ))
        
    except Exception as e:
        console.print(f"\n[red]❌ NLP test failed: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Main test runner"""
    
    console.print("\n[bold cyan]🚀 RealyticsAI Gemini Integration Test Suite[/bold cyan]")
    console.print("[dim]This will test the Gemini API integration with your price prediction system[/dim]\n")
    
    # Run tests
    test_price_prediction_nlp()
    console.print("\n" + "-"*80)
    test_chatbot()
    
    console.print("\n[bold green]✨ Testing complete![/bold green]")
    console.print("\n[yellow]To start the interactive chatbot, run:[/yellow]")
    console.print("[dim]python backend/services/chatbot_orchestrator/realytics_chatbot.py[/dim]\n")


if __name__ == "__main__":
    main()
