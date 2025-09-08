#!/usr/bin/env python3
"""
RealyticsAI CLI Chatbot Interface
==================================
Interactive command-line interface for testing the chatbot
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
import asyncio

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import chatbot
from chatbot_handler import RealyticsAIChatbot

console = Console()

def display_welcome():
    """Display welcome message"""
    welcome_text = """
# 🏠 Welcome to RealyticsAI Chatbot
    
Your intelligent real estate assistant for:
- **Price Predictions** 
- **Market Analysis**
- **Property Recommendations**
- **Location Comparisons**
    
Type your questions naturally, or type:
- `help` for assistance
- `examples` for sample queries
- `quit` to exit
"""
    
    panel = Panel(Markdown(welcome_text), title="RealyticsAI", border_style="cyan")
    console.print(panel)

def display_examples():
    """Display example queries"""
    table = Table(title="📝 Example Queries", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="yellow", no_wrap=True)
    table.add_column("Example Query", style="white")
    
    examples = [
        ("Price Prediction", "What's the price of a 2 bathroom apartment with 1 balcony?"),
        ("Price Prediction", "How much would a house with 3 bathrooms cost?"),
        ("Market Analysis", "Show me market analysis for Whitefield"),
        ("Market Analysis", "What's the average price in Electronic City?"),
        ("Recommendations", "Show me properties under 50 lakhs"),
        ("Recommendations", "What can I get for 100 lakhs?"),
        ("Comparison", "Compare Whitefield and Electronic City"),
        ("General", "Help me find a good property"),
    ]
    
    for category, query in examples:
        table.add_row(category, query)
    
    console.print(table)

def format_response(response: str) -> Panel:
    """Format chatbot response in a nice panel"""
    # Convert markdown-style bold to rich markup
    formatted = response.replace("**", "[bold]").replace("[bold]", "[/bold]", 1)
    
    return Panel(
        formatted,
        title="🤖 RealyticsAI",
        title_align="left",
        border_style="green",
        padding=(1, 2)
    )

async def main():
    """Main CLI loop"""
    # Display welcome
    display_welcome()
    
    # Initialize chatbot
    console.print("[yellow]Initializing chatbot...[/yellow]")
    
    try:
        # Check for OpenAI API key in environment
        openai_key = os.getenv("OPENAI_API_KEY")
        use_openai = openai_key is not None
        
        if use_openai:
            console.print("[green]✅ Using OpenAI GPT for enhanced understanding[/green]")
            chatbot = RealyticsAIChatbot(use_openai=True, openai_api_key=openai_key)
        else:
            console.print("[cyan]ℹ Using rule-based NLU (set OPENAI_API_KEY for GPT)[/cyan]")
            chatbot = RealyticsAIChatbot(use_openai=False)
        
        console.print("[green]✅ Chatbot ready![/green]\n")
        
    except Exception as e:
        console.print(f"[red]❌ Error initializing chatbot: {e}[/red]")
        return
    
    # Main conversation loop
    session_id = "cli_session"
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("\n[yellow]Thank you for using RealyticsAI! Goodbye! 👋[/yellow]")
                break
            
            elif user_input.lower() == 'help':
                response = await chatbot.process_query("help", session_id)
                console.print(format_response(response))
                continue
            
            elif user_input.lower() == 'examples':
                display_examples()
                continue
            
            elif user_input.lower() == 'clear':
                console.clear()
                display_welcome()
                continue
            
            # Process the query
            with console.status("[bold green]Thinking...[/bold green]"):
                response = await chatbot.process_query(user_input, session_id)
            
            # Display response
            console.print(format_response(response))
            
            # Show extracted entities in debug mode
            if os.getenv("DEBUG"):
                entities = chatbot.extract_entities(user_input)
                intent = chatbot.classify_intent(user_input)
                
                debug_info = f"\n[dim]Intent: {intent.value} | Entities: {entities}[/dim]"
                console.print(debug_info)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def run():
    """Entry point for the CLI"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    run()
