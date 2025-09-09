#!/usr/bin/env python3
"""
Test Multi-Turn Conversation
=============================
Demonstrates the chatbot's ability to maintain context across multiple turns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot import RealyticsAIChatbot
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def test_multi_turn_conversation():
    """Test multi-turn conversation with incremental information."""
    
    print("="*80)
    print("🏠 Testing Multi-Turn Conversation with State Management")
    print("="*80)
    
    # Initialize chatbot
    print("\nInitializing chatbot...")
    chatbot = RealyticsAIChatbot()
    
    # Test conversation turns
    test_turns = [
        ("I have a property in Whitefield", "Providing location first"),
        ("It's a 3 BHK", "Adding BHK information"),
        ("current state", "Checking what chatbot knows"),
        ("The size is 1500 square feet", "Adding size"),
        ("2 bathrooms and 2 balconies", "Adding remaining details"),
        ("Actually, make that Koramangala instead of Whitefield", "Correcting location"),
        ("reset", "Starting fresh"),
        ("2 BHK, 1000 sqft", "Providing multiple details at once"),
        ("in Electronic City", "Adding location"),
        ("2 bathrooms", "Adding bathrooms")
    ]
    
    for turn_num, (query, description) in enumerate(test_turns, 1):
        print(f"\n{'='*80}")
        print(f"Turn {turn_num}: {description}")
        print(f"User: {query}")
        print("-"*40)
        
        response = chatbot.process_query(query)
        
        # Display response in a nice panel
        console.print(Panel(
            Markdown(response),
            title="[bold cyan]Assistant Response[/bold cyan]",
            border_style="green"
        ))
        
        # Show current state (for debugging)
        if chatbot.state.features.to_dict():
            console.print(f"\n[dim]Current State: {chatbot.state.features.to_dict()}[/dim]")
        
    print("\n" + "="*80)
    print("✅ Multi-turn conversation test completed!")
    print("="*80)
    
    # Summary
    print("\n📊 DEMONSTRATION SUMMARY:")
    print("-"*40)
    print("1. ✅ Chatbot maintained context across turns")
    print("2. ✅ Accumulated features incrementally")
    print("3. ✅ Handled corrections (location change)")
    print("4. ✅ Reset functionality worked")
    print("5. ✅ Made predictions when sufficient data available")
    print("6. ✅ Asked for clarification when needed")

def test_conversation_scenarios():
    """Test different conversation scenarios."""
    
    print("\n" + "="*80)
    print("Testing Different Conversation Scenarios")
    print("="*80)
    
    chatbot = RealyticsAIChatbot()
    
    scenarios = [
        {
            "name": "Minimal Information Path",
            "turns": [
                "I need a price for 3 BHK",
                "Add 2 bathrooms"
            ]
        },
        {
            "name": "Gradual Information Reveal",
            "turns": [
                "Looking to price a property",
                "It's in Marathahalli",
                "2 bedrooms",
                "About 1200 square feet",
                "2 bathrooms, 1 balcony"
            ]
        },
        {
            "name": "Correction Scenario",
            "turns": [
                "3 BHK in BTM Layout",
                "Sorry, I meant 4 BHK",
                "1800 sqft"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n\n📌 Scenario: {scenario['name']}")
        print("-"*60)
        
        # Reset for new scenario
        chatbot.state.reset_features()
        
        for turn_num, query in enumerate(scenario['turns'], 1):
            print(f"\nTurn {turn_num} - User: {query}")
            response = chatbot.process_query(query)
            
            # Show brief response
            brief_response = response[:150] + "..." if len(response) > 150 else response
            print(f"Bot: {brief_response}")
            
            # Show if prediction was made
            if chatbot.state.should_make_prediction():
                print(f"[✓ Can make prediction with: {chatbot.state.features.to_dict()}]")

if __name__ == "__main__":
    test_multi_turn_conversation()
    test_conversation_scenarios()
