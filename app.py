#!/usr/bin/env python3
"""
RealyticsAI - Intelligent Real Estate Assistant
===============================================
Main application launcher for the RealyticsAI chatbot system.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.chatbot import RealyticsAIChatbot


def main():
    """Main entry point for RealyticsAI"""
    print("\n" + "="*80)
    print("🏠 RealyticsAI - Intelligent Real Estate Assistant")
    print("="*80)
    print("\nInitializing system...")
    
    try:
        # Create and start chatbot
        chatbot = RealyticsAIChatbot()
        chatbot.start()
    except KeyboardInterrupt:
        print("\n\n✨ Thank you for using RealyticsAI!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
