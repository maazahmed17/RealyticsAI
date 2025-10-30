#!/usr/bin/env python3
"""
RealyticsAI Unified System Launcher
===================================
Simple launcher script to start the unified chatbot system with both
price prediction and property recommendation capabilities.
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch the unified RealyticsAI system"""
    print("🏠 RealyticsAI Unified System Launcher")
    print("=" * 50)
    
    print("\nChoose how to run the unified system:")
    print("1. 💬 Interactive Chat (Terminal)")
    print("2. 🌐 Web Interface (Browser)")
    print("3. 🧪 Run Tests")
    print("4. ❓ Help")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting interactive chat...")
            try:
                from realytics_ai import main as chat_main
                chat_main()
            except KeyboardInterrupt:
                print("\n✨ Thank you for using RealyticsAI!")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            break
            
        elif choice == "2":
            print("\n🌐 Starting web server...")
            try:
                import multiprocessing
                import uvicorn
                
                def start_fastapi():
                    """Start FastAPI backend on port 8000"""
                    uvicorn.run(
                        "backend.main:app",
                        host="0.0.0.0",
                        port=8000,
                        log_level="info"
                    )
                
                def start_flask():
                    """Start Flask web server on port 5000"""
                    from web_server import app, initialize_system
                    
                    # Initialize system
                    if not initialize_system():
                        print("❌ Failed to initialize Flask system. Exiting.")
                        return
                    
                    app.run(host='0.0.0.0', port=5000, debug=False)
                
                # Start FastAPI in background process
                print("🚀 Starting FastAPI backend (port 8000)...")
                fastapi_process = multiprocessing.Process(target=start_fastapi, daemon=True)
                fastapi_process.start()
                
                # Give FastAPI time to start
                import time
                time.sleep(2)
                
                print("🌐 Starting Flask web server (port 5000)...")
                print("\n📱 Web interface will be available at:")
                print("   http://localhost:5000")
                print("   FastAPI backend: http://localhost:8000")
                print("\n⏹️  Press Ctrl+C to stop the servers")
                
                # Run Flask in main thread
                start_flask()
                
            except KeyboardInterrupt:
                print("\n✨ Servers stopped.")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
            break
            
        elif choice == "3":
            print("\n🧪 Running comprehensive tests...")
            try:
                from test_unified_system import main as test_main
                test_main()
            except Exception as e:
                print(f"\n❌ Test error: {e}")
            break
            
        elif choice == "4":
            show_help()
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def show_help():
    """Show help information"""
    help_text = """
🏠 RealyticsAI Unified System Help
=================================

This unified system combines:
• 💰 Price Prediction - ML-powered property valuation
• 🔍 Property Recommendations - Intelligent property search  
• 📊 Market Analysis - Real estate trends and insights
• 🤖 Smart Routing - Gemini AI determines the best service for each query

📋 Available Options:

1. Interactive Chat:
   - Terminal-based chatbot interface
   - Natural language queries
   - Real-time responses from all services
   - Perfect for testing and development

2. Web Interface:
   - Modern web application
   - User-friendly interface
   - Access all features through browser
   - Production-ready deployment

3. Run Tests:
   - Comprehensive test suite
   - Validates all system components
   - Performance benchmarking
   - Generates detailed reports

🔧 System Requirements:
- Python 3.8+
- Gemini API key configured
- All dependencies installed (pip install -r requirements.txt)

💡 Sample Queries to Try:
- "What's the price of a 3 BHK in Whitefield?"
- "Find me apartments under 50 lakhs"
- "What are the market trends in Bangalore?"
- "Recommend properties similar to HSR Layout"

🚀 Quick Start:
1. Ensure Gemini API key is set in config/settings.py
2. Run: python run_unified_system.py
3. Choose option 1 or 2 to start using the system

For technical support, check the documentation or run tests for diagnostics.
    """
    print(help_text)

if __name__ == "__main__":
    main()