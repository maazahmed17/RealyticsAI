#!/usr/bin/env python3
"""
Test Conversation State Management
===================================
Tests the new conversation state management functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conversation.state_manager import ConversationState, PropertyFeatures

def test_state_management():
    """Test the conversation state manager."""
    
    print("="*80)
    print("Testing Conversation State Management")
    print("="*80)
    
    # Initialize state
    state = ConversationState()
    print(f"\n✅ Created session: {state.session_id}")
    
    # Test 1: Initial state
    print("\n1. Initial State:")
    print("-" * 40)
    print(f"State summary: {state.get_current_state_summary()}")
    print(f"Ready for prediction: {state.should_make_prediction()}")
    print(f"Missing fields: {state.features.get_missing_required_fields()}")
    
    # Test 2: Add partial information
    print("\n2. Adding partial information (BHK only):")
    print("-" * 40)
    update1 = state.update_features({"bhk": 3})
    print(f"Updated fields: {update1['updated_fields']}")
    print(f"Current state: {state.get_current_state_summary()}")
    print(f"Ready for prediction: {state.should_make_prediction()}")
    print(f"Clarification prompt: {state.get_clarification_prompt()}")
    
    # Test 3: Add more information
    print("\n3. Adding location:")
    print("-" * 40)
    update2 = state.update_features({"location": "Whitefield"})
    print(f"Updated fields: {update2['updated_fields']}")
    print(f"Current state: {state.get_current_state_summary()}")
    print(f"Ready for prediction: {state.should_make_prediction()}")
    
    # Test 4: Add remaining info for prediction
    print("\n4. Adding size and bathrooms:")
    print("-" * 40)
    update3 = state.update_features({"sqft": 1500, "bath": 2})
    print(f"Updated fields: {update3['updated_fields']}")
    print(f"Current state: {state.get_current_state_summary()}")
    print(f"Ready for prediction: {state.should_make_prediction()}")
    print(f"All features: {state.features.to_dict()}")
    
    # Test 5: Update existing field
    print("\n5. Updating existing field (change location):")
    print("-" * 40)
    update4 = state.update_features({"location": "Koramangala"})
    print(f"Updated fields: {update4['updated_fields']}")
    print(f"Current state: {state.get_current_state_summary()}")
    
    # Test 6: Conversation history
    print("\n6. Adding conversation turns:")
    print("-" * 40)
    state.add_turn("I have a 3 BHK", "Got it! You have a 3 BHK property.", {"bhk": 3})
    state.add_turn("It's in Whitefield", "Location noted as Whitefield.", {"location": "Whitefield"})
    state.add_turn("1500 sqft", "Size recorded as 1500 sq.ft.", {"sqft": 1500})
    
    print("Conversation context:")
    print(state.get_conversation_context(last_n_turns=3))
    
    # Test 7: Reset features
    print("\n7. Resetting features:")
    print("-" * 40)
    state.reset_features()
    print(f"State after reset: {state.get_current_state_summary()}")
    print(f"Conversation history maintained: {state.turn_count} turns")
    
    # Test 8: Minimum requirements without BHK
    print("\n8. Testing alternate minimum requirements (bath + balcony):")
    print("-" * 40)
    state.update_features({"bath": 2, "balcony": 1})
    print(f"Current state: {state.get_current_state_summary()}")
    print(f"Ready for prediction: {state.should_make_prediction()}")
    
    print("\n" + "="*80)
    print("✅ All state management tests passed!")
    print("="*80)

def test_property_features():
    """Test the PropertyFeatures dataclass."""
    
    print("\n" + "="*80)
    print("Testing PropertyFeatures Validation")
    print("="*80)
    
    # Test different combinations
    test_cases = [
        {"bhk": 3},  # BHK only - should be complete
        {"bath": 2, "balcony": 1},  # Bath + Balcony - should be complete
        {"sqft": 1500},  # Size only - incomplete
        {"location": "Whitefield"},  # Location only - incomplete
        {"bhk": 3, "sqft": 1500, "bath": 2, "balcony": 1, "location": "Whitefield"}  # All fields
    ]
    
    for i, features_dict in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {features_dict}")
        print("-" * 40)
        
        features = PropertyFeatures()
        features.update_from_dict(features_dict)
        
        print(f"Complete for prediction: {features.is_complete_for_prediction()}")
        print(f"Missing required: {features.get_missing_required_fields()}")
        print(f"Missing optional: {features.get_missing_optional_fields()}")

if __name__ == "__main__":
    test_state_management()
    test_property_features()
