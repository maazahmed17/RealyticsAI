#!/usr/bin/env python3
"""
Simple test for the negotiation endpoint.
Tests different price scenarios.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_negotiation(property_id: str, asking_price: float, target_price: float):
    """Test negotiation with given prices."""
    print(f"\n{'='*60}")
    print(f"Testing Negotiation")
    print(f"Property: {property_id}")
    print(f"Asking Price: ₹{asking_price} Lakhs")
    print(f"Target Price: ₹{target_price} Lakhs")
    print(f"{'='*60}")
    
    payload = {
        "property_id": property_id,
        "target_price": target_price,
        "user_role": "buyer",
        "asking_price": asking_price,
        "initial_message": ""
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/negotiate/start",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"\nSession ID: {data['session_id']}")
            print(f"\n📊 Analysis:")
            print(f"{data['agent_opening']}")
            print(f"\n{'='*60}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("🏠 Testing Simplified Negotiation Feature")
    
    # Test Case 1: Target very close to asking (within 5%)
    test_negotiation("prop-001", asking_price=87.0, target_price=85.0)
    
    # Test Case 2: Moderate gap (5-15%)
    test_negotiation("prop-002", asking_price=100.0, target_price=90.0)
    
    # Test Case 3: Large gap (>15%)
    test_negotiation("prop-003", asking_price=120.0, target_price=95.0)
    
    # Test Case 4: Target above asking
    test_negotiation("prop-004", asking_price=80.0, target_price=85.0)
