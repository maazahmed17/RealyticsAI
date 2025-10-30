# Simplified Negotiation Feature

## Overview
The negotiation feature has been simplified to provide a **one-time price compatibility analysis** for buyers only. No multi-round back-and-forth, no seller simulation - just clean advice on whether your target price is reasonable.

## How It Works

### User Flow
1. User gets property recommendations or valuation with a price (e.g., ₹87 Lakhs)
2. A "Start Negotiation" button appears
3. User clicks the button and enters their target price (e.g., ₹80 Lakhs)
4. System analyzes the compatibility and provides **Gemini-powered advice**

### Backend Logic
The system compares:
- **Property Asking Price**: ₹87 Lakhs (from recommendation/valuation)
- **Buyer Target Price**: ₹80 Lakhs (user input)
- **Price Difference**: ₹7 Lakhs (8.0%)

Based on the percentage difference, it provides:

#### ✅ Compatible (≤5% difference)
- "Your price is reasonable - proceed with confidence"
- Seller likely to accept or negotiate minimally

#### ⚠️ Negotiable (5-15% difference)
- "There's a gap but it's negotiable"
- Suggests starting slightly higher
- Provides negotiation strategies

#### ❌ Challenging (>15% difference)
- "Gap may be too large"
- Suggests reconsidering budget or finding other properties
- Provides alternatives

#### 💰 Above Asking
- "You don't need to pay more"
- Suggests offering at or below asking price

### Response Format
The response is a clean, Gemini-generated message covering:
1. **Compatibility Assessment**: Is the target reasonable?
2. **Market Perspective**: Good negotiation starting point?
3. **Clear Recommendation**: Proceed with this offer or not?
4. **Next Steps**: Actionable advice

### Fallback Logic
If Gemini API is unavailable, the system falls back to rule-based responses with the same structure.

## API Endpoint

### POST `/api/negotiate/start`

**Request:**
```json
{
  "property_id": "prop-123",
  "target_price": 80.0,
  "asking_price": 87.0,
  "user_role": "buyer",
  "initial_message": ""
}
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "property_summary": {
    "property_id": "prop-123",
    "asking_price": 87.0,
    "target_price": 80.0,
    "price_difference": 7.0,
    "price_difference_percent": 8.05
  },
  "agent_opening": "✅ Compatible Price Range\n\nYour target of 80.0 Lakhs...",
  "initial_offer": 80.0
}
```

## Testing

Run the test script:
```bash
python test_negotiation_simple.py
```

This tests 4 scenarios:
1. Close prices (2% diff)
2. Moderate gap (10% diff)
3. Large gap (21% diff)
4. Above asking price

## Frontend Integration

When a property is shown with a price:
1. `currentNegotiationContext.askingPrice` is set
2. Negotiation bar becomes visible
3. User clicks "Start Negotiation"
4. Prompt shows asking price and asks for target
5. Analysis is displayed in chat
6. Context is reset (one-time analysis)

## Key Changes Made

### Backend (`backend/negotiation/routes.py`)
- Removed complex multi-round logic
- Simplified to single analysis
- Added percentage-based rules (5%, 15% thresholds)
- Integrated Gemini for natural language advice
- Fallback to structured rule-based responses

### Frontend (`frontend/App.js`)
- Removed multi-round negotiation logic
- Simplified to one-time analysis flow
- Shows asking price in prompt
- Validates asking price exists
- Resets context after analysis

## No Multi-Round Negotiation
This is intentional. The feature:
- ✅ Analyzes price compatibility
- ✅ Provides professional advice
- ✅ Suggests next steps
- ❌ Does NOT simulate seller responses
- ❌ Does NOT do back-and-forth negotiation
- ❌ Does NOT store negotiation history

The buyer gets advice, then takes action with the actual seller outside the system.

## Example Output

**Scenario**: Property at ₹87L, Target ₹80L (8% difference)

**Gemini Response**:
```
⚠️ Moderate Gap - Negotiable

Your target of 80.0 Lakhs is 8.0% below the asking price of 87.0 Lakhs.

While there's a gap, this is within negotiable range in current market 
conditions. Consider starting slightly higher (around 82.1 Lakhs) to 
leave room for negotiation.

Next Steps:
- Start with a counter around 82.1 Lakhs
- Highlight any property issues or market comparables
- Be prepared to meet somewhere in the middle
```

Clean, actionable, professional.
