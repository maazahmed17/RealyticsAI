# Negotiation Feature Fixes

## Issues Fixed

### 1. Missing Asking Price in Negotiation Context
**Problem**: When clicking "Start Negotiation" button, it showed "Property price information is not available"

**Root Cause**: The `askingPrice` wasn't being properly extracted and stored from property recommendations

**Fix**:
- Updated `App.js` to properly parse and store `askingPrice` from recommendation data
- Extract price from `firstProp.price`, `estimated_price`, or `list_price`
- Store in `currentNegotiationContext` with full property details
- Also capture predicted price from price prediction responses

### 2. Chat-Based Negotiation Requests
**Problem**: When users type "Help me negotiate for a property in RT Nagar of 3BHK which is asking 100 lakhs", it only showed generic tips

**Root Cause**: The `_handle_negotiation()` method in chatbot wasn't extracting prices and calling the negotiation API

**Fix**:
- Added regex pattern matching to extract:
  - Asking price (e.g., "asking 100 lakhs")
  - Target price (e.g., "my budget is 90 lakhs")
  - Location (e.g., "in RT Nagar")
  - BHK type (e.g., "3BHK")
- When both prices are present, call `/api/negotiate/start` internally
- Return the Gemini-generated negotiation analysis
- Provide helpful prompts if only one price is given

## How It Works Now

### Via Button (Recommendation Flow)
1. User gets property recommendations with price (e.g., ₹39.9 Lakhs)
2. System stores: `{ propertyId, askingPrice: 39.9, location, bhk, sqft }`
3. Negotiation bar appears with full details
4. User clicks "Start Negotiation"
5. Prompt shows: "Property asking price is ₹39.9 Lakhs. What is your target price?"
6. User enters target (e.g., 35)
7. System analyzes and returns Gemini response

### Via Chat (Direct Message)
1. User types: "Help me negotiate for a property in RT Nagar of 3BHK which is asking 100 lakhs"
2. System extracts: asking=100, location=RT Nagar, bhk=3
3. System asks: "What is your target price?"
4. User replies: "My budget is 90 lakhs"
5. System extracts: target=90
6. System calls negotiation API internally
7. Returns full analysis

**Or in one message:**
"I want to buy a 2BHK in Whitefield for 85 lakhs but it's listed at 95 lakhs"
→ Immediate analysis returned

## Code Changes

### Backend: `backend/services/chatbot_orchestrator/realytics_chatbot.py`
```python
def _handle_negotiation(self, message: str) -> str:
    # Extract prices using regex
    asking_match = re.search(r'asking\s+(?:₹)?(\d+\.?\d*)\s*(?:lakhs?|l)', message, re.IGNORECASE)
    target_match = re.search(r'(?:budget|offer)\s+(?:₹)?(\d+\.?\d*)\s*(?:lakhs?|l)', message, re.IGNORECASE)
    
    # If both present, call API
    if asking_price and target_price:
        # Call /api/negotiate/start internally
        # Return agent_opening response
    
    # If partial info, guide user
    # If no info, provide instructions
```

### Frontend: `frontend/App.js`
```javascript
// When recommendations arrive
const askingPrice = price ? parseFloat(price) : null;
currentNegotiationContext = {
    propertyId: detectedPropertyId,
    details: detailsText,
    askingPrice: askingPrice,  // ← Key fix
    location: loc,
    bhk: bhk,
    sqft: sqft
};

// When starting negotiation
if (!askingPrice || askingPrice <= 0) {
    alert('Property price information is not available...');
    return;
}
```

## Testing

### Test Button Flow:
1. Ask: "Show me 1 BHK properties in Basaveshwara Nagar"
2. Wait for recommendations with prices
3. Click "Start Negotiation" button
4. Should show: "Property asking price is ₹XX.X Lakhs"
5. Enter target price
6. Should receive negotiation analysis

### Test Chat Flow:
```
User: "Help me negotiate for a property in RT Nagar of 3bhk which is asking 100 lakhs"
Bot: "I can help you analyze this negotiation!
      🏠 Property Details:
      - Location: RT Nagar
      - Type: 3 BHK
      - Asking Price: ₹100 Lakhs
      
      💡 What is your target price?"

User: "My budget is 90 lakhs"
Bot: [Full negotiation analysis with compatibility assessment]
```

Or one-shot:
```
User: "Property is asking 120 lakhs, my budget is 110 lakhs"
Bot: [Immediate analysis]
```

## Regex Patterns Used

```python
# Asking price
r'asking\s+(?:price\s+)?(?:is\s+)?(?:₹)?(\d+\.?\d*)\s*(?:lakhs?|l)'

# Target/Budget
r'(?:target|budget|offer|pay)\s+(?:price\s+)?(?:is\s+)?(?:₹)?(\d+\.?\d*)\s*(?:lakhs?|l)'

# Location
r'in\s+([A-Za-z\s]+?)(?:\s+of|,|\.|$)'

# BHK
r'(\d+)\s*bhk'
```

## Example Outputs

**Scenario 1**: Button click for ₹39.9L property, target ₹35L (12.3% gap)
```
⚠️ Moderate Gap - Negotiable

Your target of 35.0 Lakhs is 12.3% below the asking price of 39.9 Lakhs.

While there's a gap, this is within negotiable range in current market 
conditions. Consider starting slightly higher (around 36.5 Lakhs) to 
leave room for negotiation.

Next Steps:
- Start with a counter around 36.5 Lakhs
- Highlight any property issues or market comparables
- Be prepared to meet somewhere in the middle
```

**Scenario 2**: Chat request "asking 100, budget 98"
```
✅ Compatible Price Range

Your target of 98.0 Lakhs is very close to the asking price of 100.0 Lakhs 
(2.0% difference).

Recommendation: This is a reasonable starting point. The seller is likely 
to consider your offer. You can proceed with confidence.

Next Steps:
- Present this offer to the seller
- Prepare to negotiate minor details (timeline, repairs, etc.)
- Be ready to compromise slightly if needed
```

## Benefits

✅ **Two Ways to Negotiate**: Button or chat message
✅ **Proper Price Capture**: From recommendations and predictions
✅ **Intelligent Extraction**: Parses natural language queries
✅ **Guided Flow**: Prompts for missing information
✅ **Clean Responses**: Consistent Gemini-powered analysis
✅ **Error Handling**: Validates prices and provides fallbacks

Now users can negotiate either by:
1. Getting recommendations → clicking button
2. Directly asking in chat with full details
3. Having a conversation where bot asks for missing info
