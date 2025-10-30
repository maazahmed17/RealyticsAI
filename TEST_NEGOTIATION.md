# Testing the Negotiation Feature

## Quick Start
Make sure your backend server is running on `http://localhost:8000`

## Test Method 1: Via Recommendation Button

### Steps:
1. **Open the app** in browser
2. **Ask for recommendations**: 
   ```
   "Show me 1 BHK properties in Basaveshwara Nagar"
   ```
3. **Wait for response** with property recommendations in sidebar
4. **Look for negotiation bar** at bottom (should appear automatically)
   - Should show property details with price
5. **Click "Start Negotiation"** button
6. **Prompt will appear** showing:
   ```
   Property asking price is ₹39.9 Lakhs.
   
   What is your target price in Lakhs?
   ```
7. **Enter your target** (e.g., `35`)
8. **Get analysis** in chat

### Expected Result:
Clean negotiation analysis like:
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

---

## Test Method 2: Via Chat (One-Shot)

### Example Queries:

#### Query 1: Complete Info
```
Property is asking 120 lakhs, my budget is 110 lakhs
```

**Expected**: Immediate analysis of 110 vs 120 (8.3% gap)

#### Query 2: With Location
```
Help me negotiate for a property in RT Nagar of 3BHK which is asking 100 lakhs and my budget is 90 lakhs
```

**Expected**: Analysis of 90 vs 100 (10% gap) with location context

#### Query 3: Different Phrasing
```
I want to buy a 2BHK in Whitefield for 85 lakhs but it's listed at 95 lakhs
```

**Expected**: Analysis of 85 vs 95 (10.5% gap)

---

## Test Method 3: Via Chat (Conversational)

### Conversation 1: Missing Target Price

**You:**
```
Help me negotiate for a property in RT Nagar of 3bhk which is asking 100 lakhs
```

**Bot:** 
```
I can help you analyze this negotiation!

🏠 Property Details:
- Location: RT Nagar
- Type: 3 BHK
- Asking Price: ₹100 Lakhs

💡 To provide negotiation advice, please tell me:
What is your target price?

For example: "My budget is 85 lakhs" or "I want to offer 80 lakhs"
```

**You:**
```
My budget is 90 lakhs
```

**Bot:** [Full analysis of 90 vs 100]

---

### Conversation 2: Missing Asking Price

**You:**
```
I have a budget of 85 lakhs for a 2BHK
```

**Bot:**
```
I can help you with this negotiation!

💰 Your Budget: ₹85 Lakhs
🏠 Type: 2 BHK

💡 To provide negotiation advice, please tell me:
What is the property's asking price?

For example: "The property is asking 95 lakhs"
```

**You:**
```
The property is asking 95 lakhs
```

**Bot:** [Full analysis of 85 vs 95]

---

## Test Method 4: API Direct Call

```bash
python test_negotiation_simple.py
```

This tests 4 scenarios:
1. Close prices (2% diff) → Compatible ✅
2. Moderate gap (10% diff) → Negotiable ⚠️
3. Large gap (21% diff) → Challenging ❌
4. Above asking → Don't overpay 💰

---

## What to Look For

### ✅ Success Indicators:
- Negotiation bar appears when properties shown
- Bar shows property details WITH price
- Button click shows asking price in prompt
- Chat extracts prices from natural language
- Analysis categorizes into: Compatible/Negotiable/Challenging
- Response is clean, professional, actionable
- Provides clear next steps

### ❌ Failure Indicators:
- Bar shows "Property price information is not available"
- No price captured from recommendations
- Chat shows generic tips instead of analysis
- Response is messy or incomplete
- No percentage calculation shown

---

## Common Test Phrases

### Natural Language Variations:
```
"asking 100 lakhs"
"asking price is 100 lakhs"
"asking ₹100 lakhs"
"listed at 100L"
"property is asking 100"
```

```
"my budget is 90 lakhs"
"I want to offer 90 lakhs"
"target price 90 lakhs"
"I can pay 90L"
"budget 90"
```

All should work!

---

## Expected Response Categories

### Compatible (≤5% difference)
**Example**: 95 → 98 (3.2% gap)
```
✅ Compatible Price Range
Your price is reasonable - proceed with confidence
```

### Negotiable (5-15% difference)
**Example**: 90 → 100 (10% gap)
```
⚠️ Moderate Gap - Negotiable
There's a gap but it's within negotiable range
Suggest starting at ~93 lakhs
```

### Challenging (>15% difference)
**Example**: 80 → 110 (27% gap)
```
❌ Significant Gap - Challenging
Gap may be too large for successful negotiation
Consider increasing budget or finding other properties
```

### Above Asking (negative gap)
**Example**: 105 → 100
```
💰 Above Asking Price
You don't need to offer more than asking
Start at or below asking price
```

---

## Debug Tips

### Check Browser Console:
Look for:
```
Negotiation context set: { propertyId, askingPrice, ... }
```

### Check Backend Logs:
Look for:
```
Negotiation API call failed: ...
Negotiation handling error: ...
```

### Verify Price Extraction:
In browser console after recommendations:
```javascript
console.log(currentNegotiationContext);
// Should show: { propertyId: "...", askingPrice: 39.9, ... }
```

---

## Troubleshooting

### Issue: "Property price information is not available"
**Fix**: Check that recommendations have `price`, `estimated_price`, or `list_price` field

### Issue: No negotiation bar appears
**Fix**: Check browser console for `currentNegotiationContext` object

### Issue: Chat shows generic tips
**Fix**: Ensure message contains both asking and target prices, or follow conversational flow

### Issue: Analysis seems wrong
**Fix**: Check that prices are being extracted as floats, not strings
