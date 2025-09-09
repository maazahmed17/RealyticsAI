# 🚀 How to Run RealyticsAI with Interactive Chat

## Quick Start (Recommended)

### Option 1: Using the Startup Script
```bash
cd /home/maaz/RealyticsAI
./run_app.sh
```

Then open your browser and go to: **http://localhost:5000**

### Option 2: Manual Start
```bash
cd /home/maaz/RealyticsAI
python3 api_server.py
```

## 📱 Access the Application

Once the server is running, you can access the application:

- **Local machine**: http://localhost:5000
- **From network**: http://[your-ip-address]:5000
- **Demo mode**: http://localhost:5000/demo.html

## 🎯 Features You Can Test

### 1. **Chat with the AI**
- Type any property-related question
- Examples:
  - "What's the price of a 3 BHK in Whitefield?"
  - "I have a 1500 sqft apartment in Koramangala"
  - "Show me properties in Electronic City"

### 2. **Use the Features Menu**
- Click the **+** button next to the input
- Select "Price Valuation" to get started
- Other features marked as "Coming Soon"

### 3. **Interactive Elements**
- **Left Sidebar**: Click ☰ to expand/collapse
- **Right Sidebar**: Automatically shows property details
- **Action Buttons**: Hover over AI messages to see Copy, Share, Explain, Refine buttons
- **New Chat**: Click "New Chat" to start fresh

### 4. **Different Welcome States**
- First visit: See the full welcome screen with logo
- Return visits: See simple "How can I help you today?"

## 🔧 Troubleshooting

### If the server won't start:

1. **Check dependencies**:
```bash
pip install flask flask-cors google-generativeai pandas numpy scikit-learn joblib rich
```

2. **Check Gemini API Key**:
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-actual-api-key"
```

3. **Port already in use**:
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
python3 api_server.py --port 5001
```

### If chat doesn't respond:

1. **Check console for errors**: Press F12 in browser → Console tab
2. **Verify backend is running**: Check terminal for error messages
3. **Test API directly**:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'
```

## 📊 What Happens When You Chat

1. **You type a message** → Sent to Flask API server
2. **API server** → Forwards to RealyticsAI chatbot backend
3. **Chatbot processes** → Uses Gemini AI + ML models
4. **Response generated** → Includes price estimates if applicable
5. **Frontend displays** → Streaming text animation + valuation cards

## 🎨 UI Features

### Dark Theme Interface
- Background: #121212
- Accent: Teal (#00A99D)
- Modern, professional design

### Animations
- **Streaming text**: AI responses appear word-by-word
- **Typing indicator**: Three dots while AI thinks
- **Smooth transitions**: Sidebars slide in/out
- **Hover effects**: Interactive elements respond to mouse

### Responsive Design
- Works on desktop, tablet, and mobile
- Sidebars adapt to screen size
- Touch-friendly on mobile devices

## 🛠️ Advanced Configuration

### Change Server Settings

Edit `api_server.py`:
```python
app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode
```

### Modify Chat Behavior

Edit `src/chatbot.py` to customize:
- Response style
- Guardrail thresholds
- Feature extraction

### Update Frontend Theme

Edit `frontend/styles.css`:
```css
:root {
    --accent-primary: #00A99D;  /* Change accent color */
    --bg-primary: #121212;      /* Change background */
}
```

## 📝 Testing Checklist

- [ ] Server starts without errors
- [ ] Homepage loads correctly
- [ ] Can send messages in chat
- [ ] Receive AI responses
- [ ] Features menu (+) button works
- [ ] Sidebars toggle properly
- [ ] Action buttons appear on hover
- [ ] Valuation card shows for price queries
- [ ] New chat button clears conversation
- [ ] Mobile responsive design works

## 🔗 API Endpoints

The server provides these endpoints:

- `GET /` - Main application
- `POST /api/chat` - Send chat messages
- `POST /api/reset` - Reset conversation
- `POST /api/features/<type>` - Use specific features
- `GET /api/sample_queries` - Get sample queries
- `GET /api/health` - Health check

## 💡 Tips for Best Experience

1. **Provide complete property details** for accurate estimates:
   - Location (area in Bengaluru)
   - Size (square feet)
   - Number of bedrooms (BHK)
   - Number of bathrooms
   - Number of balconies

2. **Try conversational input**:
   - "I have a property to evaluate"
   - Then: "It's in Whitefield"
   - Then: "3 BHK, 1500 sqft"
   - System remembers context!

3. **Test the guardrails**:
   - Try extreme values to see outlier detection
   - Example: "10 BHK mansion in Whitefield, 50000 sqft"

## 🎯 Ready to Start?

Run this command and start chatting:
```bash
cd /home/maaz/RealyticsAI && ./run_app.sh
```

Enjoy your AI-powered real estate assistant! 🏠✨
