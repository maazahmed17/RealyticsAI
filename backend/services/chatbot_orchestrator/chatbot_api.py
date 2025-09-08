"""
RealyticsAI Chatbot API
========================
FastAPI endpoints for natural language chatbot interface
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import uuid
from datetime import datetime
import asyncio

# Import chatbot handler
from chatbot_handler import RealyticsAIChatbot, ConversationState

# Initialize FastAPI app
app = FastAPI(
    title="RealyticsAI Chatbot API",
    description="Natural language interface for property price predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

# Session storage (in production, use Redis or database)
sessions: Dict[str, ConversationState] = {}

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    
class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    last_activity: str

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    print("🚀 Initializing RealyticsAI Chatbot...")
    
    # Initialize chatbot (set use_openai=True and provide API key for GPT)
    chatbot = RealyticsAIChatbot(use_openai=False)
    
    print("✅ Chatbot initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "RealyticsAI Chatbot",
        "status": "active",
        "endpoints": {
            "chat": "/api/chat",
            "websocket": "/ws/{session_id}",
            "session": "/api/session/{session_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for processing natural language queries
    
    Args:
        request: ChatRequest with message and optional session_id
        
    Returns:
        ChatResponse with bot response and metadata
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = ConversationState()
    
    # Process query
    response = await chatbot.process_query(request.message, session_id)
    
    # Extract intent and entities for metadata
    intent = chatbot.classify_intent(request.message)
    entities = chatbot.extract_entities(request.message)
    
    return ChatResponse(
        response=response,
        session_id=session_id,
        intent=intent.value,
        entities=entities,
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get session information
    
    Args:
        session_id: Session identifier
        
    Returns:
        SessionInfo with session details
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionInfo(
        session_id=session_id,
        created_at=session.history[0]["timestamp"] if session.history else datetime.now().isoformat(),
        message_count=len(session.history),
        last_activity=session.history[-1]["timestamp"] if session.history else datetime.now().isoformat()
    )

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session history
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat
    
    Args:
        websocket: WebSocket connection
        session_id: Session identifier
    """
    await manager.connect(websocket, session_id)
    
    # Create session if doesn't exist
    if session_id not in sessions:
        sessions[session_id] = ConversationState()
    
    # Send welcome message
    welcome = {
        "type": "system",
        "message": "Connected to RealyticsAI Chatbot. How can I help you today?",
        "timestamp": datetime.now().isoformat()
    }
    await websocket.send_json(welcome)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Process message
            response = await chatbot.process_query(data, session_id)
            
            # Send response
            response_data = {
                "type": "assistant",
                "message": response,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(response_data)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client {session_id} disconnected")

@app.post("/api/batch")
async def batch_process(queries: List[str]):
    """
    Process multiple queries at once
    
    Args:
        queries: List of natural language queries
        
    Returns:
        List of responses
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    responses = []
    session_id = str(uuid.uuid4())
    
    for query in queries:
        response = await chatbot.process_query(query, session_id)
        responses.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    return {"session_id": session_id, "responses": responses}

# Example usage endpoints
@app.get("/api/examples")
async def get_examples():
    """Get example queries for testing"""
    return {
        "examples": [
            {
                "category": "Price Prediction",
                "queries": [
                    "What's the price of a 2 bathroom apartment with 1 balcony?",
                    "How much would a house with 3 bathrooms and 2 balconies cost?",
                    "I need a property with two bathrooms and no balcony",
                    "What's the cost of a 3BHK in Whitefield?"
                ]
            },
            {
                "category": "Market Analysis",
                "queries": [
                    "Show me market analysis for Whitefield",
                    "What's the average price in Electronic City?",
                    "Give me market statistics",
                    "Which area has the most properties?"
                ]
            },
            {
                "category": "Recommendations",
                "queries": [
                    "Show me properties under 50 lakhs",
                    "What can I get for 100 lakhs?",
                    "Recommend areas within my budget of 75 lakhs",
                    "What's the best value location?"
                ]
            },
            {
                "category": "Comparisons",
                "queries": [
                    "Compare Whitefield and Electronic City",
                    "What's the price difference between 2 and 3 bathrooms?",
                    "Compare apartments vs houses",
                    "Which is better value: Koramangala or Indiranagar?"
                ]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
