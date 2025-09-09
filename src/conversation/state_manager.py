"""
Conversation State Manager
===========================
Manages conversation context and entity tracking across multiple turns.
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
import uuid


@dataclass
class PropertyFeatures:
    """Dataclass to store property features with validation."""
    bhk: Optional[int] = None
    sqft: Optional[float] = None
    bath: Optional[int] = None
    balcony: Optional[int] = None
    location: Optional[str] = None
    
    def is_complete_for_prediction(self) -> bool:
        """Check if we have minimum required features for prediction."""
        # Minimum requirements: BHK or (bath and balcony)
        has_bhk = self.bhk is not None
        has_bath_balcony = self.bath is not None and self.balcony is not None
        return has_bhk or has_bath_balcony
    
    def get_missing_required_fields(self) -> List[str]:
        """Get list of missing required fields."""
        missing = []
        
        # Check for primary requirements
        if self.bhk is None:
            missing.append("number of bedrooms (BHK)")
        
        # If no BHK, we need bath and balcony
        if self.bhk is None:
            if self.bath is None:
                missing.append("number of bathrooms")
            if self.balcony is None:
                missing.append("number of balconies")
        
        # Location is highly recommended but not required
        if self.location is None:
            missing.append("location (recommended for accurate prediction)")
            
        return missing
    
    def get_missing_optional_fields(self) -> List[str]:
        """Get list of missing optional fields that would improve accuracy."""
        missing = []
        
        if self.sqft is None:
            missing.append("size in square feet")
        
        # If we have BHK but not bath/balcony
        if self.bhk is not None:
            if self.bath is None:
                missing.append("number of bathrooms")
            if self.balcony is None:
                missing.append("number of balconies")
                
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def update_from_dict(self, updates: Dict[str, Any]) -> Set[str]:
        """Update features from dictionary, return set of updated fields."""
        updated_fields = set()
        
        for key, value in updates.items():
            if value is not None and hasattr(self, key):
                old_value = getattr(self, key)
                if old_value != value:
                    setattr(self, key, value)
                    updated_fields.add(key)
                    
        return updated_fields


class ConversationState:
    """Manages conversation state and context."""
    
    def __init__(self, session_id: str = None):
        """Initialize conversation state.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.features = PropertyFeatures()
        self.conversation_history: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.turn_count = 0
        self.last_prediction = None
        self.context_notes = []  # Store important context from conversation
    
    def update_features(self, new_features: Dict[str, Any]) -> Dict[str, Any]:
        """Update features with new information.
        
        Args:
            new_features: Dictionary of new feature values
            
        Returns:
            Dictionary with update status and details
        """
        # Filter out None values and empty strings
        cleaned_features = {
            k: v for k, v in new_features.items() 
            if v is not None and v != ""
        }
        
        # Update features
        updated_fields = self.features.update_from_dict(cleaned_features)
        
        # Update metadata
        self.last_updated = datetime.now()
        
        # Create update summary
        update_summary = {
            "updated_fields": list(updated_fields),
            "new_values": cleaned_features,
            "current_state": self.features.to_dict(),
            "is_complete": self.features.is_complete_for_prediction(),
            "missing_required": self.features.get_missing_required_fields(),
            "missing_optional": self.features.get_missing_optional_fields()
        }
        
        return update_summary
    
    def add_turn(self, user_message: str, bot_response: str, 
                 extracted_features: Dict[str, Any] = None):
        """Add a conversation turn to history.
        
        Args:
            user_message: User's input
            bot_response: Bot's response
            extracted_features: Features extracted from this turn
        """
        self.turn_count += 1
        
        turn = {
            "turn": self.turn_count,
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "bot": bot_response,
            "extracted_features": extracted_features or {}
        }
        
        self.conversation_history.append(turn)
        
        # Keep only last 10 turns to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_context(self, last_n_turns: int = 3) -> str:
        """Get conversation context as a string.
        
        Args:
            last_n_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        if not self.conversation_history:
            return "No previous conversation."
        
        recent_turns = self.conversation_history[-last_n_turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user']}")
            # Truncate long bot responses
            bot_response = turn['bot'][:200] + "..." if len(turn['bot']) > 200 else turn['bot']
            context_parts.append(f"Assistant: {bot_response}")
        
        return "\n".join(context_parts)
    
    def get_current_state_summary(self) -> str:
        """Get a human-readable summary of current state.
        
        Returns:
            String summary of current property features
        """
        if not self.features.to_dict():
            return "No property details collected yet."
        
        parts = []
        
        if self.features.bhk:
            parts.append(f"{self.features.bhk} BHK")
        
        if self.features.sqft:
            parts.append(f"{self.features.sqft} sq.ft")
        
        if self.features.bath:
            parts.append(f"{self.features.bath} bathroom(s)")
        
        if self.features.balcony:
            parts.append(f"{self.features.balcony} balcony/balconies")
        
        if self.features.location:
            parts.append(f"in {self.features.location}")
        
        if parts:
            return "Current property: " + ", ".join(parts)
        else:
            return "Partial property details collected."
    
    def reset_features(self):
        """Reset property features while maintaining conversation history."""
        self.features = PropertyFeatures()
        self.last_prediction = None
        self.context_notes = []
    
    def should_make_prediction(self) -> bool:
        """Check if we should make a prediction based on current state."""
        return self.features.is_complete_for_prediction()
    
    def get_clarification_prompt(self) -> str:
        """Generate a clarification prompt based on missing information.
        
        Returns:
            A helpful prompt asking for missing information
        """
        missing_required = self.features.get_missing_required_fields()
        missing_optional = self.features.get_missing_optional_fields()
        
        if not missing_required:
            if missing_optional:
                optional_str = ", ".join(missing_optional)
                return (f"I have enough information for a basic estimate. "
                       f"For a more accurate prediction, could you also provide: {optional_str}?")
            else:
                return "I have all the information needed for an accurate prediction."
        
        # Build clarification request
        current_info = self.get_current_state_summary()
        
        if current_info != "No property details collected yet.":
            prompt = f"{current_info}\n\n"
        else:
            prompt = ""
        
        prompt += "To provide a price estimate, I need "
        
        if len(missing_required) == 1:
            prompt += f"the {missing_required[0]}."
        else:
            prompt += "the following information: " + ", ".join(missing_required) + "."
        
        return prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "features": self.features.to_dict(),
            "conversation_history": self.conversation_history,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "turn_count": self.turn_count,
            "last_prediction": self.last_prediction,
            "context_notes": self.context_notes
        }
    
    def save_to_file(self, filepath: Path = None):
        """Save state to JSON file.
        
        Args:
            filepath: Path to save file (default: session_states/{session_id}.json)
        """
        if filepath is None:
            state_dir = Path(__file__).parent.parent.parent / "session_states"
            state_dir.mkdir(exist_ok=True)
            filepath = state_dir / f"{self.session_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ConversationState':
        """Load state from JSON file.
        
        Args:
            filepath: Path to state file
            
        Returns:
            ConversationState instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        state = cls(session_id=data['session_id'])
        state.features.update_from_dict(data['features'])
        state.conversation_history = data['conversation_history']
        state.created_at = datetime.fromisoformat(data['created_at'])
        state.last_updated = datetime.fromisoformat(data['last_updated'])
        state.turn_count = data['turn_count']
        state.last_prediction = data.get('last_prediction')
        state.context_notes = data.get('context_notes', [])
        
        return state
