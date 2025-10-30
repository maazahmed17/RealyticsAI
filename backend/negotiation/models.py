"""
Negotiation module ORM models.

Defines SQLAlchemy ORM models for negotiation sessions and turns, designed to
work with the project's async SQLAlchemy setup (AsyncSession with
expire_on_commit=False).
"""

from __future__ import annotations

from typing import Optional, List
import enum

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Enum as SAEnum,
    ForeignKey,
    Text,
    DateTime,
    Index,
)
from sqlalchemy.orm import relationship

# Import BaseModel from the project's core models
from backend.core.models.base import BaseModel


class UserRole(enum.Enum):
    """Role of the end user in the negotiation."""

    BUYER = "BUYER"
    SELLER = "SELLER"


class NegotiationStatus(enum.Enum):
    """High-level state of a negotiation session."""

    ACTIVE = "ACTIVE"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ABANDONED = "ABANDONED"


class TurnActor(enum.Enum):
    """Actor for a given negotiation turn (user vs agent)."""

    USER = "USER"
    AGENT = "AGENT"


class NegotiationSession(BaseModel):
    """Negotiation session lifecycle and parameters.

    Inherits common fields from BaseModel: id (PK), created_at, updated_at.

    Fields
    - session_id: UUID string identifier (unique)
    - property_id: Identifier of the property under negotiation
    - user_id: Optional user id for future auth integration
    - user_role: The role the end user is playing (BUYER/SELLER)
    - initial_asking_price: Initial listing/asking price
    - target_price: Target price for planned settlement
    - reservation_price: Reservation price (min seller accepts / max buyer pays)
    - current_offer: Latest numerical offer proposed by the agent (nullable)
    - round_number: Current round number, starts at 0
    - status: Session status, defaults to ACTIVE
    - valuation_data: JSON string payload for valuation context (nullable)

    Relationships
    - turns: One-to-many to NegotiationTurn
    """

    __tablename__ = "negotiation_sessions"

    # Natural/External identifiers
    session_id = Column(String(36), unique=True, index=True, nullable=False)

    # Domain linkage
    property_id = Column(String(50), index=True, nullable=False)
    user_id = Column(String(50), index=True, nullable=True)

    # Session configuration
    user_role = Column(SAEnum(UserRole), nullable=False)
    initial_asking_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    reservation_price = Column(Float, nullable=False)

    # Dynamic state
    current_offer: Optional[float] = Column(Float, nullable=True)
    round_number = Column(Integer, nullable=False, default=0)
    status = Column(SAEnum(NegotiationStatus), nullable=False, default=NegotiationStatus.ACTIVE)

    # Context
    valuation_data: Optional[str] = Column(Text, nullable=True)

    # Relationships
    turns = relationship(
        "NegotiationTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=False,
    )

    # Indexes for common queries
    __table_args__ = (
        Index("ix_negotiation_sessions_status_created_at", "status", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<NegotiationSession(session_id={self.session_id!r}, "
            f"status={self.status.value if self.status else None!r}, "
            f"round_number={self.round_number})>"
        )


class NegotiationTurn(BaseModel):
    """A single conversational turn within a negotiation session.

    Inherits common fields from BaseModel: id (PK), created_at, updated_at.

    Fields
    - turn_id: UUID string identifier (unique)
    - session_id: FK to negotiation_sessions.session_id
    - turn_number: Monotonic counter within a session
    - actor: Whether the user or the agent produced this turn
    - offer_amount: Numeric offer value if present
    - message: Raw textual content from user/agent
    - rationale: Optional agent reasoning/context
    - confidence_score: Optional confidence for agent offers (0.0 - 1.0)
    """

    __tablename__ = "negotiation_turns"

    # Natural/External identifiers
    turn_id = Column(String(36), unique=True, index=True, nullable=False)

    # Linkage
    session_id = Column(
        String(36),
        ForeignKey("negotiation_sessions.session_id", ondelete=None),
        nullable=False,
        index=True,
    )

    # Turn metadata
    turn_number = Column(Integer, nullable=False)
    actor = Column(SAEnum(TurnActor), nullable=False)

    # Content
    offer_amount: Optional[float] = Column(Float, nullable=True)
    message = Column(Text, nullable=False)
    rationale: Optional[str] = Column(Text, nullable=True)
    confidence_score: Optional[float] = Column(Float, nullable=True)

    # Relationships
    session = relationship("NegotiationSession", back_populates="turns")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_negotiation_turns_session_id_turn_number", "session_id", "turn_number"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<NegotiationTurn(turn_id={self.turn_id!r}, "
            f"turn_number={self.turn_number}, actor={self.actor.value if self.actor else None!r})>"
        )
