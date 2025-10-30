"""
Negotiation API routes.

Provides endpoints to start, respond, view history, and end a negotiation session.
Uses async SQLAlchemy sessions, rule-based policy, and Gemini LLM integration.
Includes simple in-memory rate limiting (10 req/min per user key).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import uuid
import asyncio
from collections import defaultdict, deque

import httpx
from fastapi import APIRouter, HTTPException, Depends, Path, Body, status, Header, Request
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database.connection import get_db_session
from backend.negotiation.models import (
    NegotiationSession,
    NegotiationTurn,
    UserRole,
    NegotiationStatus,
    TurnActor,
)
from backend.negotiation.policy import RuleBasedNegotiationPolicy
from backend.negotiation.llm_handler import GeminiNegotiationHandler


router = APIRouter()

# ---------------------------------------------------------------------------
# Rate limiting: 10 requests/minute per user key (user_id or client IP)
# ---------------------------------------------------------------------------
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_LIMIT_MAX_REQUESTS = 10
_request_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_RATE_LIMIT_MAX_REQUESTS))


def _rate_limit_key(request: Request, user_id: Optional[str]) -> str:
    if user_id:
        return f"user:{user_id}"
    # fallback to client host
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


async def rate_limiter(request: Request, x_user_id: Optional[str] = Header(None)) -> None:
    now = time.time()
    key = _rate_limit_key(request, x_user_id)
    dq = _request_log[key]
    # Remove old entries
    while dq and now - dq[0] > _RATE_LIMIT_WINDOW_SEC:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    dq.append(now)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class StartNegotiationRequest(BaseModel):
    property_id: str = Field(...)
    target_price: float = Field(..., gt=0)
    user_role: str = Field(..., pattern=r"^(?i)(buyer|seller)$")
    initial_message: Optional[str] = Field(None)
    asking_price: Optional[float] = Field(None, gt=0)
    property_features: Optional[Dict[str, Any]] = None

    @validator("user_role")
    def normalize_role(cls, v: str) -> str:
        return v.upper()


class StartNegotiationResponse(BaseModel):
    session_id: str
    property_summary: Dict[str, Any]
    agent_opening: str
    initial_offer: float


class RespondRequest(BaseModel):
    session_id: str = Field(...)
    user_offer: float = Field(..., gt=0)
    user_message: str = Field(..., min_length=1)


class RespondResponse(BaseModel):
    agent_offer: float
    message: str
    explanation: str
    next_steps: List[str]
    status: str
    confidence: float


class HistoryResponse(BaseModel):
    session: Dict[str, Any]
    turns: List[Dict[str, Any]]
    summary: Optional[str]


class EndResponse(BaseModel):
    session_id: str
    status: str
    summary: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
policy = RuleBasedNegotiationPolicy()
llm = GeminiNegotiationHandler()


async def _fetch_property_valuation(property_id: str, target_price: float, *, asking_price: Optional[float] = None, property_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Attempt to fetch valuation via the existing prediction API.

    Falls back to target_price as valuation if API unavailable.
    """
    # Attempt minimal call (requires features; here we fallback immediately)
    # If you have a property store, pull features by property_id and call /api/v1/predict
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Prefer provided features from frontend if available
            payload = {
                "property_features": property_features or {"area": 1000, "bath": 2, "balcony": 1, "location": "Unknown"},
                "model_type": "local",
                "return_confidence": False,
            }
            resp = await client.post("http://localhost:8000/api/v1/predict", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "property_id": property_id,
                    "valuation": float(data.get("predicted_price", target_price)),
                    "currency": data.get("currency", "INR Lakhs"),
                    "model_used": data.get("model_used", "local"),
                }
    except Exception:
        pass
    # Fallback to asking_price if provided, else target
    fallback_val = asking_price if (asking_price and asking_price > 0) else target_price
    return {"property_id": property_id, "valuation": float(fallback_val), "currency": "INR Lakhs", "model_used": "fallback"}


def _role_enum(role: str) -> UserRole:
    r = (role or "").upper()
    return UserRole.SELLER if r == "SELLER" else UserRole.BUYER


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/api/negotiate/start", response_model=StartNegotiationResponse, dependencies=[Depends(rate_limiter)])
async def start_negotiation(
    req: StartNegotiationRequest,
    session: AsyncSession = Depends(get_db_session),
    x_user_id: Optional[str] = Header(None),
):
    """Simplified buyer-only negotiation: Check if target price is compatible with property asking price."""
    try:
        # Property asking price (from recommendation or valuation output)
        property_asking_price = float(req.asking_price) if req.asking_price else float(req.target_price)
        target_price = float(req.target_price)
        
        # Calculate price difference
        price_diff = property_asking_price - target_price
        price_diff_percent = (price_diff / property_asking_price) * 100 if property_asking_price > 0 else 0
        
        # Build context for Gemini
        prompt_context = f"""
You are a real estate negotiation advisor helping a BUYER.

Property Details:
- Property ID: {req.property_id}
- Property Asking Price: {property_asking_price:.2f} Lakhs
- Buyer's Target Price: {target_price:.2f} Lakhs
- Difference: {price_diff:.2f} Lakhs ({price_diff_percent:.1f}%)

Task: Analyze whether the buyer should suggest their target price to the seller.

Provide a clean, professional response covering:
1. Compatibility assessment (Is the target reasonable?)
2. Market perspective (Is this a good negotiation starting point?)
3. Clear recommendation (Should they proceed with this offer?)
4. Next steps

Keep it concise (3-4 short paragraphs) and actionable.
"""
        
        # Generate response using Gemini
        try:
            llm_response = llm._safe_generate(
                "You are an expert real estate negotiation advisor. Be concise, clear, and actionable.",
                prompt_context
            )
            agent_message = llm._get_text_content(llm_response) or "Analysis completed."
        except Exception as e:
            # Fallback to rule-based response
            if abs(price_diff_percent) <= 5:
                agent_message = (
                    f"✅ **Compatible Price Range**\n\n"
                    f"Your target of {target_price:.2f} Lakhs is very close to the asking price of {property_asking_price:.2f} Lakhs "
                    f"({abs(price_diff_percent):.1f}% difference).\n\n"
                    f"**Recommendation:** This is a reasonable starting point. The seller is likely to consider your offer. "
                    f"You can proceed with confidence.\n\n"
                    f"**Next Steps:**\n"
                    f"- Present this offer to the seller\n"
                    f"- Prepare to negotiate minor details (timeline, repairs, etc.)\n"
                    f"- Be ready to compromise slightly if needed"
                )
            elif price_diff_percent > 5 and price_diff_percent <= 15:
                agent_message = (
                    f"⚠️ **Moderate Gap - Negotiable**\n\n"
                    f"Your target of {target_price:.2f} Lakhs is {abs(price_diff_percent):.1f}% below the asking price of {property_asking_price:.2f} Lakhs.\n\n"
                    f"**Recommendation:** While there's a gap, this is within negotiable range in current market conditions. "
                    f"Consider starting slightly higher (around {target_price + (price_diff * 0.3):.2f} Lakhs) to leave room for negotiation.\n\n"
                    f"**Next Steps:**\n"
                    f"- Start with a counter around {target_price + (price_diff * 0.3):.2f} Lakhs\n"
                    f"- Highlight any property issues or market comparables\n"
                    f"- Be prepared to meet somewhere in the middle"
                )
            elif price_diff_percent > 15:
                agent_message = (
                    f"❌ **Significant Gap - Challenging**\n\n"
                    f"Your target of {target_price:.2f} Lakhs is {abs(price_diff_percent):.1f}% below the asking price of {property_asking_price:.2f} Lakhs.\n\n"
                    f"**Recommendation:** This gap may be too large for successful negotiation. "
                    f"Consider either increasing your budget to around {target_price + (price_diff * 0.5):.2f} Lakhs or looking for similar properties in a lower price range.\n\n"
                    f"**Next Steps:**\n"
                    f"- Reassess your budget constraints\n"
                    f"- Look for comparable properties in your range\n"
                    f"- If you proceed, provide strong justification for the lower price"
                )
            else:
                # Target is higher than asking (unlikely but handle it)
                agent_message = (
                    f"💰 **Above Asking Price**\n\n"
                    f"Your target of {target_price:.2f} Lakhs is actually higher than the asking price of {property_asking_price:.2f} Lakhs.\n\n"
                    f"**Recommendation:** You don't need to offer more than the asking price unless this is a highly competitive market. "
                    f"Start at or slightly below the asking price.\n\n"
                    f"**Next Steps:**\n"
                    f"- Offer at asking price: {property_asking_price:.2f} Lakhs\n"
                    f"- Negotiate for better terms or included furnishings\n"
                    f"- Save the extra budget for closing costs"
                )
        
        # Store minimal session info (optional - can be removed if not needed)
        session_id = str(uuid.uuid4())
        new_session = NegotiationSession(
            session_id=session_id,
            property_id=req.property_id,
            user_id=x_user_id,
            user_role=UserRole.BUYER,
            initial_asking_price=property_asking_price,
            target_price=target_price,
            reservation_price=target_price,  # Simplified
            current_offer=target_price,
            round_number=1,
            status=NegotiationStatus.ACTIVE,
            valuation_data=None,
        )
        session.add(new_session)
        await session.commit()
        
        return StartNegotiationResponse(
            session_id=session_id,
            property_summary={
                "property_id": req.property_id,
                "asking_price": property_asking_price,
                "target_price": target_price,
                "price_difference": price_diff,
                "price_difference_percent": price_diff_percent,
            },
            agent_opening=agent_message,
            initial_offer=target_price,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start negotiation: {e}")


@router.post("/api/negotiate/respond", response_model=RespondResponse, dependencies=[Depends(rate_limiter)])
async def respond_negotiation(
    req: RespondRequest,
    session: AsyncSession = Depends(get_db_session),
):
    try:
        # Load session
        result = await session.execute(
            select(NegotiationSession).where(NegotiationSession.session_id == req.session_id)
        )
        sess: Optional[NegotiationSession] = result.scalar_one_or_none()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        # Fetch latest valuation (or reuse target as fallback)
        property_summary = await _fetch_property_valuation(sess.property_id, sess.target_price)
        valuation = float(property_summary.get("valuation", sess.target_price))

        # Buyer-only simple respond logic
        user_offer = float(req.user_offer)
        if user_offer < 0.98 * valuation:
            agent_offer = round(valuation, 2)
            advice = (
                f"Your offer is below our model-estimated fair price (~{valuation:.2f}). "
                f"Consider raising your budget to {agent_offer:.2f} and submit that as your counter."
            )
        elif user_offer > valuation:
            agent_offer = round(0.96 * valuation, 2)
            advice = "Current trends support a counter just below the predicted fair price."
        else:
            agent_offer = round((user_offer + valuation) / 2.0, 2)
            advice = "A midpoint near fair value is reasonable to move discussions forward."

        policy_output = {
            "offer": agent_offer,
            "concession_amount": max(0.0, agent_offer - (sess.current_offer or agent_offer)),
            "rationale_points": [
                "Buyer mode",
                f"Model valuation≈{valuation:.2f}",
                advice,
            ],
            "should_accept": False,
            "confidence": 0.65,
        }

        # Generate conversational response using LLM
        llm_payload = llm.generate_negotiation_response(sess, policy_output, property_summary)

        # Persist turn
        next_round = int((sess.round_number or 0) + 1)
        agent_offer = float(policy_output.get("offer", sess.current_offer or sess.target_price))
        new_turn = NegotiationTurn(
            turn_id=str(uuid.uuid4()),
            session_id=sess.session_id,
            turn_number=next_round,
            actor=TurnActor.AGENT,
            offer_amount=agent_offer,
            message=str(llm_payload.get("message")),
            rationale=str(llm_payload.get("explanation")),
            confidence_score=float(policy_output.get("confidence", 0.5)),
        )
        session.add(new_turn)

        # Update session state
        sess.current_offer = agent_offer
        sess.round_number = next_round

        # Determine status
        status_out = "active"
        if policy.should_accept_offer(sess, req.user_offer):
            sess.status = NegotiationStatus.ACCEPTED
            status_out = "accepted"
        elif next_round > policy.max_rounds:
            sess.status = NegotiationStatus.ABANDONED
            status_out = "rejected"

        await session.commit()

        return RespondResponse(
            agent_offer=agent_offer,
            message=str(llm_payload.get("message")),
            explanation=str(llm_payload.get("explanation")),
            next_steps=list(llm_payload.get("next_steps", [])),
            status=status_out,
            confidence=float(policy_output.get("confidence", 0.5)),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process response: {e}")


@router.get("/api/negotiate/history/{session_id}", response_model=HistoryResponse, dependencies=[Depends(rate_limiter)])
async def get_history(
    session_id: str = Path(...),
    db: AsyncSession = Depends(get_db_session),
):
    try:
        res = await db.execute(select(NegotiationSession).where(NegotiationSession.session_id == session_id))
        sess: Optional[NegotiationSession] = res.scalar_one_or_none()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        # Query turns explicitly to avoid async IO from lazy relationship access
        res_turns = await db.execute(
            select(NegotiationTurn).where(NegotiationTurn.session_id == session_id)
        )
        turns = sorted(list(res_turns.scalars().all()), key=lambda t: (t.turn_number, t.created_at or 0))
        turns_payload = [
            {
                "turn_id": t.turn_id,
                "turn_number": t.turn_number,
                "actor": getattr(t.actor, "value", t.actor),
                "offer": t.offer_amount,
                "message": t.message,
                "rationale": t.rationale,
                "confidence": t.confidence_score,
                "created_at": getattr(t, "created_at", None),
            }
            for t in turns
        ]

        summary = llm.generate_negotiation_summary(sess)
        session_payload = {
            "session_id": sess.session_id,
            "property_id": sess.property_id,
            "user_id": sess.user_id,
            "role": getattr(sess.user_role, "value", sess.user_role),
            "initial_asking_price": sess.initial_asking_price,
            "target_price": sess.target_price,
            "reservation_price": sess.reservation_price,
            "current_offer": sess.current_offer,
            "round_number": sess.round_number,
            "status": getattr(sess.status, "value", sess.status),
        }

        return HistoryResponse(session=session_payload, turns=turns_payload, summary=summary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


@router.post("/api/negotiate/end/{session_id}", response_model=EndResponse, dependencies=[Depends(rate_limiter)])
async def end_negotiation(
    session_id: str = Path(...),
    db: AsyncSession = Depends(get_db_session),
):
    try:
        res = await db.execute(select(NegotiationSession).where(NegotiationSession.session_id == session_id))
        sess: Optional[NegotiationSession] = res.scalar_one_or_none()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        # Mark as abandoned if still active
        if sess.status == NegotiationStatus.ACTIVE:
            sess.status = NegotiationStatus.ABANDONED
        await db.commit()

        summary = llm.generate_negotiation_summary(sess)
        return EndResponse(session_id=sess.session_id, status=getattr(sess.status, "value", sess.status), summary=summary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end negotiation: {e}")
