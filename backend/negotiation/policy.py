"""
Rule-based negotiation policy implementation for counteroffers and acceptance logic.

This module provides a simple, transparent set of heuristics for generating
counteroffers and acceptance decisions based on reservation and target prices.

It is designed to be used alongside the async SQLAlchemy setup (AsyncSession)
and to feed rationale/context to LLM components.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import math

from backend.negotiation.models import NegotiationSession, UserRole


class RuleBasedNegotiationPolicy:
    """Deterministic policy for negotiation decisions.

    Parameters
    - max_rounds: Maximum rounds after which the system tends to accept.
    - base_concession_factor: Scales the concession magnitude.
    """

    def __init__(self, max_rounds: int = 8, base_concession_factor: float = 0.3) -> None:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if not (0.0 < base_concession_factor <= 1.0):
            raise ValueError("base_concession_factor must be in (0, 1]")
        self.max_rounds = max_rounds
        self.base_concession_factor = base_concession_factor

    def calculate_reservation_price(self, property_valuation: Dict[str, float], user_role: str) -> float:
        """Compute reservation price from valuation and role.

        - Seller: willing to accept 10% below valuation
        - Buyer: willing to pay up to 10% above valuation

        Args:
            property_valuation: Dict with at least key "valuation" (float).
            user_role: "SELLER" or "BUYER".
        Returns:
            Reservation price as float.
        """
        valuation = float(property_valuation.get("valuation", 0.0))
        if valuation <= 0:
            raise ValueError("valuation must be positive")

        role = (user_role or "").upper().strip()
        if role == UserRole.SELLER.value:
            return valuation * 0.90
        if role == UserRole.BUYER.value:
            return valuation * 1.10
        raise ValueError("user_role must be 'SELLER' or 'BUYER'")

    def generate_counteroffer(self, session: NegotiationSession, user_offer: float) -> Dict[str, object]:
        """Generate a counteroffer with time-decay concessions and rationale.

        Concession formula:
            concession = (target - reservation) * (1 - round/max_rounds) * 0.3
        New offer direction:
            new_offer = last_offer + (concession if seller else -concession)
        Bounds:
            new_offer clamped to [reservation_price, target_price]

        Returns a structured payload usable by LLMs and UI layers.
        """
        if user_offer <= 0:
            raise ValueError("user_offer must be positive")

        role = session.user_role
        if isinstance(role, str):
            role_value = role.upper()
        else:
            role_value = role.value

        reservation = float(session.reservation_price)
        target = float(session.target_price)
        round_number = int(max(0, session.round_number or 0))
        last_offer = float(session.current_offer) if session.current_offer is not None else (
            float(session.initial_asking_price) if role_value == UserRole.SELLER.value else float(session.target_price)
        )

        # Time-decay term; never negative
        time_factor = max(0.0, 1.0 - (round_number / float(self.max_rounds)))
        concession_magnitude = (target - reservation) * time_factor * self.base_concession_factor

        if role_value == UserRole.SELLER.value:
            proposed = last_offer + concession_magnitude
        else:
            proposed = last_offer - concession_magnitude

        # Clamp to [reservation, target]
        lower, upper = (reservation, target) if reservation <= target else (target, reservation)
        new_offer = min(max(proposed, lower), upper)

        # Acceptance and confidence signals
        should_accept = self.should_accept_offer(session, user_offer)
        distance = abs(new_offer - reservation)
        confidence = 1.0 - (distance / max(reservation, 1e-6))
        confidence = float(min(max(confidence, 0.0), 1.0))

        rationale_points: List[str] = [
            f"Time-decay factor: {time_factor:.3f}",
            f"Concession magnitude: {concession_magnitude:.2f}",
            f"Role: {role_value}",
            f"Bounds: [{lower:.2f}, {upper:.2f}]",
            f"Last offer: {last_offer:.2f}",
            f"User offer: {user_offer:.2f}",
        ]

        return {
            "offer": float(new_offer),
            "concession_amount": float(abs(new_offer - last_offer)),
            "rationale_points": rationale_points,
            "should_accept": bool(should_accept),
            "confidence": confidence,
        }

    def should_accept_offer(self, session: NegotiationSession, offer: float) -> bool:
        """Decide acceptance based on proximity to reservation and elapsed rounds.

        Accept if:
        - Offer within 2% of reservation price; OR
        - round_number > max_rounds
        """
        if offer <= 0:
            return False

        reservation = float(session.reservation_price)
        if reservation <= 0:
            return False

        distance_ratio = abs(offer - reservation) / reservation
        near_reservation = distance_ratio <= 0.02
        overtime = (session.round_number or 0) > self.max_rounds
        return bool(near_reservation or overtime)

    def get_negotiation_features(self, session: NegotiationSession) -> Dict[str, float]:
        """Compute derived features used by downstream decision/LLM components.

        Returns
        - concession_rate: progress from target toward reservation (0..1)
        - time_pressure: round_number / max_rounds (0..1+)
        - distance_to_reservation: |current_offer - reservation| / reservation
        """
        reservation = float(session.reservation_price)
        target = float(session.target_price)
        round_number = float(max(0, session.round_number or 0))

        # Normalize concession as how far current_offer moved toward reservation from target
        denom = abs(target - reservation) or 1e-6
        if session.current_offer is None:
            progress = 0.0
        else:
            progress = abs(float(session.current_offer) - target) / denom
            progress = float(min(max(progress, 0.0), 1.0))

        time_pressure = round_number / float(self.max_rounds)
        if reservation <= 0:
            distance_to_reservation = 1.0
        else:
            base = float(session.current_offer) if session.current_offer is not None else target
            distance_to_reservation = abs(base - reservation) / reservation

        return {
            "concession_rate": float(progress),
            "time_pressure": float(time_pressure),
            "distance_to_reservation": float(distance_to_reservation),
        }
