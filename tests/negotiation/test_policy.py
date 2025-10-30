import math
import pytest

from backend.negotiation.policy import RuleBasedNegotiationPolicy
from backend.negotiation.models import NegotiationSession, UserRole


@pytest.fixture()
def policy() -> RuleBasedNegotiationPolicy:
    return RuleBasedNegotiationPolicy(max_rounds=8, base_concession_factor=0.3)


def make_session(role: UserRole, **overrides):
    base = dict(
        session_id="s-123",
        property_id="p-1",
        user_id=None,
        user_role=role,
        initial_asking_price=100.0,
        target_price=110.0 if role == UserRole.SELLER else 90.0,
        reservation_price=90.0 if role == UserRole.SELLER else 110.0,
        current_offer=None,
        round_number=0,
        status=None,
        valuation_data=None,
    )
    base.update(overrides)
    return NegotiationSession(**base)  # type: ignore[arg-type]


def test_reservation_price_calc(policy: RuleBasedNegotiationPolicy):
    assert policy.calculate_reservation_price({"valuation": 100}, "SELLER") == 90
    assert policy.calculate_reservation_price({"valuation": 100}, "BUYER") == 110
    with pytest.raises(ValueError):
        policy.calculate_reservation_price({"valuation": 0}, "SELLER")
    with pytest.raises(ValueError):
        policy.calculate_reservation_price({"valuation": 100}, "OTHER")


def test_generate_counteroffer_bounds_and_direction_seller(policy: RuleBasedNegotiationPolicy):
    session = make_session(UserRole.SELLER, current_offer=95.0, round_number=0)
    result = policy.generate_counteroffer(session, user_offer=92.0)
    assert "offer" in result
    assert session.reservation_price <= result["offer"] <= session.target_price
    # Seller rule: last_offer + concession (per spec)
    assert result["offer"] >= 95.0


def test_generate_counteroffer_bounds_and_direction_buyer(policy: RuleBasedNegotiationPolicy):
    session = make_session(UserRole.BUYER, current_offer=105.0, round_number=0)
    result = policy.generate_counteroffer(session, user_offer=106.0)
    assert session.target_price <= result["offer"] <= session.reservation_price
    # Buyer rule: last_offer - concession (per spec)
    assert result["offer"] <= 105.0


def test_time_decay_reduces_concession(policy: RuleBasedNegotiationPolicy):
    session_early = make_session(UserRole.SELLER, current_offer=95.0, round_number=0)
    session_late = make_session(UserRole.SELLER, current_offer=95.0, round_number=8)

    r_early = policy.generate_counteroffer(session_early, user_offer=92.0)
    r_late = policy.generate_counteroffer(session_late, user_offer=92.0)

    # Concession amount should decay over time
    assert r_early["concession_amount"] >= r_late["concession_amount"]


def test_should_accept_offer_threshold_and_timeout(policy: RuleBasedNegotiationPolicy):
    # Accept within 2% of reservation
    session = make_session(UserRole.SELLER, round_number=0)
    near = session.reservation_price * 1.018
    far = session.reservation_price * 1.05

    assert policy.should_accept_offer(session, near) is True
    assert policy.should_accept_offer(session, far) is False

    # Accept when over max rounds
    session_overtime = make_session(UserRole.SELLER, round_number=9)
    assert policy.should_accept_offer(session_overtime, far) is True


def test_get_negotiation_features(policy: RuleBasedNegotiationPolicy):
    session = make_session(UserRole.SELLER, current_offer=95.0, round_number=4)
    feats = policy.get_negotiation_features(session)
    assert 0.0 <= feats["concession_rate"] <= 1.0
    assert pytest.approx(feats["time_pressure"], rel=1e-6) == 4 / policy.max_rounds
    assert feats["distance_to_reservation"] >= 0.0
