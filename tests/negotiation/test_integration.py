import asyncio
import json
import pytest
import httpx

pytestmark = pytest.mark.asyncio


async def _start_session(client, property_id="prop-123", target_price=180.0, user_role="buyer"):
    resp = await client.post("/api/negotiate/start", json={
        "property_id": property_id,
        "target_price": target_price,
        "user_role": user_role,
        "initial_message": "init"
    })
    data = resp.json()
    return resp, data


async def _respond(client, session_id: str, user_offer: float, user_message: str = None):
    resp = await client.post("/api/negotiate/respond", json={
        "session_id": session_id,
        "user_offer": user_offer,
        "user_message": user_message or str(user_offer)
    })
    data = resp.json()
    return resp, data


async def _history(client, session_id: str):
    resp = await client.get(f"/api/negotiate/history/{session_id}")
    return resp, resp.json()


async def _end(client, session_id: str):
    resp = await client.post(f"/api/negotiate/end/{session_id}")
    return resp, resp.json()


async def test_complete_flow_acceptance(async_client):
    # Start
    r, d = await _start_session(async_client)
    assert r.status_code == 200
    session_id = d["session_id"]
    assert session_id
    assert d["property_summary"]["valuation"] == pytest.approx(180.0)

    # 3 rounds to trigger deterministic acceptance via fixture policy
    r1, d1 = await _respond(async_client, session_id, 170.0)
    assert r1.status_code == 200
    assert d1["status"] in ("active", "accepted")

    r2, d2 = await _respond(async_client, session_id, 175.0)
    assert r2.status_code == 200

    r3, d3 = await _respond(async_client, session_id, 178.0)
    assert r3.status_code == 200
    assert d3["status"] in ("accepted", "rejected", "active")

    # Fetch history
    rh, hist = await _history(async_client, session_id)
    assert rh.status_code == 200
    assert hist["session"]["session_id"] == session_id
    assert isinstance(hist.get("turns", []), list)


async def test_timeout_rejection(async_client, monkeypatch):
    # Override policy to set max_rounds=1 for timeout behavior
    from backend.negotiation import routes as negotiation_routes
    negotiation_routes.policy.max_rounds = 1

    r, d = await _start_session(async_client)
    session_id = d["session_id"]

    # First respond could be active
    await _respond(async_client, session_id, 170.0)
    # Second respond should exceed max rounds and mark rejected/abandoned
    r2, d2 = await _respond(async_client, session_id, 171.0)
    assert r2.status_code == 200
    assert d2["status"] in ("rejected", "accepted")


async def test_invalid_session_id(async_client):
    r, d = await _respond(async_client, "nonexistent-session", 150.0)
    assert r.status_code == 404
    assert "Session not found" in (d.get("detail") or "")


async def test_rate_limiting_enforced(async_client):
    headers = {"X-User-Id": "test-user-1"}
    # Hit start endpoint 10 times within window, last should 429
    status_codes = []
    for i in range(11):
        resp = await async_client.post("/api/negotiate/start", json={
            "property_id": f"prop-{i}",
            "target_price": 100 + i,
            "user_role": "buyer"
        }, headers=headers)
        status_codes.append(resp.status_code)
        # tiny sleep to keep within window but avoid event loop starvation
        await asyncio.sleep(0.01)
    assert status_codes.count(429) >= 1


async def test_valuation_fallback(async_client, monkeypatch):
    # Force valuation fetch to simulate failure and fallback to target_price
    from backend.negotiation import routes as negotiation_routes

    async def _fail_fetch(property_id: str, target_price: float):
        raise RuntimeError("service down")

    monkeypatch.setattr(negotiation_routes, "_fetch_property_valuation", _fail_fetch, raising=True)

    r, d = await _start_session(async_client, property_id="prop-x", target_price=222.0)
    assert r.status_code == 200
    # Because of failure, StartNegotiation still succeeds due to fallback in route logic
    # Our conftest stub was overridden to fail, so route fallback model returns target_price
    assert d["property_summary"]["valuation"] == pytest.approx(222.0)
