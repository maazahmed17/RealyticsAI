import asyncio
import json
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import app and dependencies
from backend.main import app as fastapi_app
from backend.core.models.base import Base
from backend.negotiation import routes as negotiation_routes
from backend.negotiation.policy import RuleBasedNegotiationPolicy


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, future=True)
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def test_db_session_factory(test_db_engine) -> sessionmaker:
    return sessionmaker(test_db_engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture(scope="function")
async def test_db(test_db_session_factory) -> AsyncGenerator[AsyncSession, None]:
    async with test_db_session_factory() as session:
        # Start a SAVEPOINT-like transaction for isolation
        trans = await session.begin()
        try:
            yield session
        finally:
            await trans.rollback()
            await session.close()


@pytest_asyncio.fixture(scope="function")
async def async_client(test_db, monkeypatch) -> AsyncGenerator[AsyncClient, None]:
    # Override DB dependency to use test session
    async def _override_get_db():
        yield test_db

    fastapi_app.dependency_overrides = getattr(fastapi_app, 'dependency_overrides', {})
    fastapi_app.dependency_overrides[negotiation_routes.get_db_session] = _override_get_db

    # Mock LLM handler to avoid external calls
    class _MockLLM:
        def generate_negotiation_response(self, session, policy_output: Dict[str, Any], property_data: Dict[str, Any]):
            return {
                "message": f"Proposed offer: {policy_output.get('offer', 0)}",
                "explanation": "Mocked explanation for testing",
                "next_steps": ["Consider counter", "Provide constraints"],
            }
        def generate_negotiation_summary(self, session):
            return "Mocked summary"

    monkeypatch.setattr(negotiation_routes, 'llm', _MockLLM(), raising=True)

    # Provide deterministic policy if needed
    class _DeterministicPolicy(RuleBasedNegotiationPolicy):
        def __init__(self):
            super().__init__(max_rounds=3, base_concession_factor=0.3)
        def should_accept_offer(self, session, offer: float) -> bool:
            # Accept on 3rd round for determinism
            return (session.round_number or 0) >= 3

    monkeypatch.setattr(negotiation_routes, 'policy', _DeterministicPolicy(), raising=True)

    # Stub valuation fetch to avoid external service
    async def _stub_fetch_valuation(property_id: str, target_price: float):
        return {"property_id": property_id, "valuation": float(target_price), "currency": "INR Lakhs", "model_used": "stub"}

    monkeypatch.setattr(negotiation_routes, '_fetch_property_valuation', _stub_fetch_valuation, raising=True)

    async with AsyncClient(app=fastapi_app, base_url="http://test") as client:
        yield client

    fastapi_app.dependency_overrides.clear()


@pytest.fixture()
def sample_property() -> Dict[str, Any]:
    return {
        "property_id": "prop-123",
        "location": "Whitefield",
        "valuation": 185.0,
        "features": {"bhk": 3, "bath": 2, "sqft": 1500}
    }


@pytest.fixture()
def mock_gemini(monkeypatch):
    class _MockLLM:
        def generate_negotiation_response(self, session, policy_output, property_data):
            return {"message": "mock", "explanation": "mock", "next_steps": ["a", "b"]}
        def generate_negotiation_summary(self, session):
            return "mock-summary"
    monkeypatch.setattr(negotiation_routes, 'llm', _MockLLM(), raising=True)
    return True
