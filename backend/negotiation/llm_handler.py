"""
Gemini-based LLM handler for negotiation assistance.

Provides high-level methods to:
- Generate conversational negotiation responses guided by rule-based policy output
- Parse user negotiation intent from natural language
- Summarize an entire negotiation session

Includes:
- Environment-based API key initialization (GEMINI_API_KEY)
- google-generativeai (Gemini 1.5 Flash) integration
- Retry with exponential backoff
- Structured logging for inputs/outputs and latency
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import time
import logging
import random

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - runtime dependency optional during unit tests
    genai = None  # type: ignore

from backend.negotiation.models import NegotiationSession, NegotiationTurn, UserRole
try:
    # Prefer configured settings if available
    from backend.core.config.settings import get_settings  # type: ignore
except Exception:
    get_settings = None  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeminiNegotiationHandler:
    """Wrapper around Gemini to support negotiation tasks.

    Attributes
    - model_name: Gemini model used for generation
    - max_retries: Number of retry attempts on transient failures
    - backoff_base: Base seconds for exponential backoff
    - request_timeout: Optional request timeout (if supported by SDK)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        max_retries: int = 3,
        backoff_base: float = 0.8,
        request_timeout: Optional[float] = None,
    ) -> None:
        # Pull from settings if available
        if get_settings:
            try:
                settings = get_settings()
                api_key = api_key or getattr(settings, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
                model_name = getattr(settings, "model_name", model_name) or model_name
            except Exception:
                api_key = api_key or os.getenv("GEMINI_API_KEY")
        else:
            api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set; LLM calls will be skipped.")
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.request_timeout = request_timeout

        if genai and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:  # pragma: no cover - external SDK init
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.model = None
        else:
            self.model = None

    # ------------------------------
    # Public API
    # ------------------------------
    def generate_negotiation_response(
        self,
        session: NegotiationSession,
        policy_output: Dict[str, Any],
        property_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate an LLM-grounded negotiation response.

        Returns a dict with keys: message, explanation, next_steps.
        Falls back to a deterministic message if LLM is unavailable.
        """
        prompt = self._build_negotiation_prompt(session, policy_output, property_data)
        system_prompt = (
            "You are an expert real estate negotiation assistant. Be concise, transparent, "
            "and professional. Reflect policy guidance but preserve user objectives. "
            "Provide a short response, a clear rationale, and concrete next steps."
        )

        start_time = time.time()
        try:
            llm_output = self._safe_generate(system_prompt, prompt)
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Gemini response received: latency_ms=%s", latency_ms,
            )
            return self._extract_response_payload(llm_output, policy_output)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback deterministic response
            message = self._fallback_message(session, policy_output, property_data)
            return {
                "message": message,
                "explanation": "LLM unavailable. Provided rule-based guidance instead.",
                "next_steps": [
                    "Consider the proposed counteroffer",
                    "Share additional constraints or preferences",
                    "Proceed to the next negotiation round",
                ],
            }

    def parse_user_negotiation_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Parse property_id, target_price, user_role from natural language.

        Uses Gemini for structured extraction; falls back to naive parsing when unavailable.
        Returns None if insufficient signal.
        """
        if not user_message or not user_message.strip():
            return None

        schema_instruction = (
            "Extract negotiation parameters as JSON with keys: property_id (string), "
            "target_price (number), user_role (BUYER|SELLER). If unsure, return null."
        )
        prompt = f"User message: {user_message}\n{schema_instruction}"

        try:
            llm_output = self._safe_generate("You extract structured data.", prompt)
            parsed = self._parse_json_from_text(llm_output)
            if parsed and {"property_id", "target_price", "user_role"}.issubset(parsed.keys()):
                return parsed
        except Exception as e:
            logger.warning(f"Intent parse via LLM failed: {e}")

        # Fallback naive extraction
        lower = user_message.lower()
        prop = None
        price = None
        role = None
        # rudimentary heuristics
        for token in lower.replace("\n", " ").split():
            if token.startswith("pid-") or token.startswith("prop-"):
                prop = token
            if token.endswith("k") and token[:-1].isdigit():
                try:
                    price = float(token[:-1]) * 1000
                except Exception:
                    pass
            if "seller" in lower:
                role = "SELLER"
            if "buyer" in lower:
                role = role or "BUYER"
        if prop and price and role:
            return {"property_id": prop, "target_price": price, "user_role": role}
        return None

    def generate_negotiation_summary(self, session: NegotiationSession) -> str:
        """Summarize the negotiation history and outcome.

        Uses LLM if available; otherwise constructs a deterministic summary.
        """
        # Avoid accessing lazy-loaded relationships here to prevent async IO in sync context
        history_lines: List[str] = [
            "History omitted in summary (async DB context required)."
        ]

        prompt = (
            "Provide a concise summary of the negotiation, including initial positions, key concessions, and current status.\n\n"
            + "\n".join(history_lines)
        )

        try:
            llm_output = self._safe_generate(
                "You are a concise negotiation summarizer.", prompt
            )
            text = self._get_text_content(llm_output)
            return text or "Summary unavailable."
        except Exception as e:
            logger.warning(f"Summary via LLM failed: {e}")
            # Fallback
            return (
                f"Negotiation Summary: rounds={len(history_lines)}; "
                f"role={getattr(session.user_role, 'value', session.user_role)}; "
                f"target={session.target_price}; reservation={session.reservation_price}."
            )

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _safe_generate(self, system_prompt: str, user_prompt: str) -> Any:
        """Call Gemini with retries, logging token usage and latency.

        Returns raw SDK response. Raises on final failure.
        """
        if not self.model:
            raise RuntimeError("Gemini model not initialized")

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                # Note: google-generativeai API evolve; we use a simple text prompt pattern here.
                response = self.model.generate_content(
                    [
                        {"role": "system", "parts": [system_prompt]},
                        {"role": "user", "parts": [user_prompt]},
                    ]
                )
                latency_ms = int((time.time() - start) * 1000)
                usage = getattr(response, "usage_metadata", None)
                logger.info(
                    "Gemini call ok: attempt=%s latency_ms=%s input_tokens=%s output_tokens=%s",
                    attempt,
                    latency_ms,
                    getattr(usage, "prompt_token_count", None),
                    getattr(usage, "candidates_token_count", None),
                )
                return response
            except Exception as e:  # pragma: no cover - network dependent
                last_exc = e
                # If rate limited or quota exhausted, don't bubble raw error; backoff and possibly fallback
                err_txt = str(e).lower()
                if "429" in err_txt or "resource exhausted" in err_txt or "quota" in err_txt:
                    if attempt >= self.max_retries:
                        # Final fallback by raising a generic runtime error that upstream converts to fallback
                        raise RuntimeError("LLM_RATE_LIMIT")
                sleep_s = self.backoff_base * (2 ** (attempt - 1)) * (1.0 + random.random() * 0.2)
                logger.warning("Gemini call failed: attempt=%s error=%s; sleeping %.2fs", attempt, e, sleep_s)
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    def _build_negotiation_prompt(
        self,
        session: NegotiationSession,
        policy_output: Dict[str, Any],
        property_data: Dict[str, Any],
    ) -> str:
        """Construct a grounded prompt with property details and policy guidance."""
        lines: List[str] = []
        # Property details
        lines.append("Property Context:")
        lines.append(json.dumps({
            "property": {
                "id": property_data.get("property_id") or session.property_id,
                "location": property_data.get("location"),
                "size": property_data.get("size"),
                "valuation": property_data.get("valuation"),
                "features": property_data.get("features"),
            }
        }, ensure_ascii=False))

        # Negotiation state
        lines.append("Negotiation State:")
        state = {
            "role": getattr(session.user_role, "value", session.user_role),
            "round": session.round_number,
            "initial_asking_price": session.initial_asking_price,
            "target_price": session.target_price,
            "reservation_price": session.reservation_price,
            "current_offer": session.current_offer,
        }
        # Do not attempt to access session.turns here; caller can inject history if needed.
        lines.append(json.dumps(state, ensure_ascii=False))

        # Policy guidance
        lines.append("Policy Decision:")
        lines.append(json.dumps(policy_output, ensure_ascii=False))

        lines.append(
            "Respond with: a short user-facing message, a clear explanation of reasoning, and 2-4 next steps."
        )
        return "\n".join(lines)

    def _extract_response_payload(self, llm_output: Any, policy_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gemini response into structured payload.

        Tries to parse JSON; otherwise falls back to text splits.
        """
        text = self._get_text_content(llm_output) or ""
        # Try JSON first
        parsed = self._parse_json_from_text(text)
        if isinstance(parsed, dict) and {"message", "explanation", "next_steps"}.issubset(parsed.keys()):
            return parsed  # already structured

        # Fallback heuristic parsing
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        message = parts[0] if parts else "Here is a counteroffer proposal based on current policy guidance."
        explanation = (parts[1] if len(parts) > 1 else "") or json.dumps(policy_output, ensure_ascii=False)
        next_steps = []
        if len(parts) > 2:
            for line in parts[2].splitlines():
                line = line.strip("- *• ")
                if line:
                    next_steps.append(line)
        if not next_steps:
            next_steps = [
                "Review the counteroffer and confirm acceptance or propose an adjustment.",
                "Specify constraints (timeline, payment terms) to refine the offer.",
            ]
        return {"message": message, "explanation": explanation, "next_steps": next_steps}

    @staticmethod
    def _get_text_content(llm_output: Any) -> Optional[str]:
        """Extract the primary text from a Gemini response object."""
        try:
            # google-generativeai returns candidates[0].content.parts[0].text typically
            candidates = getattr(llm_output, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        return text
            # Some SDK versions expose .text directly
            return getattr(llm_output, "text", None)
        except Exception:
            return None

    @staticmethod
    def _parse_json_from_text(text: Any) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from text, tolerating fenced blocks."""
        if not isinstance(text, str) or not text:
            return None
        s = text.strip()
        # Strip code fences if present
        if s.startswith("```"):
            s = s.strip("`\n ")
            # Remove potential language token
            s = "\n".join(line for line in s.splitlines() if not line.strip().startswith("{") or True)
        # Find first JSON object
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _fallback_message(
        session: NegotiationSession, policy_output: Dict[str, Any], property_data: Dict[str, Any]
    ) -> str:
        role = getattr(session.user_role, "value", session.user_role)
        offer = policy_output.get("offer")
        return (
            f"Based on current context, as a {role}, I propose a counteroffer of {offer}. "
            f"This balances your target ({session.target_price}) with market considerations."
        )
