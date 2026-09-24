"""
communication/protocol.py
Parse LLM raw output → StructuredMessage.

Changes vs original:
- Critic action: negation-aware keyword matching (fixes false disagree)
- Vague keywords no longer auto-disagree
- Calibrated confidence cap on parse errors
"""

import json
import logging
import re
from typing import Optional
from .message import StructuredMessage

logger = logging.getLogger(__name__)


def _normalize_reasoning(value) -> list:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(r).strip() for r in value if str(r).strip()]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items() if str(v).strip()]
    return []


def _salvage_reasoning_from_raw(raw: str) -> list:
    if not raw:
        return []
    text = re.sub(r"```(?:json)?", "", raw)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

    lines = [ln.strip(" -*\t") for ln in re.split(r"[\n\r]+", text)]
    lines = [
        ln for ln in lines
        if len(ln) > 3
        and not ln.lstrip().startswith(("{", "}", "[", "]"))
        and '":' not in ln
    ]
    if len(lines) <= 1 and text.strip():
        parts = re.split(r"(?<=[.!?=])\s+", text.strip())
        lines = [p.strip() for p in parts if len(p.strip()) > 3]
    return lines[:8]

_CRITIC_ERROR_KWS = [
    r"\bincorrect\b", r"\bwrong\b", r"\berror\b",
    r"\bmistake\b", r"\binvalid\b", r"\bmiscalculation\b",
    r"\bdoesn't make sense\b", r"\binconsistent\b", r"\bcontradiction\b",
]

_CRITIC_VAGUE_KWS = [r"\bplausible\b", r"\buncertain\b", r"\bnot sure\b"]

_NEGATION_PATTERN = re.compile(
    r"\b(not|no|isn'?t|aren'?t|doesn'?t|never|without)\b",
    re.IGNORECASE,
)


def _has_positive_error_signal(text: str) -> bool:
    """Return True only if an error keyword appears WITHOUT a preceding negation."""
    for kw_pattern in _CRITIC_ERROR_KWS:
        for m in re.finditer(kw_pattern, text, re.IGNORECASE):
            prefix = text[max(0, m.start() - 40): m.start()]
            if _NEGATION_PATTERN.search(prefix):
                continue  # negated → skip
            return True
    return False


def _repair_truncated_json(raw: str) -> str:
    raw = raw.strip()
    if raw.endswith(","):
        raw = raw[:-1]
    raw += "]" * (raw.count("[") - raw.count("]"))
    raw += "}" * (raw.count("{") - raw.count("}"))
    return raw


def _extract_action_fallback(raw: str, role: str) -> str:
    m = re.search(r'"action"\s*:\s*"?([^",}]*)', raw)
    val = m.group(1).strip() if m else ""

    if role == "solver_gsm":
        mm = re.search(r"-?\d+\.?\d*", val) or re.search(r"-?\d+\.?\d*", raw)
        return mm.group(0) if mm else "0"

    if role == "solver_choice":
        mm = re.search(r"[A-Ea-e]", val)
        return mm.group(0).upper() if mm else "A"

    if role == "solver_letter":
        from tasks.choice import extract_letter
        return extract_letter(val or raw)

    low = val.lower()
    if role in ("solver", "judge"):
        if "yes" in low or "no" in low:
            return "yes" if "yes" in low else "no"
        return "yes" if re.search(r"\byes\b", raw, re.IGNORECASE) else "no"

    if low:
        return "disagree" if "disagree" in low else "agree"
    return "disagree" if "disagree" in raw.lower() else "agree"


_FACTOID_SIGNALS = re.compile(
    r"\b(who|when|where|which year|how many|how much|named after|"
    r"birth|born|died|founded|invented|created|wrote|first|last|"
    r"oldest|youngest|tallest|largest|smallest|capital|president|"
    r"king|queen|prime minister|ceo|champion|record|medal)\b",
    re.IGNORECASE,
)

_REASONING_SIGNALS = re.compile(
    r"\b(could|would|should|can|is it possible|logically|therefore|"
    r"if .* then|given that|assuming)\b",
    re.IGNORECASE,
)


def classify_question(question: str) -> str:
    """
    Returns 'FACTOID', 'REASONING', or 'HYBRID'.
    Cheap heuristic — no LLM call needed.
    """
    q = question.lower()
    factoid_score = len(_FACTOID_SIGNALS.findall(q))
    reasoning_score = len(_REASONING_SIGNALS.findall(q))

    if factoid_score >= 2 and reasoning_score == 0:
        return "FACTOID"
    if reasoning_score >= 2 and factoid_score == 0:
        return "REASONING"
    if factoid_score == 0 and reasoning_score == 0:
        return "HYBRID"
    return "HYBRID"


def has_specific_fact(msg) -> bool:
    """
    True if reasoning contains a specific number, year, or named entity —
    indicating the agent is recalling rather than guessing.
    """
    text = msg.get_reasoning_text()
    has_year   = bool(re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text))
    has_qty    = bool(re.search(r"\b\d+\s*(km|kg|miles|meters|feet|pounds|years|days|hours|dollars|rupees|%)\b", text, re.IGNORECASE))
    has_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text))
    return has_year or has_qty or has_entity


_UNCERTAINTY_PHRASES = [
    "not certain", "not sure", "i believe", "i think",
    "may be", "might be", "unclear", "not confident",
    "i am unsure", "possibly", "probably", "i'm not",
]

def is_uncertain(msg, threshold: float = 0.65) -> bool:
    text = msg.get_reasoning_text().lower()
    phrase_hit = any(p in text for p in _UNCERTAINTY_PHRASES)
    return phrase_hit or msg.confidence < threshold

_NON_ANSWER_MARKERS = (
    "ready to help", "what is the problem", "what is the math problem",
    "provide the problem", "please provide", "you'd like me to solve",
    "don't see a", "do not see a", "no problem provided", "notice that there is no",
    "happy to help", "what is the math problem you",
)


def looks_like_non_answer(msg) -> bool:
    text = (msg.get_reasoning_text() + " " + (msg.content or "")).lower()
    return any(mk in text for mk in _NON_ANSWER_MARKERS)


def build_compact_prompt(question: str, answer_format: str) -> str:
    if answer_format == "number":
        fmt = ('{"reasoning":["step 1: ...","step 2: ..."],'
               '"action":"<final number only>","confidence":0.9,"content":"one sentence"}')
    elif answer_format == "yes or no":
        fmt = ('{"reasoning":["..."],"action":"yes or no",'
               '"confidence":0.9,"content":"one sentence"}')
    else:
        fmt = ('{"reasoning":["..."],"action":"<letter of one listed option>",'
               '"confidence":0.9,"content":"one sentence"}')
    return (
        "Solve the problem below. Respond with ONLY one JSON object and nothing else. "
        "Do not ask for the problem — it is given here.\n\n"
        f"Problem: {question}\n\n"
        f"Required JSON format:\n{fmt}"
    )

def parse_agent_response(
    raw: str,
    role: str,
    round_id: int,
    call_llm_fn=None,
    retry_prompt: str = None,
    max_retries: int = 2,
    question: str = None,
) -> StructuredMessage:

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    start = raw.find("{")

    data = None
    if start != -1:
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            try:
                data = json.loads(_repair_truncated_json(raw[start:]))
            except json.JSONDecodeError:
                pass

    is_parse_error = data is None
    if data is None:
        data = {
            "intent": "",
            "reasoning": [],
            "action": _extract_action_fallback(raw, role),
            "confidence": 0.5,
            "content": "",
        }

    action = str(data.get("action", "")).strip().lower()

    if role == "critic":
        reasoning_text = " ".join(str(r) for r in data.get("reasoning", [])).lower()

        if _has_positive_error_signal(reasoning_text):
            action = "disagree"
        else:
            action = "disagree" if "disagree" in action else "agree"

    elif role == "solver_gsm":
        m2 = re.search(r"-?\d+\.?\d*", action)
        action = m2.group(0) if m2 else "0"

    elif role == "solver_choice":
        m2 = re.search(r"\b([A-Ea-e])\b", action)
        if m2:
            action = m2.group(1).upper()
        else:
            content_field = str(data.get("content", ""))
            m3 = re.search(r"\b([A-Ea-e])\b", content_field)
            if m3:
                action = m3.group(1).upper()
            else:
                reasoning_text = " ".join(str(r) for r in data.get("reasoning", []))
                all_matches = re.findall(r"\b([A-Ea-e])\b", reasoning_text)
                action = all_matches[-1].upper() if all_matches else "A"

    elif role == "solver_letter":
        # option letters A-R (BBH has up to 18 options); fall back to content, then reasoning
        from tasks.choice import LETTERS, extract_letter
        for source in (action, str(data.get("content", "")),
                       " ".join(str(r) for r in data.get("reasoning", []))):
            letter = extract_letter(source)
            if len(letter) == 1 and letter in LETTERS:
                action = letter
                break
        else:
            action = "A"

    elif role in ("solver"):
        if "yes" in action:
            action = "yes"
        elif "no" in action:
            action = "no"
        else:
            action = "no"

    confidence = 0.5
    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        pass
    if is_parse_error:
        confidence = min(confidence, 0.5)


    reasoning = _normalize_reasoning(data.get("reasoning", []))

    if not reasoning:
        for alt_key in ("steps", "thought", "thoughts", "explanation",
                        "rationale", "work", "solution", "analysis"):
            reasoning = _normalize_reasoning(data.get(alt_key))
            if reasoning:
                break

    if not reasoning:
        salvaged = _salvage_reasoning_from_raw(raw)
        if salvaged:
            reasoning = salvaged
            logger.warning(
                "[parse:%s] reasoning missing in JSON -> salvaged %d line(s) from raw",
                role, len(salvaged),
            )

    if not reasoning:
        if call_llm_fn and retry_prompt:
            raw_retry, _ = call_llm_fn(retry_prompt)
            return parse_agent_response(
                raw_retry, role, round_id,
                call_llm_fn=None, retry_prompt=None,
            )
        logger.warning("[parse:%s] EMPTY reasoning. raw head=%r", role, raw[:300])
        reasoning = ["No reasoning provided", "Fallback answer only", "Confidence reduced"]
        confidence = min(confidence, 0.30)

    msg = StructuredMessage(
        role=role,
        intent=data.get("intent", ""),
        reasoning=reasoning,
        action=action,
        confidence=confidence,
        content=str(data.get("content", "")),
        round_id=round_id,
    )
    msg.independent_answer = None
    return msg