"""
src/verification/rules.py
Free (no LLM call) checks on a candidate answer and the reasoning offered for it.

Each check returns a list of issues; an empty list means nothing was found. The verifier logs
these next to the LLM verdict as a separate signal, so the OSC verifier model can learn how
much each one is worth instead of hard-coding a penalty.

  check_arithmetic      "a op b = c" steps whose left side does not evaluate to c
  check_answer_support  numeric answer never produced by the reasoning; letter / yes-no answer
                        that contradicts the conclusion stated in the last step
"""

from __future__ import annotations

import ast
import operator
import re
from typing import List, Optional

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate + - * / ** and parentheses on numbers only; None for anything else."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


_NUM = r"-?\d+(?:\.\d+)?"
# trailing arithmetic expression of a text chunk, e.g. "total = 3 × 20" -> "3 × 20"
_TRAILING_EXPR = re.compile(r"([\d\.\s\+\-\*/\(\)×÷x\^]+)$")
_LEADING_NUM = re.compile(r"^\s*\$?\s*(" + _NUM + r")")


def _clean(expr: str) -> str:
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr)              # 1,000 -> 1000
    expr = re.sub(r"(?<=[\d\)])\s*[x×]\s*(?=[\d\(])", "*", expr)
    return expr.replace("÷", "/").replace("^", "**").strip()


def _tolerance(stated: str) -> float:
    """Half a unit in the last written decimal place ("6.25" -> 0.005), at least 1e-6."""
    decimals = len(stated.split(".")[1]) if "." in stated else 0
    return max(0.5 * 10 ** (-decimals), 1e-6) if decimals else 1e-6


_LEADING_EXPR = re.compile(r"^\s*([\d\.\s\+\-\*/\(\)\^]+)")
_IS_CALC = re.compile(r"\d\s*(\*\*|[\+\-\*/])\s*[\d\(]")


def _right_value(right: str):
    """Value written after '=': a chained expression ("30×0.2083") or a plain number.
    Returns (value, text, tolerance) or None."""
    cleaned = _clean(right)
    m = _LEADING_EXPR.match(cleaned)
    if m and _IS_CALC.search(m.group(1)):
        v = safe_eval(m.group(1).strip())
        if v is not None:  # both sides are calculations: allow 0.5% for rounded intermediates
            return v, m.group(1).strip(), max(abs(v) * 5e-3, 1e-6)
    r = _LEADING_NUM.match(right)
    if not r:
        return None
    return float(r.group(1)), r.group(1), _tolerance(r.group(1))


def check_arithmetic(steps: List[str]) -> List[str]:
    issues = []
    for i, step in enumerate(steps or []):
        s = re.sub(r"(?<=\d),(?=\d{3})", "", step).replace("$", "")
        parts = s.split("=")
        for left, right in zip(parts, parts[1:]):
            m = _TRAILING_EXPR.search(_clean(left.rstrip()))
            rv = _right_value(right)
            if not m or rv is None:
                continue
            expr = m.group(1).strip()
            if not _IS_CALC.search(expr):
                continue  # a bare number, not a calculation
            value = safe_eval(expr)
            if value is None:
                continue
            stated, text, tol = rv
            if abs(value - stated) > max(tol, abs(value) * 1e-6):
                issues.append(f"step {i + 1}: {expr} = {text}, but it evaluates to {value:g}")
    return issues


_YES_END = re.compile(r"\b(?:answer is|therefore|thus|so|hence)[\s,:]*(?:the answer is\s*)?yes\b")
_NO_END = re.compile(r"\b(?:answer is|therefore|thus|so|hence)[\s,:]*(?:the answer is\s*)?no\b")
_LETTER_END = re.compile(r"\b(?:answer is|option|choice|therefore|thus|so)\s*:?\s*\(?([a-r])\)?(?:[\s\.\),]|$)")


def check_answer_support(answer: str, steps: List[str], answer_format: str) -> List[str]:
    steps = [s for s in (steps or []) if s and s.strip()]
    if not steps:
        return ["no reasoning offered"]
    text = " ".join(steps)
    last = steps[-1].lower()
    a = (answer or "").strip().lower()
    if answer_format == "number":
        try:
            target = float(a)
        except ValueError:
            return []
        nums = [float(x) for x in re.findall(_NUM, re.sub(r"(?<=\d),(?=\d{3})", "", text))]
        if nums and not any(abs(n - target) < 1e-6 for n in nums):
            return [f"answer {answer} never appears in the reasoning"]
        return []
    if answer_format == "yes or no":
        if a == "yes" and _NO_END.search(last):
            return ["last step concludes 'no' but the answer is yes"]
        if a == "no" and _YES_END.search(last):
            return ["last step concludes 'yes' but the answer is no"]
        return []
    if answer_format == "letter":
        m = _LETTER_END.search(last)
        if m and m.group(1) != a[:1]:
            return [f"last step points to ({m.group(1).upper()}) but the answer is {answer.upper()}"]
    return []


def rule_check(answer: str, steps: List[str], answer_format: str) -> List[str]:
    issues = check_answer_support(answer, steps, answer_format)
    if answer_format == "number":
        issues += check_arithmetic(steps)
    return issues
