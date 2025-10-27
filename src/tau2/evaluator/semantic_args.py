"""Semantic argument matching helpers for Tau2 evaluations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Iterable


def _norm_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "")


def _lev_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""

    a, b = _norm_token(a), _norm_token(b)
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    dist = dp[la][lb]
    return 1.0 - dist / max(1, max(la, lb))


def _numeric_close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _parse_date(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None


def _same_id(a: Any, b: Any, sim_th: float) -> bool:
    a_tok, b_tok = _norm_token(a), _norm_token(b)
    if a_tok == b_tok or a_tok in b_tok or b_tok in a_tok:
        return True
    return _lev_sim(a_tok, b_tok) >= sim_th


def _unordered_list_equal(xs: Iterable[Any], ys: Iterable[Any], sim_th: float) -> bool:
    xs_list, ys_list = list(xs), list(ys)
    if len(xs_list) != len(ys_list):
        return False
    used = [False] * len(ys_list)
    for x in xs_list:
        matched = False
        for idx, y in enumerate(ys_list):
            if used[idx]:
                continue
            if semantic_value_equal(x, y, sim_th):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


SYNONYMS: Dict[str, set[str]] = {
    "credit_card": {"cc", "visa_card", "credit"},
    "paypal": {"pypl", "pay_pal"},
    "economy": {"coach", "eco"},
    "business": {"biz", "j"},
    "first": {"f"},
    "usd": {"us_dollar", "dollar"},
}


def _enum_equal(a: Any, b: Any) -> bool:
    a_tok, b_tok = _norm_token(a), _norm_token(b)
    if a_tok == b_tok:
        return True

    def canon(token: str) -> str:
        for root, variants in SYNONYMS.items():
            if token == root or token in variants:
                return root
        return token

    return canon(a_tok) == canon(b_tok)


def semantic_value_equal(v1: Any, v2: Any, sim_th: float) -> bool:
    if isinstance(v1, (int, float)) or isinstance(v2, (int, float)):
        return _numeric_close(v1, v2)

    if isinstance(v1, dict) and isinstance(v2, dict):
        return semantic_args_equal(v1, v2, sim_th)

    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        return _unordered_list_equal(v1, v2, sim_th)

    d1 = _parse_date(str(v1))
    d2 = _parse_date(str(v2))
    if d1 and d2:
        return d1 == d2

    return _lev_sim(str(v1), str(v2)) >= sim_th


def _numeric_validator_factory(tol: float) -> Callable[[Any, Any], bool]:
    return lambda a, b: _numeric_close(a, b, tol)


def _date_equal(a: Any, b: Any) -> bool:
    return _parse_date(str(a)) == _parse_date(str(b))


def _norm_key(key: str) -> str:
    return _norm_token(key)


FIELD_VALIDATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "reservation_id": lambda a, b, th=0.9: _same_id(a, b, th),
    "payment_method_id": lambda a, b, th=0.9: _same_id(a, b, th),
    "order_id": lambda a, b, th=0.9: _same_id(a, b, th),
    "user_id": lambda a, b, th=0.9: _same_id(a, b, th),
    "currency": _enum_equal,
    "cabin": _enum_equal,
    "fare_class": _enum_equal,
    "method": _enum_equal,
    "status": _enum_equal,
    "amount": _numeric_validator_factory(1e-2),
    "price": _numeric_validator_factory(1e-2),
    "tax": _numeric_validator_factory(1e-2),
    "date": _date_equal,
    "flight_date": _date_equal,
}


def semantic_args_equal(
    pred: Dict[str, Any], gold: Dict[str, Any], sim_th: float = 0.86
) -> bool:
    keys = set(pred.keys()) | set(gold.keys())
    for key in keys:
        p_val = pred.get(key)
        g_val = gold.get(key)
        if p_val in (None, "") and g_val in (None, ""):
            continue
        validator = FIELD_VALIDATORS.get(_norm_key(key))
        if validator:
            if not validator(p_val, g_val):
                return False
        else:
            if not semantic_value_equal(p_val, g_val, sim_th):
                return False
    return True


def semantic_args_soft_score(
    pred: Dict[str, Any], gold: Dict[str, Any], sim_th: float = 0.86
) -> float:
    keys = list(set(pred.keys()) | set(gold.keys()))
    if not keys:
        return 1.0
    hits = 0
    for key in keys:
        validator = FIELD_VALIDATORS.get(_norm_key(key))
        if validator:
            ok = validator(pred.get(key), gold.get(key))
        else:
            ok = semantic_value_equal(pred.get(key), gold.get(key), sim_th)
        hits += 1 if ok else 0
    return hits / len(keys)


__all__ = [
    "semantic_args_equal",
    "semantic_args_soft_score",
    "FIELD_VALIDATORS",
]
