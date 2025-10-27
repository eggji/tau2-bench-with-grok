"""User variant generator for robustness testing."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Tuple

Turn = Dict[str, str]
Dialogue = List[Turn]


PARA_TABLE: List[Tuple[str, str]] = [
    ("split the booking", "separate the reservation"),
    ("split my booking", "separate my reservation"),
    ("reissue", "re-ticket"),
    ("carry-on", "hand luggage"),
    ("baggage", "luggage"),
    ("refund", "money back"),
    ("change my seat", "modify the seat assignment"),
    ("cancel the flight", "void the flight segment"),
    ("economy", "coach"),
    ("business", "biz class"),
]

SMALL_TALK_POOL: List[str] = [
    "By the way, is there a meal included?",
    "Sorry, my kid is crying—one second.",
    "Also, I might travel again next month.",
    "Thanks! Just making sure I won't lose miles.",
]

AMBIGUITY_REPLACERS: List[Tuple[str, str]] = [
    ("JG7FMM", "my record"),
    ("HAT028", "the morning one"),
    ("HAT277", "the later one"),
    ("credit_card", "my card"),
    ("paypal", "my online payment"),
]

GOAL_CHANGE_SNIPPETS: List[str] = [
    "Actually, can we instead keep the original flight if fees are high?",
    "Wait—if that's complicated, maybe just add 1 extra bag.",
]


def paraphrase_text(text: str, p_each: float) -> str:
    out = text
    for src, tgt in PARA_TABLE:
        if random.random() < p_each and src in out:
            out = out.replace(src, tgt)
    return out


def add_ambiguity(text: str, p_each: float) -> str:
    out = text
    for src, tgt in AMBIGUITY_REPLACERS:
        if random.random() < p_each and src in out:
            out = out.replace(src, tgt)
    return out


def inject_smalltalk(dialogue: Dialogue, prob: float) -> Dialogue:
    if random.random() >= prob or len(dialogue) < 1:
        return dialogue
    clone = dialogue[:]
    pos = random.randint(1, max(1, len(clone) - 1))
    clone.insert(pos, {"role": "user", "content": random.choice(SMALL_TALK_POOL)})
    return clone


def insert_goal_change(dialogue: Dialogue, prob: float) -> Dialogue:
    if random.random() >= prob:
        return dialogue
    clone = dialogue[:]
    user_positions = [idx for idx, turn in enumerate(clone) if turn["role"] == "user"]
    if not user_positions:
        return clone
    pos = random.choice(user_positions)
    snippet = random.choice(GOAL_CHANGE_SNIPPETS)
    clone.insert(pos + 1, {"role": "user", "content": snippet})
    return clone


def make_variants(
    seed_dialogue: Dialogue,
    n_variants: int,
    seed: int,
    paraphrase_prob: float = 0.5,
    ambiguity_prob: float = 0.3,
    smalltalk_prob: float = 0.3,
    goal_change_prob: float = 0.2,
) -> List[Dialogue]:
    random.seed(seed)
    variants: List[Dialogue] = []
    for _ in range(n_variants):
        d = copy.deepcopy(seed_dialogue)
        for turn in d:
            if turn["role"] != "user":
                continue
            text = paraphrase_text(turn["content"], paraphrase_prob)
            text = add_ambiguity(text, ambiguity_prob)
            turn["content"] = text
        d = inject_smalltalk(d, smalltalk_prob)
        d = insert_goal_change(d, goal_change_prob)
        variants.append(d)
    return variants


def stability_at_k(success_vector: List[int]) -> float:
    if not success_vector:
        return 0.0
    mean = sum(success_vector) / len(success_vector)
    variance = sum((x - mean) ** 2 for x in success_vector) / len(success_vector)
    return max(0.0, 1.0 - variance)


def evaluate_variants(
    seed_dialogue: Dialogue,
    run_agent_fn: Callable[[Dialogue], Dict[str, Any]],
    n_variants: int = 5,
    seed: int = 42,
    **variant_cfg: Any,
) -> Dict[str, Any]:
    variants = make_variants(seed_dialogue, n_variants, seed, **variant_cfg)
    successes: List[int] = []
    for dialogue in variants:
        result = run_agent_fn(dialogue) or {}
        successes.append(int(result.get("success", 0)))
    success_rate = sum(successes) / max(1, len(successes))
    return {
        "k": n_variants,
        "success_vector": successes,
        "success_rate": success_rate,
        "stability_at_k": stability_at_k(successes),
    }


__all__ = [
    "Dialogue",
    "evaluate_variants",
    "make_variants",
    "stability_at_k",
]
