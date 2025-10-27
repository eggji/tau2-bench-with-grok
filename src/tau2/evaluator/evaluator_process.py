from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.environment.toolkit import ToolKitBase, ToolType
from tau2.evaluator.evaluator_base import EvaluatorBase


ActionRecord = Dict[str, Any]


class ProcessAwareEvaluator(EvaluatorBase):
    """Process-aware scoring that rewards accurate, well-ordered tool use."""

    ARG_MATCH_THRESHOLD = 0.86

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[..., Any],
        task: Task,
        full_trajectory: list[Message],
        solo_mode: bool = False,
    ) -> RewardInfo:
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
                reward_basis=[RewardType.ACTION],
            )

        expected_actions: Optional[list[Action]] = task.evaluation_criteria.actions
        if not expected_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No expected actions"},
                reward_breakdown={RewardType.ACTION: 1.0},
                reward_basis=[RewardType.ACTION],
            )

        environment = environment_constructor(solo_mode=solo_mode)
        tool_index = cls._build_tool_index(environment)

        predicted = cls._extract_actions(full_trajectory, tool_index, solo_mode)
        expected = cls._convert_expected_actions(expected_actions, tool_index, solo_mode)

        metrics = score_dialogue(
            actions=predicted,
            expected=expected,
            critical_writes=cls._critical_writes(expected),
            arg_match_threshold=cls.ARG_MATCH_THRESHOLD,
        )

        reward = metrics["ProcessScoreNorm"]

        return RewardInfo(
            reward=reward,
            reward_breakdown={RewardType.ACTION: reward},
            reward_basis=[RewardType.ACTION],
            info={"process_metrics": metrics},
        )

    @staticmethod
    def _build_tool_index(environment: Any) -> dict[str, dict[str, bool]]:
        index = {"assistant": {}, "user": {}}

        def _ingest(toolkit: Optional[ToolKitBase], bucket: str) -> None:
            if toolkit is None:
                return
            for name in toolkit.get_tools().keys():
                try:
                    tool_type = toolkit.tool_type(name)
                except AttributeError:
                    tool_type = ToolType.GENERIC
                index[bucket][name] = tool_type == ToolType.WRITE

        _ingest(getattr(environment, "tools", None), "assistant")
        _ingest(getattr(environment, "user_tools", None), "user")
        return index

    @classmethod
    def _extract_actions(
        cls,
        trajectory: list[Message],
        tool_index: dict[str, dict[str, bool]],
        solo_mode: bool,
    ) -> list[ActionRecord]:
        actions: list[ActionRecord] = []
        for turn_idx, message in enumerate(trajectory):
            if not isinstance(message, (AssistantMessage, UserMessage)):
                continue
            if not message.is_tool_call():
                continue
            for call in message.tool_calls or []:
                actions.append(
                    {
                        "name": call.name,
                        "args": call.arguments or {},
                        "write": cls._is_write(call, tool_index, solo_mode),
                        "requestor": call.requestor,
                        "ts": message.turn_idx if message.turn_idx is not None else turn_idx,
                    }
                )
        return actions

    @classmethod
    def _convert_expected_actions(
        cls,
        expected_actions: Sequence[Action],
        tool_index: dict[str, dict[str, bool]],
        solo_mode: bool,
    ) -> list[ActionRecord]:
        converted: list[ActionRecord] = []
        for idx, action in enumerate(expected_actions):
            converted.append(
                {
                    "name": action.name,
                    "args": action.arguments or {},
                    "write": cls._lookup_write_flag(action.name, action.requestor, tool_index, solo_mode),
                    "requestor": action.requestor,
                    "ts": idx,
                }
            )
        return converted

    @staticmethod
    def _critical_writes(expected: Iterable[ActionRecord]) -> list[str]:
        return sorted({action["name"] for action in expected if action.get("write")})

    @classmethod
    def _lookup_write_flag(
        cls,
        tool_name: str,
        requestor: str,
        tool_index: dict[str, dict[str, bool]],
        solo_mode: bool,
    ) -> bool:
        if tool_name in tool_index.get(requestor, {}):
            return tool_index[requestor][tool_name]
        if solo_mode:
            other = "user" if requestor == "assistant" else "assistant"
            if tool_name in tool_index.get(other, {}):
                return tool_index[other][tool_name]
        return False

    @classmethod
    def _is_write(
        cls,
        call: ToolCall,
        tool_index: dict[str, dict[str, bool]],
        solo_mode: bool,
    ) -> bool:
        return cls._lookup_write_flag(call.name, call.requestor, tool_index, solo_mode)


def score_dialogue(
    actions: List[ActionRecord],
    expected: List[ActionRecord],
    dependency_edges: Optional[List[Tuple[str, str]]] = None,
    critical_writes: Optional[List[str]] = None,
    arg_match_threshold: float = 0.86,
) -> Dict[str, float]:
    dependency_edges = dependency_edges or []
    critical_writes = set(critical_writes or [])

    step_hits, order_hits = _align_hits(actions, expected, arg_match_threshold)
    redundancy_penalty = _redundancy_penalty(actions, expected, critical_writes)
    causal_penalty = _causal_penalty(actions, dependency_edges)

    step_reward = float(step_hits)
    order_reward = float(order_hits) * 0.5
    process_score = max(0.0, step_reward + order_reward - redundancy_penalty - causal_penalty)

    denom = max(1.0, len(expected) + 0.5 * len(expected))
    normalized = min(1.0, process_score / denom)

    return {
        "StepReward": step_reward,
        "OrderReward": order_reward,
        "RedundancyPenalty": float(redundancy_penalty),
        "CausalPenalty": float(causal_penalty),
        "ProcessScore": process_score,
        "ProcessScoreNorm": normalized,
    }


def _align_hits(
    actions: List[ActionRecord],
    expected: List[ActionRecord],
    threshold: float,
) -> Tuple[int, int]:
    i = j = 0
    step_hits = order_hits = 0
    while i < len(actions) and j < len(expected):
        if _same_call(actions[i], expected[j], threshold):
            step_hits += 1
            order_hits += 1
            i += 1
            j += 1
        else:
            i += 1
    return step_hits, order_hits


def _redundancy_penalty(
    actions: List[ActionRecord],
    expected: List[ActionRecord],
    critical_writes: set[str],
) -> int:
    need: dict[str, int] = {}
    for exp in expected:
        if exp.get("write"):
            need[exp["name"]] = need.get(exp["name"], 0) + 1

    used: dict[str, int] = {}
    penalty = 0
    for action in actions:
        if not action.get("write"):
            continue
        name = action["name"]
        used[name] = used.get(name, 0) + 1
        over = used[name] > need.get(name, 0)
        not_expected = name not in need
        if over or not_expected:
            penalty += 2 if name in critical_writes else 1
    return penalty


def _causal_penalty(actions: List[ActionRecord], edges: List[Tuple[str, str]]) -> int:
    name_seq = [action["name"] for action in actions]
    violations = 0
    for u, v in edges:
        iu = _first_idx(name_seq, u)
        iv = _first_idx(name_seq, v)
        if iu is not None and iv is not None and iv < iu:
            violations += 1
    return violations


def _same_call(a: ActionRecord, b: ActionRecord, threshold: float) -> bool:
    if a["name"] != b["name"]:
        return False
    return _args_semantic_match(a.get("args", {}), b.get("args", {}), threshold)


def _args_semantic_match(pred: Dict[str, Any], gold: Dict[str, Any], threshold: float) -> bool:
    keys = set(pred.keys()) | set(gold.keys())
    for key in keys:
        pv, gv = str(pred.get(key, "")), str(gold.get(key, ""))
        if pv == gv:
            continue
        if key in {"amount", "price", "fare"}:
            if not _numeric_close(pv, gv, tol=1e-6):
                return False
        else:
            if _edit_sim(_normalize(pv), _normalize(gv)) < threshold:
                return False
    return True


def _numeric_close(a: str, b: str, tol: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "")


def _first_idx(seq: Sequence[str], token: str) -> Optional[int]:
    try:
        return seq.index(token)
    except ValueError:
        return None


def _edit_sim(a: str, b: str) -> float:
    la, lb = len(a), len(b)
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
    return 1.0 - dist / max(1, la, lb)
