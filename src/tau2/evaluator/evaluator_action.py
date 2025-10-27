from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase
from tau2.evaluator.semantic_args import (
    semantic_args_equal,
    semantic_args_soft_score,
)


class ActionEvaluator(EvaluatorBase):
    """
    Evaluates whether or not the agent communicated the required information.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent communicated the required information.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(full_trajectory, golden_actions)

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.action_match for result in action_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
        )

    @classmethod
    def evaluate_actions(
        cls,
        full_trajectory: list[Message],
        golden_actions: list[Action],
        *,
        use_semantic: bool = False,
        sim_threshold: float = 0.86,
    ) -> list[ActionCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(golden_actions) == 0:
            return []

        predicted_tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
                or isinstance(message, UserMessage)
            ) and message.is_tool_call():
                predicted_tool_calls.extend(message.tool_calls)

        # Check if all the gold actions are in the predicted actions
        action_checks = []
        for gold_action in golden_actions:
            found = False
            semantic_match = None
            semantic_score = None
            for pred_tool_call in predicted_tool_calls:
                if use_semantic:
                    hard, soft = cls._semantic_match(
                        gold_action, pred_tool_call, sim_threshold
                    )
                    if semantic_score is None or soft > semantic_score:
                        semantic_score = soft
                        semantic_match = hard
                    if hard:
                        found = True
                        break
                else:
                    if gold_action.compare_with_tool_call(pred_tool_call):
                        found = True
                        break
            if not found:
                gold_action_reward = 0.0
                gold_action_match = False
                if use_semantic and semantic_match is None:
                    semantic_match = False
                    semantic_score = 0.0
            else:
                gold_action_reward = 1.0
                gold_action_match = True
                if use_semantic and semantic_match is None:
                    semantic_match = True
                    semantic_score = 1.0
            action_checks.append(
                ActionCheck(
                    action=gold_action,
                    action_match=gold_action_match,
                    action_reward=gold_action_reward,
                    semantic_match=semantic_match,
                    semantic_score=semantic_score,
                )
            )
        return action_checks

    @staticmethod
    def _semantic_match(
        action: Action, tool_call: ToolCall, threshold: float
    ) -> tuple[bool, float]:
        if action.name != tool_call.name:
            return False, 0.0
        gold_args = action.arguments or {}
        pred_args = tool_call.arguments or {}
        if action.compare_args is not None:
            gold_args = {k: gold_args.get(k) for k in action.compare_args}
            pred_args = {k: pred_args.get(k) for k in action.compare_args}
        hard = semantic_args_equal(pred_args, gold_args, threshold)
        soft = semantic_args_soft_score(pred_args, gold_args, threshold)
        return hard, soft


class SemanticActionEvaluator(ActionEvaluator):
    """Action evaluator that uses semantic argument matching."""

    SIM_THRESHOLD = 0.86

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(
            full_trajectory,
            golden_actions,
            use_semantic=True,
            sim_threshold=cls.SIM_THRESHOLD,
        )

        all_expectations_met = all(result.action_match for result in action_checks)
        reward = 1.0 if all_expectations_met else 0.0

        info = {"semantic_mode": True}

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
            info=info,
        )
