"""Exercise the production validity/filter block without the GPU data loader."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class RewardFilteringTest(unittest.TestCase):
    def test_invalid_rollouts_excluded_from_population_and_group_advantages(self):
        path = Path(__file__).parents[1] / "open_instruct/data_loader.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "accumulate_inference_batches"
        )
        loop = next(node for node in function.body if isinstance(node, ast.While))
        start = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "states" for target in node.targets)
        )
        end = next(
            i
            for i in range(start, len(loop.body))
            if isinstance(loop.body[i], ast.If) and ast.unparse(loop.body[i].test) == "any(invalid_rewards)"
        )
        body = loop.body[start : end + 1] + ast.parse("accepted.append(result)").body
        replay = ast.For(
            target=ast.Name(id="result", ctx=ast.Store()),
            iter=ast.Name(id="results", ctx=ast.Load()),
            body=body,
            orelse=[],
        )
        metrics = Mock()
        group = SimpleNamespace(
            prompt_id="p",
            responses=[[1], [2], [3]],
            finish_reasons=["stop"] * 3,
            reward_scores=[0.5, 0, 0],
            request_info=SimpleNamespace(
                rollout_states=[{"info": {}}, {"info": {"invalid_reward": True}}, {"timeout": True, "info": {}}]
            ),
        )
        namespace = {
            "results": [group],
            "accepted": [],
            "population_metrics": metrics,
            "logger": Mock(),
            "replenish_prompts": True,
            "replenish_accepted_prompts": False,
            "enqueue_next_prompt": Mock(),
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[replay], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        metrics.add_group.assert_called_once_with([0.5, 0], ["stop", "stop"], [1, 1])
        self.assertEqual(namespace["accepted"], [])
        namespace["enqueue_next_prompt"].assert_called_once()
