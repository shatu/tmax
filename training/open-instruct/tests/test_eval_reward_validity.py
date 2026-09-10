"""Run the complete accumulation/evaluation functions without GPU imports."""

import __future__

import ast
import unittest
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np

ROOT = Path(__file__).parents[1] / "open_instruct"


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), filename, "exec", __future__.annotations.compiler_flag),
        namespace,
    )


class EvalRewardValidityTest(unittest.TestCase):
    def run_eval(self, groups):
        queue = Queue()
        for index, (scores, valid) in enumerate(groups):
            n = len(scores)
            queue.put(
                SimpleNamespace(
                    index=index,
                    prompt_id=str(index),
                    model_step=0,
                    model_steps=None,
                    start_time=0,
                    responses=[[1]] * n,
                    masks=[[1]] * n,
                    logprobs=[[0.0]] * n,
                    finish_reasons=["stop"] * n,
                    reward_scores=scores,
                    reward_metrics={"raw": 0.5},
                    token_statistics=SimpleNamespace(num_prompt_tokens=n, num_response_tokens=n, generation_time=1),
                    request_info=SimpleNamespace(
                        rollout_states=[{"info": {} if good else {"invalid_reward": True}} for good in valid],
                        **{
                            key: [0] * n
                            for key in (
                                "num_calls",
                                "timeouts",
                                "tool_errors",
                                "tool_outputs",
                                "tool_runtimes",
                                "tool_calleds",
                                "tool_call_stats",
                            )
                        },
                    ),
                )
            )
        metrics = Mock()
        dataframe = Mock(return_value=MagicMock())
        pass_at_k = Mock(return_value={"eval/pass@1": 0.5})
        namespace = dict(
            np=np,
            cast=lambda _, value: value,
            HFDataLoader=object,
            ray_queue=SimpleNamespace(Queue=Queue),
            Empty=Empty,
            logger=Mock(),
            logging=Mock(),
            tqdm=lambda **kwargs: Mock(),
            data_types=SimpleNamespace(
                ShutdownSentinel=type("ShutdownSentinel", (), {}),
                FatalGenerationError=type("FatalGenerationError", (), {}),
                TokenStatistics=SimpleNamespace,
                RequestInfo=SimpleNamespace,
                GenerationResult=SimpleNamespace,
            ),
            Batch=SimpleNamespace,
            BatchStatistics=SimpleNamespace,
            repeat_each=lambda values, n: [value for value in values for _ in range(n)],
            combine_reward_metrics=lambda values: {"raw": 0.5},
            INPUT_IDS_PROMPT_KEY="query",
            GROUND_TRUTHS_KEY="truth",
            VERIFIER_SOURCE_KEY="source",
            RAW_PROMPT_KEY="prompt",
            TOOLS_COLUMN_KEY="tools",
            grpo_utils=SimpleNamespace(compute_pass_at_k_metrics=pass_at_k),
            pd=SimpleNamespace(DataFrame=dataframe),
            print_rich_single_line_metrics=metrics,
            print_rich_table=Mock(),
        )
        load_functions("data_loader.py", {"accumulate_inference_batches", "rollout_reward_is_valid"}, namespace)
        namespace["data_loader_lib"] = SimpleNamespace(rollout_reward_is_valid=namespace["rollout_reward_is_valid"])
        load_functions("grpo_fast.py", {"maybe_evaluate"}, namespace)
        dataset = [{"query": [9], "truth": "", "source": "env", "prompt": "task"} for _ in groups]
        result = namespace["maybe_evaluate"](
            args=SimpleNamespace(num_training_steps=10, with_tracking=False),
            training_step=1,
            evaluation_inference_results_Q=queue,
            tokenizer=SimpleNamespace(batch_decode=lambda rows, **kwargs: ["response"] * len(rows), pad_token="<pad>"),
            episode=0,
            eval_dataset=dataset,
            eval_generation_config=SimpleNamespace(n=len(groups[0][0])),
            model_dims=None,
            base_env_config=None,
            max_possible_score=1,
        )
        self.assertTrue(result)
        self.assertTrue(queue.empty())
        return metrics.call_args.args[0], dataframe.call_args.args[0], pass_at_k

    def test_single_valid_sibling_consumes_finite_eval_queue(self):
        metrics, table, pass_at_k = self.run_eval([([0.75, 0], [True, False]), ([0, 0], [False, False])])
        self.assertEqual(metrics["eval/scores"], 0.75)
        self.assertEqual(metrics["eval/scored_rollouts"], 1)
        self.assertEqual(metrics["eval/unscored_rollouts"], 3)
        self.assertEqual(table["scores"], [0.75, None, None, None])
        self.assertNotIn("eval/raw", metrics)
        pass_at_k.assert_not_called()

    def test_all_invalid_omits_reward_mean(self):
        metrics, table, pass_at_k = self.run_eval([([0, 0], [False, False])])
        self.assertNotIn("eval/scores", metrics)
        self.assertEqual(metrics["eval/scored_rollouts"], 0)
        self.assertEqual(table["scores"], [None, None])
        pass_at_k.assert_not_called()

    def test_all_valid_preserves_mean_and_pass_at_k(self):
        metrics, table, pass_at_k = self.run_eval([([0.5, 1], [True, True])])
        self.assertEqual(metrics["eval/scores"], 0.75)
        self.assertEqual(metrics["eval/raw"], 0.5)
        self.assertEqual(table["scores"], [0.5, 1])
        pass_at_k.assert_called_once()
