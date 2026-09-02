import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "calvin_train_fine_tuning_rl.py"
INFER_PATH = ROOT / "scripts" / "calvin_infer_llm_future_bc_rl.py"
ORACLE_PATH = ROOT / "scripts" / "calvin_infer_gt_future_oracle.py"


def source(path):
    return path.read_text(encoding="utf-8")


def class_node(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def method_node(node, name):
    return next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name)


class VisualOnlyTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = source(TRAIN_PATH)
        cls.tree = ast.parse(cls.text)

    def test_privileged_and_alignment_identifiers_are_absent(self):
        forbidden = (
            "priv_state",
            "priv_dim",
            "nearest_progress_index",
            "INDEX_PROGRESS_SCALE",
            "ROBOT_PROGRESS_SCALE",
            "SCENE_PROGRESS_SCALE",
        )
        for identifier in forbidden:
            with self.subTest(identifier=identifier):
                self.assertNotIn(identifier, self.text)

    def test_critic_accepts_only_visual_features_and_actions(self):
        critic = class_node(self.tree, "CriticQ")
        init_args = [arg.arg for arg in method_node(critic, "__init__").args.args]
        forward_args = [arg.arg for arg in method_node(critic, "forward").args.args]
        self.assertEqual(init_args, ["self", "feature_dim", "g_dim", "action_dim"])
        self.assertEqual(
            forward_args,
            ["self", "feature", "g_t", "base_action", "delta_action"],
        )

    def test_replay_buffer_has_no_simulator_state_fields(self):
        replay = class_node(self.tree, "ReplayBuffer")
        init_args = [arg.arg for arg in method_node(replay, "__init__").args.args]
        add_args = [arg.arg for arg in method_node(replay, "add").args.args]
        self.assertEqual(
            init_args,
            ["self", "capacity", "feature_dim", "g_dim", "action_dim", "num_shifts"],
        )
        self.assertEqual(
            add_args,
            ["self", "rafc", "delta_action", "reward", "next_rafc", "done"],
        )

    def test_sparse_reward_uses_squared_l2_sum(self):
        env = class_node(self.tree, "CalvinFineTuningEnv")
        reward = method_node(env, "_reward")
        reward_source = ast.get_source_segment(self.text, reward)
        self.assertIn("np.sum(np.square(delta[:ARM_ACTION_DIM]))", reward_source)
        self.assertNotIn("np.mean", reward_source)
        self.assertIn("SUCCESS_REWARD", reward_source)
        self.assertIn("TIMEOUT_PENALTY", reward_source)

    def test_training_rejects_gt_and_missing_generated_future(self):
        env = class_node(self.tree, "CalvinFineTuningEnv")
        future = ast.get_source_segment(self.text, method_node(env, "_future_for_mode"))
        self.assertIn("GTFuture is an oracle condition", future)
        self.assertIn("training forbids demonstration fallback", future)
        self.assertNotIn("_demo_future", future)

    def test_default_episode_length_is_200(self):
        self.assertIn("MAX_EPISODE_STEPS = 200", self.text)
        self.assertIn('os.environ.get("CALVIN_MAX_EPISODE_STEPS", "200")', self.text)


class VisualOnlyInferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = source(INFER_PATH)
        cls.oracle_text = source(ORACLE_PATH)

    def test_generated_future_has_no_demo_fallback(self):
        self.assertIn("visual-only inference", self.text)
        self.assertIn("forbids demonstration fallback", self.text)
        self.assertNotIn("falling back to demo future", self.text)

    def test_gt_access_requires_explicit_oracle_runner(self):
        self.assertIn("if self.future_mode == \"gt\" and not ALLOW_GT_ORACLE", self.text)
        self.assertIn('os.environ["CALVIN_ALLOW_GT_ORACLE"] = "1"', self.oracle_text)
        self.assertIn('os.environ["CALVIN_FUTURE_MODE"] = "gt"', self.oracle_text)

    def test_default_episode_length_is_200(self):
        self.assertIn("MAX_EPISODE_STEPS = 200", self.text)
        self.assertIn('os.environ.get("CALVIN_MAX_EPISODE_STEPS", "200")', self.text)


if __name__ == "__main__":
    unittest.main()
