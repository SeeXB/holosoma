"""CPU-only checks for checkpoint recovery and the original training endpoint."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from holosoma.agents.ppo.ppo import PPO
from scripts.resume_three_group_worker import recovery_command, runtime_config


class RecoveryTest(unittest.TestCase):
    def test_gpu_override_does_not_mutate_frozen_config(self):
        cfg = dict(gpu=1, iterations=30000, num_envs=4096)
        self.assertEqual(runtime_config(cfg, 0), dict(cfg, gpu=0))
        self.assertEqual(cfg["gpu"], 1)
        self.assertEqual(runtime_config(cfg, None), cfg)
        with self.assertRaises(ValueError):
            runtime_config(cfg, -1)

    def test_opt_in_memory_trim_preserves_update_count(self):
        fake = SimpleNamespace(
            _train_mode=lambda: None, env=SimpleNamespace(reset_all=lambda: {}),
            config=SimpleNamespace(init_at_random_ep_len=False, num_learning_iterations=2, save_interval=500),
            current_learning_iteration=3, device="cpu", is_multi_gpu=False,
            is_main_process=False,
            logging_helper=SimpleNamespace(record_collection_time=nullcontext, record_learn_time=nullcontext),
            _rollout_step=Mock(side_effect=lambda obs: obs), _training_step=Mock(return_value={}),
        )
        trim = Mock(return_value=1)
        with patch.dict("os.environ", {"HOLOSOMA_CPU_MALLOC_TRIM_INTERVAL": "2"}), \
                patch("ctypes.CDLL", return_value=SimpleNamespace(malloc_trim=trim)):
            PPO.learn(fake)
        self.assertEqual(trim.call_count, 2)  # initialization and completed iteration 4
        self.assertEqual(fake._training_step.call_count, 2)
        self.assertEqual(fake.current_learning_iteration, 4)

    def test_load_restores_optimizer_and_advances_completed_iteration(self):
        actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        actor_opt = torch.optim.AdamW(actor.parameters(), lr=0.0002)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=0.0003)
        for model, optimizer in ((actor, actor_opt), (critic, critic_opt)):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
        payload = {
            "actor_model_state_dict": actor.state_dict(),
            "critic_model_state_dict": critic.state_dict(),
            "actor_optimizer_state_dict": actor_opt.state_dict(),
            "critic_optimizer_state_dict": critic_opt.state_dict(),
            "iter": 4000,
        }
        restored_actor, restored_critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        restored = SimpleNamespace(
            device="cpu", actor=restored_actor, critic=restored_critic,
            actor_optimizer=torch.optim.AdamW(restored_actor.parameters(), lr=0.001),
            critic_optimizer=torch.optim.AdamW(restored_critic.parameters(), lr=0.001),
            empirical_normalization=False, config=SimpleNamespace(load_optimizer=True),
            _restore_env_state=lambda state: None,
        )
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model_04000.pt"
            torch.save(payload, checkpoint)
            PPO.load(restored, str(checkpoint))
        self.assertEqual(restored.current_learning_iteration, 4001)
        self.assertEqual(restored.actor_learning_rate, 0.0002)
        self.assertEqual(restored.critic_learning_rate, 0.0003)
        torch.testing.assert_close(restored.actor.weight, actor.weight)
        self.assertEqual(next(iter(restored.actor_optimizer.state.values()))["step"].item(), 1)

    def test_resume_keeps_original_final_iteration(self):
        cfg = dict(task="task", tag="tag", seed=42, num_envs=4096, iterations=30000,
                   original_motion="original.npz", b4_motion="b4.npz", object_urdf="object.urdf")
        command = recovery_command(cfg, 2, Path("work"), Path("checkpoint.pt"), 4000, "new-run")
        remaining = command[command.index("--algo.config.num-learning-iterations") + 1]
        self.assertEqual(4001 + remaining - 1, 29999)
        self.assertEqual(command[command.index("--logger.id") + 1], "new-run")
        self.assertEqual(command[command.index("--training.num-envs") + 1], 4096)
        self.assertEqual(cfg["iterations"], 30000)
        with self.assertRaises(ValueError):
            recovery_command(cfg, 2, Path("work"), Path("checkpoint.pt"), 29999, "new-run")


if __name__ == "__main__":
    unittest.main()
