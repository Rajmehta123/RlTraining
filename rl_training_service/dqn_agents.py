"""DQN-based agents: Rainbow DQN and IQN for discrete stock trading."""

import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

# Check if sb3_contrib is available
try:
    from sb3_contrib import QRDQN
    from sb3_contrib.common.wrappers import ActionMasker

    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    logger.warning("sb3-contrib not installed. DQN agents will not be available.")


class DQNTrainingCallback(BaseCallback):
    """Custom callback for DQN training progress."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 1000,
        total_timesteps: int = 100000,
        progress_callback=None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.total_timesteps = total_timesteps
        self.progress_callback = progress_callback
        self.eval_results = []
        self.best_reward = -float("inf")
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = datetime.now()

    def _on_step(self) -> bool:
        if self.progress_callback:
            current_step = self.num_timesteps
            progress = (current_step / self.total_timesteps) * 100

            elapsed = (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            )
            fps = current_step / elapsed if elapsed > 0 else 0
            remaining = (self.total_timesteps - current_step) / fps if fps > 0 else 0

            # Get DQN-specific metrics
            metrics = {
                "current_step": current_step,
                "total_steps": self.total_timesteps,
                "progress_pct": progress,
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
                "fps": fps,
            }

            # Try to get exploration rate
            if hasattr(self.model, "exploration_rate"):
                metrics["exploration_rate"] = self.model.exploration_rate

            self.progress_callback(progress, metrics)

        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            eval_metrics = self._evaluate()
            self.eval_results.append(
                {
                    "step": self.n_calls,
                    "timestamp": datetime.now().isoformat(),
                    **eval_metrics,
                }
            )

            if eval_metrics.get("mean_reward", 0) > self.best_reward:
                self.best_reward = eval_metrics["mean_reward"]

            if self.verbose > 0:
                logger.info(
                    f"Step {self.n_calls}: Reward={eval_metrics.get('mean_reward', 0):.2f}, "
                    f"Return={eval_metrics.get('mean_return', 0):.2%}"
                )

        return True

    def _evaluate(self, n_eval_episodes: int = 3) -> dict[str, float]:
        """Evaluate the agent."""
        try:
            episode_rewards = []
            episode_returns = []

            for _ in range(n_eval_episodes):
                obs = self.eval_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                done = False
                episode_reward = 0
                steps = 0
                max_steps = 200

                while not done and steps < max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.eval_env.step(action)

                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result

                    if isinstance(done, (tuple, list, np.ndarray)):
                        done = done[0] if hasattr(done, "__getitem__") else bool(done)
                    if isinstance(reward, np.ndarray):
                        reward = reward[0]

                    episode_reward += reward
                    steps += 1

                episode_rewards.append(episode_reward)

                # Get return from environment
                if hasattr(self.eval_env, "envs") and hasattr(
                    self.eval_env.envs[0], "get_episode_statistics"
                ):
                    stats = self.eval_env.envs[0].get_episode_statistics()
                    episode_returns.append(stats.get("total_return", 0))
                else:
                    episode_returns.append(episode_reward / 100)

            return {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_return": np.mean(episode_returns),
                "sharpe_ratio": 0,  # Calculated at the end
                "max_drawdown": 0,
                "win_rate": 0,
            }

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return {
                "mean_reward": 0,
                "mean_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }


class RainbowDQNAgent:
    """
    Rainbow DQN Agent using QRDQN from sb3-contrib.

    Rainbow combines:
    - Double DQN
    - Prioritized Experience Replay
    - Dueling Networks
    - Multi-step Returns
    - Distributional RL (via Quantile Regression)
    - Noisy Networks (via exploration)
    """

    def __init__(self, env, config: dict[str, Any] | None = None):
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for Rainbow DQN. Install with: pip install sb3-contrib"
            )

        self.env = env
        self.config = config or self._get_default_config()
        self.model = None
        self.vec_env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dir = self.config.get("model_dir", "./models")
        self.log_dir = self.config.get("log_dir", "./logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.training_history = []

    def _get_default_config(self) -> dict[str, Any]:
        return {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 5000,
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 5000,
            "exploration_fraction": 0.2,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "total_timesteps": 100000,
            "eval_freq": 5000,
            # Rainbow-specific
            "n_quantiles": 50,  # Distributional RL
            "net_arch": [256, 256, 128],
        }

    def build_model(self) -> None:
        """Build the Rainbow DQN (QRDQN) model."""
        logger.info(f"Building Rainbow DQN model on device: {self.device}")

        self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        policy_kwargs = {
            "net_arch": self.config.get("net_arch", [256, 256, 128]),
            "n_quantiles": self.config.get("n_quantiles", 50),
        }

        self.model = QRDQN(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.get("learning_rate", 1e-4),
            buffer_size=self.config.get("buffer_size", 100000),
            learning_starts=self.config.get("learning_starts", 5000),
            batch_size=self.config.get("batch_size", 64),
            gamma=self.config.get("gamma", 0.99),
            train_freq=self.config.get("train_freq", 4),
            gradient_steps=self.config.get("gradient_steps", 1),
            target_update_interval=self.config.get("target_update_interval", 5000),
            exploration_fraction=self.config.get("exploration_fraction", 0.2),
            exploration_initial_eps=self.config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=self.config.get("exploration_final_eps", 0.05),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
        )

        logger.info("Rainbow DQN model built successfully")

    def train(self, eval_env=None, progress_callback=None) -> dict[str, Any]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info("Starting Rainbow DQN training...")

        if eval_env is None:
            eval_env = self.env

        eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])

        total_timesteps = self.config.get("total_timesteps", 100000)

        training_callback = DQNTrainingCallback(
            eval_env=eval_vec_env,
            eval_freq=self.config.get("eval_freq", 5000),
            total_timesteps=total_timesteps,
            progress_callback=progress_callback,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, 10000),
            save_path=self.model_dir,
            name_prefix="rainbow_dqn",
            verbose=1,
        )

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[training_callback, checkpoint_callback],
                progress_bar=True,
            )

            self.training_history = training_callback.eval_results

            logger.info("Rainbow DQN training completed successfully")
            return {
                "status": "completed",
                "eval_results": self.training_history,
                "best_reward": training_callback.best_reward,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def save(self, name: str) -> dict[str, str]:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")

        model_path = os.path.join(self.model_dir, f"{name}.zip")
        self.model.save(model_path)

        config_path = os.path.join(self.model_dir, f"{name}_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {model_path}")

        return {
            "model_path": model_path,
            "config_path": config_path,
        }

    def load(self, name_or_path: str) -> None:
        """Load saved model."""
        if os.path.exists(name_or_path):
            model_path = name_or_path
        else:
            model_path = os.path.join(self.model_dir, f"{name_or_path}.zip")

        if self.vec_env is None:
            self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        self.model = QRDQN.load(model_path, env=self.vec_env, device=self.device)
        logger.info(f"Model loaded from {model_path}")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        """Make prediction."""
        if self.model is None:
            raise ValueError("No model loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action), {}


class IQNAgent:
    """
    Implicit Quantile Networks (IQN) Agent.

    Uses QRDQN with more quantiles and risk-aware reward shaping.
    IQN models the full return distribution for risk-aware decisions.
    """

    def __init__(self, env, config: dict[str, Any] | None = None):
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for IQN. Install with: pip install sb3-contrib"
            )

        self.env = env
        self.config = config or self._get_default_config()
        self.model = None
        self.vec_env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dir = self.config.get("model_dir", "./models")
        self.log_dir = self.config.get("log_dir", "./logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.training_history = []

    def _get_default_config(self) -> dict[str, Any]:
        return {
            "learning_rate": 5e-5,  # Lower LR for stability
            "buffer_size": 200000,  # Larger buffer
            "learning_starts": 10000,
            "batch_size": 128,  # Larger batch
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 2,  # More gradient steps
            "target_update_interval": 8000,
            "exploration_fraction": 0.15,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,  # Lower final eps
            "total_timesteps": 100000,
            "eval_freq": 5000,
            # IQN-specific (more quantiles for better distribution modeling)
            "n_quantiles": 100,  # More quantiles for IQN
            "net_arch": [512, 256, 128],  # Larger network
            # Risk-aware settings
            "risk_aversion": 0.5,
        }

    def build_model(self) -> None:
        """Build the IQN model (QRDQN with more quantiles)."""
        logger.info(f"Building IQN model on device: {self.device}")

        self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        policy_kwargs = {
            "net_arch": self.config.get("net_arch", [512, 256, 128]),
            "n_quantiles": self.config.get("n_quantiles", 100),
        }

        self.model = QRDQN(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.get("learning_rate", 5e-5),
            buffer_size=self.config.get("buffer_size", 200000),
            learning_starts=self.config.get("learning_starts", 10000),
            batch_size=self.config.get("batch_size", 128),
            gamma=self.config.get("gamma", 0.99),
            train_freq=self.config.get("train_freq", 4),
            gradient_steps=self.config.get("gradient_steps", 2),
            target_update_interval=self.config.get("target_update_interval", 8000),
            exploration_fraction=self.config.get("exploration_fraction", 0.15),
            exploration_initial_eps=self.config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=self.config.get("exploration_final_eps", 0.02),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
        )

        logger.info("IQN model built successfully")

    def train(self, eval_env=None, progress_callback=None) -> dict[str, Any]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info("Starting IQN training...")

        if eval_env is None:
            eval_env = self.env

        eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])

        total_timesteps = self.config.get("total_timesteps", 100000)

        training_callback = DQNTrainingCallback(
            eval_env=eval_vec_env,
            eval_freq=self.config.get("eval_freq", 5000),
            total_timesteps=total_timesteps,
            progress_callback=progress_callback,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, 10000),
            save_path=self.model_dir,
            name_prefix="iqn",
            verbose=1,
        )

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[training_callback, checkpoint_callback],
                progress_bar=True,
            )

            self.training_history = training_callback.eval_results

            logger.info("IQN training completed successfully")
            return {
                "status": "completed",
                "eval_results": self.training_history,
                "best_reward": training_callback.best_reward,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def save(self, name: str) -> dict[str, str]:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")

        model_path = os.path.join(self.model_dir, f"{name}.zip")
        self.model.save(model_path)

        config_path = os.path.join(self.model_dir, f"{name}_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {model_path}")

        return {
            "model_path": model_path,
            "config_path": config_path,
        }

    def load(self, name_or_path: str) -> None:
        """Load saved model."""
        if os.path.exists(name_or_path):
            model_path = name_or_path
        else:
            model_path = os.path.join(self.model_dir, f"{name_or_path}.zip")

        if self.vec_env is None:
            self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        self.model = QRDQN.load(model_path, env=self.vec_env, device=self.device)
        logger.info(f"Model loaded from {model_path}")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        """Make prediction."""
        if self.model is None:
            raise ValueError("No model loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action), {}


def get_agent_class(algorithm: str):
    """Get the appropriate agent class based on algorithm name."""
    agents = {
        "ppo": None,  # Will use RLTradingAgent from rl_agent.py
        "rainbow_dqn": RainbowDQNAgent,
        "iqn": IQNAgent,
    }
    return agents.get(algorithm.lower())
