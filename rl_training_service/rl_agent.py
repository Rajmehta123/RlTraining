"""RL Agent with PPO and custom attention network."""

import json
import logging
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, embed_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        self._initialize_weights()

    def _initialize_weights(self):
        for proj in [self.query, self.key, self.value]:
            nn.init.kaiming_normal_(proj.weight, mode="fan_in")
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.shape}")

        batch_size, seq_len, _ = x.size()
        residual = x
        x = self.layer_norm(x)

        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(
            attn_scores
        )
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)
        output = output + residual

        return output


class TradingNetworkV2(BaseFeaturesExtractor):
    """Custom network with attention for trading."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_market_features: int = 50,
        hidden_dims: list = None,
        use_attention: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.n_market_features = n_market_features
        self.n_portfolio_features = observation_space.shape[0] - n_market_features
        self.use_attention = use_attention

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Market feature processor
        market_layers = []
        prev_dim = n_market_features
        for dim in hidden_dims:
            market_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                ]
            )
            prev_dim = dim
        self.market_processor = nn.Sequential(*market_layers)

        # Portfolio feature processor
        portfolio_layers = []
        prev_dim = self.n_portfolio_features
        for dim in [d // 2 for d in hidden_dims]:
            portfolio_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                ]
            )
            prev_dim = dim
        self.portfolio_processor = nn.Sequential(*portfolio_layers)

        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_dims[-1])

        # Feature fusion
        fusion_dim = hidden_dims[-1] + (hidden_dims[-1] // 2)
        fusion_layers = [
            nn.Linear(fusion_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        ]
        self.feature_fusion = nn.Sequential(*fusion_layers)

        # Residual projection
        self.residual_proj = None
        if n_market_features + self.n_portfolio_features != features_dim:
            self.residual_proj = nn.Linear(
                n_market_features + self.n_portfolio_features, features_dim, bias=False
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)

        batch_size = observations.size(0)

        market_features = observations[:, : self.n_market_features]
        portfolio_features = observations[:, -self.n_portfolio_features :]

        market_encoded = self.market_processor(market_features)

        if self.use_attention and batch_size > 1:
            market_encoded = market_encoded.unsqueeze(1)
            market_encoded = self.attention(market_encoded)
            market_encoded = market_encoded.squeeze(1)

        portfolio_encoded = self.portfolio_processor(portfolio_features)

        combined = torch.cat([market_encoded, portfolio_encoded], dim=1)
        features = self.feature_fusion(combined)

        if self.residual_proj is not None:
            residual = self.residual_proj(observations)
            if residual.shape == features.shape:
                features = features + 0.1 * residual

        return features


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress."""

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
        self.best_sharpe = -float("inf")
        self.start_time = None
        self.last_log_time = None

    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
        self.last_log_time = self.start_time

    def _on_step(self) -> bool:
        # Always update progress callback with current metrics
        if self.progress_callback:
            current_step = self.num_timesteps
            progress = (current_step / self.total_timesteps) * 100

            # Calculate timing info
            elapsed = (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            )
            fps = current_step / elapsed if elapsed > 0 else 0
            remaining = (self.total_timesteps - current_step) / fps if fps > 0 else 0

            # Get PPO training metrics from logger
            ppo_metrics = {}
            if hasattr(self.model, "logger") and self.model.logger is not None:
                # Get the latest logged values
                if hasattr(self.model.logger, "name_to_value"):
                    for key, value in self.model.logger.name_to_value.items():
                        ppo_metrics[key] = value

            # Build comprehensive metrics
            metrics = {
                "current_step": current_step,
                "total_steps": self.total_timesteps,
                "progress_pct": progress,
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
                "fps": fps,
                "iterations": (
                    current_step // self.model.n_steps
                    if hasattr(self.model, "n_steps")
                    else 0
                ),
                **ppo_metrics,
            }

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

            if eval_metrics.get("sharpe_ratio", 0) > self.best_sharpe:
                self.best_sharpe = eval_metrics["sharpe_ratio"]

            if self.verbose > 0:
                logger.info(
                    f"Step {self.n_calls}: Return={eval_metrics.get('mean_return', 0):.2%}, "
                    f"Sharpe={eval_metrics.get('sharpe_ratio', 0):.2f}"
                )

        return True

    def _evaluate(self) -> dict[str, float]:
        try:
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            episode_return = 0
            done = False
            steps = 0
            max_steps = 200

            while not done and steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                if isinstance(done, np.ndarray):
                    done = done[0]
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                episode_return += reward
                steps += 1

            # Get episode stats if available
            if hasattr(self.eval_env, "envs") and hasattr(
                self.eval_env.envs[0], "get_episode_statistics"
            ):
                stats = self.eval_env.envs[0].get_episode_statistics()
            else:
                stats = {"total_return": episode_return / 100}

            return {
                "mean_return": stats.get("total_return", 0),
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
                "max_drawdown": stats.get("max_drawdown", 0),
                "win_rate": stats.get("win_rate", 0),
            }

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return {
                "mean_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }


class RLTradingAgent:
    """RL Trading Agent with PPO."""

    def __init__(self, env, config: dict[str, Any] | None = None):
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
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 100000,
            "eval_freq": 5000,
            "use_attention": True,
            "hidden_dims": [256, 128],
            "dropout_rate": 0.1,
        }

    def build_model(self) -> None:
        """Build the PPO model."""
        logger.info(f"Building model on device: {self.device}")

        self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.config.get("gamma", 0.99),
        )

        n_obs = self.vec_env.observation_space.shape[0]
        n_market = n_obs - 12  # 12 portfolio features

        policy_kwargs = {
            "net_arch": {
                "pi": self.config.get("hidden_dims", [256, 128]),
                "vf": self.config.get("hidden_dims", [256, 128]),
            },
            "activation_fn": nn.LeakyReLU,
            "features_extractor_class": TradingNetworkV2,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "n_market_features": n_market,
                "use_attention": self.config.get("use_attention", True),
                "dropout_rate": self.config.get("dropout_rate", 0.1),
            },
        }

        # Check if tensorboard is available
        try:
            import tensorboard

            tb_log = os.path.join(self.log_dir, "tensorboard")
        except ImportError:
            logger.warning("Tensorboard not installed, disabling tensorboard logging")
            tb_log = None

        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.get("learning_rate", 3e-4),
            n_steps=self.config.get("n_steps", 2048),
            batch_size=self.config.get("batch_size", 64),
            n_epochs=self.config.get("n_epochs", 10),
            gamma=self.config.get("gamma", 0.99),
            gae_lambda=self.config.get("gae_lambda", 0.95),
            clip_range=self.config.get("clip_range", 0.2),
            ent_coef=self.config.get("ent_coef", 0.01),
            vf_coef=self.config.get("vf_coef", 0.5),
            max_grad_norm=self.config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log,
            device=self.device,
        )

        logger.info("PPO model built successfully")

    def train(self, eval_env=None, progress_callback=None) -> dict[str, Any]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        logger.info("Starting training...")

        if eval_env is None:
            eval_env = self.env

        eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
        if isinstance(self.vec_env, VecNormalize):
            eval_vec_env = VecNormalize(
                eval_vec_env, norm_obs=True, norm_reward=False, training=False
            )
            eval_vec_env.obs_rms = self.vec_env.obs_rms

        # Callbacks
        total_timesteps = self.config.get("total_timesteps", 100000)
        training_callback = TrainingCallback(
            eval_env=eval_vec_env,
            eval_freq=self.config.get("eval_freq", 5000),
            total_timesteps=total_timesteps,
            progress_callback=progress_callback,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(self.config.get("total_timesteps", 100000) // 5, 10000),
            save_path=self.model_dir,
            name_prefix="ppo_trading",
            verbose=1,
        )

        try:
            self.model.learn(
                total_timesteps=self.config.get("total_timesteps", 100000),
                callback=[training_callback, checkpoint_callback],
                progress_bar=True,
            )

            self.training_history = training_callback.eval_results

            logger.info("Training completed successfully")
            return {
                "status": "completed",
                "eval_results": self.training_history,
                "best_sharpe": training_callback.best_sharpe,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def save(self, name: str) -> dict[str, str]:
        """Save model and normalization stats."""
        if self.model is None:
            raise ValueError("No model to save")

        model_path = os.path.join(self.model_dir, f"{name}.zip")
        self.model.save(model_path)

        stats_path = None
        if isinstance(self.vec_env, VecNormalize):
            stats_path = os.path.join(self.model_dir, f"{name}_vecnormalize.pkl")
            self.vec_env.save(stats_path)

        config_path = os.path.join(self.model_dir, f"{name}_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {model_path}")

        return {
            "model_path": model_path,
            "stats_path": stats_path,
            "config_path": config_path,
        }

    def load(self, name_or_path: str) -> None:
        """Load saved model."""
        if os.path.exists(name_or_path):
            model_path = name_or_path
            base_dir = os.path.dirname(name_or_path)
            base_name = os.path.basename(name_or_path).replace(".zip", "")
        else:
            model_path = os.path.join(self.model_dir, f"{name_or_path}.zip")
            base_dir = self.model_dir
            base_name = name_or_path

        config_path = os.path.join(base_dir, f"{base_name}_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

        if self.vec_env is None:
            self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])
            stats_path = os.path.join(base_dir, f"{base_name}_vecnormalize.pkl")
            if os.path.exists(stats_path):
                self.vec_env = VecNormalize.load(stats_path, self.vec_env)

        self.model = PPO.load(model_path, env=self.vec_env, device=self.device)
        logger.info(f"Model loaded from {model_path}")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        """Make prediction."""
        if self.model is None:
            raise ValueError("No model loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action), {}

    def get_model_weights(self) -> bytes:
        """Get serialized model weights."""
        import io

        buffer = io.BytesIO()
        self.model.save(buffer)
        return buffer.getvalue()

    def load_from_weights(self, weights: bytes) -> None:
        """Load model from serialized weights."""
        import io

        buffer = io.BytesIO(weights)
        self.model = PPO.load(buffer, env=self.vec_env, device=self.device)
