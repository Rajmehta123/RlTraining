"""Configuration for RL Training Service."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # Server settings
    HOST = os.getenv("RL_SERVICE_HOST", "127.0.0.1")
    PORT = int(os.getenv("RL_SERVICE_PORT", "5001"))
    DEBUG = os.getenv("RL_SERVICE_DEBUG", "false").lower() == "true"

    # Supabase settings
    SUPABASE_URL = os.getenv("VITE_SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("VITE_SUPABASE_ANON_KEY", "")

    # FMP API Key
    FMP_API_KEY = os.getenv("VITE_FMP_API_KEY", "")

    # Model storage
    MODEL_DIR = os.getenv("RL_MODEL_DIR", "./models")
    LOG_DIR = os.getenv("RL_LOG_DIR", "./logs")

    # Data defaults
    DEFAULT_YEARS = 2  # Minimum 2 years for technical indicators

    # Training defaults
    DEFAULT_TRAINING_CONFIG = {
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

    # Environment defaults
    DEFAULT_ENV_CONFIG = {
        "initial_capital": 100000.0,
        "transaction_cost": 0.001,
        "slippage": 0.0005,
        "max_position_size": 1.0,
        "lookback_window": 1,
    }
