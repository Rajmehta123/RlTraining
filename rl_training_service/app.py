"""Flask API server for RL Trading Service."""

import logging
import os
import threading
from datetime import datetime
from typing import Any

import numpy as np
from backtester import Backtester
from data_handler import DataHandler
from flask import Flask, jsonify, request
from flask_cors import CORS
from rl_agent import RLTradingAgent
from trading_environment import TradingEnvironment

from config import Config

# Import DQN agents (optional - requires sb3-contrib)
try:
    from dqn_agents import SB3_CONTRIB_AVAILABLE, IQNAgent, RainbowDQNAgent
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    RainbowDQNAgent = None
    IQNAgent = None
    logger = logging.getLogger(__name__)
    logger.warning(
        "DQN agents not available. Install sb3-contrib for Rainbow DQN and IQN."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# Training state tracking
training_jobs: dict[str, dict[str, Any]] = {}
trained_models: dict[str, Any] = {}


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_training(job_id: str, symbol: str, config: dict[str, Any]):
    """Run training in background thread."""
    try:
        training_jobs[job_id]["status"] = "loading_data"
        training_jobs[job_id]["progress"] = 5

        # Load data
        feature_tiers = config.get("feature_tiers", [1, 2])  # Default: Tier 1 & 2
        use_tiered_features = config.get("use_tiered_features", True)

        data_handler = DataHandler(
            symbol=symbol,
            fmp_api_key=Config.FMP_API_KEY,
            years=config.get("years", 2),
            feature_tiers=feature_tiers,
            use_tiered_features=use_tiered_features,
        )
        data_handler.load_data()
        data_handler.load_fundamentals()
        data_handler.calculate_features()
        data_handler.normalize_features(method="robust")

        training_jobs[job_id]["status"] = "preparing_environment"
        training_jobs[job_id]["progress"] = 15
        training_jobs[job_id]["data_summary"] = data_handler.get_data_summary()

        # Split data
        splits = data_handler.split_data(train_ratio=0.7, val_ratio=0.15)

        # Create environment
        env_config = {**Config.DEFAULT_ENV_CONFIG, **config.get("env_config", {})}

        train_env = TradingEnvironment(
            df=splits["train"],
            feature_columns=data_handler.feature_columns,
            **env_config,
        )

        val_env = TradingEnvironment(
            df=splits["val"], feature_columns=data_handler.feature_columns, **env_config
        )

        training_jobs[job_id]["status"] = "building_model"
        training_jobs[job_id]["progress"] = 20

        # Get algorithm selection
        algorithm = config.get("algorithm", "ppo").lower()
        training_jobs[job_id]["algorithm"] = algorithm

        # Create and train agent based on algorithm
        training_config = {
            **Config.DEFAULT_TRAINING_CONFIG,
            **config.get("training_config", {}),
        }
        training_config["model_dir"] = os.path.join(Config.MODEL_DIR, job_id)
        training_config["log_dir"] = os.path.join(Config.LOG_DIR, job_id)

        if algorithm == "rainbow_dqn":
            if not SB3_CONTRIB_AVAILABLE or RainbowDQNAgent is None:
                raise ImportError(
                    "Rainbow DQN requires sb3-contrib. Install with: pip install sb3-contrib"
                )
            logger.info("Using Rainbow DQN algorithm")
            agent = RainbowDQNAgent(train_env, training_config)
        elif algorithm == "iqn":
            if not SB3_CONTRIB_AVAILABLE or IQNAgent is None:
                raise ImportError(
                    "IQN requires sb3-contrib. Install with: pip install sb3-contrib"
                )
            logger.info("Using IQN algorithm")
            agent = IQNAgent(train_env, training_config)
        else:
            logger.info("Using PPO algorithm")
            agent = RLTradingAgent(train_env, training_config)

        agent.build_model()

        training_jobs[job_id]["status"] = "training"

        def progress_callback(progress: float, metrics: dict):
            training_jobs[job_id]["progress"] = 20 + int(progress * 0.6)
            training_jobs[job_id]["current_metrics"] = metrics

        result = agent.train(eval_env=val_env, progress_callback=progress_callback)

        if result["status"] != "completed":
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = result.get("error", "Unknown error")
            return

        training_jobs[job_id]["status"] = "backtesting"
        training_jobs[job_id]["progress"] = 85

        # Backtest
        test_env = TradingEnvironment(
            df=splits["test"],
            feature_columns=data_handler.feature_columns,
            **env_config,
        )

        backtester = Backtester(
            initial_capital=env_config.get("initial_capital", 100000),
            transaction_cost=env_config.get("transaction_cost", 0.001),
            slippage=env_config.get("slippage", 0.0005),
        )

        agent_metrics = backtester.backtest_agent(agent, test_env, splits["test"])
        bh_metrics = backtester.backtest_buy_hold(splits["test"])
        sma_metrics = backtester.backtest_sma_crossover(splits["test"])

        plots = backtester.generate_plots()
        comparison = backtester.compare_strategies()

        training_jobs[job_id]["status"] = "saving"
        training_jobs[job_id]["progress"] = 95

        # Save model
        model_name = f"{symbol}_{job_id}"
        save_paths = agent.save(model_name)

        # Store model reference
        trained_models[job_id] = {
            "agent": agent,
            "data_handler": data_handler,
            "symbol": symbol,
            "feature_columns": data_handler.feature_columns,
            "env_config": env_config,
            "feature_tiers": feature_tiers,
            "use_tiered_features": use_tiered_features,
        }

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["results"] = {
            "metrics": {
                "agent": agent_metrics,
                "buy_hold": bh_metrics,
                "sma_crossover": sma_metrics,
            },
            "comparison": comparison.to_dict(),
            "plots": plots,
            "equity_curve": backtester.get_equity_curve("agent"),
            "trades": backtester.get_trades("agent"),
            "training_history": result.get("eval_results", []),
            "model_path": save_paths.get("model_path"),
        }

        logger.info(f"Training job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/train", methods=["POST"])
def start_training():
    """Start a new training job."""
    try:
        data = request.json
        symbol = data.get("symbol", "AAPL").upper()
        config = data.get("config", {})
        user_id = data.get("user_id")

        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400

        job_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        training_jobs[job_id] = {
            "job_id": job_id,
            "symbol": symbol,
            "algorithm": config.get("algorithm", "ppo"),
            "user_id": user_id,
            "status": "initializing",
            "progress": 0,
            "started_at": datetime.now().isoformat(),
            "config": config,
        }

        # Start training in background thread
        thread = threading.Thread(target=run_training, args=(job_id, symbol, config))
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "job_id": job_id,
                "status": "started",
                "message": f"Training started for {symbol}",
            }
        )

    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/train/<job_id>/status", methods=["GET"])
def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = training_jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "symbol": job["symbol"],
        "algorithm": job.get("algorithm", "ppo"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }

    if "error" in job:
        response["error"] = job["error"]

    if "current_metrics" in job:
        response["current_metrics"] = convert_numpy_types(job["current_metrics"])

    if "data_summary" in job:
        response["data_summary"] = convert_numpy_types(job["data_summary"])

    return jsonify(convert_numpy_types(response))


@app.route("/api/train/<job_id>/results", methods=["GET"])
def get_training_results(job_id: str):
    """Get training results."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = training_jobs[job_id]

    if job["status"] != "completed":
        return (
            jsonify({"error": "Training not completed", "status": job["status"]}),
            400,
        )

    return jsonify(
        convert_numpy_types({"job_id": job_id, "results": job.get("results", {})})
    )


@app.route("/api/train/<job_id>/stop", methods=["POST"])
def stop_training(job_id: str):
    """Stop a training job (best effort)."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404

    training_jobs[job_id]["status"] = "stopped"

    return jsonify({"message": "Stop signal sent"})


@app.route("/api/predict/<job_id>", methods=["POST"])
def predict(job_id: str):
    """Make predictions using a trained model."""
    if job_id not in trained_models:
        return jsonify({"error": "Model not found"}), 404

    try:
        data = request.json
        symbol = data.get("symbol")

        model_info = trained_models[job_id]
        agent = model_info["agent"]

        # Get current data using same feature configuration as training
        data_handler = DataHandler(
            symbol=symbol or model_info["symbol"],
            fmp_api_key=Config.FMP_API_KEY,
            years=1,
            feature_tiers=model_info.get("feature_tiers", [1, 2]),
            use_tiered_features=model_info.get("use_tiered_features", True),
        )
        data_handler.load_data()
        data_handler.load_fundamentals()  # Load fundamentals if Tier 2+ used
        data_handler.calculate_features()
        data_handler.normalize_features(method="robust")

        # Create environment for current data
        current_env = TradingEnvironment(
            df=data_handler.processed_data.tail(100),
            feature_columns=model_info["feature_columns"],
            **model_info["env_config"],
        )

        # Run simulation
        obs = current_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        actions = []
        positions = []

        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            step_result = current_env.step(action)

            # Handle both old Gym (4 values) and new Gymnasium (5 values) API
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, _, done, info = step_result

            if isinstance(done, (tuple, list)):
                done = done[0]
            if not isinstance(info, dict):
                info = {}

            actions.append(int(action))
            positions.append(info.get("position", 0))

        stats = current_env.get_episode_statistics()

        # Get latest recommendation
        action_names = ["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"]
        latest_action = actions[-1] if actions else 2

        return jsonify(
            convert_numpy_types(
                {
                    "symbol": symbol or model_info["symbol"],
                    "recommendation": action_names[latest_action],
                    "action": latest_action,
                    "position": positions[-1] if positions else 0,
                    "predicted_return": stats.get("total_return", 0),
                    "metrics": stats,
                }
            )
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def list_models():
    """List available trained models."""
    models = []

    for job_id, job in training_jobs.items():
        if job["status"] == "completed":
            models.append(
                {
                    "job_id": job_id,
                    "symbol": job["symbol"],
                    "created_at": job["started_at"],
                    "metrics": job.get("results", {})
                    .get("metrics", {})
                    .get("agent", {}),
                }
            )

    return jsonify(convert_numpy_types({"models": models}))


@app.route("/api/models/<job_id>", methods=["DELETE"])
def delete_model(job_id: str):
    """Delete a trained model."""
    if job_id in training_jobs:
        del training_jobs[job_id]

    if job_id in trained_models:
        del trained_models[job_id]

    return jsonify({"message": "Model deleted"})


if __name__ == "__main__":
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    logger.info(f"Starting RL Training Service on {Config.HOST}:{Config.PORT}")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)
