"""Custom Gymnasium environment for RL-based trading."""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Trading environment for RL agent training."""

    metadata = {"render_modes": ["human"]}

    # Action mapping
    ACTIONS = {
        0: ("SELL_STRONG", -1.0),
        1: ("SELL_WEAK", -0.5),
        2: ("HOLD", 0.0),
        3: ("BUY_WEAK", 0.5),
        4: ("BUY_STRONG", 1.0),
    }

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 1.0,
        lookback_window: int = 1,
        reward_type: str = "balanced",
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_type = reward_type

        # Validate features
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Observation space: market features + portfolio features
        n_market_features = len(feature_columns) * lookback_window
        n_portfolio_features = 12  # Position, returns, drawdown, etc.
        obs_dim = n_market_features + n_portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.n_market_features = n_market_features
        self.n_portfolio_features = n_portfolio_features

        self.reset()

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.cash = self.initial_capital
        self.shares = 0
        self.position = 0.0  # -1 to 1
        self.portfolio_value = self.initial_capital
        self.entry_price = None

        # Tracking
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        self.position_history = []
        self.trades = []
        self.cumulative_return = 0.0
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        self.winning_trades = 0
        self.total_trades = 0

        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Get current and next prices (use iloc for integer indexing)
        current_price = self.df.iloc[self.current_step]["Close"]
        next_price = self.df.iloc[self.current_step + 1]["Close"]

        # Get target position from action
        _, target_position = self.ACTIONS[action]

        # Execute trade
        old_position = self.position
        self._execute_trade(target_position, current_price)

        # Update portfolio value
        self.current_step += 1
        current_price = self.df.iloc[self.current_step]["Close"]

        if self.shares != 0:
            self.portfolio_value = self.cash + self.shares * current_price
        else:
            self.portfolio_value = self.cash

        # Calculate return
        prev_portfolio_value = (
            self.portfolio_history[-1]
            if self.portfolio_history
            else self.initial_capital
        )
        daily_return = (
            self.portfolio_value - prev_portfolio_value
        ) / prev_portfolio_value

        # Update tracking
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(daily_return)
        self.position_history.append(self.position)

        # Update max drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = (
            self.max_portfolio_value - self.portfolio_value
        ) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Calculate cumulative return
        self.cumulative_return = (
            self.portfolio_value - self.initial_capital
        ) / self.initial_capital

        # Calculate reward
        reward = self._calculate_reward(daily_return, old_position, action)

        # Check if done
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, self._get_info()

    def _execute_trade(self, target_position: float, current_price: float) -> None:
        """Execute a trade to reach target position."""
        target_position = np.clip(
            target_position, -self.max_position_size, self.max_position_size
        )

        # Calculate position change
        position_change = target_position - self.position

        if abs(position_change) < 0.02:  # Minimum threshold
            return

        # Calculate trade details
        trade_value = abs(position_change) * self.portfolio_value

        if position_change > 0:  # Buying
            execution_price = current_price * (1 + self.slippage)
            cost = trade_value * self.transaction_cost
            shares_to_buy = (trade_value - cost) / execution_price

            self.shares += shares_to_buy
            self.cash -= trade_value

            if self.position <= 0 and target_position > 0:
                self.entry_price = execution_price

        else:  # Selling
            execution_price = current_price * (1 - self.slippage)
            shares_to_sell = (
                abs(position_change) * self.portfolio_value / execution_price
            )
            shares_to_sell = (
                min(shares_to_sell, self.shares) if self.shares > 0 else shares_to_sell
            )

            gross_proceeds = shares_to_sell * execution_price
            cost = gross_proceeds * self.transaction_cost

            self.shares -= shares_to_sell
            self.cash += gross_proceeds - cost

            # Track trade P&L
            if self.entry_price and target_position <= 0:
                trade_return = (execution_price - self.entry_price) / self.entry_price
                self.total_trades += 1
                if trade_return > 0:
                    self.winning_trades += 1

        self.position = target_position

        # Record trade with date
        trade_date = (
            str(self.df.index[self.current_step])[:10]
            if hasattr(self.df.index, "__getitem__")
            else None
        )
        self.trades.append(
            {
                "step": self.current_step,
                "date": trade_date,
                "action": "BUY" if position_change > 0 else "SELL",
                "price": execution_price,
                "position": target_position,
                "portfolio_value": self.portfolio_value,
                "cost": cost if position_change > 0 else cost,
            }
        )

    def _calculate_reward(
        self, daily_return: float, old_position: float, action: int
    ) -> float:
        """Calculate reward based on reward type."""
        if self.reward_type == "simple":
            return daily_return * 100 * abs(self.position)

        elif self.reward_type == "balanced":
            # Position-weighted return
            reward = daily_return * 100 * abs(self.position)

            # Trading activity bonus
            if action != 2:  # Not HOLD
                reward += 0.001

            # Opportunity cost for missing moves
            price_change = (
                self.df.iloc[self.current_step]["Close"]
                - self.df.iloc[self.current_step - 1]["Close"]
            ) / self.df.iloc[self.current_step - 1]["Close"]

            if abs(price_change) > 0.01 and self.position == 0:
                reward -= abs(price_change) * 10

            # Drawdown penalty
            if self.max_drawdown > 0.1:
                reward -= self.max_drawdown * 5

            return reward

        else:  # Default
            return daily_return * 100

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Market features
        market_features = []
        for i in range(self.lookback_window):
            idx = self.current_step - self.lookback_window + 1 + i
            if idx >= 0 and idx < len(self.df):
                features = self.df.iloc[idx][self.feature_columns].values
                market_features.extend(features)
            else:
                market_features.extend([0.0] * len(self.feature_columns))

        # Portfolio features
        win_rate = self.winning_trades / max(self.total_trades, 1)
        trading_freq = len(self.trades) / max(self.current_step, 1)

        # Calculate rolling Sharpe
        if len(self.returns_history) > 20:
            recent_returns = np.array(self.returns_history[-20:])
            sharpe = (
                np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Volatility regime
        if len(self.returns_history) > 20:
            vol = np.std(self.returns_history[-20:]) * np.sqrt(252)
            vol_regime = np.clip(vol / 0.3, 0, 2) - 1  # Normalize around 30% vol
        else:
            vol_regime = 0.0

        # Trend strength (using returns momentum)
        if len(self.returns_history) > 5:
            trend = np.mean(self.returns_history[-5:]) / (
                np.std(self.returns_history[-5:]) + 1e-8
            )
            trend_strength = np.clip(trend, -2, 2)
        else:
            trend_strength = 0.0

        portfolio_features = [
            self.position,
            self.cumulative_return,
            self.max_drawdown,
            (
                (self.portfolio_value - self.entry_price * self.shares)
                / self.portfolio_value
                if self.entry_price and self.shares > 0
                else 0.0
            ),
            sharpe,
            win_rate,
            trading_freq,
            vol_regime,
            trend_strength,
            self.cash / self.portfolio_value,
            (
                self.shares
                * self.df.iloc[self.current_step]["Close"]
                / self.portfolio_value
                if self.portfolio_value > 0
                else 0.0
            ),
            len(self.trades) / 100.0,  # Normalized trade count
        ]

        obs = np.array(market_features + portfolio_features, dtype=np.float32)

        # Handle NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        obs = np.clip(obs, -10, 10)

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Get current info dict."""
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
        }

    def get_episode_statistics(self) -> dict[str, Any]:
        """Get comprehensive episode statistics."""
        returns = (
            np.array(self.returns_history) if self.returns_history else np.array([0.0])
        )

        # Calculate metrics
        total_return = self.cumulative_return
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        sharpe = (
            (np.mean(returns) * 252) / (volatility + 1e-8) if volatility > 0 else 0.0
        )

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = (
            np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0
            else 0.0
        )
        sortino = (
            (np.mean(returns) * 252) / (downside_vol + 1e-8)
            if downside_vol > 0
            else 0.0
        )

        return {
            "total_return": total_return,
            "annualized_return": (1 + total_return) ** (252 / max(len(returns), 1)) - 1,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "final_value": self.portfolio_value,
            "n_steps": len(returns),
        }

    def render(self, mode: str = "human"):
        """Render current state."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Position: {self.position:.2f}")
            print(f"Return: {self.cumulative_return:.2%}")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
