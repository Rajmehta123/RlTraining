"""Backtesting engine for RL trading strategies."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade record for analysis."""

    timestamp: str
    action: str
    price: float
    size: float
    position_before: float
    position_after: float
    portfolio_value: float
    cost: float
    pnl: float = 0.0


class Backtester:
    """Backtesting engine with comprehensive metrics."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.results = {}

    def backtest_agent(
        self,
        agent,
        test_env,
        test_data: pd.DataFrame,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Backtest trained RL agent."""
        logger.info("Running agent backtest...")

        obs = test_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        actions = []
        positions = []
        portfolio_values = [self.initial_capital]

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)

            step_result = test_env.step(action)

            # Handle both old Gym (4 values) and new Gymnasium (5 values) API
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            if isinstance(done, (tuple, np.ndarray)):
                done = done[0] if hasattr(done, "__getitem__") else bool(done)
            if isinstance(info, tuple):
                info = info[0]
            if not isinstance(info, dict):
                info = {}

            actions.append(action)
            positions.append(info.get("position", 0))
            portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))

        # Get final statistics
        env_stats = (
            test_env.get_episode_statistics()
            if hasattr(test_env, "get_episode_statistics")
            else {}
        )

        # Build results DataFrame
        n_steps = len(actions)
        results_df = pd.DataFrame(
            {
                "date": test_data.index[:n_steps],
                "portfolio_value": portfolio_values[1 : n_steps + 1],
                "position": positions[:n_steps],
                "action": actions[:n_steps],
                "price": test_data["Close"].iloc[:n_steps].values,
            }
        )

        results_df["returns"] = results_df["portfolio_value"].pct_change()
        results_df["cumulative_returns"] = (
            1 + results_df["returns"].fillna(0)
        ).cumprod() - 1

        self.results["agent"] = {
            "data": results_df,
            "env_stats": env_stats,
            "trades": test_env.trades if hasattr(test_env, "trades") else [],
        }

        metrics = self._calculate_metrics(results_df, "RL Agent")
        metrics.update(env_stats)

        logger.info(
            f"Agent backtest complete. Return: {metrics.get('total_return', 0):.2%}"
        )
        return metrics

    def backtest_buy_hold(self, test_data: pd.DataFrame) -> dict[str, Any]:
        """Buy-and-hold benchmark."""
        logger.info("Running buy-and-hold backtest...")

        initial_price = test_data["Close"].iloc[0]
        initial_cost = self.initial_capital * self.transaction_cost
        invested = self.initial_capital - initial_cost

        portfolio_values = invested * (test_data["Close"] / initial_price)

        results_df = pd.DataFrame(
            {
                "date": test_data.index,
                "portfolio_value": portfolio_values,
                "position": 1,
                "price": test_data["Close"],
            }
        )

        results_df["returns"] = results_df["portfolio_value"].pct_change()
        results_df["cumulative_returns"] = (
            1 + results_df["returns"].fillna(0)
        ).cumprod() - 1

        self.results["buy_hold"] = {"data": results_df}

        return self._calculate_metrics(results_df, "Buy & Hold")

    def backtest_sma_crossover(
        self,
        test_data: pd.DataFrame,
        fast_period: int = None,
        slow_period: int = None,
    ) -> dict[str, Any]:
        """SMA crossover strategy with adaptive periods based on data length."""
        n_days = len(test_data)

        # Adapt periods based on available data - ensure at least 50% of data for trading
        if slow_period is None:
            slow_period = min(50, n_days // 3)  # Use 1/3 of data max for warmup
        if fast_period is None:
            fast_period = min(20, slow_period // 2)  # Fast is half of slow

        # Ensure minimum periods
        slow_period = max(slow_period, 10)
        fast_period = max(fast_period, 5)

        logger.info(
            f"Running SMA crossover backtest ({fast_period}/{slow_period}) on {n_days} days..."
        )

        sma_fast = (
            test_data["Close"].rolling(fast_period, min_periods=fast_period).mean()
        )
        sma_slow = (
            test_data["Close"].rolling(slow_period, min_periods=slow_period).mean()
        )

        signals = pd.Series(0, index=test_data.index)
        signals[sma_fast > sma_slow] = 1
        position_changes = signals.diff()

        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        n_trades = 0

        for i, (date, row) in enumerate(test_data.iterrows()):
            if i < slow_period or pd.isna(sma_slow.iloc[i]):
                # During warmup, just hold cash
                portfolio_values.append(cash if shares == 0 else shares * row["Close"])
            elif position_changes.iloc[i] == 1 and shares == 0:
                # Buy signal
                price = row["Close"] * (1 + self.slippage)
                cost = cash * self.transaction_cost
                shares = (cash - cost) / price
                cash = 0
                portfolio_values.append(shares * row["Close"])
                n_trades += 1
            elif position_changes.iloc[i] == -1 and shares > 0:
                # Sell signal
                price = row["Close"] * (1 - self.slippage)
                gross = shares * price
                cost = gross * self.transaction_cost
                cash = gross - cost
                shares = 0
                portfolio_values.append(cash)
                n_trades += 1
            else:
                pv = cash if shares == 0 else shares * row["Close"]
                portfolio_values.append(pv)

        logger.info(f"SMA crossover executed {n_trades} trades")

        results_df = pd.DataFrame(
            {
                "date": test_data.index,
                "portfolio_value": portfolio_values,
                "position": signals,
                "price": test_data["Close"],
            }
        )

        results_df["returns"] = results_df["portfolio_value"].pct_change()
        results_df["cumulative_returns"] = (
            1 + results_df["returns"].fillna(0)
        ).cumprod() - 1

        self.results["sma_crossover"] = {"data": results_df}

        metrics = self._calculate_metrics(results_df, "SMA Crossover")
        metrics["n_trades"] = n_trades
        metrics["fast_period"] = fast_period
        metrics["slow_period"] = slow_period
        return metrics

    def _calculate_metrics(self, results_df: pd.DataFrame, name: str) -> dict[str, Any]:
        """Calculate performance metrics."""
        returns = results_df["returns"].dropna()

        total_return = (
            results_df["portfolio_value"].iloc[-1] - self.initial_capital
        ) / self.initial_capital
        n_days = len(results_df)
        n_years = n_days / 252

        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # Handle edge cases for Sharpe ratio
        if volatility < 0.001:  # Near-zero volatility (no trades or flat portfolio)
            sharpe = 0.0  # Cannot calculate meaningful Sharpe
        else:
            sharpe = (annualized_return - self.risk_free_rate) / volatility

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Sortino
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        if downside_vol < 0.001:
            sortino = 0.0
        else:
            sortino = (annualized_return - self.risk_free_rate) / downside_vol

        # Calmar
        if abs(max_drawdown) < 0.001:
            calmar = 0.0
        else:
            calmar = annualized_return / abs(max_drawdown)

        # Get date range
        start_date = (
            str(results_df["date"].iloc[0])[:10]
            if "date" in results_df.columns
            else None
        )
        end_date = (
            str(results_df["date"].iloc[-1])[:10]
            if "date" in results_df.columns
            else None
        )

        return {
            "strategy": name,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "final_value": results_df["portfolio_value"].iloc[-1],
            "n_days": n_days,
            "n_weeks": round(n_days / 7, 1),
            "n_months": round(n_days / 30, 1),
            "start_date": start_date,
            "end_date": end_date,
        }

    def compare_strategies(self) -> pd.DataFrame:
        """Compare all backtested strategies."""
        comparisons = []
        for key, result in self.results.items():
            if "data" in result:
                metrics = self._calculate_metrics(
                    result["data"], key.replace("_", " ").title()
                )
                comparisons.append(metrics)

        return pd.DataFrame(comparisons).set_index("strategy")

    def generate_plots(self) -> dict[str, str]:
        """Generate backtest visualization plots as base64 images."""
        plots = {}

        # 1. Cumulative Returns
        fig, ax = plt.subplots(figsize=(12, 6))
        for key, result in self.results.items():
            if "data" in result:
                data = result["data"]
                label = key.replace("_", " ").title()
                ax.plot(
                    data["date"],
                    data["cumulative_returns"] * 100,
                    label=label,
                    linewidth=2,
                )

        ax.set_title("Cumulative Returns Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plots["cumulative_returns"] = self._fig_to_base64(fig)
        plt.close(fig)

        # 2. Drawdown Analysis
        fig, ax = plt.subplots(figsize=(12, 4))
        for key, result in self.results.items():
            if "data" in result:
                data = result["data"]
                returns = data["returns"].dropna()
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max

                label = key.replace("_", " ").title()
                ax.fill_between(
                    data["date"][1:], drawdown * 100, 0, alpha=0.3, label=label
                )

        ax.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plots["drawdown"] = self._fig_to_base64(fig)
        plt.close(fig)

        # 3. Returns Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, result in self.results.items():
            if "data" in result:
                data = result["data"]
                returns = data["returns"].dropna() * 100
                label = key.replace("_", " ").title()
                ax.hist(returns, bins=50, alpha=0.5, label=label, density=True)

        ax.set_title("Returns Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plots["returns_distribution"] = self._fig_to_base64(fig)
        plt.close(fig)

        # 4. Rolling Sharpe
        fig, ax = plt.subplots(figsize=(12, 4))
        for key, result in self.results.items():
            if "data" in result:
                data = result["data"]
                returns = data["returns"].dropna()
                rolling_sharpe = returns.rolling(60).apply(
                    lambda x: (
                        np.sqrt(252) * x.mean() / (x.std() + 1e-8) if x.std() > 0 else 0
                    )
                )
                label = key.replace("_", " ").title()
                ax.plot(data["date"][1:], rolling_sharpe, label=label, alpha=0.7)

        ax.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Sharpe = 1")
        ax.set_title("Rolling Sharpe Ratio (60-day)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plots["rolling_sharpe"] = self._fig_to_base64(fig)
        plt.close(fig)

        # 5. Risk-Return Scatter
        if len(self.results) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            comparison = self.compare_strategies()

            colors = plt.cm.Set1(np.linspace(0, 1, len(comparison)))
            for i, (strategy, metrics) in enumerate(comparison.iterrows()):
                ax.scatter(
                    metrics["volatility"] * 100,
                    metrics["annualized_return"] * 100,
                    s=200,
                    c=[colors[i]],
                    alpha=0.7,
                    label=strategy,
                )
                ax.annotate(
                    strategy,
                    (metrics["volatility"] * 100, metrics["annualized_return"] * 100),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

            ax.set_title("Risk-Return Profile", fontsize=14, fontweight="bold")
            ax.set_xlabel("Annualized Volatility (%)")
            ax.set_ylabel("Annualized Return (%)")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plots["risk_return"] = self._fig_to_base64(fig)
            plt.close(fig)

        # 6. Metrics Heatmap
        if len(self.results) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            comparison = self.compare_strategies()

            metrics_cols = [
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "total_return",
                "max_drawdown",
            ]
            available_cols = [c for c in metrics_cols if c in comparison.columns]
            heatmap_data = comparison[available_cols].T

            # Normalize for visualization
            heatmap_norm = (
                heatmap_data - heatmap_data.min(axis=1).values.reshape(-1, 1)
            ) / (
                heatmap_data.max(axis=1) - heatmap_data.min(axis=1) + 1e-8
            ).values.reshape(
                -1, 1
            )

            sns.heatmap(heatmap_norm, annot=heatmap_data.round(2), cmap="RdYlGn", ax=ax)
            ax.set_title(
                "Performance Metrics Comparison", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            plots["metrics_heatmap"] = self._fig_to_base64(fig)
            plt.close(fig)

        # 7. Portfolio Value Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        for key, result in self.results.items():
            if "data" in result:
                data = result["data"]
                label = key.replace("_", " ").title()
                ax.plot(data["date"], data["portfolio_value"], label=label, linewidth=2)

        ax.axhline(
            y=self.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial Capital",
        )
        ax.set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        plt.tight_layout()

        plots["portfolio_value"] = self._fig_to_base64(fig)
        plt.close(fig)

        return plots

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def get_equity_curve(self, strategy: str = "agent") -> list[dict]:
        """Get equity curve data for a strategy."""
        if strategy not in self.results:
            return []

        data = self.results[strategy]["data"]
        return [
            {
                "date": str(row["date"]),
                "value": row["portfolio_value"],
                "return": row["cumulative_returns"],
            }
            for _, row in data.iterrows()
        ]

    def get_trades(self, strategy: str = "agent") -> list[dict]:
        """Get trade list for a strategy."""
        if strategy not in self.results:
            return []

        trades = self.results[strategy].get("trades", [])
        return [
            {
                "step": t.get("step", 0),
                "date": t.get("date", ""),
                "action": t.get("action", ""),
                "price": t.get("price", 0),
                "position": t.get("position", 0),
                "portfolio_value": t.get("portfolio_value", 0),
            }
            for t in trades
        ]
