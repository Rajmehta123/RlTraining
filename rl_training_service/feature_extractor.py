"""Tiered Feature Extraction for Stock Trading RL."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TieredFeatureExtractor:
    """
    Feature extraction following tiered importance:
    - Tier 1: Price/Technical (Most predictive) ~50-60%
    - Tier 2: Fundamentals/Catalyst ~25-30%
    - Tier 3: Market Context ~10-15%
    - Tier 4: Alpha Signals ~5-10%
    """

    def __init__(self, include_tiers: list[int] = None):
        """
        Initialize feature extractor.

        Args:
            include_tiers: List of tiers to include (1-4). Default [1, 2] for balanced approach.
        """
        self.include_tiers = include_tiers or [1, 2]
        self.feature_names = []
        self._build_feature_list()

    def _build_feature_list(self):
        """Build list of feature names based on selected tiers."""
        self.feature_names = []

        if 1 in self.include_tiers:
            self.feature_names.extend(
                [
                    # Price Momentum
                    "return_1d",
                    "return_5d",
                    "return_10d",
                    "return_20d",
                    # Volatility
                    "volatility_5d",
                    "volatility_20d",
                    "atr_normalized",
                    # Moving Average Position
                    "price_vs_sma20",
                    "price_vs_sma50",
                    "price_vs_sma200",
                    "sma20_vs_sma50",
                    "sma50_vs_sma200",
                    # Technical Indicators
                    "rsi",
                    "rsi_oversold",
                    "rsi_overbought",
                    "macd_signal",
                    "macd_histogram",
                    # Volume
                    "volume_ratio",
                    "volume_trend",
                    "obv_trend",
                    # Price Patterns
                    "high_low_range",
                    "close_position",
                    "upper_shadow",
                    "lower_shadow",
                    # Bollinger Bands
                    "bb_position",
                    "bb_width",
                ]
            )

        if 2 in self.include_tiers:
            self.feature_names.extend(
                [
                    # Valuation
                    "pe_normalized",
                    "pb_normalized",
                    "ps_normalized",
                    # Quality
                    "roe",
                    "gross_margin",
                    "net_margin",
                    "debt_to_equity",
                    # Earnings
                    "earnings_surprise",
                    "earnings_momentum",
                    # Insider Activity
                    "insider_net_signal",
                ]
            )

        if 3 in self.include_tiers:
            self.feature_names.extend(
                [
                    # Market Regime
                    "market_return",
                    "market_trend",
                    "sector_relative_strength",
                    # Macro
                    "rate_environment",
                ]
            )

        if 4 in self.include_tiers:
            self.feature_names.extend(
                [
                    # Alpha signals
                    "analyst_revision",
                    "dcf_discount",
                ]
            )

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def calculate_tier1_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tier 1: Price/Technical features (most important)."""
        features = pd.DataFrame(index=df.index)

        # === PRICE MOMENTUM ===
        features["return_1d"] = df["Close"].pct_change(1)
        features["return_5d"] = df["Close"].pct_change(5)
        features["return_10d"] = df["Close"].pct_change(10)
        features["return_20d"] = df["Close"].pct_change(20)

        # === VOLATILITY ===
        features["volatility_5d"] = df["Close"].pct_change().rolling(5).std() * np.sqrt(
            252
        )
        features["volatility_20d"] = df["Close"].pct_change().rolling(
            20
        ).std() * np.sqrt(252)

        # ATR normalized
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        features["atr_normalized"] = atr / df["Close"]

        # === MOVING AVERAGES ===
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200, min_periods=50).mean()

        features["price_vs_sma20"] = (df["Close"] - sma20) / sma20
        features["price_vs_sma50"] = (df["Close"] - sma50) / sma50
        features["price_vs_sma200"] = (df["Close"] - sma200) / sma200
        features["sma20_vs_sma50"] = (sma20 - sma50) / sma50
        features["sma50_vs_sma200"] = (sma50 - sma200) / sma200

        # === RSI ===
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features["rsi"] = rsi / 100  # Normalize to 0-1
        features["rsi_oversold"] = (rsi < 30).astype(float)
        features["rsi_overbought"] = (rsi > 70).astype(float)

        # === MACD ===
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features["macd_signal"] = (macd > signal).astype(float)
        features["macd_histogram"] = (macd - signal) / df["Close"]

        # === VOLUME ===
        features["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        features["volume_trend"] = (
            df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()
        )

        # OBV trend
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        features["obv_trend"] = obv.pct_change(10)

        # === PRICE PATTERNS ===
        features["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        features["close_position"] = (df["Close"] - df["Low"]) / (
            df["High"] - df["Low"] + 1e-10
        )
        features["upper_shadow"] = (
            df["High"] - df[["Open", "Close"]].max(axis=1)
        ) / df["Close"]
        features["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df[
            "Close"
        ]

        # === BOLLINGER BANDS ===
        bb_middle = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        features["bb_position"] = (df["Close"] - bb_lower) / (
            bb_upper - bb_lower + 1e-10
        )
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle

        return features

    def calculate_tier2_features(
        self, df: pd.DataFrame, fundamentals: dict = None
    ) -> pd.DataFrame:
        """Calculate Tier 2: Fundamentals/Catalyst features."""
        features = pd.DataFrame(index=df.index)

        if fundamentals:
            # === VALUATION ===
            pe = fundamentals.get("peRatio", 20)
            pb = fundamentals.get("pbRatio", 2)
            ps = fundamentals.get("priceToSalesRatio", 3)

            features["pe_normalized"] = np.clip(pe, 0, 100) / 100 if pe else 0.2
            features["pb_normalized"] = np.clip(pb, 0, 20) / 20 if pb else 0.1
            features["ps_normalized"] = np.clip(ps, 0, 20) / 20 if ps else 0.15

            # === QUALITY ===
            features["roe"] = fundamentals.get("returnOnEquity", 0.1) or 0.1
            features["gross_margin"] = fundamentals.get("grossProfitMargin", 0.3) or 0.3
            features["net_margin"] = fundamentals.get("netProfitMargin", 0.1) or 0.1
            features["debt_to_equity"] = (
                np.clip(fundamentals.get("debtToEquity", 0.5) or 0.5, 0, 5) / 5
            )

            # === EARNINGS ===
            earnings_surprise = fundamentals.get("earningsSurprise", 0) or 0
            features["earnings_surprise"] = np.clip(earnings_surprise / 100, -0.5, 0.5)
            features["earnings_momentum"] = fundamentals.get("earningsGrowth", 0) or 0

            # === INSIDER ===
            insider_buys = fundamentals.get("insiderBuys", 0) or 0
            insider_sells = fundamentals.get("insiderSells", 0) or 0
            total = insider_buys + insider_sells + 1
            features["insider_net_signal"] = (insider_buys - insider_sells) / total
        else:
            # Default values when no fundamentals
            features["pe_normalized"] = 0.2
            features["pb_normalized"] = 0.1
            features["ps_normalized"] = 0.15
            features["roe"] = 0.1
            features["gross_margin"] = 0.3
            features["net_margin"] = 0.1
            features["debt_to_equity"] = 0.25
            features["earnings_surprise"] = 0.0
            features["earnings_momentum"] = 0.0
            features["insider_net_signal"] = 0.0

        return features

    def calculate_tier3_features(
        self, df: pd.DataFrame, market_data: dict = None
    ) -> pd.DataFrame:
        """Calculate Tier 3: Market Context features."""
        features = pd.DataFrame(index=df.index)

        if market_data and "spy" in market_data:
            spy = market_data["spy"]
            features["market_return"] = spy.get("changePercent", 0) / 100
            features["market_trend"] = 1 if spy.get("changePercent", 0) > 0 else -1
        else:
            # Use stock's own momentum as proxy
            features["market_return"] = df["Close"].pct_change(5).fillna(0)
            features["market_trend"] = (df["Close"] > df["Close"].shift(20)).astype(
                float
            ) * 2 - 1

        # Sector relative strength (use stock's relative performance as proxy)
        features["sector_relative_strength"] = (
            df["Close"].pct_change(20) - df["Close"].pct_change(60) / 3
        )

        # Rate environment (simplified)
        features["rate_environment"] = (
            0.05  # Placeholder - could be fetched from macro data
        )

        return features

    def calculate_tier4_features(
        self, df: pd.DataFrame, alpha_data: dict = None
    ) -> pd.DataFrame:
        """Calculate Tier 4: Alpha Signals."""
        features = pd.DataFrame(index=df.index)

        if alpha_data:
            features["analyst_revision"] = alpha_data.get("analystRevision", 0)
            features["dcf_discount"] = alpha_data.get("dcfDiscount", 0)
        else:
            features["analyst_revision"] = 0.0
            features["dcf_discount"] = 0.0

        return features

    def extract_features(
        self,
        df: pd.DataFrame,
        fundamentals: dict = None,
        market_data: dict = None,
        alpha_data: dict = None,
    ) -> pd.DataFrame:
        """
        Extract all features based on selected tiers.

        Returns DataFrame with all feature columns.
        """
        all_features = []

        if 1 in self.include_tiers:
            tier1 = self.calculate_tier1_features(df)
            all_features.append(tier1)
            logger.info(f"Tier 1 features: {len(tier1.columns)} columns")

        if 2 in self.include_tiers:
            tier2 = self.calculate_tier2_features(df, fundamentals)
            all_features.append(tier2)
            logger.info(f"Tier 2 features: {len(tier2.columns)} columns")

        if 3 in self.include_tiers:
            tier3 = self.calculate_tier3_features(df, market_data)
            all_features.append(tier3)
            logger.info(f"Tier 3 features: {len(tier3.columns)} columns")

        if 4 in self.include_tiers:
            tier4 = self.calculate_tier4_features(df, alpha_data)
            all_features.append(tier4)
            logger.info(f"Tier 4 features: {len(tier4.columns)} columns")

        # Combine all features
        features_df = pd.concat(all_features, axis=1)

        # Handle NaN and Inf
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.ffill().bfill().fillna(0)

        # Clip extreme values
        for col in features_df.columns:
            features_df[col] = np.clip(features_df[col], -10, 10)

        logger.info(f"Total features extracted: {len(features_df.columns)}")
        return features_df


# Risk-aware reward shaping for IQN
class RiskAwareRewardShaper:
    """Shapes rewards for risk-aware trading (used with IQN)."""

    def __init__(
        self,
        risk_aversion: float = 0.5,
        drawdown_penalty: float = 2.0,
        sharpe_bonus: float = 1.0,
    ):
        self.risk_aversion = risk_aversion
        self.drawdown_penalty = drawdown_penalty
        self.sharpe_bonus = sharpe_bonus
        self.returns_history = []
        self.peak_value = 0

    def shape_reward(
        self,
        base_reward: float,
        portfolio_value: float,
        daily_return: float,
    ) -> float:
        """
        Shape reward to be risk-aware.

        - Penalizes drawdowns
        - Rewards consistent returns (Sharpe-like)
        - Penalizes high volatility
        """
        self.returns_history.append(daily_return)
        self.peak_value = max(self.peak_value, portfolio_value)

        # Drawdown penalty
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        drawdown_penalty = -self.drawdown_penalty * drawdown**2

        # Volatility penalty (risk aversion)
        if len(self.returns_history) > 5:
            recent_vol = (
                np.std(self.returns_history[-20:])
                if len(self.returns_history) >= 20
                else np.std(self.returns_history)
            )
            vol_penalty = -self.risk_aversion * recent_vol * 10
        else:
            vol_penalty = 0

        # Sharpe-like bonus for consistent positive returns
        if len(self.returns_history) > 10:
            mean_return = np.mean(self.returns_history[-10:])
            std_return = np.std(self.returns_history[-10:]) + 1e-8
            sharpe_bonus = (
                self.sharpe_bonus * (mean_return / std_return) if std_return > 0 else 0
            )
        else:
            sharpe_bonus = 0

        shaped_reward = base_reward + drawdown_penalty + vol_penalty + sharpe_bonus
        return shaped_reward

    def reset(self):
        """Reset for new episode."""
        self.returns_history = []
        self.peak_value = 0
