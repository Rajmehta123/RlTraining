"""Data handling and feature engineering for RL trading."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import tiered feature extractor
try:
    from feature_extractor import TieredFeatureExtractor

    TIERED_FEATURES_AVAILABLE = True
except ImportError:
    TIERED_FEATURES_AVAILABLE = False
    logger.warning("TieredFeatureExtractor not available, using legacy features")


class DataHandler:
    """Handles data collection, preprocessing, and feature engineering."""

    MIN_YEARS = 2  # Minimum years required for technical indicators
    FMP_BASE_URL = "https://financialmodelingprep.com/stable"  # New stable API base URL

    def __init__(
        self,
        symbol: str,
        fmp_api_key: str = None,
        years: int = 2,
        feature_tiers: list[int] = None,
        use_tiered_features: bool = True,
    ):
        self.symbol = symbol.upper()
        self.fmp_api_key = fmp_api_key
        self.years = max(years, self.MIN_YEARS)  # Enforce minimum
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        self.fundamentals = {}
        self.feature_tiers = feature_tiers or [1, 2]  # Default: Tier 1 & 2
        self.use_tiered_features = use_tiered_features and TIERED_FEATURES_AVAILABLE

    def _make_fmp_request(
        self, endpoint: str, params: dict[str, str] = None
    ) -> dict | None:
        """
        Make a request to the FMP stable API.

        Args:
            endpoint: API endpoint (e.g., '/income-statement')
            params: Query parameters (symbol will be added automatically)

        Returns:
            JSON response or None if request failed
        """
        if not self.fmp_api_key:
            return None

        if params is None:
            params = {}

        # Add symbol and API key to params
        params["symbol"] = self.symbol
        params["apikey"] = self.fmp_api_key

        # Build URL with query parameters
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.FMP_BASE_URL}{endpoint}?{query_string}"

        # Mask API key for logging
        masked_url = url.replace(self.fmp_api_key, "***")

        try:
            response = requests.get(url, timeout=15)

            if response.status_code == 429:
                logger.warning(f"FMP Rate Limited: {masked_url}")
                return None

            if response.status_code in (401, 403):
                logger.error(f"FMP Auth Error ({response.status_code}): {masked_url}")
                return None

            if response.status_code != 200:
                logger.error(f"FMP API Error ({response.status_code}): {masked_url}")
                return None

            data = response.json()

            # Check for API error messages in response
            if isinstance(data, dict) and "Error Message" in data:
                logger.error(f"FMP API returned error: {data['Error Message']}")
                return None

            return data

        except requests.exceptions.Timeout:
            logger.error(f"FMP Request timeout: {masked_url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"FMP Request failed: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"FMP JSON decode error: {str(e)}")
            return None

    def load_data(self) -> pd.DataFrame:
        """Load historical price data from yfinance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.years)

            logger.info(f"Downloading data from yfinance for {self.symbol}")
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
            )

            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Handle missing values (pandas 2.0+ compatible)
            df.ffill(inplace=True)
            df.bfill(inplace=True)

            self.raw_data = df
            logger.info(f"Successfully loaded {len(df)} rows of data")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

    def load_fundamentals(self) -> dict:
        """Load fundamental data from FMP stable API."""
        if not self.fmp_api_key:
            logger.warning("No FMP API key provided, skipping fundamentals")
            return {}

        try:
            logger.info(f"Loading fundamentals from FMP stable API for {self.symbol}")

            # Get income statement (quarterly, last 8 quarters)
            income_data = self._make_fmp_request(
                "/income-statement", {"period": "quarter", "limit": "8"}
            )
            if income_data:
                self.fundamentals["income_statement"] = income_data
                logger.debug(
                    f"Loaded {len(income_data) if isinstance(income_data, list) else 1} income statements"
                )

            # Get balance sheet (quarterly, last 8 quarters)
            balance_data = self._make_fmp_request(
                "/balance-sheet-statement", {"period": "quarter", "limit": "8"}
            )
            if balance_data:
                self.fundamentals["balance_sheet"] = balance_data
                logger.debug(
                    f"Loaded {len(balance_data) if isinstance(balance_data, list) else 1} balance sheets"
                )

            # Get key metrics (quarterly, last 8 quarters)
            metrics_data = self._make_fmp_request(
                "/key-metrics", {"period": "quarter", "limit": "8"}
            )
            if metrics_data:
                self.fundamentals["key_metrics"] = metrics_data
                logger.debug(
                    f"Loaded {len(metrics_data) if isinstance(metrics_data, list) else 1} key metrics"
                )

            # Get TTM key metrics for most current data
            metrics_ttm_data = self._make_fmp_request("/key-metrics-ttm")
            if metrics_ttm_data:
                self.fundamentals["key_metrics_ttm"] = metrics_ttm_data
                logger.debug("Loaded TTM key metrics")

            # Get TTM ratios
            ratios_ttm_data = self._make_fmp_request("/ratios-ttm")
            if ratios_ttm_data:
                self.fundamentals["ratios_ttm"] = ratios_ttm_data
                logger.debug("Loaded TTM ratios")

            # Get earnings data (new stable endpoint)
            earnings_data = self._make_fmp_request("/earnings", {"limit": "20"})
            if earnings_data:
                self.fundamentals["earnings"] = earnings_data
                logger.debug(
                    f"Loaded {len(earnings_data) if isinstance(earnings_data, list) else 1} earnings records"
                )

            # Get company profile for additional context
            profile_data = self._make_fmp_request("/profile")
            if profile_data:
                self.fundamentals["profile"] = profile_data
                logger.debug("Loaded company profile")

            if self.fundamentals:
                logger.info(
                    f"Successfully loaded fundamentals for {self.symbol}: {list(self.fundamentals.keys())}"
                )
            else:
                logger.warning(f"No fundamental data loaded for {self.symbol}")

            return self.fundamentals

        except Exception as e:
            logger.error(f"Error loading fundamentals: {str(e)}")
            return {}

    def _get_latest_metric(self, metric_name: str, default=None):
        """
        Get the latest value for a metric from fundamentals data.
        Checks TTM data first, then falls back to quarterly data.
        """
        # Try TTM key metrics first
        ttm_metrics = self.fundamentals.get("key_metrics_ttm", [])
        if isinstance(ttm_metrics, list) and len(ttm_metrics) > 0:
            value = ttm_metrics[0].get(metric_name)
            if value is not None:
                return value

        # Try TTM ratios
        ttm_ratios = self.fundamentals.get("ratios_ttm", [])
        if isinstance(ttm_ratios, list) and len(ttm_ratios) > 0:
            value = ttm_ratios[0].get(metric_name)
            if value is not None:
                return value

        # Fall back to quarterly key metrics
        quarterly_metrics = self.fundamentals.get("key_metrics", [])
        if isinstance(quarterly_metrics, list) and len(quarterly_metrics) > 0:
            value = quarterly_metrics[0].get(metric_name)
            if value is not None:
                return value

        return default

    def calculate_features(self) -> pd.DataFrame:
        """Calculate technical indicators and features."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Calculating technical features...")
        df = self.raw_data.copy()

        # Use tiered feature extraction if available and enabled
        if self.use_tiered_features:
            return self._calculate_tiered_features(df)

        # Legacy feature extraction below

        # Determine available data length for adaptive lookback
        data_len = len(df)
        use_long_lookback = data_len > 250

        # Returns
        df["returns_1d"] = df["Close"].pct_change()
        df["returns_5d"] = df["Close"].pct_change(5)
        df["returns_20d"] = df["Close"].pct_change(20)
        df["log_returns_1d"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_returns_5d"] = np.log(df["Close"] / df["Close"].shift(5))

        # Volatility measures
        df["volatility_20d"] = df["returns_1d"].rolling(20).std()
        df["volatility_60d"] = df["returns_1d"].rolling(min(60, data_len // 4)).std()
        df["realized_vol"] = df["returns_1d"].rolling(20).std() * np.sqrt(252)
        vol_window = min(252, data_len - 50) if data_len > 100 else 50
        df["vol_percentile"] = (
            df["realized_vol"].rolling(vol_window, min_periods=20).rank(pct=True)
        )

        # Simple Moving Averages (adaptive)
        df["sma_10"] = df["Close"].rolling(10).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["sma_50"] = df["Close"].rolling(min(50, data_len // 5)).mean()
        sma_long = (
            min(200, data_len // 3) if use_long_lookback else min(100, data_len // 3)
        )
        df["sma_200"] = df["Close"].rolling(sma_long, min_periods=20).mean()

        # SMA ratios
        df["sma_ratio_20_50"] = df["sma_20"] / df["sma_50"]
        df["sma_ratio_50_200"] = df["sma_50"] / df["sma_200"]
        df["price_to_sma20"] = df["Close"] / df["sma_20"] - 1
        df["price_to_sma50"] = df["Close"] / df["sma_50"] - 1

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["Close"], period=14)
        df["rsi_7"] = self._calculate_rsi(df["Close"], period=7)

        # MACD
        df["macd"], df["macd_signal"] = self._calculate_macd(df["Close"])
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = (
            self._calculate_bollinger_bands(df["Close"])
        )
        df["bb_position"] = (df["Close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-8
        )
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Volume features
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["volume_trend"] = (
            df["Volume"].rolling(20).mean() / df["Volume"].rolling(60).mean()
        )

        # Price features
        df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_location"] = (df["Close"] - df["Low"]) / (
            df["High"] - df["Low"] + 1e-8
        )

        # Momentum (adaptive)
        df["momentum_10d"] = df["Close"] / df["Close"].shift(10) - 1
        df["momentum_30d"] = (
            df["Close"] / df["Close"].shift(min(30, data_len // 10)) - 1
        )
        df["momentum_60d"] = df["Close"] / df["Close"].shift(min(60, data_len // 5)) - 1

        # ADX, ATR, Stochastic
        df["adx"] = self._calculate_adx(df, period=14)
        df["atr"] = self._calculate_atr(df, period=14)
        df["atr_ratio"] = df["atr"] / df["Close"]
        df["stoch_k"], df["stoch_d"] = self._calculate_stochastic(df, period=14)

        # Time-based features
        df["day_of_week"] = df.index.dayofweek / 4.0 - 1.0
        df["month_of_year"] = (df.index.month - 6.5) / 5.5

        # Market microstructure
        df["daily_range"] = (df["High"] - df["Low"]) / df["Open"]
        df["overnight_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

        # Define feature columns
        self.feature_columns = [
            "returns_1d",
            "returns_5d",
            "returns_20d",
            "log_returns_1d",
            "log_returns_5d",
            "volatility_20d",
            "volatility_60d",
            "vol_percentile",
            "sma_ratio_20_50",
            "sma_ratio_50_200",
            "price_to_sma20",
            "price_to_sma50",
            "rsi_14",
            "rsi_7",
            "macd_signal",
            "macd_histogram",
            "bb_position",
            "bb_width",
            "adx",
            "atr_ratio",
            "stoch_k",
            "stoch_d",
            "volume_ratio",
            "volume_trend",
            "high_low_spread",
            "close_location",
            "daily_range",
            "overnight_gap",
            "momentum_10d",
            "momentum_30d",
            "momentum_60d",
            "day_of_week",
            "month_of_year",
        ]

        # Add fundamental features if available
        if self.fundamentals.get("key_metrics") or self.fundamentals.get(
            "key_metrics_ttm"
        ):
            df = self._add_fundamental_features(df)

        # Fill remaining NaN values instead of dropping rows
        # First forward fill, then backward fill, then fill with 0
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)

        # Drop only rows where essential columns are still NaN
        essential_cols = ["Close", "returns_1d"]
        df.dropna(subset=essential_cols, inplace=True)

        # Ensure we have enough data
        if len(df) < 50:
            logger.warning(
                f"Only {len(df)} rows after feature calculation. Need at least 50."
            )

        self.processed_data = df

        logger.info(f"Features calculated. Dataset size: {len(df)}")
        return df

    def _add_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental data features from the new stable API format."""

        # Get PE ratio from TTM metrics or quarterly metrics
        pe_ratio = self._get_latest_metric("peRatio") or self._get_latest_metric(
            "peRatioTTM"
        )
        if pe_ratio is not None:
            df["pe_ratio"] = pe_ratio
            if "pe_ratio" not in self.feature_columns:
                self.feature_columns.append("pe_ratio")

        # Get PB ratio
        pb_ratio = self._get_latest_metric("pbRatio") or self._get_latest_metric(
            "priceToBookRatioTTM"
        )
        if pb_ratio is not None:
            df["pb_ratio"] = pb_ratio
            if "pb_ratio" not in self.feature_columns:
                self.feature_columns.append("pb_ratio")

        # Get ROE
        roe = self._get_latest_metric("returnOnEquity") or self._get_latest_metric(
            "returnOnEquityTTM"
        )
        if roe is not None:
            df["roe"] = roe
            if "roe" not in self.feature_columns:
                self.feature_columns.append("roe")

        # Get Debt to Equity
        debt_to_equity = self._get_latest_metric(
            "debtToEquity"
        ) or self._get_latest_metric("debtToEquityTTM")
        if debt_to_equity is not None:
            df["debt_to_equity"] = debt_to_equity
            if "debt_to_equity" not in self.feature_columns:
                self.feature_columns.append("debt_to_equity")

        # Map quarterly metrics to dates if available for time-varying fundamentals
        metrics = self.fundamentals.get("key_metrics", [])
        if isinstance(metrics, list) and len(metrics) > 0:
            pe_ratios = []
            for m in metrics:
                if "date" in m and "peRatio" in m:
                    pe_ratios.append(
                        {
                            "date": pd.to_datetime(m["date"]),
                            "pe": m.get("peRatio", np.nan),
                        }
                    )

            if pe_ratios:
                pe_df = pd.DataFrame(pe_ratios).set_index("date").sort_index()
                pe_df = pe_df.reindex(df.index, method="ffill")
                # Only update if we have more granular data
                if not pe_df["pe"].isna().all():
                    df["pe_ratio"] = pe_df["pe"].ffill().bfill()

        return df

    def _calculate_tiered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features using the tiered feature extraction system."""
        logger.info(f"Using tiered feature extraction (Tiers: {self.feature_tiers})")

        # Initialize feature extractor with selected tiers
        extractor = TieredFeatureExtractor(include_tiers=self.feature_tiers)

        # Prepare fundamentals dict for Tier 2 if available
        fundamentals_dict = None

        # Try to get data from TTM sources first (most current)
        ttm_metrics = self.fundamentals.get("key_metrics_ttm", [])
        ttm_ratios = self.fundamentals.get("ratios_ttm", [])
        quarterly_metrics = self.fundamentals.get("key_metrics", [])

        if ttm_metrics or ttm_ratios or quarterly_metrics:
            fundamentals_dict = {}

            # Get from TTM key metrics
            if isinstance(ttm_metrics, list) and len(ttm_metrics) > 0:
                m = ttm_metrics[0]
                fundamentals_dict.update(
                    {
                        "peRatio": m.get("peRatioTTM"),
                        "pbRatio": m.get("pbRatioTTM"),
                        "priceToSalesRatio": m.get("priceToSalesRatioTTM"),
                        "returnOnEquity": m.get("roeTTM") or m.get("returnOnEquityTTM"),
                        "returnOnAssets": m.get("roaTTM") or m.get("returnOnAssetsTTM"),
                        "debtToEquity": m.get("debtToEquityTTM"),
                        "currentRatio": m.get("currentRatioTTM"),
                        "evToEbitda": m.get("enterpriseValueOverEBITDATTM"),
                    }
                )

            # Get from TTM ratios
            if isinstance(ttm_ratios, list) and len(ttm_ratios) > 0:
                r = ttm_ratios[0]
                if fundamentals_dict.get("peRatio") is None:
                    fundamentals_dict["peRatio"] = r.get("priceEarningsRatioTTM")
                if fundamentals_dict.get("pbRatio") is None:
                    fundamentals_dict["pbRatio"] = r.get("priceToBookRatioTTM")
                fundamentals_dict.update(
                    {
                        "grossProfitMargin": r.get("grossProfitMarginTTM"),
                        "netProfitMargin": r.get("netProfitMarginTTM"),
                        "operatingProfitMargin": r.get("operatingProfitMarginTTM"),
                        "dividendYield": r.get("dividendYieldTTM"),
                    }
                )

            # Fallback to quarterly metrics
            if isinstance(quarterly_metrics, list) and len(quarterly_metrics) > 0:
                m = quarterly_metrics[0]
                if fundamentals_dict.get("peRatio") is None:
                    fundamentals_dict["peRatio"] = m.get("peRatio")
                if fundamentals_dict.get("pbRatio") is None:
                    fundamentals_dict["pbRatio"] = m.get("pbRatio")
                if fundamentals_dict.get("returnOnEquity") is None:
                    fundamentals_dict["returnOnEquity"] = m.get("roe")
                if fundamentals_dict.get("debtToEquity") is None:
                    fundamentals_dict["debtToEquity"] = m.get("debtToEquity")

            # Add earnings data if available (new stable API format)
            earnings = self.fundamentals.get("earnings", [])
            if isinstance(earnings, list) and len(earnings) > 0:
                latest_earnings = earnings[0]
                # New stable API uses epsActual/epsEstimated instead of actualEarningResult/estimatedEarning
                actual = latest_earnings.get("epsActual")
                estimated = latest_earnings.get("epsEstimated")
                if actual is not None and estimated is not None and estimated != 0:
                    fundamentals_dict["earningsSurprise"] = (
                        (actual - estimated) / abs(estimated)
                    ) * 100

                # Revenue data
                actual_revenue = latest_earnings.get("revenueActual")
                estimated_revenue = latest_earnings.get("revenueEstimated")
                if (
                    actual_revenue is not None
                    and estimated_revenue is not None
                    and estimated_revenue != 0
                ):
                    fundamentals_dict["revenueSurprise"] = (
                        (actual_revenue - estimated_revenue) / abs(estimated_revenue)
                    ) * 100

                fundamentals_dict["revenueActual"] = actual_revenue

            # Clean up None values
            fundamentals_dict = {
                k: v for k, v in fundamentals_dict.items() if v is not None
            }

            if not fundamentals_dict:
                fundamentals_dict = None

        # Extract features
        features_df = extractor.extract_features(
            df=df,
            fundamentals=fundamentals_dict,
            market_data=None,  # Could add SPY data fetching here
            alpha_data=None,  # Could add analyst data here
        )

        # Merge features with original OHLCV data
        result_df = df.copy()
        for col in features_df.columns:
            result_df[col] = features_df[col]

        # Set feature columns (these will be used by the environment)
        self.feature_columns = list(features_df.columns)

        # Drop rows where essential columns are NaN
        essential_cols = (
            ["Close", "return_1d"] if "return_1d" in result_df.columns else ["Close"]
        )
        result_df.dropna(subset=essential_cols, inplace=True)

        # Ensure we have enough data
        if len(result_df) < 50:
            logger.warning(
                f"Only {len(result_df)} rows after feature calculation. Need at least 50."
            )

        self.processed_data = result_df

        logger.info(
            f"Tiered features calculated. Dataset size: {len(result_df)}, Features: {len(self.feature_columns)}"
        )
        return result_df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / (atr + 1e-8)))

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
        adx = dx.rolling(period).mean()
        return adx

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_stochastic(
        self, df: pd.DataFrame, period: int = 14
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df["Low"].rolling(window=period).min()
        high_max = df["High"].rolling(window=period).max()
        k = 100 * ((df["Close"] - low_min) / (high_max - low_min + 1e-8))
        k = k.rolling(window=3).mean()
        d = k.rolling(window=3).mean()
        return k, d

    def normalize_features(self, method: str = "robust") -> pd.DataFrame:
        """Normalize features for neural network input."""
        if self.processed_data is None:
            raise ValueError("No processed data. Run calculate_features() first.")

        df = self.processed_data.copy()
        original_features = self.feature_columns.copy()

        logger.info(f"Normalizing features using {method} method")

        for col in original_features:
            if col not in df.columns:
                continue

            if method == "robust":
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                df[f"{col}_norm"] = ((df[col] - median) / (iqr + 1e-8)).clip(-3, 3)
            elif method == "standard":
                mean = df[col].mean()
                std = df[col].std()
                df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-8)
            elif method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                df[f"{col}_norm"] = (
                    2 * (df[col] - min_val) / (max_val - min_val + 1e-8) - 1
                )

        self.feature_columns = [
            f"{col}_norm" for col in original_features if col in df.columns
        ]
        self.processed_data = df

        return df

    def split_data(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        if self.processed_data is None:
            raise ValueError("No processed data available")

        n = len(self.processed_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            "train": self.processed_data.iloc[:train_end].copy(),
            "val": self.processed_data.iloc[train_end:val_end].copy(),
            "test": self.processed_data.iloc[val_end:].copy(),
        }

    def get_data_summary(self) -> dict:
        """Get summary of loaded data."""
        if self.raw_data is None:
            return {}

        return {
            "symbol": self.symbol,
            "start_date": str(self.raw_data.index[0]),
            "end_date": str(self.raw_data.index[-1]),
            "total_rows": len(self.raw_data),
            "processed_rows": (
                len(self.processed_data) if self.processed_data is not None else 0
            ),
            "feature_count": len(self.feature_columns),
            "has_fundamentals": bool(self.fundamentals),
            "fundamentals_loaded": (
                list(self.fundamentals.keys()) if self.fundamentals else []
            ),
        }
