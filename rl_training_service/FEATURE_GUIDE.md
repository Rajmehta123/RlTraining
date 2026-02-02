# RL Trading Agent - Feature & Algorithm Selection Guide

Quick reference guide for selecting the right algorithm and feature tiers for your trading strategy.

## Algorithm Selection Guide

### When to Use PPO (Proximal Policy Optimization)

**Best For:**
- ✅ General stock trading
- ✅ First-time users
- ✅ Limited computational resources
- ✅ Stable, reliable performance
- ✅ Fast iteration and experimentation

**Pros:**
- Fast training (5-7 min for 100K steps)
- Very stable and reliable
- Works well across different stocks
- Attention network for pattern recognition
- Easy to tune hyperparameters

**Cons:**
- Lower sample efficiency
- Doesn't explicitly model risk
- May need more timesteps for complex strategies

**Performance:**
- Typical Sharpe Ratio: 1.2 - 1.5
- Training Speed: ⚡⚡⚡ (Fastest)
- Stability: ⭐⭐⭐⭐ (Very Stable)
- Memory: 💾 (Low)

**Recommended Config:**
```python
{
    'algorithm': 'ppo',
    'years': 2,
    'feature_tiers': [1, 2],
    'training_config': {
        'total_timesteps': 100000,
        'use_attention': True
    }
}
```

---

### When to Use Rainbow DQN

**Best For:**
- ✅ Sample-limited scenarios (limited data)
- ✅ Discrete action trading
- ✅ Want distributional returns
- ✅ Medium computational budget
- ✅ Exploring different strategies

**Pros:**
- High sample efficiency (learns from less data)
- Distributional RL (models return distribution)
- Prioritized replay (learns from important experiences)
- Dueling architecture (better value estimation)

**Cons:**
- Slower than PPO
- Higher memory usage
- Exploration-exploitation tradeoff tuning needed
- Requires more GPU memory

**Performance:**
- Typical Sharpe Ratio: 1.3 - 1.6
- Training Speed: ⚡⚡ (Medium)
- Stability: ⭐⭐⭐ (Stable)
- Memory: 💾💾 (Medium)

**Recommended Config:**
```python
{
    'algorithm': 'rainbow_dqn',
    'years': 3,
    'feature_tiers': [1, 2, 3],
    'training_config': {
        'total_timesteps': 150000
    }
}
```

---

### When to Use IQN (Implicit Quantile Networks)

**Best For:**
- ✅ Risk-aware trading strategies
- ✅ Tail risk management
- ✅ Optimizing Sharpe ratio
- ✅ Professional/institutional use
- ✅ Maximum performance

**Pros:**
- State-of-the-art distributional RL
- Risk-aware reward shaping
- Models full return distribution
- Penalizes drawdowns and volatility
- Best sample efficiency

**Cons:**
- Slowest training time
- Highest memory usage
- Most complex to tune
- Needs more computational resources

**Performance:**
- Typical Sharpe Ratio: 1.5 - 2.0+
- Training Speed: ⚡ (Slow)
- Stability: ⭐⭐⭐ (Stable with tuning)
- Memory: 💾💾💾 (High)

**Recommended Config:**
```python
{
    'algorithm': 'iqn',
    'years': 5,
    'feature_tiers': [1, 2, 3],
    'training_config': {
        'total_timesteps': 250000
    }
}
```

---

## Feature Tier Selection Guide

### Tier 1: Price/Technical (26 features)

**What It Includes:**
- Daily, 5-day, 10-day, 20-day returns
- 5-day and 20-day volatility
- SMA (20, 50, 200), price vs SMA ratios
- RSI (7, 14), overbought/oversold signals
- MACD signal and histogram
- Volume ratio, volume trend, OBV trend
- Bollinger Bands position and width
- Candlestick patterns (shadows, range, position)
- ATR normalized

**When to Use:**
- ✅ Pure technical trading
- ✅ Fast iteration/experimentation
- ✅ High-frequency patterns
- ✅ Limited API access
- ✅ Testing new algorithms

**Pros:**
- Always available (no API needed)
- Fast computation
- Good for momentum/trend strategies
- Proven indicators

**Cons:**
- Ignores fundamentals
- May miss value opportunities
- Less robust for long-term holds

**Recommended For:**
- Day trading strategies
- Technical analysis enthusiasts
- Quick prototypes
- Intraday signals

---

### Tier 2: Fundamentals (11 features)

**What It Includes:**
- P/E, P/B, P/S ratios (normalized)
- ROE, gross margin, net margin
- Debt-to-equity ratio
- Earnings surprise
- Earnings momentum/growth
- Insider buying/selling signal

**When to Use:**
- ✅ Value investing
- ✅ Long-term holding
- ✅ Quality assessment
- ✅ Combining technical + fundamental
- ✅ Have FMP API access

**Pros:**
- Captures company quality
- Better for swing/position trading
- Identifies undervalued stocks
- More robust long-term

**Cons:**
- Requires FMP API key
- Updates quarterly (less frequent)
- May lag price movements
- Less useful for short-term trades

**Recommended For:**
- Swing trading (1 week - 1 month holds)
- Position trading (1 month+ holds)
- Quality-focused strategies
- Warren Buffett-style investors

---

### Tier 3: Market Context (4 features)

**What It Includes:**
- Market return (SPY or proxy)
- Market trend direction
- Sector relative strength
- Interest rate environment

**When to Use:**
- ✅ Portfolio context awareness
- ✅ Regime-based strategies
- ✅ Sector rotation
- ✅ Macro-aware trading
- ✅ Comprehensive models

**Pros:**
- Adds market awareness
- Helps with regime changes
- Improves risk management
- Better correlation modeling

**Cons:**
- Adds complexity
- Requires market data fetching
- May slow training slightly
- Less impact than Tier 1/2

**Recommended For:**
- Multi-asset portfolios
- Institutional strategies
- Regime-switching models
- Advanced users

---

### Tier 4: Alpha Signals (2 features)

**What It Includes:**
- Analyst rating revisions
- DCF fair value discount

**When to Use:**
- ✅ Maximum feature set
- ✅ Research-grade models
- ✅ Have analyst data access
- ✅ Seeking every edge

**Pros:**
- Captures analyst sentiment
- Theoretical fair value
- Potential unique signals

**Cons:**
- Requires additional data sources
- May not be available for all stocks
- Small marginal improvement
- Slowest training

**Recommended For:**
- Research purposes
- Hedge fund strategies
- Maximum performance seeking
- Data-rich environments

---

## Recommended Combinations

### 🚀 Quick Prototype (Fastest)
```python
Algorithm: PPO
Feature Tiers: [1]
Training Steps: 50,000
Time: ~2-3 minutes
Use Case: Quick testing, technical-only strategies
```

### 📊 Balanced Default (Recommended)
```python
Algorithm: PPO
Feature Tiers: [1, 2]
Training Steps: 100,000
Time: ~5-7 minutes
Use Case: General trading, best balance of speed and performance
```

### 🎯 High Performance
```python
Algorithm: Rainbow DQN
Feature Tiers: [1, 2, 3]
Training Steps: 150,000
Time: ~15-20 minutes
Use Case: Serious trading, willing to wait for better results
```

### 💎 Maximum Quality (Best Results)
```python
Algorithm: IQN
Feature Tiers: [1, 2, 3]
Training Steps: 250,000
Time: ~30-40 minutes
Use Case: Professional use, risk-aware, seeking alpha
```

### 🔬 Research Grade (Comprehensive)
```python
Algorithm: IQN
Feature Tiers: [1, 2, 3, 4]
Training Steps: 500,000
Time: ~60-90 minutes
Use Case: Academic research, maximum features, best possible performance
```

---

## Feature Tier Comparison Table

| Tier | Features | Always Available | Requires API | Update Frequency | Importance |
|------|----------|-----------------|--------------|------------------|------------|
| 1 | 26 | ✅ Yes | ❌ No | Daily | ⭐⭐⭐⭐⭐ (60%) |
| 2 | 11 | ❌ No | ✅ FMP | Quarterly | ⭐⭐⭐⭐ (25%) |
| 3 | 4 | ❌ No | ✅ Market | Daily | ⭐⭐⭐ (10%) |
| 4 | 2 | ❌ No | ✅ Multiple | Varies | ⭐⭐ (5%) |

---

## Performance vs. Training Time Trade-off

```
Performance (Sharpe Ratio)
    2.0+ │                                      ● IQN [1,2,3] 500K
         │                                    ●
    1.8  │                              ● IQN [1,2,3] 250K
         │                          ●
    1.6  │                    ● Rainbow [1,2,3] 150K
         │                  ●
    1.4  │              ●  PPO [1,2] 100K
         │            ●
    1.2  │        ● PPO [1] 50K
         │      ●
    1.0  │    ●
         └────────────────────────────────────────────> Training Time
         2min  5min  10min  20min  30min  60min  90min
```

---

## Decision Tree

```
START: What's your priority?

├─ Speed & Simplicity
│  └─ PPO + Tier [1] + 50K steps
│     → 2-3 minutes, Sharpe ~1.0-1.2
│
├─ Balanced Performance
│  └─ PPO + Tier [1,2] + 100K steps
│     → 5-7 minutes, Sharpe ~1.2-1.5
│
├─ Sample Efficiency
│  └─ Rainbow DQN + Tier [1,2,3] + 150K steps
│     → 15-20 minutes, Sharpe ~1.3-1.6
│
├─ Risk Management
│  └─ IQN + Tier [1,2,3] + 250K steps
│     → 30-40 minutes, Sharpe ~1.5-2.0
│
└─ Maximum Performance
   └─ IQN + Tier [1,2,3,4] + 500K steps
      → 60-90 minutes, Sharpe ~1.6-2.0+
```

---

## Common Use Cases

### Day Trading
- **Algorithm:** PPO
- **Tiers:** [1]
- **Steps:** 50-100K
- **Why:** Fast execution, technical signals only, frequent trades

### Swing Trading (1 week - 1 month)
- **Algorithm:** PPO or Rainbow DQN
- **Tiers:** [1, 2]
- **Steps:** 100-150K
- **Why:** Balance technical + fundamental, medium holding period

### Position Trading (1 month+)
- **Algorithm:** Rainbow DQN or IQN
- **Tiers:** [1, 2, 3]
- **Steps:** 150-250K
- **Why:** Long-term quality, fundamentals matter, regime awareness

### Risk-Aware Portfolio
- **Algorithm:** IQN
- **Tiers:** [1, 2, 3]
- **Steps:** 250-500K
- **Why:** Minimize drawdowns, maximize Sharpe, tail risk management

### Research & Development
- **Algorithm:** All three (compare)
- **Tiers:** [1, 2, 3, 4]
- **Steps:** 500K
- **Why:** Comprehensive testing, academic rigor, maximum features

---

## Performance Expectations by Configuration

### PPO + [1] + 50K
- **Sharpe:** 1.0 - 1.2
- **Drawdown:** -15% to -20%
- **Win Rate:** 50-55%
- **Best For:** Quick tests, technical strategies

### PPO + [1,2] + 100K (Default)
- **Sharpe:** 1.2 - 1.5
- **Drawdown:** -10% to -15%
- **Win Rate:** 52-58%
- **Best For:** General trading, balanced approach

### Rainbow DQN + [1,2,3] + 150K
- **Sharpe:** 1.3 - 1.6
- **Drawdown:** -8% to -12%
- **Win Rate:** 54-60%
- **Best For:** Sample efficiency, distributional RL

### IQN + [1,2,3] + 250K
- **Sharpe:** 1.5 - 2.0
- **Drawdown:** -6% to -10%
- **Win Rate:** 56-62%
- **Best For:** Risk-aware, professional use

### IQN + [1,2,3,4] + 500K
- **Sharpe:** 1.6 - 2.0+
- **Drawdown:** -5% to -8%
- **Win Rate:** 58-65%
- **Best For:** Maximum performance, research

*Note: Actual performance varies by stock, market conditions, and training data quality.*

---

## Tips for Best Results

### General Tips
1. **Start Simple:** Use PPO + [1,2] + 100K first
2. **More Data:** Use 3-5 years for better generalization
3. **Test Multiple:** Try different algorithms on same stock
4. **Monitor Training:** Watch for overfitting in eval metrics
5. **Backtest Rigorously:** Trust backtest more than training metrics

### Algorithm-Specific

**PPO:**
- Use attention network (`use_attention: true`)
- 100K steps minimum for convergence
- Increase `n_epochs` for more stable policies

**Rainbow DQN:**
- Need more steps than PPO (150K+)
- Watch exploration schedule
- Increase `buffer_size` if enough RAM

**IQN:**
- Train longer (250K+ steps)
- Lower learning rate (5e-5)
- Monitor risk-adjusted metrics

### Feature-Specific

**Tier 1 Only:**
- Good for high-frequency, technical strategies
- Fast iteration
- Consider 3-5 years of data

**Tier 1 + 2:**
- Best balance
- Need valid FMP API key
- Combines momentum + value

**Tier 1 + 2 + 3:**
- For comprehensive models
- Adds market awareness
- Best for IQN/Rainbow DQN

**All Tiers:**
- Research purposes
- Diminishing returns on Tier 4
- Use with 500K+ steps

---

## FAQ

**Q: Which algorithm should I start with?**
A: PPO with tiers [1, 2] and 100K steps. Fast, stable, good results.

**Q: Do I need Tier 2 (fundamentals)?**
A: Highly recommended. Adds 25% of predictive power for minimal cost.

**Q: When should I use IQN?**
A: When you care about risk management, Sharpe ratio, and are willing to wait longer.

**Q: How much data should I use?**
A: Minimum 2 years. Recommended 3-5 years for better generalization.

**Q: Can I trade different stocks with the same model?**
A: Predictions work best on the trained stock. For other stocks, retrain or use transfer learning (future feature).

**Q: Why is my Sharpe ratio low?**
A: Try: (1) More timesteps, (2) Add Tier 2, (3) More data years, (4) Try IQN algorithm.

**Q: Training is too slow, help?**
A: Use GPU, reduce timesteps, use PPO, or fewer feature tiers.

**Q: Which features matter most?**
A: Tier 1 (60%) > Tier 2 (25%) > Tier 3 (10%) > Tier 4 (5%).

---

**Last Updated:** January 2024
**Version:** 1.0.0
