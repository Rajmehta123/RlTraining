# RL Trading Agent - Frontend Interface Guide

Complete guide to using the "Train your RL Agent Trader" interface.

---

## 📍 Accessing the Interface

1. Navigate to the sidebar
2. Click **"Train your RL Agent Trader"** (Brain icon 🧠)
3. Service status indicator shows connection to backend (green = connected)

---

## 🎛️ Training Tab - Configuration Panel

### Symbol Input
- **Location:** Top of configuration panel
- **Format:** Stock ticker (e.g., AAPL, TSLA, MSFT)
- **Validation:** Automatically converted to uppercase
- **Tip:** Use major stocks for best results (more data available)

### Algorithm Selection
- **Location:** Dropdown below symbol
- **Options:**
  - `PPO` - **Recommended for beginners**
  - `Rainbow DQN` - Sample-efficient
  - `IQN` - Risk-aware trading
- **Default:** PPO
- **Visual:** Badge shows selected algorithm during training

### Data Years Slider
- **Location:** Below algorithm selector
- **Range:** 2-10 years
- **Default:** 2 years
- **Minimum:** 2 years (required for technical indicators)
- **Display:** Shows current value (e.g., "2 years")
- **Tip:** More years = better generalization, longer download time

### Initial Capital
- **Location:** Currency input field
- **Range:** $1,000 - $1,000,000
- **Default:** $100,000
- **Format:** Displays with dollar sign
- **Purpose:** Starting portfolio value for backtesting

### Training Steps Slider
- **Location:** Mid-panel
- **Range:** 50,000 - 500,000 steps
- **Default:** 100,000 steps
- **Display:** Shows in thousands (e.g., "100K")
- **Helper Text:** "More steps = better training but longer time"
- **Quick Guide:**
  - 50K: Quick test (~2-3 min)
  - 100K: Default (~5-7 min)
  - 250K: High quality (~15-20 min)
  - 500K: Maximum (~30-40 min)

### Feature Tiers Selector
- **Location:** After training steps
- **Display:** 2×2 grid of tier cards
- **Interaction:** Click to toggle selection
- **Visual States:**
  - **Selected:** Blue border, blue indicator dot, blue background
  - **Unselected:** Gray border, gray indicator dot
  - **Disabled:** Grayed out during training

**Tier Cards:**

**Tier 1: Price/Technical**
- Label: "Tier 1"
- Description: "Price/Technical"
- Subdescription: "Returns, volatility, MAs, RSI, MACD"
- Features: 26 technical indicators
- Always recommended

**Tier 2: Fundamentals**
- Label: "Tier 2"
- Description: "Fundamentals"
- Subdescription: "PE, ROE, margins, earnings"
- Features: 11 fundamental metrics
- Requires FMP API key
- Highly recommended

**Tier 3: Market Context**
- Label: "Tier 3"
- Description: "Market Context"
- Subdescription: "Market regime, sector strength"
- Features: 4 market indicators
- Optional enhancement

**Tier 4: Alpha Signals**
- Label: "Tier 4"
- Description: "Alpha Signals"
- Subdescription: "Analyst revisions, DCF"
- Features: 2 alpha signals
- Advanced use

**Selection Counter:**
- Shows "X tier(s) selected"
- Warning if 0 tiers selected: "Select at least one tier"

### Use Attention Network (PPO only)
- **Location:** Below feature tiers
- **Type:** Toggle switch
- **Default:** ON
- **Visible:** Only when PPO algorithm selected
- **Purpose:** Enables attention mechanism for pattern recognition
- **Recommendation:** Keep ON for better performance

### Action Buttons

**Start Training Button**
- **State - Ready:**
  - Green button
  - Text: "Start Training" with Play icon
  - Enabled when: Service available + Symbol entered + Tiers selected
- **State - Training:**
  - Red button
  - Text: "Stop Training" with Square icon
  - Click to send stop signal (best effort)

**Disabled Conditions:**
- Service not available (backend down)
- No symbol entered
- No feature tiers selected

---

## 📊 Training Progress Panel

### Header
- **Title:** "Training Progress"
- **Algorithm Badge:** Shows selected algorithm (PPO / Rainbow DQN / IQN)
- **Status:** Shows current state and symbol being trained

### Progress Bar
- **Top Line:** Status (colored) and percentage
- **Colors:**
  - 🟡 Yellow: Initializing, loading, preparing
  - 🔵 Blue: Training
  - 🟢 Green: Completed
  - 🔴 Red: Failed
  - ⚪ Gray: Stopped

### Status Phases
1. **Initializing (0-5%)** - Job created
2. **Loading Data (5-15%)** - Downloading from yfinance + FMP
3. **Preparing Environment (15-20%)** - Creating Gym environment
4. **Building Model (20%)** - Initializing neural network
5. **Training (20-85%)** - Active learning
6. **Backtesting (85-95%)** - Running evaluation
7. **Saving (95-100%)** - Saving model files
8. **Completed (100%)** - Done!

### Data Summary (appears after data loaded)
Four-column grid showing:
- **Start Date:** First date of data
- **End Date:** Last date of data
- **Data Points:** Total rows
- **Features:** Number of features extracted

### TQDM-Style Progress Display

**Format:**
```
[45000/100000 | 02:03 < 02:35 | 374 fps | it: 702]
```

**Breakdown:**
- `45000/100000` - Current step / Total steps
- `02:03` - Elapsed time (HH:MM:SS or MM:SS)
- `< 02:35` - Estimated time remaining
- `374 fps` - Frames per second (training speed)
- `it: 702` - Iterations completed

### Metrics Panels (during training)

**📊 Rollout Metrics** (PPO)
- Episode Reward Mean
- Episode Length Mean
- Shows average episode performance

**⏱️ Time Metrics** (PPO)
- FPS (Frames Per Second)
- Iterations
- Total Timesteps
- Time Elapsed

**🎓 Train Metrics** (PPO)
- Loss
- Policy Loss
- Value Loss
- Entropy Loss
- Approx KL
- Clip Fraction
- Shows training dynamics

**🎯 Evaluation Metrics** (Custom)
- Mean Return (%)
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)
- Updated every eval checkpoint

### Error Display
- **Red Alert Box** at top of panel
- **Icon:** Alert circle
- **Message:** Detailed error description
- **Action:** Check error, fix issue, restart training

---

## 📈 Backtest Results Tab

### When Available
- Unlocked after training completes (100%)
- Shows comprehensive backtest metrics
- Includes comparison to baseline strategies

### Backtest Period Alert
**Blue info box** at top shows:
```
📅 Jan 15, 2023 → Jan 15, 2024
📊 365 days (52 weeks, 12 months)
💼 87 trades executed
⏱️ Test period: 15% of total data
```

### Performance Metrics Table

**Three-Strategy Comparison:**
1. **RL Agent** - Your trained model
2. **Buy & Hold** - Baseline buy and hold
3. **SMA Crossover** - Technical baseline

**Columns:**
- Total Return (%)
- Annualized Return (%)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown (%)
- Volatility (%)
- Win Rate (%)
- Total Trades
- Final Value ($)

**Color Coding:**
- 🟢 Green: Positive values (returns, ratios)
- 🔴 Red: Negative values (drawdown)
- Best in each column highlighted

### Visualization Plots

**1. Cumulative Returns**
- Line chart comparing all strategies
- X-axis: Date
- Y-axis: Cumulative return (%)
- Shows compounding effect
- Legend: RL Agent, Buy & Hold, SMA Cross

**2. Drawdown Chart**
- Area chart showing underwater periods
- X-axis: Date
- Y-axis: Drawdown from peak (%)
- Shows max drawdown visually
- Darker = deeper drawdown

**3. Returns Distribution**
- Histogram of daily returns
- X-axis: Return (%)
- Y-axis: Frequency
- Shows return distribution shape
- Identifies skewness/outliers

**4. Rolling Sharpe Ratio**
- Line chart of 20-day rolling Sharpe
- X-axis: Date
- Y-axis: Sharpe ratio
- Shows consistency over time
- Highlights stable periods

**5. Portfolio Value**
- Line chart of portfolio value over time
- X-axis: Date
- Y-axis: Portfolio value ($)
- Shows growth trajectory
- All strategies included

**6. Metrics Heatmap** (optional)
- Color-coded comparison matrix
- Rows: Strategies
- Columns: Metrics
- Green = good, Red = bad

### Image Display
- Full-width responsive images
- Base64 encoded PNG
- Zoom on click (optional)
- Download button for each plot

---

## 📋 Trade History Tab

### Trade Log Table

**Columns:**
1. **Step** - Environment step number
2. **Date** - Trade execution date (YYYY-MM-DD)
3. **Action** - BUY, SELL, STRONG BUY, STRONG SELL
4. **Price** - Execution price ($)
5. **Position** - Position size (0.0 - 1.0)
6. **P&L** - Profit/Loss from trade ($)
7. **Portfolio** - Portfolio value after trade ($)

**Visual:**
- Clean table with alternating row colors
- Scrollable (for many trades)
- Sortable by column (click header)
- Color-coded actions:
  - 🟢 BUY, STRONG BUY (green text)
  - 🔴 SELL, STRONG SELL (red text)
  - ⚪ HOLD (gray text)

**Features:**
- Shows all trades executed during backtest
- Includes entry/exit points
- Tracks position changes
- Running portfolio value

**Empty State:**
- "No trades to display"
- Shows when backtest not run yet

---

## 💾 Saved Models Tab

### Model List

**Card Layout:**
Each saved model shown as a card with:
- **Header:** Symbol - Date/Time
- **Algorithm Badge:** PPO / Rainbow DQN / IQN
- **Metrics Summary:**
  - Total Return (%)
  - Sharpe Ratio
  - Max Drawdown (%)
  - Win Rate (%)
- **Action Buttons:**
  - 🔮 **Predict** - Use model for predictions
  - 🗑️ **Delete** - Remove model

**Sorting:**
- Newest first (by creation date)
- Can filter by symbol (future feature)

### Prediction Panel

**Shown after clicking "Predict":**

**Inputs:**
- Symbol input (can be different from training symbol)
- Default: Training symbol
- "Get Prediction" button

**Output Card:**
- **Symbol:** Stock predicted
- **Recommendation:** STRONG SELL → STRONG BUY
  - Visual: Color-coded badge
  - 🔴 STRONG SELL
  - 🟠 SELL
  - 🟡 HOLD
  - 🟢 BUY
  - 🟢🟢 STRONG BUY
- **Action Value:** 0-4 (numeric)
- **Position:** Recommended position size
- **Predicted Return:** Expected return (%)
- **Metrics Summary:**
  - Total return from simulation
  - Sharpe ratio
  - Max drawdown

**Loading State:**
- Spinner while fetching prediction
- "Generating prediction..." text

**Error State:**
- Red alert box
- Error message
- "Try again" button

---

## 🎨 UI Elements & Interactions

### Service Status Indicator
- **Location:** Top-right of page header
- **States:**
  - 🟢 Green dot: Service available
  - 🔴 Red dot: Service unavailable
  - 🟡 Yellow dot: Checking...
- **Auto-refresh:** Every 30 seconds

### Real-time Updates
- Progress updates every 2 seconds during training
- Smooth progress bar animation
- Live metric updates
- No page refresh needed

### Responsive Design
- Mobile-friendly layout
- Tablet optimized
- Desktop full-featured
- Grid layout adjusts to screen size

### Loading States
- Spinner for async operations
- Skeleton loaders for data fetching
- Progress bars for training
- Disabled buttons during operations

### Color Scheme
- **Primary:** Purple/Blue gradient
- **Success:** Green (#10b981)
- **Warning:** Yellow (#eab308)
- **Error:** Red (#ef4444)
- **Muted:** Gray (#6b7280)

### Typography
- **Headers:** Bold, large
- **Metrics:** Monospace font
- **Descriptions:** Regular weight
- **Numbers:** Tabular figures

---

## ⌨️ Keyboard Shortcuts (future feature)

- `Ctrl/Cmd + Enter` - Start training
- `Escape` - Stop training
- `Tab` - Navigate through inputs
- `Space` - Toggle tier selection

---

## 📱 Mobile Experience

### Optimizations
- Vertical layout for narrow screens
- Swipeable tabs
- Touch-friendly buttons
- Collapsible panels
- Optimized table scrolling

### Limitations
- Charts may be smaller
- Table horizontal scroll
- Some metrics stacked vertically

---

## 🎯 Best Practices for UI

### Before Training
1. ✅ Check service status (green dot)
2. ✅ Enter valid stock symbol
3. ✅ Select appropriate algorithm
4. ✅ Choose at least Tier 1 + 2
5. ✅ Set reasonable training steps (100K+)

### During Training
1. 👀 Monitor progress percentage
2. 👀 Watch Sharpe ratio in eval metrics
3. 👀 Check FPS (should be >100)
4. 👀 Ensure eval metrics improving
5. ⚠️ Don't close browser tab

### After Training
1. 📊 Review all metrics in table
2. 📈 Check cumulative returns plot
3. 📉 Verify max drawdown acceptable
4. 📋 Inspect trade history
5. 💾 Save model if satisfied

---

## 🐛 Common UI Issues

### "Service Unavailable" (Red Dot)
**Problem:** Backend not running
**Solution:**
1. Start backend: `python app.py`
2. Check port 5001 is free
3. Verify no firewall blocking

### Training Progress Stuck
**Problem:** UI not updating
**Solution:**
1. Check browser console for errors
2. Verify network tab shows requests
3. Refresh page (training continues in backend)
4. Check backend logs

### Charts Not Displaying
**Problem:** Base64 image error
**Solution:**
1. Check browser console
2. Verify backtest completed (100%)
3. Refresh results tab
4. Clear browser cache

### Table Empty After Training
**Problem:** Results not loaded
**Solution:**
1. Wait for 100% completion
2. Switch to different tab and back
3. Click "Backtest Results" tab again
4. Check for error alerts

### Tiers Not Selecting
**Problem:** Click not registering
**Solution:**
1. Ensure not currently training
2. Click directly on tier card
3. Check disabled state
4. Refresh page

---

## 💡 UI Tips & Tricks

1. **Watch the FPS** - Higher FPS = faster training
2. **Eval Metrics** - If Sharpe decreasing, may be overfitting
3. **Progress Bar** - Color indicates status (blue = training)
4. **Trade History** - Sort by date to see chronological order
5. **Model Cards** - Newer models appear first
6. **Prediction** - Try on different symbols (experimental)
7. **Comparison Table** - Best strategy for each metric auto-highlighted
8. **Plots** - Hover for exact values (tooltip)
9. **Mobile** - Rotate to landscape for better chart view
10. **Browser** - Chrome/Firefox recommended for best experience

---

## 🎓 Learning Path

### Beginner (Day 1)
1. Train first model (AAPL, PPO, [1,2], 100K)
2. Watch progress panel
3. View backtest results
4. Check trade history
5. Save model

### Intermediate (Week 1)
1. Try different symbols
2. Compare PPO vs Rainbow DQN
3. Experiment with tiers
4. Analyze metrics table
5. Optimize timesteps

### Advanced (Month 1)
1. Master all algorithms
2. Optimize feature combinations
3. Understand metric trade-offs
4. Build model portfolio
5. Use for actual trading ideas

---

## 📸 Screenshots & Examples

### Training Panel Example
```
┌─────────────────────────────────────────┐
│ Configuration                           │
├─────────────────────────────────────────┤
│ Symbol: AAPL                            │
│ Algorithm: ● PPO ○ Rainbow DQN ○ IQN   │
│ Data Years: ▬▬▬▬○────────── 2 years    │
│ Initial Capital: $100,000               │
│ Training Steps: ▬▬▬▬▬○───── 100K       │
│                                         │
│ Feature Tiers:                          │
│ ┌──────────┐ ┌──────────┐              │
│ │● Tier 1  │ │● Tier 2  │              │
│ │Technical │ │Fundament.│              │
│ └──────────┘ └──────────┘              │
│ ┌──────────┐ ┌──────────┐              │
│ │○ Tier 3  │ │○ Tier 4  │              │
│ │ Market   │ │  Alpha   │              │
│ └──────────┘ └──────────┘              │
│ 2 tiers selected                        │
│                                         │
│ Use Attention: [ON]                     │
│                                         │
│ ┌─────────────────────┐                │
│ │  ▶ Start Training   │                │
│ └─────────────────────┘                │
└─────────────────────────────────────────┘
```

### Progress Display Example
```
TRAINING PROGRESS (45%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45%

[45000/100000 | 02:03 < 02:35 | 374 fps | it: 702]

┌─ Rollout ─────────────────┐
│ Ep Reward: 156.2          │
│ Ep Length: 245            │
└───────────────────────────┘

┌─ Evaluation ──────────────┐
│ Return: 2.34%             │
│ Sharpe: 1.45              │
│ Drawdown: -8.23%          │
│ Win Rate: 58.3%           │
└───────────────────────────┘
```

---

**Last Updated:** January 2024
**Version:** 1.0.0
