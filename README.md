# 量化之神 (God of Quant Trading)

Unified quantitative trading system combining the best architectures from 5 quant trading projects.

## Architecture Overview

```
量化之神
├── core/                    # Core engine (from quant_v13)
│   ├── brain.py            # TitanBrain (RAG-based cognitive engine)
│   ├── cortex.py           # TitanCortex (Self-healing system)
│   ├── event_bus.py        # Event-driven architecture
│   └── ...
├── five_force/             # Five Force Cognitive Architecture (THE CORE DIFFERENTATOR)
│   ├── cognitive_unit.py  # Individual cognitive agent
│   ├── hive_mind.py        # Collective intelligence
│   ├── dream_engine.py     # Dream state processing
│   └── consensus.py        # Consensus mechanism
├── gene_lab/              # Gene Lab (Evolutionary strategy breeding)
│   ├── gene_bank.py       # Strategy gene storage
│   ├── strategy_breeder.py # Evolutionary algorithm
│   └── gene_mutator.py    # Gene mutation operators
├── strategy/              # Trading strategies
│   ├── advanced/         # Academic paper strategies
│   │   ├── hawkes_order_flow.py    # arXiv:2601.23172
│   │   ├── info_theory.py          # arXiv:2602.14575
│   │   ├── pa_amm.py               # arXiv:2602.09887
│   │   └── onchain_momentum.py
│   └── classic/          # Classic strategies
├── options/              # Options trading module
│   ├── pricing/
│   │   ├── black_scholes.py
│   │   └── greeks.py
│   ├── strategies.py     # Jade Lizard, Iron Condor, etc.
│   └── risk_manager.py
├── knowledge/           # Knowledge layer (from 量化交易 Node.js)
│   ├── loss_recovery.py  # Loss compound interest mechanism
│   ├── obsidian_integrator.py
│   └── sentiment_analyzer.py  # AI-powered sentiment analysis
├── optimization/        # Optimization module (from quant_trading_system_v13)
│   ├── kelly_calculator.py
│   ├── position_sizer.py
│   └── strategy_selector.py
├── risk/               # Risk management
│   └── manager.py
├── execution/          # Trade execution
│   └── executor.py
└── utils/              # Utilities
    └── logger.py
```

## Features

### Core Intelligence
- **TitanBrain**: RAG-based cognitive engine for decision making
- **TitanCortex**: Self-healing system for error recovery
- **EventBus**: Event-driven architecture for loose coupling
- **HiveMind**: Five Force collective intelligence model

### Strategy Modules
- **Hawkes Order Flow** (arXiv:2601.23172): Hawkes process for order flow modeling
- **Info Theory** (arXiv:2602.14575): Information-theoretic trading signals
- **PA-AMM** (arXiv:2602.09887): Prediction augment AMM strategy
- **Onchain Momentum**: On-chain data momentum indicators

### Options Trading
- Black-Scholes pricing model
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Options strategies: Jade Lizard, Iron Condor, Bull Call Spread, etc.
- Risk-managed options execution

### Knowledge Layer
- **Loss Recovery Manager**: Compound learning from losses
- **Obsidian Integrator**: Knowledge base synchronization
- **Sentiment Analyzer**: AI-powered market sentiment analysis

### Optimization
- Kelly Criterion calculator
- Dynamic position sizer
- Strategy selector with performance metrics

## Installation

```bash
# Clone the repository
cd /d/量化交易系统/量化之神

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Required Environment Variables

```env
# API Configuration
GATE_API_KEY=your_api_key
GATE_API_SECRET=your_api_secret

# AI API (for Sentiment Analyzer)
GLM_API_KEY=your_glm_api_key
GLM_API_URL=https://open.bigmodel.cn/api/paas/v4

# Database
DATABASE_URL=sqlite:///./data/trading.db

# Obsidian (optional)
OBSIDIAN_VAULT_PATH=D:/Obsidian Vault

# Logging
LOG_LEVEL=INFO
```

## Usage

### CLI Commands

```bash
# Show system info
python -m quant_trading.cli info

# Initialize core modules
python -m quant_trading.cli brain
python -m quant_trading.cli cortex
python -m quant_trading.cli hive-mind

# Run backtest
python -m quant_trading.cli backtest -s hawkes_order_flow

# Options pricing
python -m quant_trading.cli options --spot 100 --strike 105 --time-to-expiry 0.25 --volatility 0.2 --call

# Kelly criterion
python -m quant_trading.cli kelly --win-rate 0.55 --avg-win 100 --avg-loss 50

# Risk report
python -m quant_trading.cli risk

# Loss recovery analysis
python -m quant_trading.cli loss-recovery -d ./data

# Sentiment analysis
python -m quant_trading.cli sentiment --text "市场即将迎来大涨"
```

### Python API

```python
from quant_trading import (
    TitanBrain, TitanCortex, EventBus,
    HiveMind, GeneBank,
    HawkesOrderFlowStrategy,
    BlackScholes, GreeksCalculator,
    KellyCalculator, RiskManager
)

# Initialize core
brain = TitanBrain()
cortex = TitanCortex()
event_bus = EventBus()

# Initialize strategy
strategy = HawkesOrderFlowStrategy()

# Calculate options price
bs = BlackScholes()
price = bs.price(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")

# Calculate Kelly fraction
kelly = KellyCalculator()
fraction = kelly.calculate_kelly_fraction(win_rate=0.55, avg_win=100, avg_loss=50)

# Risk management
risk_mgr = RiskManager()
check = risk_mgr.check_trade_allowed(symbol="BTC_USDT", potential_loss=-50)
```

## Source Projects

This unified system merges the best features from:

1. **quant_v13** - Core cognitive architecture (Five Force, Gene Lab, TitanBrain)
2. **quant-trading-system** - Academic paper strategies (Hawkes, Info Theory, PA-AMM)
3. **AI_Trading_System** - Options trading (Black-Scholes, Greeks, strategies)
4. **量化交易 (Node.js)** - Knowledge layer (Loss Recovery, Obsidian, Sentiment)
5. **quant_trading_system_v13** - Optimization (Kelly, Position Sizer, Strategy Selector)

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black quant_trading/
isort quant_trading/

# Type checking
mypy quant_trading/

# Lint
ruff check quant_trading/
```

## License

MIT License
