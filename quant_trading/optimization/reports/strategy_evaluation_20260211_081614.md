# Strategy Evaluation Report - Iterations 1-5

**Generated**: 2026-02-11 08:16:14

## Summary

- **Total Strategies Tested**: 20
- **Qualified Strategies**: 9
- **Selected Strategies**: 5

## Top 10 Strategies

| Rank | Strategy | ROI (%) | Sharpe | MaxDD (%) | WinRate (%) | Trades |
|------|----------|---------|--------|-----------|-------------|--------|
| 1 | MartinBinance | 23.47 | 2.43 | 26.17 | 52.6 | 72 |
| 2 | DualThrust | 19.15 | 1.73 | 28.04 | 48.1 | 122 |
| 3 | MACDCross | 18.18 | 1.81 | 24.95 | 66.0 | 59 |
| 4 | WilliamsR | 17.45 | 1.73 | 21.24 | 44.5 | 81 |
| 5 | RSIDivergence | 15.34 | 1.55 | 23.49 | 57.3 | 143 |
| 6 | ParabolicSAR | 11.34 | 1.62 | 25.15 | 66.1 | 60 |
| 7 | SqueezeMomentum | 10.08 | 2.42 | 19.99 | 52.9 | 83 |
| 8 | WeightedGrid | 6.24 | 2.40 | 24.64 | 60.0 | 122 |
| 9 | BinanceAccessory | 4.00 | 1.75 | 27.01 | 53.7 | 37 |

## Final Portfolio Selection

1. **MartinBinance**
   - ROI: 23.47%
   - Sharpe: 2.43
   - MaxDD: 26.17%
   - WinRate: 52.6%

2. **MACDCross**
   - ROI: 18.18%
   - Sharpe: 1.81
   - MaxDD: 24.95%
   - WinRate: 66.0%

3. **WilliamsR**
   - ROI: 17.45%
   - Sharpe: 1.73
   - MaxDD: 21.24%
   - WinRate: 44.5%

4. **RSIDivergence**
   - ROI: 15.34%
   - Sharpe: 1.55
   - MaxDD: 23.49%
   - WinRate: 57.3%

5. **ParabolicSAR**
   - ROI: 11.34%
   - Sharpe: 1.62
   - MaxDD: 25.15%
   - WinRate: 66.1%

## Selection Criteria

- Min ROI: 0%
- Max Drawdown: <30%
- Min Sharpe: >1.0
- Min Trades: >10
- Max Correlation: <0.7

## Completed Tasks

- [x] Iteration 1: Batch backtest all strategies
- [x] Iteration 2: Strategy ranking and filtering
- [x] Iteration 3: Multi-period validation (simulated)
- [x] Iteration 4: Correlation analysis
- [x] Iteration 5: Generate evaluation report

