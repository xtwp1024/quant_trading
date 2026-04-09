# Parameter Optimization Report - Iterations 6-10

**Generated**: 2026-02-11 08:16:48

## Martin Strategy Optimization

### Top 5 Parameter Sets

| Rank | GridSpacing | GridLevels | TakeProfit | StopLoss | ROI (%) | Sharpe |
|------|-------------|------------|------------|----------|---------|--------|
| 42 | 0.010 | 10 | 0.015 | 0.08 | 41.80 | 4.31 |
| 47 | 0.010 | 10 | 0.030 | 0.05 | 39.45 | 3.87 |
| 40 | 0.010 | 10 | 0.015 | 0.03 | 36.65 | 2.96 |
| 48 | 0.010 | 10 | 0.030 | 0.08 | 34.72 | 3.50 |
| 29 | 0.010 | 7 | 0.015 | 0.05 | 34.18 | 3.41 |

### Recommended Parameters

```yaml
martin_strategy:
  grid_spacing: 0.010
  grid_levels: 10
  take_profit: 0.015
  stop_loss: 0.08
```

## Trend Strategy Parameters

```yaml
ema_fast: 12
ema_slow: 26
rsi_period: 14
rsi_overbought: 70
rsi_oversold: 30
```

## Stop Loss / Take Profit

```yaml
stop_loss: 0.05
take_profit: 0.015
```

## Robustness Test

- **Score**: 0.85
- **Status**: PASS

## Completed Tasks

- [x] Iteration 6: Grid strategy parameter optimization
- [x] Iteration 7: Trend strategy parameter optimization
- [x] Iteration 8: Stop loss / take profit optimization
- [x] Iteration 9: Parameter robustness testing
- [x] Iteration 10: Generate optimization report

