import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("DreamEngine")

class DreamEngine:
    """
    The Dream Engine: Generates synthetic 'Market Dreams' using Stochastic Differential Equations (SDEs).
    Used to train Cognitive Units on counterfactual histories (What COULD have happened).
    """
    def __init__(self, initial_price=2000.0, volatility=0.2, drift=0.0):
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        
    def weave_dream(self, duration_days=30, resolution_mins=15):
        """
        Synthesize a parallel market reality.
        Uses Geometric Brownian Motion (GBM) + Poission Jumps.
        """
        logger.info(f"🛌 Weaving a {duration_days}-day Dream Sequence...")
        
        # Parameters
        dt = (resolution_mins / (24 * 60)) / 365 # Annualized time step
        steps = int((duration_days * 24 * 60) / resolution_mins)
        
        # GBM Model: dS = S * (mu * dt + sigma * dW)
        prices = [self.initial_price]
        timestamps = pd.date_range(start="2025-01-01", periods=steps, freq=f"{resolution_mins}min")
        
        for _ in range(1, steps):
            prev_price = prices[-1]
            shock = np.random.normal(0, 1)
            
            # Jump Diffusion (Sudden events)
            jump = 0
            if np.random.random() > 0.995: # 0.5% chance of black swan shock
                jump_magnitude = np.random.normal(-0.05, 0.02) # -5% crash typically
                jump = jump_magnitude * prev_price
                
            # GBM Formula
            change = prev_price * (self.drift * dt + self.volatility * np.sqrt(dt) * shock) + jump
            new_price = max(0.01, prev_price + change)
            prices.append(new_price)
            
        # Structure as OHLCV DataFrame
        df = pd.DataFrame(index=timestamps)
        df['close'] = prices
        # Synthetic OHLC
        noise = self.volatility * 0.01 * np.array(prices)
        df['open'] = df['close'].shift(1).fillna(prices[0]) + np.random.normal(0, noise.mean(), size=steps)
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, noise.mean(), size=steps))
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, noise.mean(), size=steps))
        df['volume'] = np.random.randint(100, 5000, size=steps) * (1 + np.abs(np.diff(prices, prepend=prices[0]))) # Vol follows volatility
        
        logger.info("💤 Dream Woven. Reality constructed.")
        return df

    def run_backtest(self, dream_df: pd.DataFrame, signals: list = None) -> dict:
        """
        在梦境数据上运行回测

        Args:
            dream_df: 梦境DataFrame，需要包含 close, open, high, low, volume 列
            signals: 信号列表，如果为None则使用简单MA策略

        Returns:
            回测结果字典
        """
        if 'close' not in dream_df.columns:
            logger.error("dream_df缺少close列")
            return {}

        returns = dream_df['close'].pct_change().fillna(0)

        if signals is None:
            # 简单MA策略: 价格>MA20买入，<MA20卖出
            ma20 = dream_df['close'].rolling(20).mean()
            signals = []
            position = 0
            for i in range(len(dream_df)):
                if pd.isna(ma20.iloc[i]):
                    signals.append("HOLD")
                elif dream_df['close'].iloc[i] > ma20.iloc[i]:
                    signals.append("BUY")
                else:
                    signals.append("SELL")
        else:
            # 确保signals长度匹配
            if len(signals) < len(dream_df):
                signals = list(signals) + ["HOLD"] * (len(dream_df) - len(signals))
            elif len(signals) > len(dream_df):
                signals = signals[:len(dream_df)]

        # 计算策略收益
        strategy_returns = []
        position = 0

        for i in range(len(signals)):
            signal = signals[i]

            if signal == "BUY" and position == 0:
                position = 1
            elif signal == "SELL" and position == 0:
                position = -1
            elif signal == "HOLD":
                pass
            else:
                position = 0

            if position == 1:
                strategy_returns.append(returns.iloc[i])
            elif position == -1:
                strategy_returns.append(-returns.iloc[i])
            else:
                strategy_returns.append(0)

        strategy_returns = np.array(strategy_returns)

        # 计算指标
        total_return = (1 + strategy_returns).prod() - 1
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe = mean_return / std_return * np.sqrt(252 * 24) if std_return > 0 else 0

        # 最大回撤
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 胜率
        win_trades = sum(1 for r in strategy_returns if r > 0)
        total_trades = sum(1 for s in signals if s in ["BUY", "SELL"])
        win_rate = win_trades / max(total_trades, 1)

        result = {
            "total_return": total_return * 100,  # 百分比
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown * 100,  # 百分比
            "win_rate": win_rate * 100,  # 百分比
            "total_trades": total_trades,
            "avg_return_per_trade": mean_return * 100 if total_trades > 0 else 0,
        }

        logger.info(f"📊 回测完成: 总收益={result['total_return']:.2f}%, 夏普={sharpe:.2f}, 最大回撤={max_drawdown*100:.2f}%")
        return result

if __name__ == "__main__":
    # Test Dream
    engine = DreamEngine()
    dream_world = engine.weave_dream()
    print(dream_world.head())
    print(dream_world.tail())
