"""
Maverick-style risk management tools for position sizing, stop loss calculation, and portfolio risk analysis.
Adapted from maverick-mcp risk_management.py for use with generic pandas DataFrames.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) indicator.

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)

    Returns:
        ATR Series
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range = max(H - L, |H - PC|, |L - PC|)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_ma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        series: Price series
        period: MA period

    Returns:
        MA Series
    """
    return series.rolling(window=period).mean()


def calculate_swing_low(prices: pd.Series, lookback: int) -> float:
    """
    Calculate swing low (lowest low in lookback period).

    Args:
        prices: Price series
        lookback: Number of periods to look back

    Returns:
        Swing low value
    """
    return float(prices.iloc[-lookback:].min())


def calculate_support_level(lows: pd.Series, lookback: int) -> float:
    """
    Calculate support level (rolling minimum).

    Args:
        lows: Low price series
        lookback: Lookback period

    Returns:
        Support level
    """
    return float(lows.rolling(window=lookback).min().iloc[-1])


class PositionSizeTool:
    """
    Calculate position size based on risk management rules with Kelly Criterion support.
    """

    def __init__(self):
        """Initialize position size tool."""
        pass

    def kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_size: float,
        risk_percentage: float = 2.0
    ) -> dict:
        """
        Calculate position size using Kelly Criterion formula.

        Kelly formula: f* = (bp - q) / b
        where b = odds (avg_win / avg_loss), p = win_rate, q = 1 - p

        Args:
            win_rate: Win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive value)
            account_size: Total account size
            risk_percentage: Percentage of account to risk (default 2%)

        Returns:
            Dictionary with position sizing results
        """
        if avg_loss == 0:
            return {"error": "Average loss cannot be zero"}

        # Calculate Kelly fraction
        b = avg_win / avg_loss  # odds
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0.0, min(1.0, kelly_fraction))  # Bound to 0-1

        # Risk-based position sizing
        risk_amount = account_size * (risk_percentage / 100)

        # Calculate shares based on risk amount and avg_loss
        risk_per_share = avg_loss
        base_shares = risk_amount / risk_per_share

        # Apply Kelly fraction
        kelly_shares = base_shares * kelly_fraction

        # Conservative multipliers
        half_kelly = kelly_shares * 0.5
        quarter_kelly = kelly_shares * 0.25

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "odds": b,
            "kelly_fraction": round(kelly_fraction, 4),
            "half_kelly_fraction": round(kelly_fraction * 0.5, 4),
            "quarter_kelly_fraction": round(kelly_fraction * 0.25, 4),
            "risk_amount": round(risk_amount, 2),
            "base_shares": int(base_shares),
            "kelly_shares": int(kelly_shares),
            "half_kelly_shares": int(half_kelly),
            "quarter_kelly_shares": int(quarter_kelly),
            "recommended_shares": int(half_kelly),  # Conservative recommendation
        }

    def persona_adjusted_size(
        self,
        base_shares: int,
        persona: str = "moderate"
    ) -> dict:
        """
        Adjust position size based on investor persona/risk tolerance.

        Args:
            base_shares: Base number of shares from Kelly calculation
            persona: Risk persona ("conservative", "moderate", "aggressive")

        Returns:
            Dictionary with adjusted position sizing
        """
        persona_multipliers = {
            "conservative": 0.25,
            "moderate": 0.5,
            "aggressive": 0.75,
            "day_trader": 1.0
        }

        multiplier = persona_multipliers.get(persona.lower(), 0.5)

        return {
            "persona": persona,
            "base_shares": base_shares,
            "adjusted_shares": int(base_shares * multiplier),
            "multiplier": multiplier
        }

    def calculate_position_size(
        self,
        account_size: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percentage: float = 2.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> dict:
        """
        Calculate position size based on account risk and optional Kelly parameters.

        Args:
            account_size: Total account size in dollars
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_percentage: Percentage of account to risk (default 2%)
            win_rate: Optional win rate for Kelly calculation
            avg_win: Optional average win for Kelly calculation
            avg_loss: Optional average loss for Kelly calculation

        Returns:
            Dictionary with position sizing results
        """
        # Basic risk calculation
        risk_amount = account_size * (risk_percentage / 100)
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return {"error": "Entry and stop loss prices cannot be the same"}

        # Calculate base position size from price risk
        base_shares = risk_amount / price_risk
        base_position_value = base_shares * entry_price

        result = {
            "account_size": account_size,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "risk_percentage": risk_percentage,
            "risk_amount": round(risk_amount, 2),
            "price_risk_per_share": round(price_risk, 2),
            "base_shares": int(base_shares),
            "base_position_value": round(base_position_value, 2),
            "position_percentage": round((base_position_value / account_size) * 100, 2),
        }

        # Add Kelly-based calculation if parameters provided
        if all(x is not None for x in [win_rate, avg_win, avg_loss]):
            kelly_result = self.kelly_position_size(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                account_size=account_size,
                risk_percentage=risk_percentage
            )
            result["kelly"] = kelly_result
            # Use Kelly shares as recommended if available
            if "recommended_shares" in kelly_result:
                result["recommended_shares"] = kelly_result["recommended_shares"]

        return result


class TechnicalStopsTool:
    """
    Calculate stop loss levels based on technical analysis (ATR, support, MA).
    """

    def __init__(self):
        """Initialize technical stops tool."""
        pass

    def calculate_atr_stop(
        self,
        df: pd.DataFrame,
        atr_multiplier: float = 2.0,
        period: int = 14,
        side: str = "long"
    ) -> dict:
        """
        Calculate ATR-based stop loss.

        Args:
            df: DataFrame with High, Low, Close columns
            atr_multiplier: ATR multiplier for stop distance
            period: ATR period
            side: "long" or "short" position

        Returns:
            Dictionary with ATR stop level
        """
        if len(df) < period:
            return {"error": f"Need at least {period} periods for ATR calculation"}

        current_price = float(df['Close'].iloc[-1])
        atr = calculate_atr(df, period=period)
        atr_value = float(atr.iloc[-1])

        if side == "long":
            atr_stop = current_price - (atr_value * atr_multiplier)
        else:
            atr_stop = current_price + (atr_value * atr_multiplier)

        stop_distance_pct = (atr_value * atr_multiplier / current_price) * 100

        return {
            "current_price": round(current_price, 2),
            "atr_value": round(atr_value, 2),
            "atr_multiplier": atr_multiplier,
            "stop_level": round(atr_stop, 2),
            "stop_distance_pct": round(stop_distance_pct, 2),
            "side": side
        }

    def calculate_support_stop(
        self,
        df: pd.DataFrame,
        lookback_days: int = 20,
        side: str = "long"
    ) -> dict:
        """
        Calculate support-based stop loss.

        Args:
            df: DataFrame with Low column
            lookback_days: Days to look back for support
            side: "long" or "short" position

        Returns:
            Dictionary with support stop level
        """
        current_price = float(df['Close'].iloc[-1])
        support_level = calculate_support_level(df['Low'], lookback_days)

        if side == "long":
            support_stop = support_level
        else:
            support_stop = support_level * 2 - current_price  # Resistance for short

        stop_distance_pct = ((current_price - support_stop) / current_price) * 100

        return {
            "current_price": round(current_price, 2),
            "support_level": round(support_level, 2),
            "lookback_days": lookback_days,
            "stop_level": round(support_stop, 2),
            "stop_distance_pct": round(abs(stop_distance_pct), 2),
            "side": side
        }

    def calculate_swing_low_stop(
        self,
        df: pd.DataFrame,
        lookback_days: int = 20,
        buffer_pct: float = 0.0,
        side: str = "long"
    ) -> dict:
        """
        Calculate swing low-based stop loss.

        Args:
            df: DataFrame with Low column
            lookback_days: Days to look back for swing low
            buffer_pct: Optional buffer percentage to add to stop
            side: "long" or "short" position

        Returns:
            Dictionary with swing low stop level
        """
        current_price = float(df['Close'].iloc[-1])
        swing_low = calculate_swing_low(df['Low'], lookback_days)

        if side == "long":
            swing_stop = swing_low * (1 - buffer_pct / 100) if buffer_pct > 0 else swing_low
        else:
            swing_high = float(df['High'].iloc[-lookback_days:].max())
            swing_stop = swing_high * (1 + buffer_pct / 100) if buffer_pct > 0 else swing_high

        stop_distance_pct = ((current_price - swing_stop) / current_price) * 100

        return {
            "current_price": round(current_price, 2),
            "swing_low": round(swing_low, 2) if side == "long" else None,
            "swing_high": round(swing_low, 2) if side == "short" else None,
            "lookback_days": lookback_days,
            "buffer_pct": buffer_pct,
            "stop_level": round(swing_stop, 2),
            "stop_distance_pct": round(abs(stop_distance_pct), 2),
            "side": side
        }

    def calculate_ma_stop(
        self,
        df: pd.DataFrame,
        ma_period: int = 20,
        side: str = "long",
        use_ma_as_stop: bool = True
    ) -> dict:
        """
        Calculate MA-based trailing stop.

        Args:
            df: DataFrame with Close column
            ma_period: MA period
            side: "long" or "short" position
            use_ma_as_stop: If True, use MA as stop level; if False, calculate offset

        Returns:
            Dictionary with MA stop level
        """
        current_price = float(df['Close'].iloc[-1])
        ma_value = float(calculate_ma(df['Close'], ma_period).iloc[-1])

        if use_ma_as_stop:
            ma_stop = ma_value
        else:
            # Price has moved away from MA
            if side == "long":
                ma_stop = ma_value  # Trail below MA
            else:
                ma_stop = ma_value  # Trail above MA

        stop_distance_pct = ((current_price - ma_stop) / current_price) * 100

        return {
            "current_price": round(current_price, 2),
            "ma_period": ma_period,
            "ma_value": round(ma_value, 2),
            "stop_level": round(ma_stop, 2),
            "stop_distance_pct": round(abs(stop_distance_pct), 2),
            "side": side
        }

    def calculate_all_stops(
        self,
        df: pd.DataFrame,
        atr_multiplier: float = 2.0,
        lookback_days: int = 20,
        side: str = "long"
    ) -> dict:
        """
        Calculate all technical stop levels.

        Args:
            df: DataFrame with High, Low, Close columns
            atr_multiplier: ATR multiplier for stop distance
            lookback_days: Days to look back for support/swing
            side: "long" or "short" position

        Returns:
            Dictionary with all calculated stops
        """
        atr_stop = self.calculate_atr_stop(df, atr_multiplier, side=side)
        support_stop = self.calculate_support_stop(df, lookback_days, side=side)
        swing_stop = self.calculate_swing_low_stop(df, lookback_days, side=side)
        ma_stop = self.calculate_ma_stop(df, 20, side=side)  # 20-period MA
        ma50_stop = self.calculate_ma_stop(df, 50, side=side) if len(df) >= 50 else None

        # Recommend best stop based on side
        if side == "long":
            # For long: use tightest reasonable stop
            stops = [atr_stop, support_stop, swing_stop, ma_stop]
            if ma50_stop:
                stops.append(ma50_stop)
            recommended = min(s['stop_level'] for s in stops if 'stop_level' in s)
        else:
            # For short: use highest stop
            stops = [atr_stop, support_stop, swing_stop, ma_stop]
            if ma50_stop:
                stops.append(ma50_stop)
            recommended = max(s['stop_level'] for s in stops if 'stop_level' in s)

        current_price = float(df['Close'].iloc[-1])

        return {
            "current_price": current_price,
            "atr_stop": atr_stop,
            "support_stop": support_stop,
            "swing_low_stop": swing_stop,
            "ma_20_stop": ma_stop,
            "ma_50_stop": ma50_stop,
            "recommended_stop": round(recommended, 2),
            "recommended_stop_pct": round(abs((current_price - recommended) / current_price) * 100, 2),
            "side": side
        }


class RiskMetricsTool:
    """
    Calculate portfolio risk metrics including correlations, beta, and VaR.
    """

    def __init__(self):
        """Initialize risk metrics tool."""
        pass

    def calculate_portfolio_risk(
        self,
        returns_df: pd.DataFrame,
        weights: Optional[list[float]] = None
    ) -> dict:
        """
        Calculate portfolio risk metrics.

        Args:
            returns_df: DataFrame of returns (columns = assets, index = dates)
            weights: Optional portfolio weights (equal weight if not provided)

        Returns:
            Dictionary with portfolio risk metrics
        """
        if returns_df.empty:
            return {"error": "Empty returns DataFrame"}

        # Normalize weights
        n_assets = len(returns_df.columns)
        if weights is None:
            weights = [1.0 / n_assets] * n_assets
        else:
            weights = list(np.array(weights) / np.sum(weights))

        weights = weights[:n_assets]  # Match asset count

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Annualized volatility
        portfolio_std = portfolio_returns.std() * np.sqrt(252)

        # VaR (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)

        # CVaR (Expected Shortfall)
        var_95_value = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95_value].mean() * np.sqrt(252)

        return {
            "annualized_volatility": round(portfolio_std * 100, 2),
            "value_at_risk_95": round(var_95 * 100, 2),
            "conditional_var_95": round(cvar_95 * 100, 2) if not np.isnan(cvar_95) else None,
            "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, weights)}
        }

    def correlation_analysis(
        self,
        returns_df: pd.DataFrame
    ) -> dict:
        """
        Perform correlation analysis on returns.

        Args:
            returns_df: DataFrame of returns (columns = assets, index = dates)

        Returns:
            Dictionary with correlation matrix and summary stats
        """
        if returns_df.empty:
            return {"error": "Empty returns DataFrame"}

        corr_matrix = returns_df.corr()

        # Get upper triangle correlations (excluding diagonal)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        corr_values = upper_tri.stack().values
        avg_correlation = float(np.mean(corr_values)) if len(corr_values) > 0 else 0.0
        max_correlation = float(np.max(corr_values)) if len(corr_values) > 0 else 0.0
        min_correlation = float(np.min(corr_values)) if len(corr_values) > 0 else 0.0

        # Find highest correlation pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_pairs.append({
                    "asset1": corr_matrix.columns[i],
                    "asset2": corr_matrix.columns[j],
                    "correlation": round(corr_matrix.iloc[i, j], 4)
                })

        corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)

        return {
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "average_correlation": round(avg_correlation, 4),
            "max_correlation": round(max_correlation, 4),
            "min_correlation": round(min_correlation, 4),
            "highest_correlation_pairs": corr_pairs[:5] if corr_pairs else []
        }

    def beta_calculation(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> dict:
        """
        Calculate beta of an asset vs benchmark.

        Beta = Cov(asset, benchmark) / Var(benchmark)

        Args:
            asset_returns: Asset return series
            benchmark_returns: Benchmark return series

        Returns:
            Dictionary with beta and related metrics
        """
        # Align dates
        common_idx = asset_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) == 0:
            return {"error": "No common dates between asset and benchmark"}

        asset_aligned = asset_returns[common_idx]
        benchmark_aligned = benchmark_returns[common_idx]

        # Calculate covariance and variance
        covariance = asset_aligned.cov(benchmark_aligned)
        benchmark_var = benchmark_aligned.var()

        if benchmark_var == 0:
            return {"error": "Benchmark variance is zero"}

        beta = covariance / benchmark_var

        # Calculate correlation
        correlation = asset_aligned.corr(benchmark_aligned)

        # Calculate alpha (annualized)
        asset_mean = asset_aligned.mean()
        benchmark_mean = benchmark_aligned.mean()
        alpha = (asset_mean - beta * benchmark_mean) * 252

        return {
            "beta": round(beta, 4),
            "correlation": round(correlation, 4),
            "alpha_annualized": round(alpha, 4),
            "covariance": round(covariance, 6),
            "benchmark_variance": round(benchmark_var, 6),
            "n_observations": len(common_idx)
        }

    def diversification_score(
        self,
        returns_df: pd.DataFrame,
        weights: Optional[list[float]] = None
    ) -> dict:
        """
        Calculate portfolio diversification score.

        Score based on:
        - Average correlation (lower = more diversified)
        - Number of assets
        - Weight concentration (HHI)

        Args:
            returns_df: DataFrame of returns
            weights: Optional portfolio weights

        Returns:
            Dictionary with diversification metrics
        """
        n_assets = len(returns_df.columns)

        if weights is None:
            weights = [1.0 / n_assets] * n_assets
        weights = np.array(weights[:n_assets])
        weights = weights / weights.sum()

        # Herfindahl-HHI concentration index
        hhi = np.sum(weights ** 2)

        # Average correlation
        corr_matrix = returns_df.corr()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper_tri.stack().mean()

        # Diversification ratio (weighted avg vol / portfolio vol)
        vols = returns_df.std() * np.sqrt(252)
        weighted_avg_vol = np.sum(weights * vols)
        portfolio_vol = self.calculate_portfolio_risk(returns_df, weights.tolist())["annualized_volatility"] / 100

        if portfolio_vol > 0:
            div_ratio = weighted_avg_vol / portfolio_vol
        else:
            div_ratio = 1.0

        # Overall diversification score (0-1, higher = more diversified)
        corr_score = 1 - min(1.0, max(0.0, (avg_corr + 1) / 2))  # Convert corr -1,1 to score 1,0
        conc_score = 1 - hhi  # HHI 0-1, score 1-0
        ratio_score = min(1.0, div_ratio / 2) if div_ratio > 1 else div_ratio

        overall_score = (corr_score * 0.4 + conc_score * 0.3 + ratio_score * 0.3)

        return {
            "diversification_score": round(overall_score, 4),
            "components": {
                "correlation_score": round(corr_score, 4),
                "concentration_score": round(conc_score, 4),
                "ratio_score": round(ratio_score, 4)
            },
            "average_correlation": round(float(avg_corr), 4) if not np.isnan(avg_corr) else 0.0,
            "hhi_concentration": round(float(hhi), 4),
            "diversification_ratio": round(div_ratio, 4),
            "n_assets": n_assets,
            "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, weights)}
        }

    def calculate_full_risk_analysis(
        self,
        returns_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        weights: Optional[list[float]] = None
    ) -> dict:
        """
        Perform full risk analysis on a portfolio.

        Args:
            returns_df: DataFrame of asset returns
            benchmark_returns: Optional benchmark returns for beta calculation
            weights: Optional portfolio weights

        Returns:
            Comprehensive risk analysis dictionary
        """
        portfolio_risk = self.calculate_portfolio_risk(returns_df, weights)
        correlations = self.correlation_analysis(returns_df)
        diversification = self.diversification_score(returns_df, weights)

        result = {
            "portfolio_risk": portfolio_risk,
            "correlations": correlations,
            "diversification": diversification
        }

        if benchmark_returns is not None:
            # Calculate portfolio returns for beta
            n_assets = len(returns_df.columns)
            if weights is None:
                weights = [1.0 / n_assets] * n_assets
            portfolio_returns = (returns_df * weights[:n_assets]).sum(axis=1)
            beta_result = self.beta_calculation(portfolio_returns, benchmark_returns)
            result["portfolio_beta"] = beta_result

        return result
