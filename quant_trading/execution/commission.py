"""Commission models — fee structures for backtesting and live trading.

Provides:
- CommissionModel     : Abstract base
- FixedCommission    : Flat fee per trade (e.g. 0.1% spot)
- MakerTakerCommission : Maker/taker fee schedule (CEX standard)
- TieredCommission  : Volume-based tiered fees (like Binance VIP)
- CryptoCommission   : Per-asset commission with rebate tiers

Usage
-----
```python
from quant_trading.execution.commission import MakerTakerCommission

# Binance standard spot fees
comm = MakerTakerCommission(maker=0.001, taker=0.001)
fee = comm.calculate(symbol="BTC/USDT", side="buy", quantity=1.0, price=50000)
print(f"Commission: ${fee:.2f}")

# Tiered fees
tiered = TieredCommission(tiers=[
    (10_000, 0.0009, 0.0011),   # (volume_usdt, maker, taker)
    (100_000, 0.0007, 0.0009),
    (1_000_000, 0.0005, 0.0007),
])
fee = tiered.calculate("BTC/USDT", "buy", 1.0, 50000, 30_day_volume=200_000)
```
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Commission Models
# ---------------------------------------------------------------------------

class CommissionModel(ABC):
    """Abstract base for commission/fee models."""

    @abstractmethod
    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        **kwargs,
    ) -> float:
        """Calculate commission for a trade.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTC/USDT")
        side : str
            "buy" or "sell"
        quantity : float
            Base asset quantity
        price : float
            Execution price

        Returns
        -------
        float
            Commission in quote currency (e.g. USDT)
        """
        ...

    def calculate_portfolio_fees(self, trades: list) -> float:
        """Sum commissions across multiple trades."""
        return sum(
            self.calculate(t["symbol"], t["side"], t["qty"], t["price"])
            for t in trades
        )


@dataclass
class FixedCommission(CommissionModel):
    """Fixed percentage commission per trade.

    Applies the same rate to both buy and sell sides.
    """
    rate: float = 0.001  # 0.1% default

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        **kwargs,
    ) -> float:
        notional = quantity * price
        return notional * self.rate


@dataclass
class MakerTakerCommission(CommissionModel):
    """Standard maker/taker fee schedule.

    Maker fees are charged when the order adds liquidity (limit orders).
    Taker fees are charged when the order removes liquidity (market orders).
    """
    maker: float = 0.001  # 0.1%
    taker: float = 0.001  # 0.1%
    is_maker: bool = True  # If True, assume maker (limit order); set False for market

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        is_maker: Optional[bool] = None,
        **kwargs,
    ) -> float:
        notional = quantity * price
        rate = self.maker if (is_maker if is_maker is not None else self.is_maker) else self.taker
        return notional * rate

    def effective_rate(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Return the effective fee rate (commission / notional)."""
        notional = quantity * price
        if notional == 0:
            return 0.0
        return self.calculate(symbol, side, quantity, price) / notional


@dataclass
class TieredCommission(CommissionModel):
    """Volume-based tiered commission schedule.

    Commission tiers are based on 30-day trading volume in quote currency.
    Higher volume → lower fees.
    """
    tiers: list = None  # List of (min_volume_usdt, maker_rate, taker_rate)

    def __post_init__(self):
        if self.tiers is None:
            # Default Binance VIP tiers (USD volume in last 30 days)
            self.tiers = [
                (0, 0.001, 0.001),          # VIP 0
                (10_000, 0.0009, 0.001),    # VIP 1
                (100_000, 0.0007, 0.0009),  # VIP 2
                (1_000_000, 0.0005, 0.0007), # VIP 3
                (10_000_000, 0.0003, 0.0005), # VIP 4
            ]

    def _get_tier(self, volume_30d: float) -> tuple:
        """Get the best tier for a given 30-day volume."""
        best = self.tiers[0]
        for min_vol, maker, taker in self.tiers:
            if volume_30d >= min_vol:
                best = (min_vol, maker, taker)
            else:
                break
        return best

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        is_maker: Optional[bool] = None,
        volume_30d: float = 0.0,
        **kwargs,
    ) -> float:
        notional = quantity * price
        _, maker, taker = self._get_tier(volume_30d)
        rate = maker if (is_maker if is_maker is not None else False) else taker
        return notional * rate

    def tier_info(self, volume_30d: float) -> dict:
        """Return tier info for a given 30-day volume."""
        min_vol, maker, taker = self._get_tier(volume_30d)
        return {
            "volume_30d": volume_30d,
            "tier_min_volume": min_vol,
            "maker_rate": maker,
            "taker_rate": taker,
        }


@dataclass
class CryptoCommission(CommissionModel):
    """Per-asset commission with optional rebate.

    Allows specifying different fees per base/quote asset and
    a maker rebate (negative fee for providing liquidity).
    """
    # Per-base-asset fee in base currency (e.g. BTC)
    base_fees: Dict[str, float] = None
    # Per-quote-asset fee in quote currency (e.g. USDT)
    quote_fees: Dict[str, float] = None
    # Maker rebate (negative fee)
    maker_rebate_rate: float = 0.0

    def __post_init__(self):
        if self.base_fees is None:
            self.base_fees = {}
        if self.quote_fees is None:
            self.quote_fees = {}

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        is_maker: Optional[bool] = False,
        base_asset: Optional[str] = None,
        quote_asset: Optional[str] = None,
        **kwargs,
    ) -> float:
        if base_asset is None:
            base_asset = symbol.split("/")[0] if "/" in symbol else symbol
        if quote_asset is None:
            quote_asset = symbol.split("/")[1] if "/" in symbol else "USDT"

        notional = quantity * price

        # Quote fee (on USDT volume)
        quote_fee = notional * self.quote_fees.get(quote_asset, 0.0)

        # Base fee (on BTC/ETH quantity)
        base_fee_qty = quantity * self.base_fees.get(base_asset, 0.0)
        base_fee_value = base_fee_qty * price if base_asset != quote_asset else 0.0

        # Maker rebate (reduce commission)
        total = quote_fee + base_fee_value
        if is_maker:
            rebate = notional * self.maker_rebate_rate
            total -= rebate

        return max(0.0, total)

    def add_base_fee(self, asset: str, rate: float) -> None:
        """Add or update a base asset fee rate."""
        self.base_fees[asset] = rate

    def add_quote_fee(self, asset: str, rate: float) -> None:
        """Add or update a quote asset fee rate."""
        self.quote_fees[asset] = rate


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

BINANCE_SPOT_COMMISSION = MakerTakerCommission(maker=0.001, taker=0.001)
BINANCE_FUTURES_COMMISSION = MakerTakerCommission(maker=0.0002, taker=0.0004)
COINBASE_COMMISSION = MakerTakerCommission(maker=0.004, taker=0.006)
KRAKEN_COMMISSION = MakerTakerCommission(maker=0.0016, taker=0.0026)
