import os
import random
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
import time
import importlib.util
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from .database import DatabaseManager
# from backtest_engine import BacktestExchange, BacktestEventBus
# from ..modules.five_force.strategy_parser import StrategyGenomeParser

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - ENCLAVE - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestEnclave")

class BacktestEnclave:
    def __init__(self, config_path="config.yaml"):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.db = DatabaseManager(self.config)
        self.output_dir = "modules/strategies/validated"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache for historical data
        self.data_cache = {}
        
        # Thresholds (Lowered for Operation Phoenix Initiation)
        self.min_roi = -5.0 
        self.min_sharpe = 0.0 
        self.max_drawdown = 50.0

    async def prepare_engine(self, symbol="ETH-USDT-SWAP", days=90):
        await self.db.connect()
        logger.info(f"📥 Loading historical data for {symbol} ({days} days)...")
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (days * 24 * 3600 * 1000)
        
        with self.db.pg_conn.cursor() as cur:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_candles
                WHERE symbol = %s AND timeframe = '15m'
                AND timestamp >= %s
                ORDER BY timestamp ASC
            """
            cur.execute(query, (symbol, start_ts))
            rows = cur.fetchall()
            
        if not rows:
            logger.error("No data found! Run backfill first.")
            return False
            
        self.data_cache[symbol] = [list(r) for r in rows]
        logger.info(f"✅ Cached {len(rows)} 15m candles for {symbol}.")
        return True

    async def run_fast_backtest(self, strat_class, symbol="ETH-USDT-SWAP") -> Dict[str, Any]:
        """
        Executes a minimalist backtest on the cached data.
        """
        data = self.data_cache.get(symbol)
        if not data: return {"roi": -100}

        bus = BacktestEventBus()
        exchange = BacktestExchange(initial_capital=1000.0)
        strategy = strat_class(bus, {'symbol': symbol, 'timeframe': '15m'})
        
        if hasattr(strategy, 'start'):
            await strategy.start()
            
        data_buffer = []
        for r in data:
            ts, o, h, l, c, v = r
            data_buffer.append([float(x) for x in r])
            if len(data_buffer) > 500: data_buffer.pop(0)
            if len(data_buffer) < 50: continue
            
            # Tick
            await strategy.on_candle(type('Event', (), {'payload': {'symbol': symbol, 'data': data_buffer}}))
            
            # Process Signals (Interal Logic of backtest_engine simplified)
            for sig in bus.trade_history:
                if not sig.get('processed'):
                    exchange.execute_signal(sig['signal'], float(c), ts)
                    sig['processed'] = True
                    
        # Stats
        final_equity = exchange.get_equity({symbol: float(data[-1][4])})
        roi = (final_equity - 1000.0) / 1000.0 * 100
        return {
            "roi": roi,
            "trades": len(exchange.trades),
            "equity": final_equity
        }

    async def run_vectorized_backtest(self, filename, symbol="ETH-USDT-SWAP", overrides: Dict[str, Any] = None, return_masks: bool = False) -> Dict[str, Any]:
        """
        Ultra-fast vectorized backtest for Phoenix strategies.
        Extracts logic DNA and evaluates using pandas masks.
        """
        import re
        import textwrap
        import importlib.util
        try:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
            
            # Extract indicators and buy/sell logic from Phoenix template structure
            indicator_match = re.search(r"# --- Indicator Calculation ---(.*?)current_price =", code, re.DOTALL)
            
            dna_section = re.search(r"# --- Logical DNA ---(.*?)except Exception:", code, re.DOTALL)
            if not dna_section: return {"roi": -100, "trades": 0}
            dna_code = dna_section.group(1).strip()
            
            buy_match = re.search(r"(?<!el)\bif\b (.*?):\s+signals\.append", dna_code, re.DOTALL)
            sell_match = re.search(r"\belif\b (.*?):\s+signals\.append", dna_code, re.DOTALL)
            
            if not buy_match or not sell_match:
                return {"roi": -100, "trades": 0}
            
            buy_logic = buy_match.group(1).strip()
            sell_logic = sell_match.group(1).strip()
            
            # Data
            data = self.data_cache.get(symbol)
            if not data: return {"roi": -100, "trades": 0}
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Calculate Indicators via extracted code
            if indicator_match:
                indicator_code = textwrap.dedent(indicator_match.group(1))
                # Remove leading empty lines that break dedent
                indicator_lines = indicator_code.split("\n")
                while indicator_lines and not indicator_lines[0].strip():
                    indicator_lines.pop(0)
                indicator_code = textwrap.dedent("\n".join(indicator_lines)).strip()
                
                import pandas_ta as ta
                import core.qtpylib as qtpylib
                exec_globals = {"df": df, "ta": ta, "np": np, "pd": pd, "qtpylib": qtpylib}
                try:
                    exec(indicator_code, exec_globals)
                except Exception:
                    return {"roi": -100, "trades": 0}
            
            # Vectorized Logic Translation
            v_buy = buy_logic.replace("last_row", "df").replace("df.iloc[-1]", "df")
            v_sell = sell_logic.replace("last_row", "df").replace("df.iloc[-1]", "df")
            
            # Convert Python logical operators to Pandas bitwise for vectorized eval
            # Using regex to ensure we only replace 'or'/'and' word-boundaries
            v_buy = re.sub(r"\bor\b", "|", v_buy)
            v_buy = re.sub(r"\band\b", "&", v_buy)
            v_buy = re.sub(r"\bnot\b", "~", v_buy)
            
            v_sell = re.sub(r"\bor\b", "|", v_sell)
            v_sell = re.sub(r"\band\b", "&", v_sell)
            v_sell = re.sub(r"\bnot\b", "~", v_sell)
            
            # Column Alignment & Fallbacks
            # Many strategies use inconsistent naming for Bollinger Bands
            if 'bb_lowerband' in df.columns:
                df['lband'] = df['bb_lowerband']
            if 'bb_upperband' in df.columns:
                df['uband'] = df['bb_upperband']
            if 'bb_middleband' in df.columns:
                df['mband'] = df['bb_middleband']
            if 'macdsignal' in df.columns:
                df['macd_signal'] = df['macdsignal']

            # Dynamic Import Class to get 'self' context for attributes like lookback
            name = os.path.basename(filename).replace(".py", "")
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                strat_class = getattr(module, name)
                # Pass a dummy bus and the config
                strategy = strat_class(None, {'symbol': symbol, 'timeframe': '15m'})
                
                # Apply Hyperparameter Overrides
                if overrides:
                    for key, val in overrides.items():
                        if hasattr(strategy, key):
                            setattr(strategy, key, val)
                            # logger.info(f"   🔧 Overriding {key} = {val} for {name}")
            except Exception as e:
                logger.error(f"Failed to instantiate {name}: {e}")
                return {"roi": -100, "trades": 0}
            
            try:
                env = {"df": df, "np": np, "pd": pd, "qtpylib": qtpylib, "self": strategy, "ta": ta}
                
                # Check for missing columns in logic and inject defaults OR calculate them
                all_logic = v_buy + " " + v_sell
                potential_cols = re.findall(r"(?:df|dataframe)\[['\"](\w+)['\"]\]", all_logic)
                logger.info(f"Checking indicators for {strategy.name}: {potential_cols}")
                for col in set(potential_cols):
                    if col not in df.columns:
                        try:
                            if col == 'rsi': df['rsi'] = ta.rsi(df['close'], length=14)
                            elif col == 'cci': df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
                            elif col == 'adx': df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
                            # ... (other indicators) ...
                            elif 'ema_' in col:
                                p_match = re.search(r'(\d+)', col)
                                if p_match:
                                    df[col] = ta.ema(df['close'], length=int(p_match.group(1)))
                            elif 'sma_' in col:
                                p_match = re.search(r'(\d+)', col)
                                if p_match:
                                    df[col] = ta.sma(df['close'], length=int(p_match.group(1)))
                            else:
                                if col not in ['close', 'open', 'high', 'low', 'volume']:
                                    logger.warning(f"Defaulting missing indicator {col} to False")
                                    df[col] = False 
                        except Exception as e:
                            logger.error(f"Failed to auto-calc {col}: {e}")
                            df[col] = False
                
                try:
                    buy_mask = eval(v_buy, env).fillna(False).astype(bool)
                    sell_mask = eval(v_sell, env).fillna(False).astype(bool)
                except Exception as e:
                    logger.error(f"EVAL ERROR in {strategy.name}: {e}\nLogic: {v_buy}")
                    return {'roi': -10000, 'trades': 0}

                if return_masks:
                    return {"buy_mask": buy_mask, "sell_mask": sell_mask}
                
                # Pre-calculated Loop Simulation (Fastest safe way)
                # Convert to numpy for maximum speed
                buys = buy_mask.values
                sells = sell_mask.values
                prices = df['close'].values
                
                in_position = False
                entry_price = 0
                total_roi = 0
                trades = 0
                
                for i in range(len(df)):
                    if not in_position:
                        if buys[i]:
                            in_position = True
                            entry_price = prices[i]
                            trades += 1
                    else:
                        if sells[i]:
                            in_position = False
                            pnl = (prices[i] - entry_price) / entry_price
                            total_roi += pnl
                
                return {"roi": float(total_roi * 100), "trades": int(trades)}
            except Exception as e:
                logger.error(f"Vectorized eval failed for {filename}: {e}")
                return {"roi": -100, "trades": 0}
        except Exception as e:
            logger.error(f"Vectorized backtest process failed for {filename}: {e}")
            return {"roi": -100, "trades": 0}

    async def validate_and_promote(self, name, code, buy_logic=None, sell_logic=None, symbol="ETH-USDT-SWAP"):
        """
        Headless validation and promotion to /validated folder.
        Uses vectorized evaluation if logic is provided, otherwise falls back to fast backtest.
        """
        temp_path = os.path.join(os.getcwd(), f"tmp_enclave_{name}_{random.randint(1000,9999)}.py")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Dynamic Import
            spec = importlib.util.spec_from_file_location(name, temp_path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                # Keep one failed file for debugging if it happens
                debug_path = f"debug_fail_{name}.py"
                import shutil
                shutil.copy(temp_path, debug_path)
                raise e
            strat_class = getattr(module, name)
            
            # VECTORIZED HEARTBEAT CHECK
            if buy_logic and sell_logic:
                data = self.data_cache.get(symbol)
                if not data: return False
                
                df_base = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                strategy = strat_class(symbol)
                
                # Pre-calculate indicators (once!)
                df = strategy.calculate_indicators(df_base)
                
                # Prepare vectorized logic
                v_buy = buy_logic.replace("last_row", "df").replace("df.iloc[-1]", "df")
                v_sell = sell_logic.replace("last_row", "df").replace("df.iloc[-1]", "df")
                
                try:
                    # Use pandas.eval() for safer evaluation of vectorized logic
                    buy_mask = pd.eval(v_buy, engine='python')
                    sell_mask = pd.eval(v_sell, engine='python')
                    
                    # Ensure they are boolean series
                    if isinstance(buy_mask, pd.Series) and isinstance(sell_mask, pd.Series):
                        total_signals = buy_mask.sum() + sell_mask.sum()
                    else:
                        total_signals = 1 if (buy_mask or sell_mask) else 0
                        
                    if total_signals >= 1:
                        # PROMOTION
                        final_path = os.path.join(self.output_dir, f"{name}.py")
                        with open(final_path, "w", encoding="utf-8") as f:
                            f.write(code)
                        logger.info(f"🏆 PROMOTE (V): {name} | Signals: {total_signals}")
                        return True
                    return False
                except Exception as e:
                    logger.debug(f"Vectorization failed for {name}: {e}. Falling back to slow mode.")
            
            # FALLBACK TO SLOW BACKTEST (Original Logic)
            results = await self.run_fast_backtest(strat_class, symbol)
            if results['trades'] >= 1:
                final_path = os.path.join(self.output_dir, f"{name}.py")
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(code)
                logger.info(f"🏆 PROMOTE (S): {name} | Trades: {results['trades']}")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"💥 ERROR: {name} - {e}")
            return False
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except PermissionError:
                pass

    def cleanup_temp_files(self):
        """Final cleanup of any leftover locked files."""
        import glob
        files = glob.glob("tmp_enclave_*.py")
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

if __name__ == "__main__":
    # Test stub
    pass
