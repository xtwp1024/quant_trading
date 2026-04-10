# -*- coding: utf-8 -*-
"""
Unified Experiment Runner
========================

Central entry point for running all Phase 1 and Phase 2 experiments.

Usage:
    python run_all_experiments.py --all
    python run_all_experiments.py --category v36
    python run_all_experiments.py --category multi_strategy
    python run_all_experiments.py --category agent
    python run_all_experiments.py --category rl
    python run_all_experiments.py --category survival
    python run_all_experiments.py --category production
    python run_all_experiments.py --experiment v36_winrate_improvement

Categories:
    v36: v36_winrate_improvement, v36_multi_symbol_backtest
    multi_strategy: multi_strategy_backtest, multi_strategy_comparison
    agent: agent_backtest
    rl: rl_backtest
    survival: survival_test, stress_tests, monte_carlo_sim
    production: production_backtest, test_binance_connection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ExperimentRunner")


# ===================== Experiment Definitions =====================

@dataclass
class Experiment:
    """Experiment definition"""
    name: str
    module: str
    category: str
    description: str
    command: List[str] = field(default_factory=list)
    required_files: List[str] = field(default_factory=list)


# Experiment registry
EXPERIMENTS: Dict[str, Experiment] = {
    # V36 category
    "v36_winrate_improvement": Experiment(
        name="v36_winrate_improvement",
        module="v36_winrate_improvement",
        category="v36",
        description="V36策略胜率改进回测 - 比较原始V36、v2、v3版本的胜率表现",
        required_files=["v36_winrate_improvement.py"],
    ),
    "v36_multi_symbol_backtest": Experiment(
        name="v36_multi_symbol_backtest",
        module="v36_multi_symbol_backtest",
        category="v36",
        description="V36多币种回测 - 在多个加密货币上测试V36策略",
        required_files=["v36_multi_symbol_backtest.py"],
    ),

    # Multi-strategy category
    "multi_strategy_backtest": Experiment(
        name="multi_strategy_backtest",
        module="multi_strategy_backtest",
        category="multi_strategy",
        description="多策略回测 - V36、Grid及组合策略对比",
        required_files=["multi_strategy_backtest.py"],
    ),
    "multi_strategy_comparison": Experiment(
        name="multi_strategy_comparison",
        module="multi_strategy_comparison",
        category="multi_strategy",
        description="多策略比较 - 4种策略(V36, DaveLandry, GridHODL, RSI)对比",
        required_files=["multi_strategy_comparison.py"],
    ),

    # Agent category
    "agent_backtest": Experiment(
        name="agent_backtest",
        module="agent_backtest",
        category="agent",
        description="Agent系统回测 - 市场分析Agent与基线策略对比",
        required_files=["agent_backtest.py"],
    ),

    # RL category
    "rl_backtest": Experiment(
        name="rl_backtest",
        module="rl_backtest_demo",
        category="rl",
        description="RL Agent回测 - 演示模式（无需训练模型）",
        required_files=["rl_backtest_demo.py"],
    ),

    # Survival category
    "survival_test": Experiment(
        name="survival_test",
        module="survival_test",
        category="survival",
        description="30天生存测试 - 模拟实盘交易验证系统生存能力",
        required_files=["survival_test.py"],
    ),
    "stress_tests": Experiment(
        name="stress_tests",
        module="stress_tests",
        category="survival",
        description="压力测试 - 多场景压力测试组合",
        required_files=["stress_tests.py"],
    ),
    "monte_carlo_sim": Experiment(
        name="monte_carlo_sim",
        module="monte_carlo_sim",
        category="survival",
        description="蒙特卡洛模拟 - 随机市场环境下的策略表现",
        required_files=["monte_carlo_sim.py"],
    ),

    # Production category
    "production_backtest": Experiment(
        name="production_backtest",
        module="production_backtest",
        category="production",
        description="生产回测 - 进入实盘前的完整验证",
        required_files=["production_backtest.py"],
    ),
    "test_binance_connection": Experiment(
        name="test_binance_connection",
        module="test_binance_connection",
        category="production",
        description="Binance连接测试 - 验证实盘交易设置",
        required_files=["test_binance_connection.py"],
    ),
}

CATEGORIES = {
    "v36": ["v36_winrate_improvement", "v36_multi_symbol_backtest"],
    "multi_strategy": ["multi_strategy_backtest", "multi_strategy_comparison"],
    "agent": ["agent_backtest"],
    "rl": ["rl_backtest"],
    "survival": ["survival_test", "stress_tests", "monte_carlo_sim"],
    "production": ["production_backtest", "test_binance_connection"],
}

ALL_EXPERIMENTS = list(EXPERIMENTS.keys())


# ===================== Database Logging =====================

def get_experiment_db_path(output_dir: str) -> str:
    """Get the experiment database path"""
    return os.path.join(output_dir, "experiment_db.sqlite")


def init_experiment_db(output_dir: str) -> None:
    """Initialize experiment database"""
    db_path = get_experiment_db_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            category TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT NOT NULL,
            duration_seconds REAL,
            output_dir TEXT,
            error_message TEXT,
            results_json TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            metric_name TEXT,
            metric_value REAL,
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Initialized experiment database at {db_path}")


def log_experiment_start(
    output_dir: str,
    experiment_name: str,
    category: str
) -> int:
    """Log experiment start and return run_id"""
    db_path = get_experiment_db_path(output_dir)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO experiment_runs (experiment_name, category, start_time, status, output_dir)
        VALUES (?, ?, ?, ?, ?)
    """, (experiment_name, category, datetime.now().isoformat(), "running", output_dir))

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def log_experiment_end(
    output_dir: str,
    run_id: int,
    status: str,
    duration_seconds: float,
    error_message: Optional[str] = None,
    results_json: Optional[str] = None
) -> None:
    """Log experiment end"""
    db_path = get_experiment_db_path(output_dir)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE experiment_runs
        SET end_time = ?, status = ?, duration_seconds = ?, error_message = ?, results_json = ?
        WHERE id = ?
    """, (datetime.now().isoformat(), status, duration_seconds, error_message, results_json, run_id))

    conn.commit()
    conn.close()


# ===================== Summary Table =====================

def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print a summary table of experiment results"""
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    header = f"| {'Experiment':<30} | {'Category':<15} | {'Status':<10} | {'Duration':<12} | {'Exit Code':<10} |"
    print(header)
    print("|" + "-" * 32 + "|" + "-" * 17 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 12 + "|")

    for r in results:
        status_str = r.get("status", "unknown")
        status_display = "SUCCESS" if status_str == "success" else ("FAILED" if status_str == "failed" else status_str)
        duration = r.get("duration", 0.0)
        duration_str = f"{duration:.2f}s" if duration else "N/A"
        exit_code = r.get("exit_code", "N/A")

        print(f"| {r.get('name', 'unknown'):<30} | {r.get('category', 'unknown'):<15} | {status_display:<10} | {duration_str:<12} | {exit_code:<10} |")

    print("=" * 100)

    # Summary stats
    total = len(results)
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    print(f"\nTotal: {total} | Succeeded: {succeeded} | Failed: {failed}")

    # Print errors if any
    failed_results = [r for r in results if r.get("status") == "failed"]
    if failed_results:
        print("\nFailed Experiments:")
        for r in failed_results:
            error = r.get("error", "Unknown error")
            print(f"  - {r.get('name')}: {error}")


# ===================== Experiment Runners =====================

def run_experiment(
    experiment: Experiment,
    output_dir: str,
    extra_args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run a single experiment"""
    import time

    result = {
        "name": experiment.name,
        "category": experiment.category,
        "status": "failed",
        "duration": 0.0,
        "exit_code": 1,
        "error": None,
    }

    start_time = time.time()

    # Ensure output directory exists
    exp_output_dir = os.path.join(output_dir, experiment.category, experiment.name)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Log to database
    run_id = log_experiment_start(output_dir, experiment.name, experiment.category)

    try:
        # Build command
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, f"{experiment.module}.py")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = [sys.executable, "-m", f"quant_trading.experiments.{experiment.module}", "--output-dir", exp_output_dir]

        # Add any extra arguments
        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Running: {' '.join(cmd)}")

        # Run the experiment from project root so quant_trading package is visible
        project_root = os.path.dirname(os.path.dirname(script_dir))  # 量化之神 root
        process_result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        result["exit_code"] = process_result.returncode
        result["duration"] = time.time() - start_time

        # Check for "fatal: bad revision 'HEAD'" in stderr - this is an environment issue, not script error
        stderr_lower = (process_result.stderr or "").lower()
        if "fatal: bad revision" in stderr_lower and "head" in stderr_lower:
            # Git environment issue - script likely ran fine
            if process_result.returncode == 0:
                result["status"] = "success"
                result["error"] = "git environment issue (ignored)"
                logger.warning(f"Experiment {experiment.name} succeeded (git environment issue detected)")
            else:
                result["status"] = "failed"
                result["error"] = process_result.stderr[:500] if process_result.stderr else "Unknown error"
                logger.error(f"Experiment {experiment.name} failed: {result['error']}")
        elif process_result.returncode == 0:
            result["status"] = "success"
            logger.info(f"Experiment {experiment.name} succeeded in {result['duration']:.2f}s")
        else:
            result["status"] = "failed"
            result["error"] = process_result.stderr[:500] if process_result.stderr else "Unknown error"
            logger.error(f"Experiment {experiment.name} failed: {result['error']}")

        # Try to load results JSON if exists
        results_json_path = os.path.join(exp_output_dir, "results.json")
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r', encoding='utf-8') as f:
                result["results_file"] = results_json_path

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (>1 hour)"
        result["duration"] = time.time() - start_time
        logger.error(f"Experiment {experiment.name} timed out")
    except Exception as e:
        result["error"] = str(e)[:500]
        result["duration"] = time.time() - start_time
        logger.error(f"Experiment {experiment.name} error: {e}")

    # Log to database
    log_experiment_end(
        output_dir,
        run_id,
        result["status"],
        result["duration"],
        result.get("error"),
        json.dumps(result) if result.get("results_file") else None
    )

    return result


def run_category(
    category: str,
    output_dir: str,
    extra_args: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Run all experiments in a category"""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}")

    experiment_names = CATEGORIES[category]
    results = []

    print(f"\n{'=' * 80}")
    print(f"Running Category: {category.upper()}")
    print(f"Experiments: {', '.join(experiment_names)}")
    print(f"{'=' * 80}\n")

    for name in experiment_names:
        if name not in EXPERIMENTS:
            logger.warning(f"Experiment {name} not found in registry, skipping")
            continue

        experiment = EXPERIMENTS[name]
        result = run_experiment(experiment, output_dir, extra_args)
        results.append(result)

        # Print result
        status_icon = "[OK]" if result["status"] == "success" else "[FAIL]"
        print(f"  {status_icon} {name}: {result['status']} ({result['duration']:.2f}s)")

    return results


# ===================== Main =====================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for Phase 1 & Phase 2 experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py --all
  python run_all_experiments.py --category v36
  python run_all_experiments.py --category multi_strategy
  python run_all_experiments.py --category agent
  python run_all_experiments.py --category rl
  python run_all_experiments.py --category survival
  python run_all_experiments.py --category production
  python run_all_experiments.py --experiment v36_winrate_improvement
  python run_all_experiments.py --all --output-dir ./results

Categories:
  v36           - V36策略相关实验
  multi_strategy - 多策略回测
  agent         - Agent系统实验
  rl            - 强化学习实验
  survival      - 生存能力测试
  production    - 生产环境验证
        """
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments"
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORIES.keys()),
        help="Run all experiments in a specific category"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=ALL_EXPERIMENTS,
        help="Run a specific experiment"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="Output directory for results (default: ./experiment_results)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and categories"
    )

    return parser.parse_args()


def list_experiments():
    """List all available experiments"""
    print("\n" + "=" * 80)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 80)

    for category, experiments in CATEGORIES.items():
        print(f"\n[{category.upper()}]")
        for exp_name in experiments:
            if exp_name in EXPERIMENTS:
                exp = EXPERIMENTS[exp_name]
                print(f"  - {exp_name}: {exp.description}")

    print("\n" + "=" * 80)
    print(f"Total: {len(ALL_EXPERIMENTS)} experiments in {len(CATEGORIES)} categories")
    print("=" * 80)


def main():
    """Main entry point"""
    args = parse_args()

    # List experiments if requested
    if args.list:
        list_experiments()
        return 0

    # Initialize output directory and database
    os.makedirs(args.output_dir, exist_ok=True)
    init_experiment_db(args.output_dir)

    # Determine what to run
    results = []

    if args.all:
        # Run all experiments
        for category in CATEGORIES.keys():
            category_results = run_category(category, args.output_dir)
            results.extend(category_results)

    elif args.category:
        # Run specific category
        results = run_category(args.category, args.output_dir)

    elif args.experiment:
        # Run specific experiment
        if args.experiment not in EXPERIMENTS:
            logger.error(f"Unknown experiment: {args.experiment}")
            return 1

        experiment = EXPERIMENTS[args.experiment]
        print(f"\nRunning experiment: {experiment.name}")
        print(f"Description: {experiment.description}")

        result = run_experiment(experiment, args.output_dir)
        results.append(result)

    else:
        logger.error("Please specify --all, --category, or --experiment")
        return 1

    # Print summary table
    print_summary_table(results)

    # Return exit code based on results
    if all(r.get("status") == "success" for r in results):
        logger.info("All experiments completed successfully!")
        return 0
    else:
        logger.error("Some experiments failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
