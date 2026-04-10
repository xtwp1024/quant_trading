# -*- coding: utf-8 -*-
"""
Gate.io 持仓仪表板
Web Dashboard for Gate.io Position Visualization

Usage:
    python -m quant_trading.dashboard.gate_dashboard
    然后打开 http://localhost:8050
"""

import asyncio
import os
from pathlib import Path

# 自动加载 .env
_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from quant_trading.connectors.gate_sync import GateSync


# 初始化 Dash 应用
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Gate.io 持仓仪表板"

# 存储全局数据
data_cache = {
    'positions': [],
    'balance': {},
    'prices': {},
    'last_update': None
}


def fetch_gate_data():
    """获取 Gate.io 数据"""
    try:
        gate = GateSync()
        with gate:
            # 获取余额
            balance = gate.balance()

            # 获取各币种价格
            symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX']
            prices = {}
            for sym in symbols:
                try:
                    prices[sym] = gate.price(sym)
                except:
                    prices[sym] = 0

            # 获取持仓
            positions = []
            for sym in symbols:
                try:
                    pos_list = gate.positions(sym)
                    for pos in pos_list:
                        if pos.size != 0:
                            pos.current_price = prices.get(sym, 0)
                            pos.unrealized_pnl = pos.size * (pos.current_price - pos.entry_price)
                            pos.pnl_percent = (pos.current_price / pos.entry_price - 1) * 100 if pos.entry_price else 0
                            positions.append(pos)
                except:
                    pass

        return {
            'positions': positions,
            'balance': balance,
            'prices': prices,
            'last_update': str(gate._closed)
        }
    except Exception as e:
        return {
            'positions': [],
            'balance': {},
            'prices': {},
            'error': str(e)
        }


# 布局
app.layout = html.Div([
    dbc.Container([
        # 标题
        dbc.Row([
            dbc.Col([
                html.H1("Gate.io 持仓仪表板", className="text-center my-4"),
                html.H5("实时持仓与收益监控", className="text-center text-muted"),
            ])
        ]),

        # 状态栏
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='status-indicator', children="连接中..."),
                        html.Small("最后更新", id='last-update')
                    ])
                ], className="text-center")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='total-pnl', children="$0.00"),
                        html.Small("总未实现盈亏")
                    ])
                ], className="text-center")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='total-balance', children="$0.00"),
                        html.Small("账户余额 (USDT)")
                    ])
                ], className="text-center")
            ], width=4),
        ], className="mb-4"),

        # 刷新按钮
        dbc.Row([
            dbc.Col([
                dbc.Button("刷新数据", id='refresh-btn', color="primary", n_clicks=0),
                html.Span(id='refresh-status', className="ms-2")
            ], className="text-center my-3")
        ]),

        # 持仓表格
        dbc.Row([
            dbc.Col([
                html.H3("当前持仓", className="mt-3 mb-2"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("币种"),
                            html.Th("方向"),
                            html.Th("数量"),
                            html.Th("开仓价"),
                            html.Th("当前价"),
                            html.Th("强平价"),
                            html.Th("未实现盈亏"),
                            html.Th("收益率"),
                        ])
                    ]),
                    html.Tbody(id='positions-table')
                ], bordered=True, hover=True, striped=True, responsive=True)
            ])
        ]),

        # 价格行情
        dbc.Row([
            dbc.Col([
                html.H3("实时行情", className="mt-4 mb-2"),
                dcc.Interval(id='price-interval', interval=30000, n_intervals=0),  # 30秒刷新
                html.Div(id='prices-grid')
            ])
        ]),

    ], fluid=True),

    # 隐藏存储
    dcc.Store(id='gate-data'),
    dcc.Interval(id='auto-refresh', interval=60000, n_intervals=0),  # 60秒自动刷新
], className="bg-light min-vh-100 p-4")


@callback(
    Output('gate-data', 'data'),
    Input('refresh-btn', 'n_clicks'),
    Input('auto-refresh', 'n_intervals')
)
def update_data(n_clicks, n_intervals):
    """更新数据"""
    data = fetch_gate_data()
    return data


@callback(
    Output('status-indicator', 'children'),
    Output('last-update', 'children'),
    Output('total-pnl', 'children'),
    Output('total-balance', 'children'),
    Output('positions-table', 'children'),
    Output('prices-grid', 'children'),
    Input('gate-data', 'data')
)
def update_ui(data):
    if not data:
        return "无数据", "", "$0.00", "$0.00", [], []

    error = data.get('error')
    if error:
        return f"错误: {error[:30]}", "", "$0.00", "$0.00", [], []

    # 更新状态
    status = "✅ 已连接"

    # 更新时间
    import datetime
    last_update = datetime.datetime.now().strftime("%H:%M:%S")

    # 计算总盈亏
    positions = data.get('positions', [])
    total_pnl = sum(p.unrealized_pnl for p in positions if hasattr(p, 'unrealized_pnl'))

    # 余额
    balance_data = data.get('balance', {})
    total_balance = balance_data.get('total', 0)
    if isinstance(total_balance, dict):
        total_balance = total_balance.get('USDT', 0)

    # 持仓表格
    if not positions:
        positions_rows = [html.Tr(html.Td("暂无持仓", colSpan=8))]
    else:
        positions_rows = []
        for p in positions:
            size = p.size if hasattr(p, 'size') else 0
            entry = p.entry_price if hasattr(p, 'entry_price') else 0
            current = p.current_price if hasattr(p, 'current_price') else 0
            liq = p.liq_price if hasattr(p, 'liq_price') else 0
            pnl = p.unrealized_pnl if hasattr(p, 'unrealized_pnl') else 0
            pnl_pct = p.pnl_percent if hasattr(p, 'pnl_percent') else 0

            direction = "📈 多" if size > 0 else "📉 空"
            pnl_color = "text-success" if pnl >= 0 else "text-danger"

            positions_rows.append(html.Tr([
                html.Td(f"{p.symbol.replace('_USDT', '')}"),
                html.Td(direction),
                html.Td(f"{abs(size):.4f}"),
                html.Td(f"${entry:.2f}"),
                html.Td(f"${current:.2f}"),
                html.Td(f"${liq:.2f}"),
                html.Td(className=pnl_color, children=f"${pnl:.2f}"),
                html.Td(className=pnl_color, children=f"{pnl_pct:+.2f}%"),
            ]))

    # 价格网格
    prices = data.get('prices', {})
    prices_cards = []
    for sym, price in prices.items():
        if price > 0:
            prices_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(sym),
                        html.H3(f"${price:,.2f}")
                    ], className="text-center")
                ], className="m-2", style={"min-width": "120px"})
            )

    prices_grid = dbc.Row([dbc.Col(c) for c in prices_cards])

    return status, f"最后更新: {last_update}", f"${total_pnl:.2f}", f"${total_balance:.2f}", positions_rows, prices_grid


if __name__ == '__main__':
    print("=" * 50)
    print("Gate.io 持仓仪表板")
    print("=" * 50)
    print("打开浏览器访问: http://localhost:8050")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)

    # 先获取一次数据测试
    print("\n正在获取数据...")
    test_data = fetch_gate_data()
    if 'error' in test_data:
        print(f"⚠️ 警告: {test_data['error']}")
    else:
        print(f"✅ 获取到 {len(test_data['positions'])} 个持仓")
        print(f"   价格数据: {test_data['prices']}")

    print("\n启动服务器...")
    app.run(debug=False, host='0.0.0.0', port=8050)
