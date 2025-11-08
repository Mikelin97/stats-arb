from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import streamlit as st

DATA_DIR = Path("data")
@dataclass(frozen=True)
class AssetConfig:
    price_col: str
    display: str
    tick_size: float
    tick_value: float
    contract_size: float

    @property
    def volume_col(self) -> str:
        return f"{self.price_col}_volume"

    @property
    def position_col(self) -> str:
        return f"position_{self.price_col}"

    @property
    def pnl_col(self) -> str:
        return f"pnl_{self.price_col}"


@dataclass(frozen=True)
class PairConfig:
    pair_id: str
    asset_x: AssetConfig
    asset_y: AssetConfig
    candle_size: str = "1min"


@dataclass
class BacktestParams:
    entry_threshold: float
    exit_threshold: float
    stop_entry_threshold: float
    initial_margin: float
    max_volume_take_rate: float
    num_contracts: int


DEFAULT_ENTRY_THRESHOLD = 1.5
DEFAULT_EXIT_THRESHOLD = 0.5
DEFAULT_INITIAL_MARGIN = 0.1
DEFAULT_MAX_VOLUME_TAKE_RATE = 0.1
DEFAULT_NUM_CONTRACTS = 1
DEFAULT_STOP_ENTRY_THRESHOLD = 4.0
AVG_TRADING_HOURS = 6.5
DEFAULT_COINTEGRATION_LOOKBACK = 100
DEFAULT_COINTEGRATION_P_THRESHOLD = 0.05


PAIRS: Dict[str, PairConfig] = {
    "WTI Future vs. Brent Future": PairConfig(
        pair_id="pair1",
        asset_x=AssetConfig(
            price_col="pair1_wti_oil_future_ohlcv-1m",
            display="WTI Future",
            tick_size=0.01,
            tick_value=10.0,
            contract_size=1000.0,
        ),
        asset_y=AssetConfig(
            price_col="pair1_brent_oil_future_ohlcv-1m",
            display="Brent Future",
            tick_size=0.01,
            tick_value=10.0,
            contract_size=1000.0,
        ),
    ),
    "NatGas HH vs. NatGas LS": PairConfig(
        pair_id="pair6",
        asset_x=AssetConfig(
            price_col="pair6_natgas_hh_future_ohlcv-1m",
            display="Henry Hub NG",
            tick_size=0.001,
            tick_value=10.0,
            contract_size=10000.0,
        ),
        asset_y=AssetConfig(
            price_col="pair6_natgas_ls_future_ohlcv-1m",
            display="Louisiana NG",
            tick_size=0.001,
            tick_value=10.0,
            contract_size=10000.0,
        ),
    ),
    "MSTR vs. IBIT": PairConfig(
        pair_id="pair7",
        asset_x=AssetConfig(
            price_col="pair7_mstr_spot_ohlcv-1m",
            display="MSTR",
            tick_size=0.01,
            tick_value=1.0,
            contract_size=1.0,
        ),
        asset_y=AssetConfig(
            price_col="pair7_ibit_etf_ohlcv-1m",
            display="IBIT ETF",
            tick_size=0.01,
            tick_value=1.0,
            contract_size=1.0,
        ),
    ),
    "Gold Future vs. Micro Gold Future": PairConfig(
        pair_id="pair10",
        asset_x=AssetConfig(
            price_col="pair10_micro_gold_future_ohlcv-1m",
            display="Micro Gold",
            tick_size=0.1,
            tick_value=1.0,
            contract_size=10.0,
        ),
        asset_y=AssetConfig(
            price_col="pair10_gold_future_ohlcv-1m",
            display="Gold Future",
            tick_size=0.1,
            tick_value=10.0,
            contract_size=100.0,
        ),
    ),
    "Silver Future vs. Micro Silver Future": PairConfig(
        pair_id="pair11",
        asset_x=AssetConfig(
            price_col="pair11_micro_silver_future_ohlcv-1m",
            display="Micro Silver",
            tick_size=0.005,
            tick_value=5.0,
            contract_size=1000.0,
        ),
        asset_y=AssetConfig(
            price_col="pair11_silver_future_ohlcv-1m",
            display="Silver Future",
            tick_size=0.005,
            tick_value=25.0,
            contract_size=5000.0,
        ),
    ),
}


def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    format1 = workbook.add_format({"num_format": "0.00"})
    worksheet.set_column("A:A", None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def find_asset_file(price_col: str) -> Path:
    preferred_names = [
        f"{price_col}_ohlcv-1m.csv",
        f"{price_col}_ohlcv.csv",
        f"{price_col}.csv",
    ]
    for name in preferred_names:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    matches = sorted(DATA_DIR.glob(f"{price_col}_*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find OHLCV data for {price_col}. "
        f"Expected file like {price_col}.csv in {DATA_DIR}."
    )


def load_asset_ohlcv(price_col: str, candle_size: str) -> pd.DataFrame:
    path = find_asset_file(price_col)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df.sort_index()
    if candle_size == "1min":
        return df
    agg_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(candle_size).agg(agg_map).ffill()


def build_price_volume_frame(
    asset_df: pd.DataFrame, price_col: str
) -> pd.DataFrame:
    required_cols = {"close", "volume"}
    missing = required_cols.difference(asset_df.columns)
    if missing:
        raise ValueError(f"Asset data for {price_col} missing columns: {', '.join(sorted(missing))}")
    out = asset_df[["close", "volume"]].rename(
        columns={"close": price_col, "volume": f"{price_col}_volume"}
    )
    return out


def compute_cointegration_fields(
    pair_close: pd.DataFrame,
    asset_x_col: str,
    asset_y_col: str,
    lookback: int,
    p_threshold: float,
) -> pd.DataFrame:
    pair_close = pair_close.copy()
    pair_close["cointegrated"] = 0
    pair_close["residual"] = 0.0
    pair_close["zscore"] = 0.0

    if len(pair_close) < lookback:
        return pair_close

    lr = LinearRegression()
    is_cointegrated = False

    for i in range(lookback, len(pair_close), lookback):
        x = pair_close[asset_x_col].iloc[i - lookback : i].values[:, None]
        y = pair_close[asset_y_col].iloc[i - lookback : i].values[:, None]

        if is_cointegrated:
            x_new = pair_close[asset_x_col].iloc[i : i + lookback].values[:, None]
            y_new = pair_close[asset_y_col].iloc[i : i + lookback].values[:, None]
            spread_back = y - lr.coef_ * x
            spread_forward = y_new - lr.coef_ * x_new
            spread_std = spread_back.std()
            if spread_std == 0 or np.isnan(spread_std):
                zscore = np.zeros_like(spread_forward)
            else:
                zscore = (spread_forward - spread_back.mean()) / spread_std

            pair_close.iloc[
                i : i + lookback, pair_close.columns.get_loc("cointegrated")
            ] = 1
            pair_close.iloc[i : i + lookback, pair_close.columns.get_loc("residual")] = spread_forward
            pair_close.iloc[i : i + lookback, pair_close.columns.get_loc("zscore")] = zscore

        _, p_value, _ = coint(x, y)
        is_cointegrated = p_value < p_threshold
        lr.fit(x, y)

    return pair_close


def build_backtest_source_df(
    pair_cfg: PairConfig,
    lookback: int,
    p_threshold: float,
) -> pd.DataFrame:
    asset_x_df = load_asset_ohlcv(pair_cfg.asset_x.price_col, pair_cfg.candle_size)
    asset_y_df = load_asset_ohlcv(pair_cfg.asset_y.price_col, pair_cfg.candle_size)
    asset_x_close = build_price_volume_frame(asset_x_df, pair_cfg.asset_x.price_col)
    asset_y_close = build_price_volume_frame(asset_y_df, pair_cfg.asset_y.price_col)
    pair_close = asset_x_close.join(asset_y_close, how="inner").dropna()
    if pair_close.empty:
        raise ValueError("No overlapping timestamps between assets; cannot run backtest.")
    pair_close = compute_cointegration_fields(
        pair_close,
        pair_cfg.asset_x.price_col,
        pair_cfg.asset_y.price_col,
        lookback,
        p_threshold,
    )
    pair_close.index = pd.to_datetime(pair_close.index)
    return pair_close


def ensure_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def select_date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    idx = pd.to_datetime(df.index)
    unique_dates = idx.normalize().unique()
    default_start = pd.to_datetime(unique_dates[0]).date()
    default_end = pd.to_datetime(unique_dates[-1]).date()

    selected_range = st.sidebar.date_input(
        "Select Backtest Data Range",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=default_end,
    )

    if isinstance(selected_range, tuple):
        if len(selected_range) == 2:
            start_date = ensure_naive_timestamp(selected_range[0])
            end_date = ensure_naive_timestamp(selected_range[1])
        elif len(selected_range) == 1:
            start_date = ensure_naive_timestamp(selected_range[0])
            end_date = ensure_naive_timestamp(selected_range[0])
        else:
            start_date = ensure_naive_timestamp(default_start)
            end_date = ensure_naive_timestamp(default_end)
    else:
        start_date = ensure_naive_timestamp(selected_range)
        end_date = ensure_naive_timestamp(selected_range)

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    return start_date, end_date


def filter_df_by_dates(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    start_ts = ensure_naive_timestamp(start_date).normalize()
    end_ts = ensure_naive_timestamp(end_date).normalize() + pd.Timedelta(days=1)
    mask = (df.index >= start_ts) & (df.index < end_ts)
    return df.loc[mask]


def apply_trading_logic(
    df: pd.DataFrame,
    pair_cfg: PairConfig,
    params: BacktestParams,
) -> pd.DataFrame:
    asset_x = pair_cfg.asset_x
    asset_y = pair_cfg.asset_y
    df = df.copy()
    df[asset_x.position_col] = 0.0
    df[asset_y.position_col] = 0.0

    df[f"{asset_x.price_col}_max_volume"] = np.floor(
        df[asset_x.volume_col] * params.max_volume_take_rate
    )
    df[f"{asset_y.price_col}_max_volume"] = np.floor(
        df[asset_y.volume_col] * params.max_volume_take_rate
    )
    max_volume = np.minimum(
        df[f"{asset_x.price_col}_max_volume"],
        df[f"{asset_y.price_col}_max_volume"],
    ).fillna(0.0)

    zscores = df["zscore"]
    if np.isinf(params.stop_entry_threshold):
        within_stop = pd.Series(True, index=df.index)
    else:
        within_stop = zscores.abs() <= params.stop_entry_threshold

    long_condition = (zscores <= -params.entry_threshold) & within_stop
    short_condition = (zscores >= params.entry_threshold) & within_stop

    df.loc[long_condition, asset_y.position_col] = params.num_contracts
    df.loc[short_condition, asset_y.position_col] = -params.num_contracts
    hold_long = zscores <= -params.exit_threshold
    hold_short = zscores >= params.exit_threshold

    df[asset_y.position_col] = df[asset_y.position_col].mask(
        (df[asset_y.position_col].shift() == -params.num_contracts) & hold_short,
        -params.num_contracts,
    )
    df[asset_y.position_col] = df[asset_y.position_col].mask(
        (df[asset_y.position_col].shift() == params.num_contracts) & hold_long,
        params.num_contracts,
    )

    df[asset_y.position_col] = df[asset_y.position_col] * max_volume
    df[asset_x.position_col] = -df[asset_y.position_col]

    return df


def calculate_pnl_for_asset(df: pd.DataFrame, asset: AssetConfig) -> None:
    price_diff = df[asset.price_col].diff().shift(-1).fillna(0.0)
    df[asset.pnl_col] = (
        df[asset.position_col] * price_diff / asset.tick_size * asset.tick_value
    )


def calculate_cash_and_margin(
    df: pd.DataFrame,
    pair_cfg: PairConfig,
    params: BacktestParams,
) -> None:
    margin_cols = []
    for asset in (pair_cfg.asset_x, pair_cfg.asset_y):
        exposure_col = f"{asset.price_col}_exposure"
        margin_col = f"{asset.price_col}_margin"
        exposure = (
            df[asset.position_col].abs()
            * df[asset.price_col]
            * asset.contract_size
        )
        df[exposure_col] = exposure
        df[margin_col] = exposure * params.initial_margin
        margin_cols.append(margin_col)
    df["cash_deployed"] = df[margin_cols].sum(axis=1)


def identify_trades(df: pd.DataFrame, pair_cfg: PairConfig) -> list[tuple[int, int]]:
    positions = df[pair_cfg.asset_y.position_col].fillna(0.0).to_numpy()
    trades: list[tuple[int, int]] = []
    in_trade = False
    start_idx = 0
    for idx, pos in enumerate(positions):
        has_position = abs(pos) > 1e-9
        if not in_trade and has_position:
            in_trade = True
            start_idx = idx
        elif in_trade and not has_position:
            trades.append((start_idx, idx))
            in_trade = False
    return trades


def calculate_trade_pnls(
    df: pd.DataFrame, pair_cfg: PairConfig, trades: list[tuple[int, int]] | None = None
) -> list[float]:
    if df.empty:
        return []
    trades = trades or identify_trades(df, pair_cfg)
    pnl_series = df["gross_pnl"].fillna(0.0).to_numpy()
    trade_pnls: list[float] = []
    for start_idx, end_idx in trades:
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(pnl_series))
        if start_idx >= end_idx:
            continue
        trade_pnls.append(float(pnl_series[start_idx:end_idx].sum()))
    return trade_pnls


def run_backtest(df: pd.DataFrame, pair_cfg: PairConfig, params: BacktestParams) -> pd.DataFrame:
    managed_df = apply_trading_logic(df, pair_cfg, params)
    for asset in (pair_cfg.asset_x, pair_cfg.asset_y):
        calculate_pnl_for_asset(managed_df, asset)
    calculate_cash_and_margin(managed_df, pair_cfg, params)
    managed_df["gross_pnl"] = (
        managed_df[pair_cfg.asset_x.pnl_col] + managed_df[pair_cfg.asset_y.pnl_col]
    )
    return managed_df


def compute_performance_metrics(
    df: pd.DataFrame,
    pair_cfg: PairConfig,
) -> Dict[str, float]:
    if df.empty:
        return {
            "num_trades": 0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "avg_cash_deployed": 0.0,
            "max_cash_deployed": 0.0,
            "win_rate": 0.0,
        }

    trades = identify_trades(df, pair_cfg)
    trade_pnls = calculate_trade_pnls(df, pair_cfg, trades)
    trade_count = len(trades)
    win_rate = (
        sum(1 for pnl in trade_pnls if pnl > 0.0) / trade_count if trade_count > 0 else 0.0
    )
    gross_pnl_series = df["gross_pnl"]
    pnl_cumsum = gross_pnl_series.cumsum()
    total_pnl = pnl_cumsum.iloc[-1] if not pnl_cumsum.empty else 0.0
    pnl_std = gross_pnl_series.std()
    if pnl_std and not np.isnan(pnl_std):
        sharpe_ratio = (
            gross_pnl_series.mean()
            / pnl_std
            * np.sqrt(252 * AVG_TRADING_HOURS * 60)
        )
    else:
        sharpe_ratio = 0.0
    metrics = {
        "num_trades": trade_count,
        "total_pnl": float(total_pnl),
        "sharpe_ratio": float(sharpe_ratio),
        "avg_cash_deployed": float(df["cash_deployed"].mean()),
        "max_cash_deployed": float(df["cash_deployed"].max()),
        "win_rate": float(win_rate),
    }
    return metrics


def build_blotter(df: pd.DataFrame, pair_cfg: PairConfig) -> pd.DataFrame:
    asset_x = pair_cfg.asset_x
    asset_y = pair_cfg.asset_y
    blotter_columns = ["trade_id", "timestamp", "action", "asset", "quantity", "price", "status"]
    trades = identify_trades(df, pair_cfg)
    if not trades:
        return pd.DataFrame(columns=blotter_columns)

    records = []
    for trade_num, (start_idx, end_idx) in enumerate(trades, start=1):
        if start_idx >= len(df) or end_idx >= len(df):
            continue
        entry_row = df.iloc[start_idx]
        exit_row = df.iloc[end_idx]
        qty_x = entry_row[asset_x.position_col]
        qty_y = entry_row[asset_y.position_col]
        if qty_x == 0 or qty_y == 0:
            continue

        records.append(
            {
                "trade_id": trade_num,
                "timestamp": entry_row.name,
                "action": "BUY" if qty_x > 0 else "SHORT",
                "asset": asset_x.display.upper(),
                "quantity": qty_x,
                "price": entry_row[asset_x.price_col],
                "status": "ENTRY",
            }
        )
        records.append(
            {
                "trade_id": trade_num,
                "timestamp": entry_row.name,
                "action": "BUY" if qty_y > 0 else "SHORT",
                "asset": asset_y.display.upper(),
                "quantity": qty_y,
                "price": entry_row[asset_y.price_col],
                "status": "ENTRY",
            }
        )
        records.append(
            {
                "trade_id": trade_num,
                "timestamp": exit_row.name,
                "action": "SELL" if qty_x > 0 else "COVER",
                "asset": asset_x.display.upper(),
                "quantity": -qty_x,
                "price": exit_row[asset_x.price_col],
                "status": "EXIT",
            }
        )
        records.append(
            {
                "trade_id": trade_num,
                "timestamp": exit_row.name,
                "action": "SELL" if qty_y > 0 else "COVER",
                "asset": asset_y.display.upper(),
                "quantity": -qty_y,
                "price": exit_row[asset_y.price_col],
                "status": "EXIT",
            }
        )

    blotter_df = pd.DataFrame.from_records(records, columns=blotter_columns)
    blotter_df["trade_id"] = blotter_df["trade_id"].astype(str)
    return blotter_df


def main() -> None:
    st.sidebar.header("Backtest Controls")
    pair_name = st.sidebar.selectbox("Select Pair", list(PAIRS.keys()))
    pair_cfg = PAIRS[pair_name]
    lookback = st.sidebar.number_input(
        "Cointegration Lookback (bars)",
        min_value=20,
        max_value=2000,
        value=DEFAULT_COINTEGRATION_LOOKBACK,
        step=10,
    )
    p_threshold = st.sidebar.number_input(
        "Cointegration p-value threshold",
        min_value=0.001,
        max_value=0.5,
        value=float(DEFAULT_COINTEGRATION_P_THRESHOLD),
        step=0.01,
        format="%.3f",
    )

    slider_min = float(DEFAULT_ENTRY_THRESHOLD)
    slider_max = float(max(slider_min + 0.1, 10.0))
    slider_default = DEFAULT_STOP_ENTRY_THRESHOLD
    if slider_default is None or not np.isfinite(slider_default):
        slider_default = slider_min
    slider_default = float(max(slider_min, min(slider_default, slider_max)))
    stop_entry_threshold = st.sidebar.slider(
        "Max Entry Z-Score (stop new entries beyond this)",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_default,
        step=0.1,
        help="Blocks new trades if |z-score| exceeds this risk guardrail.",
    )
    disable_stop = st.sidebar.checkbox("Disable stop-entry guard", value=False)
    if disable_stop:
        stop_entry_threshold = float("inf")

    params = BacktestParams(
        entry_threshold=DEFAULT_ENTRY_THRESHOLD,
        exit_threshold=DEFAULT_EXIT_THRESHOLD,
        stop_entry_threshold=stop_entry_threshold,
        initial_margin=DEFAULT_INITIAL_MARGIN,
        max_volume_take_rate=DEFAULT_MAX_VOLUME_TAKE_RATE,
        num_contracts=DEFAULT_NUM_CONTRACTS,
    )

    try:
        raw_df = build_backtest_source_df(pair_cfg, int(lookback), float(p_threshold))
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    start_date, end_date = select_date_range(raw_df)
    bt_df = run_backtest(raw_df, pair_cfg, params)
    selected_df = filter_df_by_dates(bt_df, start_date, end_date)

    start_label = start_date.strftime("%Y-%m-%d")
    end_label = end_date.strftime("%Y-%m-%d")

    # Plot all-data z-score
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3)
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["zscore"], name="Z-Score"), row=1, col=1)
    fig.update_layout(title="All Data Z-Score of Residuals")
    st.plotly_chart(fig)

    # Plot selected range z-score
    fig_selected = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3)
    fig_selected.add_trace(
        go.Scatter(x=selected_df.index, y=selected_df["zscore"], name="Z-Score"),
        row=1,
        col=1,
    )
    fig_selected.update_layout(title="Selected Data Z-Score of Residuals")
    st.plotly_chart(fig_selected)

    st.write(selected_df)

    df_display = selected_df[
        [
            pair_cfg.asset_x.price_col,
            pair_cfg.asset_y.price_col,
            "zscore",
            pair_cfg.asset_y.position_col,
            "gross_pnl",
            "cash_deployed",
        ]
    ].rename(
        columns={
            pair_cfg.asset_y.position_col: f"Long({pair_cfg.asset_y.display})/Short({pair_cfg.asset_x.display}) Position",
            "gross_pnl": "Gross PnL",
            "cash_deployed": "Cash Deployed",
            "zscore": "Z-Score",
            pair_cfg.asset_x.price_col: f"{pair_cfg.asset_x.display} Close Price",
            pair_cfg.asset_y.price_col: f"{pair_cfg.asset_y.display} Close Price",
        }
    )
    st.dataframe(df_display)

    # Positions over time
    fig_positions = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3)
    fig_positions.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df[pair_cfg.asset_y.position_col],
            name=f"Position {pair_cfg.asset_y.display}",
        ),
        row=1,
        col=1,
    )
    fig_positions.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df[pair_cfg.asset_x.position_col],
            name=f"Position {pair_cfg.asset_x.display}",
        ),
        row=1,
        col=1,
    )
    fig_positions.update_layout(title=f"Positions Over Time for {pair_name}")
    st.plotly_chart(fig_positions)

    # Cumulative gross PnL
    fig_cum_pnl = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3)
    fig_cum_pnl.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df["gross_pnl"].cumsum(),
            name="Gross PnL",
        ),
        row=1,
        col=1,
    )
    fig_cum_pnl.update_layout(title=f"Cumulative Gross PnL Over Time for {pair_name}")
    st.plotly_chart(fig_cum_pnl)

    # Bar gross PnL
    fig_gross_pnl = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3)
    fig_gross_pnl.add_trace(
        go.Bar(
            x=selected_df.index,
            y=selected_df["gross_pnl"],
            name="Gross PnL",
            marker_color=np.where(selected_df["gross_pnl"] >= 0, "green", "red"),
        ),
        row=1,
        col=1,
    )
    fig_gross_pnl.update_layout(title=f"Gross PnL Over Time for {pair_name}")
    st.plotly_chart(fig_gross_pnl)

    metrics = compute_performance_metrics(selected_df, pair_cfg)
    st.subheader("Performance Metrics")
    st.markdown(
        f"**Number of Trades Executed From {start_label} to {end_label}:** {metrics['num_trades']}"
    )
    st.markdown("**Average Hold Period:** 1 minute")
    st.markdown(f"**Total Gross PnL:** ${metrics['total_pnl']:,.2f}")
    st.markdown(f"**Annualized Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}")
    st.markdown(
        f"**Average Cash Deployed per Minute:** ${metrics['avg_cash_deployed']:,.2f}"
    )
    st.markdown(f"**Maximum Cash Deployed:** ${metrics['max_cash_deployed']:,.2f}")
    st.markdown(f"**Win Rate:** {metrics['win_rate']:.2%}")

    blotter_df = build_blotter(selected_df, pair_cfg)
    df_xlsx = to_excel(blotter_df)
    st.sidebar.download_button(
        label="Download Blotter as Excel",
        data=df_xlsx,
        file_name="blotter.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
