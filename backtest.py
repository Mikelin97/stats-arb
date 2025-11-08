from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


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
    asset_x: AssetConfig
    asset_y: AssetConfig
    data_file: str


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


PAIRS: Dict[str, PairConfig] = {
    "WTI Future vs. Brent Future": PairConfig(
        asset_x=AssetConfig(
            price_col="pair1_wti_oil_future",
            display="WTI Future",
            tick_size=0.01,
            tick_value=10.0,
            contract_size=1000.0,
        ),
        asset_y=AssetConfig(
            price_col="pair1_brent_oil_future",
            display="Brent Future",
            tick_size=0.01,
            tick_value=10.0,
            contract_size=1000.0,
        ),
        data_file="pair_1_cointegration_1min",
    ),
    "NatGas HH vs. NatGas LS": PairConfig(
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
        data_file="pair_6_cointegration_1min",
    ),
    "MSTR vs. IBIT": PairConfig(
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
        data_file="pair_7_cointegration_1min",
    ),
    "Gold Future vs. Micro Gold Future": PairConfig(
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
        data_file="pair_10_cointegration_1min",
    ),
    "Silver Future vs. Micro Silver Future": PairConfig(
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
        data_file="pair_11_cointegration_1min",
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


def load_backtest_data(pair_cfg: PairConfig) -> pd.DataFrame:
    df = pd.read_csv(f"data/{pair_cfg.data_file}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df.sort_index()
    return df


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


def calculate_trade_pnls(df: pd.DataFrame, pair_cfg: PairConfig) -> list[float]:
    if df.empty:
        return []
    pos_series = df[pair_cfg.asset_y.position_col].fillna(0.0).to_numpy()
    pnl_series = df["gross_pnl"].fillna(0.0).to_numpy()
    states = np.where(np.abs(pos_series) > 1e-9, np.sign(pos_series), 0.0)
    trade_pnls: list[float] = []
    current_pnl = 0.0
    in_trade = False
    prev_state = 0.0
    for state, row_pnl in zip(states, pnl_series):
        if state != prev_state:
            if prev_state != 0.0 and in_trade:
                trade_pnls.append(current_pnl)
                current_pnl = 0.0
                in_trade = False
        if state != 0.0 and not in_trade:
            in_trade = True
        if in_trade:
            current_pnl += row_pnl
        prev_state = state
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

    trades = calculate_trade_pnls(df, pair_cfg)
    trade_count = len(trades)
    win_rate = (
        sum(1 for pnl in trades if pnl > 0.0) / trade_count if trade_count > 0 else 0.0
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
    base_cols = [
        asset_x.price_col,
        asset_y.price_col,
        asset_x.position_col,
        asset_y.position_col,
        "gross_pnl",
    ]
    blotter_columns = ["trade_id", "timestamp", "action", "asset", "quantity", "price", "status"]
    if df.empty:
        return pd.DataFrame(columns=blotter_columns)

    blotter_raw_df = df[base_cols].copy()
    blotter_raw_df["status"] = np.where(
        blotter_raw_df[asset_x.position_col] != 0,
        "ENTRY",
        "NO ACTION",
    )

    trade_id = 1
    records = []
    for i in range(len(blotter_raw_df) - 1):
        row = blotter_raw_df.iloc[i]
        next_row = blotter_raw_df.iloc[i + 1] if i + 1 < len(blotter_raw_df) else None

        if row.status != "ENTRY":
            continue

        record1 = {
            "trade_id": trade_id,
            "timestamp": row.name,
            "action": "BUY" if row[asset_x.position_col] > 0 else "SHORT",
            "asset": asset_x.display.upper(),
            "quantity": row[asset_x.position_col],
            "price": row[asset_x.price_col],
            "status": "ENTRY",
        }
        record2 = {
            "trade_id": trade_id,
            "timestamp": row.name,
            "action": "BUY" if row[asset_y.position_col] > 0 else "SHORT",
            "asset": asset_y.display.upper(),
            "quantity": row[asset_y.position_col],
            "price": row[asset_y.price_col],
            "status": "ENTRY",
        }
        record3 = {
            "trade_id": trade_id,
            "timestamp": next_row.name if next_row is not None else None,
            "action": "SELL" if row[asset_x.position_col] > 0 else "COVER",
            "quantity": -row[asset_x.position_col],
            "asset": asset_x.display.upper(),
            "price": next_row[asset_x.price_col] if next_row is not None else None,
            "status": "EXIT",
        }
        record4 = {
            "trade_id": trade_id,
            "timestamp": next_row.name if next_row is not None else None,
            "action": "SELL" if row[asset_y.position_col] > 0 else "COVER",
            "quantity": -row[asset_y.position_col],
            "asset": asset_y.display.upper(),
            "price": next_row[asset_y.price_col] if next_row is not None else None,
            "status": "EXIT",
        }
        records.extend([record1, record2, record3, record4])
        trade_id += 1

    blotter_df = pd.DataFrame.from_records(records, columns=blotter_columns)
    if blotter_df.empty:
        return blotter_df
    blotter_df["trade_id"] = blotter_df["trade_id"].astype(str)
    return blotter_df


def main() -> None:
    st.sidebar.header("Backtest Controls")
    pair_name = st.sidebar.selectbox("Select Pair", list(PAIRS.keys()))
    pair_cfg = PAIRS[pair_name]

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

    raw_df = load_backtest_data(pair_cfg)
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
