import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import streamlit as st

DATA_DIR = Path("data")
LOOKBACK_DEFAULT = 100
P_THRESHOLD_DEFAULT = 0.05

PAIRS: Dict[str, str] = {
    "WTI vs. Brent": "pair1",
    "Gold vs. Silver": "pair2",
    "SOFR 3M Future vs. DUK Spot": "pair3",
    "Corn vs. Soybean Oil": "pair4",
    "Bitcoin ETF vs. Ethereum ETF": "pair5",
    "NatGas HH vs. NatGas LS": "pair6",
    "MSTR vs. IBIT": "pair7",
    "TXN vs. ADI": "pair8",
    "RBOB Gas vs. ULSD Gas": "pair9",
    "Gold Future vs. Micro Gold Future": "pair10",
    "Silver Future vs. Micro Silver Future": "pair11",
}
PAIR_IDS = {v: k for k, v in PAIRS.items()}

CANDLE_SIZES = {
    "1 minute": "1min",
    "2 minutes": "2min",
    "3 minutes": "3min",
    "4 minutes": "4min",
    "5 minutes": "5min",
    "15 minutes": "15min",
    "1 hour": "1h",
}


def list_pairs_cli() -> None:
    print("Available pairs for analysis/backtests:")
    for label, pair_id in PAIRS.items():
        print(f"  {pair_id:<6} {label}")


def resample_asset(df: pd.DataFrame, candle_size: str) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if candle_size == "1min":
        return df
    return df.resample(candle_size).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).ffill()


def load_pair_assets(pair_id: str, candle_size: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    files = sorted(
        f
        for f in DATA_DIR.glob(f"{pair_id}_*.csv")
        if "cointegration" not in f.stem
    )
    if len(files) < 2:
        raise FileNotFoundError(
            f"Expected at least two raw data files for {pair_id}. "
            "Run FetchData.py to download the required assets."
        )
    asset_frames = [pd.read_csv(file, index_col=0).ffill() for file in files[:2]]
    asset_frames = [resample_asset(df, candle_size) for df in asset_frames]
    asset_keys = [file.stem for file in files[:2]]
    return asset_frames[0], asset_frames[1], asset_keys


def prepare_pair_close(
    asset1: pd.DataFrame, asset2: pd.DataFrame, asset_keys: List[str]
) -> Tuple[pd.DataFrame, str, str]:
    asset1_key, asset2_key = asset_keys[:2]
    asset1_close = asset1[["close", "volume"]].rename(
        columns={"close": asset1_key, "volume": f"{asset1_key}_volume"}
    )
    asset2_close = asset2[["close", "volume"]].rename(
        columns={"close": asset2_key, "volume": f"{asset2_key}_volume"}
    )
    pair_close = asset1_close.join(asset2_close, how="outer").dropna()
    return pair_close, asset1_key, asset2_key


def compute_cointegration_fields(
    pair_close: pd.DataFrame,
    asset1_col: str,
    asset2_col: str,
    lookback: int = LOOKBACK_DEFAULT,
    p_threshold: float = P_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    pair_close = pair_close.copy()
    pair_close["cointegrated"] = 0
    pair_close["residual"] = 0.0
    pair_close["zscore"] = 0.0

    is_cointegrated = False
    lr = LinearRegression()

    for i in range(lookback, len(pair_close), lookback):
        x = pair_close[asset1_col].iloc[i - lookback : i].values[:, None]
        y = pair_close[asset2_col].iloc[i - lookback : i].values[:, None]

        if is_cointegrated:
            x_new = pair_close[asset1_col].iloc[i : i + lookback].values[:, None]
            y_new = pair_close[asset2_col].iloc[i : i + lookback].values[:, None]
            spread_back = y - lr.coef_ * x
            spread_forward = y_new - lr.coef_ * x_new
            spread_std = spread_back.std()
            if spread_std == 0:
                zscore = np.zeros_like(spread_forward)
            else:
                zscore = (spread_forward - spread_back.mean()) / spread_std

            pair_close.iloc[i : i + lookback, pair_close.columns.get_loc("cointegrated")] = 1
            pair_close.iloc[i : i + lookback, pair_close.columns.get_loc("residual")] = spread_forward
            pair_close.iloc[i : i + lookback, pair_close.columns.get_loc("zscore")] = zscore

        _, p, _ = coint(x, y)
        is_cointegrated = p < p_threshold
        lr.fit(x, y)

    return pair_close


def pretty_asset_label(pair_id: str, asset_key: str) -> str:
    cleaned = asset_key.replace(f"{pair_id}_", "")
    cleaned = cleaned.replace("ohlcv-1m", "").replace("ohlcv-1min", "")
    return " ".join(part.capitalize() for part in cleaned.split("_") if part)


def export_cointegration_file(
    pair_id: str,
    candle_size: str,
    output_dir: Path,
    lookback: int = LOOKBACK_DEFAULT,
    p_threshold: float = P_THRESHOLD_DEFAULT,
) -> Path:
    asset1, asset2, asset_keys = load_pair_assets(pair_id, candle_size)
    pair_close, asset1_col, asset2_col = prepare_pair_close(asset1, asset2, asset_keys)
    pair_close = compute_cointegration_fields(pair_close, asset1_col, asset2_col, lookback, p_threshold)
    pair_close.index = pd.to_datetime(pair_close.index)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{pair_id}_cointegration_{candle_size}.csv"
    pair_close.to_csv(file_path)
    return file_path


def run_streamlit_app() -> None:
    st.sidebar.header("Pairs Cointegration Analysis")
    selected_pair = st.sidebar.selectbox("Select Pair", list(PAIRS.keys()))
    current_pair = PAIRS[selected_pair]

    selected_candle_size = st.sidebar.selectbox("Select Candle Size", list(CANDLE_SIZES.keys()))
    current_candle_size = CANDLE_SIZES[selected_candle_size]

    asset1, asset2, asset_keys = load_pair_assets(current_pair, current_candle_size)
    pair_close, asset1_col, asset2_col = prepare_pair_close(asset1, asset2, asset_keys)
    pair_close = compute_cointegration_fields(pair_close, asset1_col, asset2_col)

    export_df = pair_close.copy()
    st.write(export_df)

    export_df.set_index(pd.to_datetime(export_df.index), inplace=True)
    export_df.to_csv(
        DATA_DIR / f"{current_pair}_cointegration_{current_candle_size}.csv",
        index=True,
    )

    temp = pd.to_datetime(pair_close.index)
    unique_dates = temp.normalize().unique()
    unique_dates_fmt = [d.strftime("%Y-%m-%d") for d in unique_dates]

    selected_date = st.sidebar.selectbox("Select Date", unique_dates_fmt)
    selected_index = unique_dates_fmt.index(selected_date)

    if selected_index + 1 < len(unique_dates_fmt):
        end_date = unique_dates_fmt[selected_index + 1]
        selected_data = pair_close.loc[selected_date:end_date]
        selected_asset1 = asset1.loc[selected_data.index]
        selected_asset2 = asset2.loc[selected_data.index]
    else:
        selected_data = pair_close.loc[selected_date:]
        selected_asset1 = asset1.loc[selected_data.index]
        selected_asset2 = asset2.loc[selected_data.index]

    blocks = (selected_data["cointegrated"].diff().fillna(0) != 0).cumsum()
    coint_blocks = blocks[selected_data["cointegrated"] == 1]
    coint_period_ids = coint_blocks.unique()

    asset1_label = pretty_asset_label(current_pair, asset1_col)
    asset2_label = pretty_asset_label(current_pair, asset2_col)

    if len(coint_period_ids) == 0:
        st.write("#### No cointegrated periods found for the selected date.")
        st.write(selected_data)
        st.write(f"#### {asset1_label} data:")
        st.write(selected_asset1)
        st.write(f"#### {asset2_label} data:")
        st.write(selected_asset2)
        return

    for i, block_id in enumerate(coint_period_ids):
        mask = (blocks == block_id) & (selected_data["cointegrated"] == 1)
        period_df = selected_data[mask]
        start_time = period_df.index[0]
        end_time = period_df.index[-1]

        st.write(f"#### Close price data from {start_time} to {end_time}")
        st.write(period_df)
        st.write(f"#### {asset1_label} data:")
        st.write(asset1.loc[period_df.index])
        st.write(f"#### {asset2_label} data:")
        st.write(asset2.loc[period_df.index])

        num_plots = 3
        fig = make_subplots(
            rows=num_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.3,
            specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=("Z-Score of Residuals", f"{asset1_label} Price", f"{asset2_label} Price"),
        )
        fig.add_trace(go.Scatter(x=period_df.index, y=period_df["zscore"], name="Z-Score"), row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=asset1.loc[period_df.index].index,
                y=asset1.loc[period_df.index]["volume"],
                name="Volume",
                marker_color="lightgray",
                opacity=0.4,
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Candlestick(
                x=asset1.loc[period_df.index].index,
                open=asset1.loc[period_df.index]["open"],
                high=asset1.loc[period_df.index]["high"],
                low=asset1.loc[period_df.index]["low"],
                close=asset1.loc[period_df.index]["close"],
                name=asset1_label,
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=asset2.loc[period_df.index].index,
                y=asset2.loc[period_df.index]["volume"],
                name="Volume",
                marker_color="lightgray",
                opacity=0.4,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Candlestick(
                x=asset2.loc[period_df.index].index,
                open=asset2.loc[period_df.index]["open"],
                high=asset2.loc[period_df.index]["high"],
                low=asset2.loc[period_df.index]["low"],
                close=asset2.loc[period_df.index]["close"],
                name=asset2_label,
            ),
            row=3,
            col=1,
            secondary_y=False,
        )

        fig.update_layout(
            height=1000,
            width=800,
            title_text=f"Cointegrated Period {i + 1} Analysis : from {start_time} to {end_time}",
            hovermode="x unified",
            barmode="overlay",
        )
        fig.update_xaxes(showticklabels=True, row=1, col=1)
        st.plotly_chart(fig)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairs cointegration analysis utilities.")
    parser.add_argument("--list", action="store_true", help="List available pairs and exit.")
    parser.add_argument("--export-pair", type=str, help="Pair identifier to export cointegration file (e.g. pair7).")
    parser.add_argument("--candle-size", type=str, default="1min", help="Candle size for export (default: 1min).")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Directory for exported files.")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DEFAULT, help="Lookback window for cointegration.")
    parser.add_argument("--p-threshold", type=float, default=P_THRESHOLD_DEFAULT, help="P-value threshold for cointegration.")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_cli_args()
    if args.list:
        list_pairs_cli()
        if not args.export_pair:
            return

    if args.export_pair:
        pair_id = args.export_pair
        if pair_id not in PAIR_IDS:
            raise ValueError(f"Unknown pair identifier: {pair_id}")
        export_path = export_cointegration_file(
            pair_id=pair_id,
            candle_size=args.candle_size,
            output_dir=args.output_dir,
            lookback=args.lookback,
            p_threshold=args.p_threshold,
        )
        print(f"Saved cointegration file to {export_path}")
        return

    run_streamlit_app()


if __name__ == "__main__":
    main()
