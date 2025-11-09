import numpy as np
import pandas as pd
import streamlit as st

from backtest import (
    DEFAULT_STOP_ENTRY_THRESHOLD,
    PAIRS,
    BacktestParams,
    AVG_TRADING_HOURS,
    DEFAULT_COINTEGRATION_LOOKBACK,
    DEFAULT_COINTEGRATION_P_THRESHOLD,
    DEFAULT_ENTRY_THRESHOLD,
    DEFAULT_EXIT_THRESHOLD,
    DEFAULT_INITIAL_MARGIN,
    DEFAULT_MAX_VOLUME_TAKE_RATE,
    DEFAULT_NUM_CONTRACTS,
    build_backtest_source_df,
    run_backtest,
    filter_df_by_dates,
    identify_trades,
    calculate_trade_pnls,
    compute_performance_metrics,
)

SAFETY_MARGIN_DEFAULT = 1.5
DEFAULT_PORTFOLIO_CAPITAL = 50_000_000
DEFAULT_MONEY_MARKET_RATE = 0.04
DEFAULT_BENCHMARK_WEEKLY_RETURN = 0.09 / 52


def calculate_max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    equity_curve = pnl_series.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(drawdown.min())


def aggregate_portfolio_metrics(
    portfolio_pnl: pd.Series,
    trade_pnls: list[float],
    initial_capital: float,
    money_market_capital: float,
    benchmark_capital: float,
    years: float,
    money_market_rate: float,
    benchmark_weekly_return: float,
) -> dict:
    metrics = {
        "total_pnl": 0.0,
        "sharpe_ratio": 0.0,
        "carg": 0.0,
        "win_rate": 0.0,
        "money_market_pnl": 0.0,
        "benchmark_pnl": 0.0,
    }
    if portfolio_pnl.empty:
        return metrics

    total_minutes = len(portfolio_pnl)
    annualization_factor = 252 * AVG_TRADING_HOURS * 60
    pnl_mean = portfolio_pnl.mean()
    pnl_std = portfolio_pnl.std()
    if pnl_std and not np.isnan(pnl_std):
        metrics["sharpe_ratio"] = float(pnl_mean / pnl_std * np.sqrt(annualization_factor))
    metrics["total_pnl"] = float(portfolio_pnl.sum())

    metrics["money_market_pnl"] = float(money_market_capital * money_market_rate * years)
    metrics["benchmark_pnl"] = float(benchmark_capital * benchmark_weekly_return)

    ending_equity = (
        initial_capital
        + portfolio_pnl.cumsum().iloc[-1]
        + metrics["money_market_pnl"]
        + metrics["benchmark_pnl"]
    )
    if initial_capital > 0 and years > 0:
        metrics["carg"] = float((ending_equity / initial_capital) ** (1 / years) - 1)

    total_trades = len(trade_pnls)
    if total_trades > 0:
        metrics["win_rate"] = float(
            sum(1 for pnl in trade_pnls if pnl > 0.0) / total_trades
        )
    return metrics


def main():
    st.sidebar.header("Portfolio Controls")
    available_pairs = list(PAIRS.keys())
    selected_pairs = st.sidebar.multiselect(
        "Select Pairs to Include", available_pairs, default=available_pairs
    )
    if not selected_pairs:
        st.warning("Select at least one pair to build the portfolio.")
        st.stop()

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
    stop_entry_threshold = st.sidebar.slider(
        "Max Entry Z-Score",
        min_value=float(DEFAULT_ENTRY_THRESHOLD),
        max_value=10.0,
        value=float(DEFAULT_STOP_ENTRY_THRESHOLD),
        step=0.1,
    )
    safety_margin_multiple = st.sidebar.number_input(
        "Safety Margin Multiple", min_value=1.0, max_value=5.0, value=SAFETY_MARGIN_DEFAULT, step=0.1
    )
    portfolio_capital = st.sidebar.number_input(
        "Total Portfolio Capital (USD)",
        min_value=0.0,
        value=float(DEFAULT_PORTFOLIO_CAPITAL),
        step=10000.0,
    )

    params = BacktestParams(
        entry_threshold=DEFAULT_ENTRY_THRESHOLD,
        exit_threshold=DEFAULT_EXIT_THRESHOLD,
        stop_entry_threshold=stop_entry_threshold,
        initial_margin=DEFAULT_INITIAL_MARGIN,
        max_volume_take_rate=DEFAULT_MAX_VOLUME_TAKE_RATE,
        num_contracts=DEFAULT_NUM_CONTRACTS,
    )

    pair_results: dict[str, pd.DataFrame] = {}
    min_dates = []
    max_dates = []
    for pair_name in selected_pairs:
        pair_cfg = PAIRS[pair_name]
        try:
            source_df = build_backtest_source_df(pair_cfg, int(lookback), float(p_threshold))
        except (FileNotFoundError, ValueError) as exc:
            st.warning(f"{pair_name}: {exc}")
            continue
        bt_df = run_backtest(source_df, pair_cfg, params)
        if bt_df.empty:
            continue
        pair_results[pair_name] = bt_df
        min_dates.append(bt_df.index.min())
        max_dates.append(bt_df.index.max())

    if not pair_results:
        st.error("Unable to build portfolio. None of the selected pairs produced trades.")
        st.stop()

    default_start = max(min_dates)
    default_end = min(max_dates)
    if default_start > default_end:
        default_start = min(min_dates)
        default_end = max(max_dates)

    date_range = st.sidebar.date_input(
        "Portfolio Date Range",
        value=(default_start.date(), default_end.date()),
        min_value=min_dates and min(min_dates).date(),
        max_value=max_dates and max(max_dates).date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = pd.Timestamp(date_range)
        end_date = pd.Timestamp(date_range)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    money_market_rate = st.sidebar.number_input(
        "Money Market Annual Yield",
        min_value=0.0,
        max_value=0.2,
        value=DEFAULT_MONEY_MARKET_RATE,
        step=0.005,
        format="%.4f",
    )
    benchmark_weekly_return = st.sidebar.number_input(
        "Benchmark Weekly Return",
        min_value=-0.2,
        max_value=0.2,
        value=DEFAULT_BENCHMARK_WEEKLY_RETURN,
        step=0.001,
        format="%.4f",
    )

    pair_rows = []
    portfolio_series = []
    all_trade_pnls: list[float] = []
    total_reserved_capital = 0.0

    for pair_name, bt_df in pair_results.items():
        pair_cfg = PAIRS[pair_name]
        selected_df = filter_df_by_dates(bt_df, start_date, end_date)
        if selected_df.empty:
            continue
        metrics = compute_performance_metrics(selected_df, pair_cfg)
        max_drawdown = calculate_max_drawdown(selected_df["gross_pnl"])
        safe_capital = float(
            selected_df["cash_deployed"].max() * safety_margin_multiple
            if not selected_df["cash_deployed"].empty
            else 0.0
        )
        total_reserved_capital += safe_capital
        trades = identify_trades(selected_df, pair_cfg)
        trade_pnls = calculate_trade_pnls(selected_df, pair_cfg, trades)
        all_trade_pnls.extend(trade_pnls)
        pair_rows.append(
            {
                "Pair": pair_name,
                "Gross PnL": metrics["total_pnl"],
                "Number of Trades": metrics["num_trades"],
                "Annualized Sharpe": metrics["sharpe_ratio"],
                "Max Drawdown": max_drawdown,
                "Win Rate": metrics["win_rate"],
                "Safe Capital Reserved": safe_capital,
            }
        )
        portfolio_series.append(selected_df["gross_pnl"].rename(pair_name))

    if not pair_rows:
        st.error("No trades found within the selected date range.")
        st.stop()

    pair_table = pd.DataFrame(pair_rows)
    st.subheader("Pair-Level Summary")
    st.dataframe(
        pair_table.style.format(
            {
                "Gross PnL": "${:,.0f}",
                "Annualized Sharpe": "{:.2f}",
                "Max Drawdown": "${:,.0f}",
                "Win Rate": "{:.2%}",
                "Safe Capital Reserved": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    remaining_capital = max(portfolio_capital - total_reserved_capital, 0.0)
    money_market_buffer = remaining_capital * 0.5
    benchmark_allocation = remaining_capital - money_market_buffer
    st.caption(f"Total Safe Capital Reserved: ${total_reserved_capital:,.0f}")
    st.caption(f"Money Market Allocation: ${money_market_buffer:,.0f}")
    st.caption(f"Benchmark Allocation: ${benchmark_allocation:,.0f}")

    portfolio_df = pd.concat(portfolio_series, axis=1).fillna(0.0).sort_index()
    portfolio_pnl = portfolio_df.sum(axis=1)
    total_minutes = len(portfolio_pnl)
    annualization_factor = 252 * AVG_TRADING_HOURS * 60
    years = total_minutes / annualization_factor if annualization_factor else 0
    portfolio_metrics = aggregate_portfolio_metrics(
        portfolio_pnl,
        all_trade_pnls,
        portfolio_capital,
        money_market_buffer,
        benchmark_allocation,
        years,
        money_market_rate,
        benchmark_weekly_return,
    )

    portfolio_table = pd.DataFrame(
        [
            {
                "Trading Gross PnL": portfolio_metrics["total_pnl"],
                "Money Market PnL": portfolio_metrics["money_market_pnl"],
                "Benchmark PnL": portfolio_metrics["benchmark_pnl"],
                "Total Gross PnL": (
                    portfolio_metrics["total_pnl"]
                    + portfolio_metrics["money_market_pnl"]
                    + portfolio_metrics["benchmark_pnl"]
                ),
                "Portfolio Sharpe": portfolio_metrics["sharpe_ratio"],
                "CARG": portfolio_metrics["carg"],
                "Win Rate": portfolio_metrics["win_rate"],
            }
        ]
    )
    st.subheader("Portfolio Performance")
    st.dataframe(
        portfolio_table.style.format(
            {
                "Trading Gross PnL": "${:,.0f}",
                "Money Market PnL": "${:,.0f}",
                "Benchmark PnL": "${:,.0f}",
                "Total Gross PnL": "${:,.0f}",
                "Portfolio Sharpe": "{:.2f}",
                "CARG": "{:.2%}",
                "Win Rate": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.line_chart(portfolio_pnl.cumsum().rename("Portfolio Equity Curve"))


if __name__ == "__main__":
    main()
