import pandas as pd
from io import BytesIO
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def to_excel(df):
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


### read in backtest data
df = pd.read_csv(
    "data/pair_1_cointegration_1min.csv",
    index_col=0,
)

### ENV VARIABLES
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.5
TICK_SIZE_X = 0.01
TICK_SIZE_Y = 0.01
TICK_VALUE_X = 10.0
TICK_VALUE_Y = 10.0
NOTIONAL_SIZE = 1000
INITIAL_MARGIN = 0.1
MAX_VOLUME_TAKE_RATE = 0.1
NUM_CONTRACT = 1

PAIRS = {
    "WTI vs. Brent": ["pair1_wti_oil_future", "pair1_brent_oil_future"],
}
BACKTEST_DURATION = {"1 Hour": "1h", "1 Day": "1d", "1 Week": "1w"}

selected_pair = st.sidebar.selectbox("Select Pair", list(PAIRS.keys()))
pair = PAIRS[selected_pair]


temp = pd.to_datetime(df.index)
unique_dates = temp.normalize().unique()
start_date = unique_dates[0].strftime("%Y-%m-%d")
end_date = unique_dates[-1].strftime("%Y-%m-%d")
# unique_dates = [d.strftime('%Y-%m-%d') for d in unique_dates]
selected_backtest_data_range = st.sidebar.date_input(
    "Select Backtest Data Range",
    value=(unique_dates[0], unique_dates[-1]),
    min_value=unique_dates[0],
    max_value=unique_dates[-1],
)
if (
    len(selected_backtest_data_range) == 2
    and selected_backtest_data_range[0] != selected_backtest_data_range[1]
):
    start_date = selected_backtest_data_range[0].strftime("%Y-%m-%d")
    end_date = selected_backtest_data_range[1].strftime("%Y-%m-%d")


position_x = f"position_{pair[0]}"
position_y = f"position_{pair[1]}"


df[position_x] = 0
df[position_y] = 0
df[pair[0] + "_max_volume"] = np.floor(df[f"{pair[0]}_volume"] * MAX_VOLUME_TAKE_RATE)
df[pair[1] + "_max_volume"] = np.floor(df[f"{pair[1]}_volume"] * MAX_VOLUME_TAKE_RATE)

# Entry positions based on z-score thresholds
df.loc[df.zscore < -ENTRY_THRESHOLD, position_y] = NUM_CONTRACT
df.loc[df.zscore > ENTRY_THRESHOLD, position_y] = -NUM_CONTRACT
df[position_x] = -df[position_y]

# Exit positions when z-score crosses +/-0.5
hold_y_long = df["zscore"].apply(lambda z: 1 if z <= -EXIT_THRESHOLD else 0)
hold_y_short = df["zscore"].apply(lambda z: 1 if z >= EXIT_THRESHOLD else 0)

### so what happen with this mask it is only holding for one more period, rather than holding until the exit condition is met
df[position_y] = df[position_y].mask(
    (df[position_y].shift() == -NUM_CONTRACT) & (hold_y_short == 1), -NUM_CONTRACT
)
df[position_y] = df[position_y].mask(
    (df[position_y].shift() == NUM_CONTRACT) & (hold_y_long == 1), NUM_CONTRACT
)

### turn to the max volume allowed
df[position_y] = df[position_y] * np.minimum(
    df[pair[1] + "_max_volume"], df[pair[0] + "_max_volume"]
)

df[position_x] = -df[position_y]


df[f"pnl_{pair[0]}"] = 0.0
df[f"pnl_{pair[1]}"] = 0.0
df.iloc[:-1, df.columns.get_loc(f"pnl_{pair[0]}")] = (
    df[position_x].iloc[:-1].values
    * df[pair[0]].diff()[1:].values
    / TICK_SIZE_X
    * TICK_VALUE_X
)
df.iloc[:-1, df.columns.get_loc(f"pnl_{pair[1]}")] = (
    df[position_y].iloc[:-1].values
    * df[pair[1]].diff()[1:].values
    / TICK_SIZE_Y
    * TICK_VALUE_Y
)

df["gross_pnl"] = df[f"pnl_{pair[0]}"] + df[f"pnl_{pair[1]}"]

long_x = df[position_x] > 0
long_y = df[position_y] > 0

df.loc[long_x, "x_long_cash"] = df[long_x][position_x] * df[long_x][pair[0]]
df.loc[long_y, "y_long_cash"] = df[long_y][position_y] * df[long_y][pair[1]]

### TODO: need to think about if the side doesnt have the same notional size, so cant just 2x the long margin, short margin should be calculated based on its own notional size
df.loc[long_x, "x_margin"] = (
    df.loc[long_x, "x_long_cash"] * NOTIONAL_SIZE * INITIAL_MARGIN * 2
)
df.loc[long_y, "y_margin"] = (
    df.loc[long_y, "y_long_cash"] * NOTIONAL_SIZE * INITIAL_MARGIN * 2
)

df["cash_deployed"] = df[["x_long_cash", "y_long_cash", "x_margin", "y_margin"]].sum(
    axis=1
)

### filtered data with selected date range
selected_df = df[start_date:end_date]


fig = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.3,
    specs=[
        [{}],  # Row 1: no secondary axis
    ],
    subplot_titles=("All Data Z-Score of Residuals",),
)
fig.add_trace(go.Scatter(x=df.index, y=df["zscore"], name="Z-Score"), row=1, col=1)

st.plotly_chart(fig)


fig1 = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.3,
    specs=[
        [{}],  # Row 1: no secondary axis
    ],
    subplot_titles=("Selected Data Z-Score of Residuals",),
)
fig1.add_trace(
    go.Scatter(x=selected_df.index, y=selected_df["zscore"], name="Z-Score"),
    row=1,
    col=1,
)
st.plotly_chart(fig1)

# Debug view only
# st.write(selected_df)

### Display selected data in a table
df_display = selected_df[
    [pair[0], pair[1], "zscore", f"position_{pair[1]}", "gross_pnl", "cash_deployed"]
].rename(
    columns={
        f"position_{pair[1]}": f'Long({pair[1].split("_")[1]})/Short({pair[0].split("_")[1]}) Position',
        "gross_pnl": "Gross PnL",
        "cash_deployed": "Cash Deployed",
        "zscore": "Z-Score",
        pair[0]: f'{pair[0].split("_")[1]} Close Price',
        pair[1]: f'{pair[1].split("_")[1]} Close Price',
    }
)

st.dataframe(df_display)


### Plot Positions Over Time
fig3 = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.3,
    specs=[
        [{}],  # Row 1: no secondary axis
    ],
    subplot_titles=(f"Positions Over Time for {selected_pair}",),
)
fig3.add_trace(
    go.Scatter(
        x=selected_df.index,
        y=selected_df[f"position_{pair[1]}"],
        name=f'Position {pair[1].split("_")[1]}',
    ),
    row=1,
    col=1,
)
fig3.add_trace(
    go.Scatter(
        x=selected_df.index,
        y=selected_df[f"position_{pair[0]}"],
        name=f'Position {pair[0].split("_")[1]}',
    ),
    row=1,
    col=1,
)
st.plotly_chart(fig3)


### Plot Cumulative Gross PnL Over Time
fig4 = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.3,
    specs=[
        [{}],  # Row 1: no secondary axis
    ],
    subplot_titles=(f"Cumulative Gross PnL Over Time for {selected_pair}",),
)
fig4.add_trace(
    go.Scatter(
        x=selected_df.index,
        y=selected_df["gross_pnl"].cumsum(),
        name="Gross PnL",
    ),
    row=1,
    col=1,
)
st.plotly_chart(fig4)


### Plot Gross PnL Over Time as Bar Chart
fig5 = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.3,
    specs=[
        [{}],  # Row 1: no secondary axis
    ],
    subplot_titles=(f"Gross PnL Over Time for {selected_pair}",),
)
fig5.add_trace(
    go.Bar(
        x=selected_df.index,
        y=selected_df["gross_pnl"],
        name="Gross PnL",
    ),
    row=1,
    col=1,
)
### make bars green for positive pnl and red for negative pnl
fig5.update_traces(marker_color=np.where(selected_df["gross_pnl"] >= 0, "green", "red"))
st.plotly_chart(fig5)


### Performance Metrics
num_trades = (selected_df[f"position_{pair[1]}"] != 0).sum()
average_hold_period = "1 minute"
total_pnl = selected_df["gross_pnl"].cumsum().iloc[-1]
### TODO: can try to calculate based on many hourse are actively trading in a day, and take average over that
avg_trading_hr = 6.5
sharpe_ratio = (
    selected_df["gross_pnl"].mean()
    / selected_df["gross_pnl"].std()
    * np.sqrt(252 * avg_trading_hr * 60)
)  # annualized Sharpe ratio assuming 1 min data and 6.5 trading hours per day

avg_cash_deployed_per_minute = selected_df["cash_deployed"].mean()
max_cash_deployed = selected_df["cash_deployed"].max()

st.subheader("Performance Metrics")
st.markdown(
    f"**Number of Trades Executed From {start_date} to {end_date}:** {num_trades}"
)
st.markdown(f"**Average Hold Period:** {average_hold_period}")
st.markdown(f"**Total Gross PnL:** ${total_pnl:,.2f}")
st.markdown(f"**Annualized Sharpe Ratio:** {sharpe_ratio:.2f}")
st.markdown(
    f"**Average Cash Deployed per Minute:** ${avg_cash_deployed_per_minute:,.2f}"
)
st.markdown(f"**Maximum Cash Deployed:** ${max_cash_deployed:,.2f}")


### Generate blotter records
blotter_raw_df = selected_df[[pair[0], pair[1], position_x, position_y, "gross_pnl"]]

### perform vectorized to output a blotter with trade entries and No action
blotter_raw_df["status"] = np.where(
    blotter_raw_df[position_x] != 0, "ENTRY", "NO ACTION"
)

### it entries a trade every minute when status is entry, and exit a trade the next minute

trade_id = 1
records = []
for i in range(len(blotter_raw_df) - 1):
    ### TODO: working, but need to handle the last trade exit properly
    row = blotter_raw_df.iloc[i]

    next_row = blotter_raw_df.iloc[i + 1] if i + 1 < len(blotter_raw_df) else None

    if row.status == "ENTRY":
        record1 = {
            "trade_id": trade_id,
            "timestamp": row.name,
            "action": "BUY" if row[position_x] > 0 else "SHORT",
            "quantity": row[position_x],
            "price": row[pair[0]],
            "status": "ENTRY",
        }
        record2 = {
            "trade_id": trade_id,
            "timestamp": row.name,
            "action": "BUY" if row[position_y] > 0 else "SHORT",
            "quantity": row[position_y],
            "price": row[pair[1]],
            "status": "ENTRY",
        }
        record3 = {
            "trade_id": trade_id,
            "timestamp": next_row.name if next_row is not None else None,
            "action": "SELL" if row[position_x] > 0 else "COVER",
            "quantity": -row[position_x],
            "price": next_row[pair[0]] if next_row is not None else None,
            "status": "EXIT",
        }
        record4 = {
            "trade_id": trade_id,
            "timestamp": next_row.name if next_row is not None else None,
            "action": "SELL" if row[position_y] > 0 else "COVER",
            "quantity": -row[position_y],
            "price": next_row[pair[1]] if next_row is not None else None,
            "status": "EXIT",
        }
        records.extend([record1, record2, record3, record4])
        trade_id += 1

blotter_df = pd.DataFrame.from_records(records)
blotter_df["trade_id"] = blotter_df["trade_id"].astype(str)

df_xlsx = to_excel(blotter_df)


st.sidebar.download_button(
    label="Download Blotter as Excel",
    data=df_xlsx,
    file_name="blotter.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
