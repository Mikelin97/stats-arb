import pandas as pd 
import os 
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pairs_data = {}
pairs_key = {}



PAIRS = {'WTI vs. Brent': 'pair1', 'Gold vs. Silver': 'pair2', 
         'SOFR 3M Future vs. DUK Spot': 'pair3', 'Corn vs. Soybean Oil': 'pair4', 
         'Bitcoin ETF vs. Ethereum ETF': 'pair5'}
CANDLE_SIZES = {'1 minute': '1T', '2 minutes': '2T', '3 minutes': '3T',
                '4 minutes': '4T', '5 minutes': '5T', '15 minutes': '15T',
                '1 hour': '1H'}

selected_pair = st.sidebar.selectbox("Select Pair", list(PAIRS.keys())) 
current_pair = PAIRS[selected_pair]

selected_candle_size = st.sidebar.selectbox("Select Candle Size", list(CANDLE_SIZES.keys()))
current_candle_size = CANDLE_SIZES[selected_candle_size]



lookback = 100 
p_threshold = 0.05



def load_pairs_data(current_pair, candle_size='1T'):
    pairs_data[current_pair] = [
        pd.read_csv(f"data/{f}", index_col=0) for f in os.listdir('data')
        if f.startswith(current_pair) and f.endswith('.csv')
    ]

    pairs_key[current_pair] = [
        f.split('.')[0] for f in os.listdir('data')
        if f.startswith(current_pair) and f.endswith('.csv')
    ]



    asset1 = pairs_data[current_pair][0].ffill()
    asset2 = pairs_data[current_pair][1].ffill()

    if candle_size != '1T':
        asset1.index = pd.to_datetime(asset1.index)
        asset2.index = pd.to_datetime(asset2.index)

        asset1 = asset1.resample(candle_size).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill()

        asset2 = asset2.resample(candle_size).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).ffill()

        asset1.index = asset1.index.astype(str)
        asset2.index = asset2.index.astype(str)


    return asset1, asset2


asset1, asset2 = load_pairs_data(current_pair, current_candle_size)  


### getting close price at each candle 

asset1_close = asset1['close'].to_frame(name=pairs_key[current_pair][0])
asset2_close = asset2['close'].to_frame(name=pairs_key[current_pair][1])

pair_close = asset1_close.join(asset2_close, how='outer').dropna()

    




### verifying cointegration and stationarity
pair_close["cointegrated"] = 0
pair_close["residual"] = 0.0
pair_close["zscore"] = 0.0 

is_cointegrated = False 
lr = LinearRegression()

for i in range(lookback, len(pair_close), lookback):  
    x = pair_close[pairs_key[current_pair][0]].iloc[i-lookback:i].values[:,None]
    y = pair_close[pairs_key[current_pair][1]].iloc[i-lookback:i].values[:,None]

    if is_cointegrated:
        # Compute and normalize signal on forward window
        x_new = pair_close[pairs_key[current_pair][0]].iloc[i:i+lookback].values[:,None]
        y_new = pair_close[pairs_key[current_pair][1]].iloc[i:i+lookback].values[:,None]
        spread_back = y - lr.coef_ * x
        spread_forward = y_new - lr.coef_ * x_new
        zscore = (spread_forward - spread_back.mean()) / spread_back.std()
        pair_close.iloc[i:i+lookback, pair_close.columns.get_loc("cointegrated")] = 1
        pair_close.iloc[i:i+lookback, pair_close.columns.get_loc("residual")] = spread_forward
        pair_close.iloc[i:i+lookback, pair_close.columns.get_loc("zscore")] = zscore

    _, p, _ = coint(x,y)
    is_cointegrated = p < p_threshold
    lr.fit(x,y)


temp = pd.to_datetime(pair_close.index) 
unique_dates = temp.normalize().unique()
unique_dates = [d.strftime('%Y-%m-%d') for d in unique_dates]


selected_date = st.sidebar.selectbox("Select Date", unique_dates)

selected_index = unique_dates.index(selected_date)

if selected_index + 1 < len(unique_dates):
    end_date = unique_dates[selected_index + 1]
    selected_data = pair_close.loc[unique_dates[selected_index]:end_date]
    selected_asset1 = asset1.loc[selected_data.index]
    selected_asset2 = asset2.loc[selected_data.index]
else:
    selected_data = pair_close.loc[unique_dates[selected_index]:]
    selected_asset1 = asset1.loc[selected_data.index]
    selected_asset2 = asset2.loc[selected_data.index]
blocks = (selected_data['cointegrated'].diff().fillna(0) != 0).cumsum()

coint_blocks = blocks[selected_data['cointegrated'] == 1]
coint_period_ids = coint_blocks.unique()




if len(coint_period_ids) == 0:
    st.write("#### No cointegrated periods found for the selected date.")
    st.write(selected_data)
    st.write(f"#### {pairs_key[current_pair][0]} data:")
    st.write(selected_asset1) 
    st.write(f"#### {pairs_key[current_pair][1]} data:")
    st.write(selected_asset2)

else: 

    for i, block_id in enumerate(coint_period_ids):
        # Create a mask for the current cointegrated period
        mask = (blocks == block_id) & (selected_data['cointegrated'] == 1)
        period_df = selected_data[mask]


        

        start_time = period_df.index[0]
        end_time = period_df.index[-1]
        st.write(f"#### Close price data from {start_time} to {end_time}")
        st.write(period_df)
        st.write(f"#### {pairs_key[current_pair][0]} data:")
        st.write(asset1.loc[period_df.index])
        st.write(f"#### {pairs_key[current_pair][1]} data:")
        st.write(asset2.loc[period_df.index])

        num_plots = 3

        fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.3,
                            specs=[
                                [{}],                    # Row 1: no secondary axis
                                [{"secondary_y": True}], # Row 2: price + volume
                                [{"secondary_y": True}]  # Row 3: price + volume
                            ],
                            subplot_titles=("Z-Score of Residuals", f"{pairs_key[current_pair][0]} Price", f"{pairs_key[current_pair][1]} Price"))
        fig.add_trace(go.Scatter(x=period_df.index, y=period_df['zscore'], name='Z-Score'), row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=asset1.loc[period_df.index].index,
                y=asset1.loc[period_df.index]['volume'],
                name='Volume',
                marker_color='lightgray', 
                opacity=0.4
            ),
            row=2, col=1, secondary_y=True
        )
        fig.add_trace(go.Candlestick(x=asset1.loc[period_df.index].index,
                                    open=asset1.loc[period_df.index]['open'],
                                    high=asset1.loc[period_df.index]['high'],
                                    low=asset1.loc[period_df.index]['low'],
                                    close=asset1.loc[period_df.index]['close'],
                                    name=pairs_key[current_pair][0]), row=2, col=1, secondary_y=False)
        fig.add_trace(
            go.Bar(
                x=asset2.loc[period_df.index].index,
                y=asset2.loc[period_df.index]['volume'],
                name='Volume',
                marker_color='lightgray',
                opacity=0.4
            ),
            row=3, col=1, secondary_y=True
        )
        fig.add_trace(go.Candlestick(x=asset2.loc[period_df.index].index,
                                    open=asset2.loc[period_df.index]['open'],
                                    high=asset2.loc[period_df.index]['high'],
                                    low=asset2.loc[period_df.index]['low'],
                                    close=asset2.loc[period_df.index]['close'],
                                    name=pairs_key[current_pair][1]), row=3, col=1, secondary_y=False)
        
        

        fig.update_layout(height=1000, width=800, 
                        title_text=f"Cointegrated Period {i+1} Analysis : from {start_time} to {end_time}",
                        hovermode='x unified', barmode='overlay',)


        fig.update_xaxes(showticklabels=True, row=1, col=1)

        
        

        
        st.plotly_chart(fig)


