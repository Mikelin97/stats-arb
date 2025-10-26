import pandas as pd 
import os 
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pairs_data = {}
pairs_key = {}



pairs = {'WTI vs. Brent': 'pair1', 'Gold vs. Silver': 'pair2'}
selected_pair = st.sidebar.selectbox("Select Pair", list(pairs.keys())) 
current_pair = pairs[selected_pair]

lookback = 100 
p_threshold = 0.05



def load_pairs_data(current_pair):
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



    ### getting close price at each candle 

    asset1_close = asset1['close'].to_frame(name=pairs_key[current_pair][0])
    asset2_close = asset2['close'].to_frame(name=pairs_key[current_pair][1])

    pair_close = asset1_close.join(asset2_close, how='outer').dropna()

    return asset1, asset2, pair_close 


asset1, asset2, pair_close = load_pairs_data(current_pair)



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
else:
    selected_data = pair_close.loc[unique_dates[selected_index]:]

blocks = (selected_data['cointegrated'].diff().fillna(0) != 0).cumsum()

coint_blocks = blocks[selected_data['cointegrated'] == 1]
coint_period_ids = coint_blocks.unique()





for i, block_id in enumerate(coint_period_ids):
    # Create a mask for the current cointegrated period
    mask = (blocks == block_id) & (selected_data['cointegrated'] == 1)
    period_df = selected_data[mask]


    

    start_time = period_df.index[0]
    end_time = period_df.index[-1]
    
    st.write(period_df)
    asset1.loc[period_df.index]
    asset2.loc[period_df.index]

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