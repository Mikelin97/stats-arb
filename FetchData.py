import databento as db
import pandas as pd
from dotenv import load_dotenv 
import os 
load_dotenv() 

DATA_BENTO_KEY = os.getenv("DATA_BENTO_KEY") 







def fetch_pairs_data(identifier: str, pairs: dict, location: str) -> str:

    default_schema = "ohlcv-1m"

    client = db.Historical(key=DATA_BENTO_KEY)


    for asset, variables in pairs.items():

        dataset = variables['dataset'] 
        symbol = variables['symbol']
        start = variables['start']
        end = variables['end']


        data = client.timeseries.get_range(
            dataset=dataset,
            schema=default_schema,
            stype_in="raw_symbol",
            symbols=[symbol],
            start=start,
            end=end,
        )

        df = data.to_df() 
        df.to_csv(f"{location}{identifier}_{asset}.csv") 

    return "success"



if __name__ == "__main__": 

    start = "2025-10-20" 
    end = "2025-10-24T22:05:00.000000000Z"
    location = "./data/"
    
    # pairs = {
    #     "wti_oil_future": {
    #         "dataset": "GLBX.MDP3",
    #         "symbol": "CLZ5", 
    #         "start": start, 
    #         "end": end
    #     }, 
    #     "brent_oil_future": {
    #         "dataset": "IFEU.IMPACT", 
    #         "symbol": "BRN FMF0026!", 
    #         "start": start, 
    #         "end": end
    #     }
    # }


    pairs = {
        "gold_future": {
            "dataset": "GLBX.MDP3",
            "symbol": "GCZ5", 
            "start": start, 
            "end": end
        }, 
        "silver_future": {
            "dataset": "GLBX.MDP3", 
            "symbol": "SIZ5", 
            "start": start, 
            "end": end
        }
    }

    # fetch_pairs_data("pair1", pairs, location)
    fetch_pairs_data("pair2", pairs, location)