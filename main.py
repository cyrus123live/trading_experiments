import requests
import time
from stable_baselines3 import A2C
import pandas as pd
import numpy as np
import math


def get_btc_price():
    data = requests.get("https://api.bitget.com/api/v2/spot/market/tickers?symbol=BTCUSDT").json()["data"][0]
    return round((float(data["bidPr"]) + float(data["askPr"])) / 2, 2)


if __name__ == "__main__":

    print("Initializing...")

    history = pd.DataFrame(columns=["price"])
    model = A2C.load("A2C_10")

    cash = 100000.0
    held = 0.0

    price = 0
    while True:
        new_price = get_btc_price()
        if price != new_price:
            # history = pd.concat([history, pd.DataFrame({"price":[new_price]})])
            history.loc[len(history), history.columns] = new_price
            normalized_df = (new_price - history["price"].rolling(window=20).mean()) / history["price"].rolling(window=20).std()
            normalized = float(normalized_df.loc[len(normalized_df) - 1])

            if not math.isnan(normalized):
                decision = model.predict([normalized, held / 1, cash / 100])[0]

                if decision == 1 and cash > 0:
                    held = cash / new_price
                    cash = 0.0
                elif decision == 0 and held > 0:
                    cash = new_price * held
                    held = 0.0

                print(float(history.loc[len(history) - 1].iloc[0]), normalized, decision, cash, held)
                print(f"Current Price: {float(history.loc[len(history) - 1].iloc[0])}")
                
            price = new_price