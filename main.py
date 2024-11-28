import requests
import time
from stable_baselines3 import A2C
import pandas as pd
import numpy as np
import math
import datetime as dt
import os
import sys


def get_btc_price():
    # data = requests.get("https://api.bitget.com/api/v2/spot/market/tickers?symbol=BTCUSDT").json()["data"][0]
    data = requests.get("https://api.mexc.com/api/v3/ticker/bookTicker?symbol=BTCUSDT").json()

    return round((float(data["bidPrice"]) + float(data["askPrice"])) / 2, 2)


if __name__ == "__main__":

    os.makedirs("runs", exist_ok=True)

    print("Initializing model...")

    history = pd.DataFrame(columns=["price", "value"])
    model = A2C.load("A2C_10")

    cash = 100000.0
    held = 0.0

    price = 0

    print("Initializing price normalization...")

    while True:
        try:
            new_price = get_btc_price()
        except KeyboardInterrupt:
            history.to_csv(f"runs/{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv")
            quit()
        except Exception as e:
            print("Failed to get price...")
            continue
        if price != new_price:
            
            history.loc[dt.datetime.now().timestamp()] = [new_price, held * new_price + cash]
            normalized_df = (new_price - history["price"].rolling(window=20).mean()) / history["price"].rolling(window=20).std()
            normalized = float(normalized_df.iloc[len(normalized_df) - 1])

            if not math.isnan(normalized):
                decision = model.predict([normalized, held / 1, cash / 100000])[0]
                

                if decision == 1 and cash > 0:
                    held = cash / new_price
                    cash = 0.0
                elif decision == 0 and held > 0:
                    cash = new_price * held
                    held = 0.0

                print("------------------------------------------------------")
                print(f"Price: {new_price}\n")
                print(f"Decision: {decision}")
                print(f"Held: {held:0.2f}")
                print(f"Cash: {cash:0.2f}\n")
                print(f"Value: {held * new_price + cash:0.2f}")
                print("\n")
                
            price = new_price