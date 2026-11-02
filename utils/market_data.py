import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataLoader:
    def __init__(self, ticker: str, start: str, end: str | None = None):
        self.ticker = ticker
        self.start = start
        self.end = end

    def load(self):
        price = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )["Close"]
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]

        # dropna avant le diff pour ne pas sauter de rendement
        data = price.dropna().rename("price").to_frame()
        data["ret"] = np.log(data["price"]).diff()
        data = data.dropna(subset=["ret"])
        print(
            f"{len(data)} jours | {data.index[0].date()} -> "
            f"{data.index[-1].date()}"
        )
        return data
