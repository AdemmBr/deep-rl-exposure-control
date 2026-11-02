import numpy as np
import pandas as pd


class ExposureStrategy:
    def __init__(
        self,
        vol_window: int = 21,
        target_vol: float = 0.10,
        max_exposure: float = 2.0,
        vol_method: str = "rolling",
    ):
        if vol_method not in ("rolling", "ewm"):
            raise ValueError(
                f"vol_method doit valoir 'rolling' ou 'ewm', recu {vol_method!r}"
            )
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_exposure = max_exposure
        self.vol_method = vol_method

    def build(self, data: pd.DataFrame):
        frame = data.copy()
        frame["signal"] = self._signal(frame)
        frame["forecast_vol"] = self._forecast_vol(frame)
        frame["w_base"] = self._scale(frame["signal"], frame["forecast_vol"])
        return frame

    def exposure(self, data: pd.DataFrame):
        return self.build(data)["w_base"]

    def _signal(self, frame: pd.DataFrame):
        raise NotImplementedError

    def _forecast_vol(self, frame: pd.DataFrame):
        returns = frame["ret"]
        if self.vol_method == "rolling":
            deviation = returns.rolling(self.vol_window).std()
        else:
            deviation = returns.ewm(span=self.vol_window).std()
        return deviation * np.sqrt(252)

    def _scale(self, signal: pd.Series, volatility: pd.Series):
        return (signal * self.target_vol / volatility.replace(0.0, np.nan)).clip(
            -self.max_exposure, self.max_exposure
        )


class TrendVolatilityBaseline(ExposureStrategy):
    def __init__(self, trend_window: int = 63, **kwargs):
        super().__init__(**kwargs)
        self.trend_window = trend_window

    def _signal(self, frame: pd.DataFrame):
        return np.sign(frame["ret"].rolling(self.trend_window).sum())


class LongOnly(ExposureStrategy):
    def _signal(self, frame: pd.DataFrame):
        return pd.Series(1.0, index=frame.index)


class SignReturn(ExposureStrategy):
    def __init__(self, lookback: int = 252, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def _signal(self, frame: pd.DataFrame):
        return np.sign(frame["ret"].rolling(self.lookback).sum())


class MACDSignal(ExposureStrategy):
    # Baz et al., eq. 12
    TIMESCALES = ((8, 24), (16, 48), (32, 96))
    RESPONSE_SCALE = 0.89

    def _signal(self, frame: pd.DataFrame):
        price = frame["price"]
        trend = sum(
            self._macd(price, short, long_) for short, long_ in self.TIMESCALES
        ) / len(self.TIMESCALES)
        return self._response(trend)

    def _macd(self, price: pd.Series, short: int, long_: int):
        spread = self._average(price, short) - self._average(price, long_)
        q = spread / price.rolling(63).std().replace(0.0, np.nan)
        return q / q.rolling(252).std().replace(0.0, np.nan)

    @staticmethod
    def _average(price: pd.Series, scale: int):
        halflife = np.log(0.5) / np.log(1.0 - 1.0 / scale)
        return price.ewm(halflife=halflife).mean()

    def _response(self, trend: pd.Series):
        return trend * np.exp(-(trend**2) / 4.0) / self.RESPONSE_SCALE
