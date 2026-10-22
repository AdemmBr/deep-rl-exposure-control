"""Environnement Gymnasium du controleur d'exposition."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .scaling import StateScaler


class ExposureEnv(gym.Env):
    """Applique les decisions SAC au portefeuille et calcule leur reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        scaler: StateScaler,
        transaction_cost_bps: float = 1.0,
        episode_length: int | None = None,
        a_max: float = 1.0,
        max_exposure: float = 2.0,
        lambda_risk: float = 0.0,
    ):
        super().__init__()
        self.F = scaler.transform(data).to_numpy(dtype=np.float32)
        prices = data["price"].to_numpy()
        self.R = prices[1:] / prices[:-1] - 1.0
        self.P = data["w_base"].to_numpy()
        self.dates = data.index
        self.cost_rate = transaction_cost_bps * 1e-4
        self.episode_length = episode_length
        self.a_max = a_max
        self.max_exposure = max_exposure
        self.lambda_risk = lambda_risk
        self._exposure_scale = float(scaler.std_["w_base"])

        self.action_space = spaces.Box(0.0, a_max, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.F.shape[1] + 1,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Ouvre un bloc d'evaluation, ou tire un episode d'entrainement."""
        super().reset(seed=seed)
        if self.episode_length is None:
            self._index = 0
            self._end = len(self.R)
            self._previous_exposure = (options or {}).get("previous_exposure", 0.0)
        else:
            self._index = int(
                self.np_random.integers(0, len(self.R) - self.episode_length + 1)
            )
            self._end = self._index + self.episode_length
            self._previous_exposure = self._exposure(
                self.np_random.uniform(0.0, self.a_max)
            )
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """Applique une action au rendement du jour suivant puis avance d'un jour."""
        multiplier = float(action[0])
        baseline_exposure = self.P[self._index]
        exposure = self._exposure(multiplier)
        asset_return = self.R[self._index]
        turnover = abs(exposure - self._previous_exposure)
        market_pnl = exposure * asset_return
        net_return = market_pnl - self.cost_rate * turnover
        self._previous_exposure = exposure
        risk_penalty = self.lambda_risk * market_pnl**2

        info = {
            "date": self.dates[self._index],
            "multiplier": multiplier,
            "baseline_exposure": baseline_exposure,
            "exposure": exposure,
            "net_return": net_return,
            "risk_penalty": risk_penalty,
        }

        self._index += 1
        truncated = self._index >= self._end
        reward = 100.0 * (net_return - risk_penalty)
        return self._get_observation(), reward, False, truncated, info

    def portfolio_state(self):
        return {"previous_exposure": self._previous_exposure}

    def _exposure(self, multiplier: float):
        return float(
            np.clip(
                multiplier * self.P[self._index],
                -self.max_exposure,
                self.max_exposure,
            )
        )

    def _get_observation(self):
        return np.append(
            self.F[self._index],
            [self._previous_exposure / self._exposure_scale],
        ).astype(np.float32)
