"""Les neuf indicateurs de Zhang, Zohren et Roberts (2019), section 4.4,
arXiv:1911.10107. Noms de colonnes repris du papier, pour comparer directement
avec ses tables 2 et 3."""

import numpy as np
import pandas as pd


class PaperMetrics:
    """Calcule les neuf indicateurs du papier pour chaque colonne de rendements."""

    COLUMNS = [
        "E(R)",
        "Std(R)",
        "DD",
        "Sharpe",
        "Sortino",
        "MDD",
        "Calmar",
        "% +ve Returns",
        "Ave. P / Ave. L",
    ]

    DOWNSIDE_DEFINITIONS = ("semi_deviation", "negative_std")

    def __init__(
        self, trading_days: int = 252, downside: str = "semi_deviation"
    ):
        if downside not in self.DOWNSIDE_DEFINITIONS:
            raise ValueError(
                f"downside doit valoir l'un de {self.DOWNSIDE_DEFINITIONS}, "
                f"recu {downside!r}"
            )
        self.trading_days = trading_days
        self.downside = downside

    def table(self, returns: pd.DataFrame | pd.Series):
        """Retourne une ligne par strategie et une colonne par indicateur."""
        frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
        root = np.sqrt(self.trading_days)

        expected = frame.mean() * self.trading_days
        volatility = frame.std() * root
        downside = self._downside_deviation(frame) * root
        drawdown = self._max_drawdown(frame)

        gains = frame.where(frame > 0.0).mean()
        losses = frame.where(frame < 0.0).mean()
        observed = self._nonzero(frame.notna().sum())
        positive = frame.gt(0.0).sum()

        table = pd.DataFrame(
            {
                "E(R)": expected,
                "Std(R)": volatility,
                "DD": downside,
                "Sharpe": expected / self._nonzero(volatility),
                "Sortino": expected / self._nonzero(downside),
                "MDD": drawdown,
                "Calmar": expected / self._nonzero(drawdown),
                "% +ve Returns": 100.0 * positive / observed,
                "Ave. P / Ave. L": gains / self._nonzero(losses.abs()),
            }
        )
        return table[self.COLUMNS]

    def _downside_deviation(self, frame: pd.DataFrame):
        if self.downside == "semi_deviation":
            return frame.clip(upper=0.0).pow(2).mean().pow(0.5)
        return frame.where(frame < 0.0).std()

    def _max_drawdown(self, frame: pd.DataFrame):
        equity = (1.0 + frame).cumprod()
        return -(equity / equity.cummax() - 1.0).min() + 0.0

    @staticmethod
    def _nonzero(values: pd.Series):
        return values.replace(0.0, np.nan)
