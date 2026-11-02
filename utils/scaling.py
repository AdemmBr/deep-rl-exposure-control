import pandas as pd


class StateScaler:
    def fit(self, train_data: pd.DataFrame, columns: list[str]):
        self.columns = columns
        self.mean_ = train_data[columns].mean()
        self.std_ = train_data[columns].std().replace(0.0, 1.0)
        return self

    def transform(self, data: pd.DataFrame):
        return (data[self.columns] - self.mean_) / self.std_
