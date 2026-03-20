import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def Normalize(self, features):
        scaler = MinMaxScaler()
        self.df[features] = scaler.fit_transform(self.df[features])

    def GetProcessedData(self):
        return self.df