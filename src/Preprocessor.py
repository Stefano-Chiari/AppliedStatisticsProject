import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def Normalize(self, features):
        self.minMaxScaler = MinMaxScaler()
        self.df[features] = self.minMaxScaler.fit_transform(self.df[features])
    
    # TODO
    def FindNaNRows(self):
        pass

    def RemoveRows(self):   # used with FindNaNRows
        pass

    def Standardize(self, features):
        pass

    def LabelEncoding(self, features):
        pass

    def OneHotEncoding(self, features):
        pass

    # feature engineering suggeriti da Gemini
    def AddNeighborhoodMeanPrice(self):    # associare alle varie righe il prezzo medio della loro zona
        pass

    def AddDistanceFromCityCenter(self):   # calcolo della distanza dal centro città (NYC) (eventualmente anche distanza dai landmark importanti)
        pass

    def DateToInt(self, features):         # convertire last_review da Date a intero (esempio contando il numero di giorni dal minimo per avere un ordinamento)
        pass

    def LogTransform(self, features):       # comoda per i prezzi che possono avere distribuzione esponenziale, aggiungere una colonna con il loro log
        pass
    # eventualmente, aggiungere colonne filtrando le keyword importanti nei nomi (Luxury, Designer, Renovated, View, Balcony, ...)