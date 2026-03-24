import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        # Usiamo .copy() per evitare il SettingWithCopyWarning di Pandas
        self.df = df.copy()

    def Normalize(self, features):
        scaler = MinMaxScaler()
        self.df[features] = scaler.fit_transform(self.df[features])
    
    def FindNaNRows(self):        # Restituisce le righe che contengono almeno un NaN.
        return self.df[self.df.isnull().any(axis=1)]

    def RemoveRows(self, columns=None): 
        if columns is not None:
            self.df.dropna(subset=columns, inplace=True)
        else:
            self.df.dropna(inplace=True)

    def Standardize(self, features):
        scaler = StandardScaler()
        self.df[features] = scaler.fit_transform(self.df[features])

    def LabelEncoding(self, features):
        for col in features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
    ##!! # WARNING: usare solo con modelli ad alberi (Random Forest, XGBoost), altrimenti si rischia di introdurre un ordine artificiale tra le categorie (es. 0 < 1 < 2) 

    def OneHotEncoding(self, features):
        self.df = pd.get_dummies(self.df, columns=features, drop_first=True, dtype=int)

    # feature engineering suggeriti da Gemini
    def AddNeighborhoodMeanPrice(self):    # associare alle varie righe il prezzo medio della loro zona,  Questa funzione calcola il prezzo medio per ogni quartiere
        self.df['neighborhood_mean_price'] = self.df.groupby('neighbourhood')['price'].transform('mean')
    #!!!! prezzo medio calcolato su TUTTO il dataset, quindi è una feature che potrebbe causare data leakage se usata in un modello predittivo.

    def AddDistanceFromCityCenter(self):   # calcolo della distanza dal centro città (NYC) (eventualmente anche distanza dai landmark importanti)
        #Il prezzo a New York crolla man mano che ci si allontana dal centro (Times Square)
        city_center_lat, city_center_lon = 40.7580, -73.9855
        
        # Calcolo della distanza (semplificata)
        self.df['distance_from_center'] = np.sqrt(
            (self.df['latitude'] - city_center_lat)**2 + 
            (self.df['longitude'] - city_center_lon)**2
        )

    def DateToInt(self, features):
        for col in features:
            self.df[col] = pd.to_datetime(self.df[col])     # Troviamo la data più recente nel dataset
            latest_date = self.df[col].max()                # Calcoliamo i giorni di differenza
            self.df[f'{col}_days_ago'] = (latest_date - self.df[col]).dt.days
            # Se un appartamento non ha recensioni (NaN), mettiamo un valore alto (es. 3000 giorni)
            self.df[f'{col}_days_ago'] = self.df[f'{col}_days_ago'].fillna( self.df[f'{col}_days_ago'].median())


    def LogTransform(self, features):       # comoda per i prezzi che possono avere distribuzione esponenziale, aggiungere una colonna con il loro log
        for col in features:
            # Usiamo log1p invece di log perché se il valore è 0, 
            # log(0) dà errore, mentre log(1+0) fa 0.
            self.df[f'log_{col}'] = np.log1p(self.df[col])
    # eventualmente, aggiungere colonne filtrando le keyword importanti nei nomi (Luxury, Designer, Renovated, View, Balcony, ...)

    def AddAvailabilityRatio(self):                                                     # La disponibilità di un appartamento è un fattore importante per il prezzo
        if 'availability_365' in self.df.columns:
            self.df['availability_ratio'] = self.df['availability_365'] / 365

    def AddReviewsPerYear(self):
        if 'last_review_days_ago' in self.df.columns:
            self.df['reviews_per_year'] = self.df['number_of_reviews'] / (
                self.df['last_review_days_ago'] + 1
            )

    def AddHostIsProfessional(self):
        self.df['host_is_professional'] = (self.df['calculated_host_listings_count'] > 1).astype(int)       #host professionale se ha più di 1 appartamento in affitto


    def AddRoomTypeOrdinal(self):
        mapping = {
            "Shared room": 0,
            "Private room": 1,
            "Entire home/apt": 2
        }
        if 'room_type' in self.df.columns:
            self.df['room_type_ordinal'] = self.df['room_type'].map(mapping)
    
    def AddListingsPerArea(self):                       #misura quanto è turistica una zona, più è turistica più è probabile che i prezzi siano alti
        if 'neighbourhood' in self.df.columns:
            counts = self.df['neighbourhood'].value_counts()
            self.df['listings_in_neighbourhood'] = self.df['neighbourhood'].map(counts)