import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

class Preprocessor:

    def CleanData(self):
        """
        Esegue la pulizia dei dati
        """
        initial_count = len(self.df)

        # Prezzi validi
        if 'price' in self.df.columns:                  
            self.df = self.df[self.df['price'] > 0]

            # Clip outlier (1° e 99° percentile)
            lower = self.df['price'].quantile(0.01)
            upper = self.df['price'].quantile(0.99)
            self.df['price'] = self.df['price'].clip(lower, upper)

        #Minimum nights realistici
        if 'minimum_nights' in self.df.columns:
            self.df = self.df[self.df['minimum_nights'] <= 365]

        #Coordinate valide (NYC)
        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            self.df = self.df[
                (self.df['latitude'].between(40.5, 41)) &
                (self.df['longitude'].between(-74.3, -73.5))
            ]

        # (Opzionale) drop colonne inutili
        # self.df.drop(columns=['id', 'host_name'], inplace=True, errors='ignore')
        # print(f"Cleaning completato: rimosse {initial_count - len(self.df)} righe non valide.")

        # Pulizia testi per Keyword Analysis
        if 'name' in self.df.columns:
            self.df['name'] = self.df['name'].fillna("Unknown")

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



class FullPipeline(Preprocessor):
    def __init__(self, filepath):
        # Inizializziamo caricando direttamente i dati
        self.df = pd.read_csv(filepath)
        
    def transform(self):
        """
        Coordina i metodi della classe padre e aggiunge i passaggi mancanti.
        """
        # A. Metodi della classe padre (Preprocessor.py)
        self.DateToInt(['last_review'])
        self.CleanData()
        self.AddAvailabilityRatio()
        self.AddDistanceFromCityCenter()
        
        # Gestione sicura per evitare errori se le colonne mancano
        if 'last_review_days_ago' in self.df.columns:
            self.AddReviewsPerYear()
            
        if 'calculated_host_listings_count' in self.df.columns:
            self.AddHostIsProfessional()
            
        self.AddRoomTypeOrdinal()
        self.AddListingsPerArea() 
        
        # B. Modifiche aggiuntive (Encoding, Log-Transform, Drop)
        
        # One-Hot Encoding per il distretto (variabile nominale, non ordinale)
        if 'neighbourhood_group' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['neighbourhood_group'], drop_first=True, dtype=int)
            
        # Trasformazione logaritmica del target
        if 'price' in self.df.columns:
            self.df['log_price'] = np.log1p(self.df['price'])
            
        # Eliminazione delle variabili testuali, ridondanti o affette da data leakage
        to_drop = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 
                   'last_review', 'price', 'latitude', 'longitude', 'room_type']
        
        self.df.drop(columns=to_drop, inplace=True, errors='ignore')
        
        # Riempimento finale di eventuali NaN (es. appartamenti senza recensioni)
        self.df.fillna(0, inplace=True)
        
        return self.df


class TreePipeline(Preprocessor):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        
    def transform(self):
        self.DateToInt(['last_review'])
        self.CleanData()
        self.AddAvailabilityRatio()
        self.AddDistanceFromCityCenter()
        
        if 'last_review_days_ago' in self.df.columns:
            self.AddReviewsPerYear()
            
        if 'calculated_host_listings_count' in self.df.columns:
            self.AddHostIsProfessional()
            
        self.AddRoomTypeOrdinal()
        self.AddListingsPerArea() 
        
        # One-Hot Encoding per i macro-quartieri (aiuta l'albero nei primissimi split)
        if 'neighbourhood_group' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['neighbourhood_group'], drop_first=False, dtype=int)
            
        if 'price' in self.df.columns:
            self.df['log_price'] = np.log1p(self.df['price'])
            
        # ATTENZIONE: Qui NON eliminiamo 'latitude' e 'longitude'
        to_drop = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 
                   'last_review', 'price', 'room_type']
        
        self.df.drop(columns=to_drop, inplace=True, errors='ignore')
        self.df.fillna(0, inplace=True)
        
        return self.df