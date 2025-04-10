import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

def funzione_scaler(path):
    df_vect = pd.read_csv(path, sep=';', header=None) 
    split_df = df_vect[0].str.split(',', expand=True)
    
    #remove first row
    split_df.columns = split_df.iloc[0]
    split_df = split_df[1:].reset_index(drop=True)
    
    #name for the columns
    column_name =['MRN']
    for i in range(0, 100):
        column_name.append('encoded_subseq'+str(i))
    
    split_df.columns= column_name
    X_split = split_df
    
    #standard scaler
    X = split_df.iloc[:, 1:]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X_split