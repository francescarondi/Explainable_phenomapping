'''
sys.argv[1] is for the input data
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sys

import data_for_dim_red 

input_path = sys.argv[1]
X_scaled, X_split = data_for_dim_red.funzione_scaler(input_path)


#pca
pca = PCA()
pca.fit(X_scaled)

#cumulative variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

n_components = np.argmax(cumulative_variance >= 0.9) + 1
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

X_pca_df = pd.DataFrame(X_pca)
X_pca_df.to_csv('data_pca.csv', index=False)