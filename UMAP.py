'''
sys.argv[1] is for the input data
sys.argv[2] n_neigbors
sys.argv[3] min_dist
'''

import pandas as pd
import umap
import sys

#sys.path.append('...')
import data_for_dim_red 

#scaler
input_path = sys.argv[1]
X_scaled, X_split= data_for_dim_red.funzione_scaler(input_path)

#param
n_neighbors_par= sys.argv[2]
min_dist_par = sys.argv[3]

#umap
umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=int(n_neighbors_par), min_dist=float(min_dist_par)) 
X_umap = umap_model.fit_transform(X_scaled) 

X_umap_df = pd.DataFrame(X_umap)
X_umap_df.to_csv(f'data_umap_n_neighbors-{n_neighbors_par}_min_dist-{min_dist_par}.csv', index=False)