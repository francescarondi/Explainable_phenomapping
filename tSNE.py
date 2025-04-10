'''
sys.argv[1] is for the input data
sys.argv[2] perplexity
sys.argv[3] early_exageration
'''
import pandas as pd
from sklearn.manifold import TSNE
import sys

#sys.path.append('...')
import data_for_dim_red 

#scaler
input_path = sys.argv[1]
X_scaled, X_split= data_for_dim_red.funzione_scaler(input_path)

#early_exaggeration e perplexity
perplexity_param= sys.argv[2]
early_exaggeration_param= sys.argv[3]

#tSNE
tsne = TSNE(n_components=2, random_state=42, learning_rate=400, perplexity=float(perplexity_param), n_iter=5000, early_exaggeration=float(early_exaggeration_param)) 
X_tsne = tsne.fit_transform(X_scaled)

X_tsne_df = pd.DataFrame(X_tsne)
X_tsne_df.to_csv(f'data_tsne_perplexity-{perplexity_param}_early_exaggeration-{early_exaggeration_param}.csv', index=False)