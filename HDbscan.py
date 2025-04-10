'''
sys.argv[1] is for the input data
sys.argv[2] min_cluster_size
sys.argv[3] cluster_selection_method
sys.argv[4] min_samples
sys.argv[5] original data for the later interpretation
'''
import pandas as pd
import hdbscan
import sys
import scores 
import numpy as np

import data_for_dim_red 

#data
path_data = sys.argv[1]
data = pd.read_csv(filepath_or_buffer=path_data)

#HDBSCAN clustering
min_cluster_size_param = sys.argv[2]
cluster_selection_method_param = sys.argv[3]
min_samples_param = sys.argv[4]
clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size_param), cluster_selection_method=cluster_selection_method_param, min_samples = int(min_samples_param))
clusterer.fit(data)

#cluster labels
labels = clusterer.labels_

#scores
silhouette = scores.sil_computation(data, labels)

#count unique clusters (noise is label=-1)
unique_labels = np.unique(labels)
n_clusters = len(unique_labels[unique_labels != -1])
print(f'Number of clusters: {n_clusters}')

with open("metriche_clusters.txt", "w") as my_file:
    my_file.write(f"silhouette {silhouette} \n")
    my_file.write(f"number of cluster {n_clusters} \n")


path_data_no_dim_red  = sys.argv[5]
X_scaled, X_split = data_for_dim_red.funzione_scaler(path_data_no_dim_red )
X_split['Cluster']= labels

filtered_df = X_split[['MRN', 'Cluster']]

for label in np.unique(labels):
    if label != -1:  
        #MRN of the considered cluster 
        mrn_in_cluster = filtered_df[filtered_df['Cluster'] == label]['MRN']
        print(f"Cluster {label}:")
        print(mrn_in_cluster)
        print("\n")

#saving
filtered_df.to_csv(f'cluster_finali_HDBSCAN-{min_cluster_size_param}-{cluster_selection_method_param}.csv', index=False)