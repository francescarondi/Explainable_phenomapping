'''
sys.argv[1] : cluster data
sys.argv[2] : dictionary for code meaning (part of the best model parameters)
sys.argv[3] : best model

'''
import sys
import torch
import shap
import pandas as pd
import numpy as np
import os 
from collections import defaultdict

from lime.lime_text import LimeTextExplainer


sys.path.append('...') #convAE code
import utils as ut
import evaluate

sys.path.append('...') #convAE code
import net
import data_loader


#best model
model_param = ut.model_param

embs = None
path_diz=sys.argv[2]
vocab = pd.read_csv(path_diz)
checkpoint_best_model=sys.argv[3]
checkpoint = torch.load(checkpoint_best_model, weights_only=False)
saved_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
vocab_size = saved_vocab_size
model = net.ehrEncoding(vocab_size=vocab_size,
                        max_seq_len=ut.len_padded,
                        emb_size=ut.model_param['embedding_size'],
                        kernel_size=ut.model_param['kernel_size'],
                        pre_embs=embs,
                        vocab=vocab)
model.to('cuda')  # model on GPU

optimizer = torch.optim.Adam(model.parameters(),
                             lr=ut.model_param['learning_rate'],
                             weight_decay=ut.model_param['weight_decay']) 

checkpoint = torch.load(checkpoint_best_model, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

def model_predict(input_sequences):
    #sequence of integers
    converted_sequences = [
        [int(num) for num in seq.split(',') if num.strip().isdigit()] if isinstance(seq, str) else seq 
        for seq in input_sequences
    ]
    
    #padding 
    len_padded = ut.len_padded  
    padded_sequences = [
        torch.cat([torch.tensor(seq, dtype=torch.long, device='cuda'), 
                   torch.zeros(len_padded - len(seq), dtype=torch.long, device='cuda')]) 
        if len(seq) < len_padded 
        else torch.tensor(seq[:len_padded], dtype=torch.long, device='cuda') 
        for seq in converted_sequences
    ]
    
    #construct the trajectory
    input_tensor = torch.stack(padded_sequences, dim=0)

    #model outcome
    with torch.no_grad():
        reconstruction = model(input_tensor)
    reconstruction = reconstruction[:, 0, :]  

    reconstruction_error = torch.mean((input_tensor - reconstruction) ** 2, dim=1).cpu().numpy()
    
    #reconstruction_error as bidimensional
    reconstruction_error = reconstruction_error.reshape(-1, 1) 
    reconstruction_error = np.concatenate([reconstruction_error, 1 - reconstruction_error], axis=1)
    
    return reconstruction_error

explainer = LimeTextExplainer(class_names=['reconstruction_error'])


path_data = sys.argv[1]
train_df = pd.read_csv(path_data)


all_explanations = []

num_features = 32 #number of feaures to consider

for i, input_example in enumerate(train_df['EHRseq']):
    #LIME explaination    
    explanation = explainer.explain_instance(
        input_example,          #data
        model_predict,          # model
        num_features=num_features  #number of feaures to consider
    )

    all_explanations.append(explanation)
    print(f"Spiegazione generata per il campione {i + 1}/{len(train_df)}")
    

#avg weights
feature_counts = defaultdict(int)
feature_weights_sum = defaultdict(float)

for explanation in all_explanations:
    for feature, weight in explanation.as_list():
        feature_counts[feature] += 1
        feature_weights_sum[feature] += weight

feature_weights_avg = {feature: feature_weights_sum[feature] / feature_counts[feature] 
                       for feature in feature_counts}

df_weights= pd.DataFrame(list(feature_weights_avg.items()), columns=["Feature", "Average Weight"])
df_weights.to_csv("all_explanations.csv", index=False) 
