# encoding=utf-8

import pickle
import numpy as np

embedding_size = 64

############# protein name to protein id
with open('../data/biogrid_dict.pkl', 'rb') as file:
    biogrid_dict = pickle.load(file)
biogrid_dict_reverse = {value: key for key, value in biogrid_dict.items()}

########### 7 centrality features
with open('../data/degree_centrality.pkl', 'rb') as file:
    degree_centrality = pickle.load(file)

with open('../data/closeness_centrality.pkl', 'rb') as file:
    closeness_centrality = pickle.load(file)

with open('../data/eigenvector_centrality.pkl', 'rb') as file:
    eigenvector_centrality = pickle.load(file)

with open('../data/betweenness_centrality.pkl', 'rb') as file:
    betweenness_centrality = pickle.load(file)

with open('../data/harmonic_centrality.pkl', 'rb') as file:
    harmonic_centrality = pickle.load(file)

with open('../data/load_centrality.pkl', 'rb') as file:
    load_centrality = pickle.load(file)

with open('../data/subgraph_centrality.pkl', 'rb') as file:
    subgraph_centrality = pickle.load(file)
#####################
emb_dict = dict()
with open('../data/biogrid_network_bin2.emb') as file:
    for line in file.readlines()[1:]:
        columns = line.strip().split()
        protein_id = columns[0]
        protein_emb = columns[1:]
        protein_name = biogrid_dict_reverse[protein_id]
        emb_dict[protein_name] = protein_emb

key_protein_set = set()
with open('../data/essential_proteins.txt') as file:
    for line in file.readlines():
        key_protein = line.strip()
        key_protein_set.add(key_protein)

protein_matrix = []
protein_label = []
protein_emb = []
protein_centrality = []
with open('../data/GSE3431_T36.txt') as file:
    for line in file.readlines():
        columns = line.strip().split()
        protein_name = columns[0]
        features = columns[1:]
        assert len(features) == 36

        if protein_name in emb_dict:
            protein_matrix.append(features)
            protein_emb.append(emb_dict[protein_name])
            id = int(biogrid_dict[protein_name])
            centrality_list = []
            centrality_list.append(degree_centrality[id])  # 0
            centrality_list.append(closeness_centrality[id])  # 1
            centrality_list.append(eigenvector_centrality[id])  # 2
            centrality_list.append(betweenness_centrality[id])  # 3
            centrality_list.append(harmonic_centrality[id])  # 4
            centrality_list.append(load_centrality[id])  # 5
            centrality_list.append(subgraph_centrality[id])  # delete
            protein_centrality.append(centrality_list)
            print(centrality_list)
            if protein_name in key_protein_set:
                protein_label.append(1)  # 1266
            else:
                protein_label.append(0)  # 5510

protein_matrix = np.array(protein_matrix).reshape(-1, 3, 12)
np.save('./data/protein_matrix.npy', protein_matrix)
print(protein_matrix.shape)

protein_emb = np.array(protein_emb)
np.save('./data/protein_emb.npy', protein_emb)
print(protein_emb.shape)

protein_centrality = np.array(protein_centrality).reshape(-1, 7)
np.save('./data/protein_centrality.npy', protein_centrality)
print(protein_centrality.shape)

protein_label = np.array(protein_label).reshape(-1, 1)
np.save('./data/protein_label.npy', protein_label)
print(protein_label.shape)
