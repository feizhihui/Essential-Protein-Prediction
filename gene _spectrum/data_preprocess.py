# encoding=utf-8

import pickle
import numpy as np

key_protein_set = set()
with open('./data/essential_proteins.txt') as file:
    for line in file.readlines():
        key_protein = line.strip()
        key_protein_set.add(key_protein)

protein_matrix = []
protein_label = []
with open('./data/GSE3431_T36.txt') as file:
    for line in file.readlines():
        columns = line.strip().split()
        protein_name = columns[0]
        features = columns[1:]
        assert len(features) == 36
        protein_matrix.append(features)
        if protein_name in key_protein_set:
            protein_label.append(1)  # 1266
        else:
            protein_label.append(0)  # 5510

protein_matrix = np.array(protein_matrix).reshape(-1, 3, 12)
np.save('./data/protein_matrix.npy', protein_matrix)
protein_label = np.array(protein_label).reshape(-1, 1)
np.save('./data/protein_label.npy', protein_label)
