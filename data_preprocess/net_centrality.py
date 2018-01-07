import networkx as nx
import pickle

with open("../data/biogrid_network_bin.txt", "r") as file:
    edges = []
    for line in file.readlines():
        line = line.split()
        # print(line)
        # print(len(line))
        line[0] = int(line[0])
        line[1] = int(line[1])
        edges.append(line)

# print(edges)

G = nx.Graph()
G.add_edges_from(edges)
# net = nx.draw(G)
# plt.show(net)

degree_centrality = nx.degree_centrality(G)
print(degree_centrality)
with open('../data/degree_centrality.pkl', 'wb') as file:
    pickle.dump(degree_centrality, file)

# closeness_centrality is very waste of time
closeness_centrality = nx.closeness_centrality(G)
print(closeness_centrality)
with open('../data/closeness_centrality.pkl', 'wb') as file:
    pickle.dump(closeness_centrality, file)

eigenvector_centrality = nx.eigenvector_centrality(G)
print(eigenvector_centrality)
with open('../data/eigenvector_centrality.pkl', 'wb') as file:
    pickle.dump(eigenvector_centrality, file)

# betweenness_centrality is very waste of time
betweenness_centrality = nx.betweenness_centrality(G)
print(betweenness_centrality)
with open('../data/betweenness_centrality.pkl', 'wb') as file:
    pickle.dump(betweenness_centrality, file)

harmonic_centrality = nx.harmonic_centrality(G)
print(harmonic_centrality)
with open('../data/harmonic_centrality.pkl', 'wb') as file:
    pickle.dump(harmonic_centrality, file)

load_centrality = nx.load_centrality(G)
print(load_centrality)
with open('../data/load_centrality.pkl', 'wb') as file:
    pickle.dump(load_centrality, file)

subgraph_centrality = nx.subgraph_centrality(G)
print(subgraph_centrality)
with open('../data/subgraph_centrality.pkl', 'wb') as file:
    pickle.dump(subgraph_centrality, file)
