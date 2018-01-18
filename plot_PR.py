# encoding=utf-8
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import pickle
import numpy as np

image_list = ['multi_source.pkl',
              'total6.pkl',
              'degree_centrality.pkl',
              'closeness_centrality.pkl',
              'eigenvector_centrality.pkl',
              'betweenness_centrality.pkl',
              'harmonic_centrality.pkl',
              'load_centrality.pkl']

colors = ['green', 'red', 'cyan', 'magenta', 'lightgray', 'black', 'coral', 'blue']
legends = ['Doc2Vec',
           'Totals',
           'Degree Centrality',
           'Closeness Centrality',
           'Eigenvector Centrality',
           'Betweenness Centrality',
           'Harmonic Centrality',
           'Load Centrality']

plt.title('Precision/Recall Curve')  # give plot a title
plt.xlabel('Recall')  # make axis labels
plt.ylabel('Precision')
for i, filename in enumerate(image_list):
    with open('plot_cache/' + filename, 'rb') as file:
        y_score, y_label = pickle.load(file)
        precision, recall, _ = precision_recall_curve(y_label, y_score)
        aupr = auc(recall, precision)
        plt.plot(recall, precision, 'k--', color=colors[i], label='%s AUPR:%.3f' % (legends[i], aupr), lw=2)

plt.legend(loc="lower right", fontsize='small')  # after plt.plot

plt.savefig('PR.jpg')
plt.show()