# encoding=utf-8
from sklearn.metrics import roc_curve, auc
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

plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for i, filename in enumerate(image_list):
    with open('plot_cache/' + filename, 'rb') as file:
        y_pred, y_label = pickle.load(file)
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', color=colors[i], label='%s AUC:%.3f' % (legends[i], roc_auc), lw=2)

plt.legend(loc="lower right", fontsize='medium')  # after plt.plot
plt.savefig('ROC.jpg')
plt.show()
